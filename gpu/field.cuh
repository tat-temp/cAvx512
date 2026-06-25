// field.cuh -- secp256k1 prime-field arithmetic for the device.
//
// Fp with p = 2^256 - 2^32 - 977 (the secp256k1 field prime, matching
// SECP256K1.cpp's Int::SetupField). Elements are 4 x 64-bit little-endian limbs
// (v[0] = least significant) and are kept fully reduced in [0, p) after every op,
// so v[0]&1 is the true parity and the limbs serialize directly.
//
// Multiplication is schoolbook 256x256 -> 512 (operand scanning with __umul64hi)
// followed by the secp256k1 fast reduction: 2^256 == C (mod p), C = 2^32 + 977,
// so a 512-bit product folds in two passes (high*C + low) into < 2p, then a single
// conditional subtract of p. The modular inverse is Fermat (a^(p-2)) via the
// libsecp256k1 addition chain. Every routine here is validated lane-for-lane in
// cyclone_gpu.cu --selftest against an independent host big-integer oracle that
// reduces by generic long division, so the fast fold is cross-checked.
#pragma once
#include <stdint.h>

namespace fe {

struct Elem { uint64_t v[4]; };   // little-endian, [0, p)

// p limbs (compile-time immediates -> no constant-memory loads on the hot path).
#define FE_P0 0xFFFFFFFEFFFFFC2FULL
#define FE_P1 0xFFFFFFFFFFFFFFFFULL
#define FE_P2 0xFFFFFFFFFFFFFFFFULL
#define FE_P3 0xFFFFFFFFFFFFFFFFULL
// C = 2^256 mod p = 2^32 + 977.
#define FE_C  0x1000003D1ULL

// --- low-level multi-limb primitives ----------------------------------------

__device__ __forceinline__ void mul64(uint64_t a, uint64_t b,
                                       uint64_t &lo, uint64_t &hi) {
    lo = a * b;
    hi = __umul64hi(a, b);
}

// r[0..3] = a + b, returns carry-out (0/1).
__device__ __forceinline__ uint64_t add4(const uint64_t a[4], const uint64_t b[4],
                                          uint64_t r[4]) {
    uint64_t carry = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t s = a[k] + b[k];
        uint64_t c = (s < a[k]);
        s += carry;
        c += (s < carry);
        r[k] = s;
        carry = c;
    }
    return carry;
}

// r[0..3] = a - b, returns borrow-out (0/1).
__device__ __forceinline__ uint64_t sub4(const uint64_t a[4], const uint64_t b[4],
                                          uint64_t r[4]) {
    uint64_t borrow = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t d  = a[k] - b[k];
        uint64_t br = (a[k] < b[k]);
        uint64_t d2 = d - borrow;
        br += (d < borrow);
        r[k] = d2;
        borrow = br;
    }
    return borrow;
}

// 256x256 -> 512 schoolbook on 8x32-bit limbs. Each partial product is
// (uint64)(uint32 * uint32), which nvcc emits as a single native mul.wide.u32
// (IMAD.WIDE) -- far cheaper than the emulated mul.hi.u64 a 4x64 schoolbook needs.
// 64-bit accumulation handles carries; (a*b + limb + carry) <= 2^64-1, no overflow.
// Output is repacked to 4 x 64-bit limbs for the (unchanged) 64-bit reduce().
__device__ __forceinline__ void mul256(const uint64_t a[4], const uint64_t b[4],
                                        uint64_t r[8]) {
    uint32_t A[8], B[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        A[2*k] = (uint32_t)a[k]; A[2*k+1] = (uint32_t)(a[k] >> 32);
        B[2*k] = (uint32_t)b[k]; B[2*k+1] = (uint32_t)(b[k] >> 32);
    }
    uint32_t T[16];
#pragma unroll
    for (int k = 0; k < 16; k++) T[k] = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t cur = (uint64_t)A[i] * B[j] + T[i + j] + carry;
            T[i + j] = (uint32_t)cur; carry = cur >> 32;
        }
        T[i + 8] = (uint32_t)carry;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) r[k] = (uint64_t)T[2*k] | ((uint64_t)T[2*k+1] << 32);
}

// 256-bit squaring -> 512 on 8x32-bit limbs, branchless (no data-dependent ripple).
// Native mul.wide.u32 partial products (see mul256). Upper-triangle cross products
// via operand scanning, doubled, then diagonal squares a_i^2 at limb 2i added with a
// single fixed carry sweep (each diagonal spans limbs 2i,2i+1, so one carry threads
// into the next). 28 cross + 8 diagonal = 36 native 32-bit muls. Output repacked to
// 4 x 64-bit for the (unchanged) 64-bit reduce().
__device__ __forceinline__ void sqr256(const uint64_t a[4], uint64_t r[8]) {
    uint32_t A[8];
#pragma unroll
    for (int k = 0; k < 4; k++) { A[2*k] = (uint32_t)a[k]; A[2*k+1] = (uint32_t)(a[k] >> 32); }
    uint32_t cr[16];
#pragma unroll
    for (int k = 0; k < 16; k++) cr[k] = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
#pragma unroll
        for (int j = i + 1; j < 8; j++) {
            uint64_t cur = (uint64_t)A[i] * A[j] + cr[i + j] + carry;
            cr[i + j] = (uint32_t)cur; carry = cur >> 32;
        }
        cr[i + 8] = (uint32_t)carry;
    }
    uint32_t T[16];
    uint32_t hi = 0;                        // T = 2 * cr
#pragma unroll
    for (int k = 0; k < 16; k++) { uint32_t nc = cr[k] >> 31; T[k] = (cr[k] << 1) | hi; hi = nc; }
    uint64_t carry = 0;                     // + diagonal squares a_i^2 at limb 2i
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t sq = (uint64_t)A[i] * A[i];
        uint64_t cur = (uint64_t)T[2*i] + (uint32_t)sq + carry;
        T[2*i] = (uint32_t)cur;
        cur = (uint64_t)T[2*i + 1] + (uint32_t)(sq >> 32) + (uint32_t)(cur >> 32);
        T[2*i + 1] = (uint32_t)cur;
        carry = (uint32_t)(cur >> 32);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) r[k] = (uint64_t)T[2*k] | ((uint64_t)T[2*k+1] << 32);
}

// value (carry:r) is < 2p; subtract p once (branchless) to land in [0, p).
__device__ __forceinline__ void csub_p(uint64_t r[4], uint64_t hicarry) {
    const uint64_t Pl[4] = {FE_P0, FE_P1, FE_P2, FE_P3};
    uint64_t t[4];
    uint64_t borrow = sub4(r, Pl, t);
    // value >= p  iff  hicarry==1 (value >= 2^256 > p)  OR  borrow==0 (r >= p).
    uint64_t ge   = hicarry | (borrow ^ 1);   // 0/1
    uint64_t mask = (uint64_t)0 - ge;         // all-ones if ge
#pragma unroll
    for (int k = 0; k < 4; k++)
        r[k] = (r[k] & ~mask) | (t[k] & mask);
}

// 512 -> 256 mod p via the secp256k1 two-pass fold.
__device__ __forceinline__ void reduce(const uint64_t t[8], uint64_t r[4]) {
    // fold1: m = t[hi] * C  (4-limb x 64-bit -> 5 limbs), then s = t[lo] + m.
    uint64_t m[5];
    {
        unsigned __int128 carry = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned __int128 cur = (unsigned __int128)t[4 + k] * FE_C + carry;
            m[k] = (uint64_t)cur; carry = cur >> 64;
        }
        m[4] = (uint64_t)carry;       // < 2^34
    }
    uint64_t s[5];
    {
        unsigned __int128 carry = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned __int128 cur = (unsigned __int128)t[k] + m[k] + carry;
            s[k] = (uint64_t)cur; carry = cur >> 64;
        }
        s[4] = m[4] + (uint64_t)carry; // < 2^34
    }
    // fold2: r = s[lo] + s[4]*C   (s[4]*C < 2^67)
    unsigned __int128 sc = (unsigned __int128)s[4] * FE_C;   // < 2^67
    uint64_t lo = (uint64_t)sc, hi = (uint64_t)(sc >> 64);   // hi < 8
    unsigned __int128 cur = (unsigned __int128)s[0] + lo;            r[0] = (uint64_t)cur;
    cur = (unsigned __int128)s[1] + hi + (uint64_t)(cur >> 64);      r[1] = (uint64_t)cur;
    cur = (unsigned __int128)s[2] + (uint64_t)(cur >> 64);           r[2] = (uint64_t)cur;
    cur = (unsigned __int128)s[3] + (uint64_t)(cur >> 64);           r[3] = (uint64_t)cur;
    csub_p(r, (uint64_t)(cur >> 64));                 // value (carry:r) < 2p
}

// --- field ops --------------------------------------------------------------

__device__ __forceinline__ Elem fe_add(const Elem &a, const Elem &b) {
    Elem r;
    uint64_t carry = add4(a.v, b.v, r.v);
    csub_p(r.v, carry);
    return r;
}

__device__ __forceinline__ Elem fe_sub(const Elem &a, const Elem &b) {
    Elem r;
    uint64_t borrow = sub4(a.v, b.v, r.v);
    // if a < b, add p back (a - b + p in [0, p); discard the carry-out).
    const uint64_t Pl[4] = {FE_P0, FE_P1, FE_P2, FE_P3};
    uint64_t mask = (uint64_t)0 - borrow;
    uint64_t addend[4] = {Pl[0] & mask, Pl[1] & mask, Pl[2] & mask, Pl[3] & mask};
    add4(r.v, addend, r.v);
    return r;
}

__device__ __forceinline__ Elem fe_neg(const Elem &a) {
    Elem z = {{0, 0, 0, 0}};
    return fe_sub(z, a);
}

__device__ __forceinline__ Elem fe_mul(const Elem &a, const Elem &b) {
    uint64_t t[8];
    mul256(a.v, b.v, t);
    Elem r;
    reduce(t, r.v);
    return r;
}

__device__ __forceinline__ Elem fe_sqr(const Elem &a) {
    uint64_t t[8];
    sqr256(a.v, t);
    Elem r;
    reduce(t, r.v);
    return r;
}

__device__ __forceinline__ Elem fe_sqrn(Elem a, int n) {
#pragma unroll 1
    for (int i = 0; i < n; i++) a = fe_sqr(a);
    return a;
}

// Modular inverse: a^(p-2) mod p via the libsecp256k1 addition chain.
// (p-2) is 5 runs of 1-bits with block lengths drawn from {1, 2, 22, 223}.
// Deliberately NOT force-inlined: it is large (~255 squarings) and called once per
// batch, so keeping it out of line frees a lot of registers in k_step (raising
// occupancy) at negligible call-overhead cost.
__device__ __noinline__ Elem fe_inv(const Elem &a) {
    Elem x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    x2 = fe_mul(fe_sqr(a), a);                 // a^(2^2-1)
    x3 = fe_mul(fe_sqr(x2), a);                // a^(2^3-1)

    x6  = fe_mul(fe_sqrn(x3, 3), x3);          // a^(2^6-1)
    x9  = fe_mul(fe_sqrn(x6, 3), x3);          // a^(2^9-1)
    x11 = fe_mul(fe_sqrn(x9, 2), x2);          // a^(2^11-1)
    x22 = fe_mul(fe_sqrn(x11, 11), x11);       // a^(2^22-1)
    x44 = fe_mul(fe_sqrn(x22, 22), x22);       // a^(2^44-1)
    x88 = fe_mul(fe_sqrn(x44, 44), x44);       // a^(2^88-1)
    x176 = fe_mul(fe_sqrn(x88, 88), x88);      // a^(2^176-1)
    x220 = fe_mul(fe_sqrn(x176, 44), x44);     // a^(2^220-1)
    x223 = fe_mul(fe_sqrn(x220, 3), x3);       // a^(2^223-1)

    t = fe_mul(fe_sqrn(x223, 23), x22);
    t = fe_mul(fe_sqrn(t, 5), a);
    t = fe_mul(fe_sqrn(t, 3), x2);
    t = fe_mul(fe_sqrn(t, 2), a);
    return t;
}

// --- small helpers ----------------------------------------------------------

__device__ __forceinline__ Elem fe_set_u64(uint64_t x) {
    Elem r = {{x, 0, 0, 0}};
    return r;
}
__device__ __forceinline__ bool fe_is_zero(const Elem &a) {
    return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0;
}
__device__ __forceinline__ bool fe_eq(const Elem &a, const Elem &b) {
    return ((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) |
            (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3])) == 0;
}
__device__ __forceinline__ unsigned fe_is_odd(const Elem &a) {
    return (unsigned)(a.v[0] & 1ULL);
}

// Big-endian serialization: out[0] is the most-significant byte (limb v[3] high).
__device__ __forceinline__ void fe_to_bytes_be(const Elem &a, uint8_t out[32]) {
#pragma unroll
    for (int limb = 0; limb < 4; limb++) {
        uint64_t w = a.v[3 - limb];
#pragma unroll
        for (int b = 0; b < 8; b++)
            out[limb * 8 + b] = (uint8_t)(w >> (56 - 8 * b));
    }
}
__device__ __forceinline__ Elem fe_from_bytes_be(const uint8_t in[32]) {
    Elem r;
#pragma unroll
    for (int limb = 0; limb < 4; limb++) {
        uint64_t w = 0;
#pragma unroll
        for (int b = 0; b < 8; b++)
            w = (w << 8) | (uint64_t)in[limb * 8 + b];
        r.v[3 - limb] = w;
    }
    return r;
}

} // namespace fe
