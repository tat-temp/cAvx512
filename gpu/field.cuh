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

// 256x256 -> 512 schoolbook (operand scanning) via unsigned __int128, so nvcc
// emits native mad.lo/hi.cc + addc carry chains. The running (a*b + limb + carry)
// total is <= 2^128-1, so the high word never overflows.
__device__ __forceinline__ void mul256(const uint64_t a[4], const uint64_t b[4],
                                        uint64_t r[8]) {
#pragma unroll
    for (int k = 0; k < 8; k++) r[k] = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned __int128 cur = (unsigned __int128)a[i] * b[j]
                                  + (unsigned __int128)r[i + j] + carry;
            r[i + j] = (uint64_t)cur;
            carry = cur >> 64;
        }
        r[i + 4] = (uint64_t)carry;
    }
}

// 256-bit squaring -> 512, BRANCHLESS (no data-dependent carry ripple, so no warp
// divergence). Upper-triangle cross products via operand scanning -- carries flow
// in the fixed, fully-unrolled inner loops -- then double, then add the diagonal
// squares with a single fixed carry sweep (each diagonal occupies limbs 2i,2i+1, so
// one carry threads straight into the next). 10 multiplies, same as the prior
// triangle form, but with zero data-dependent control flow. Square is < 2^512 and
// only additions accumulate, so the top limb never overflows.
__device__ __forceinline__ void sqr256(const uint64_t a[4], uint64_t t[8]) {
    uint64_t cr[8];
#pragma unroll
    for (int k = 0; k < 8; k++) cr[k] = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
#pragma unroll
        for (int j = i + 1; j < 4; j++) {
            unsigned __int128 cur = (unsigned __int128)a[i] * a[j] + cr[i + j] + carry;
            cr[i + j] = (uint64_t)cur;
            carry = cur >> 64;
        }
        cr[i + 4] = (uint64_t)carry;
    }
    uint64_t hi = 0;                        // t = 2 * cr
#pragma unroll
    for (int k = 0; k < 8; k++) { uint64_t nc = cr[k] >> 63; t[k] = (cr[k] << 1) | hi; hi = nc; }
    uint64_t carry = 0;                     // + diagonal squares a_i^2 at limb 2i
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned __int128 sq = (unsigned __int128)a[i] * a[i];
        unsigned __int128 cur = (unsigned __int128)t[2 * i] + (uint64_t)sq + carry;
        t[2 * i] = (uint64_t)cur;
        cur = (unsigned __int128)t[2 * i + 1] + (uint64_t)(sq >> 64) + (uint64_t)(cur >> 64);
        t[2 * i + 1] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
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
    // Strength-reduce the constant multiply: C = 2^32 + 977, so
    //   m = t_hi*C = (t_hi << 32) + t_hi*977   (shift is free; 977 is 10-bit).
    uint64_t m[5];
    {
        uint64_t p[5];                // t_hi * 977
        unsigned __int128 carry = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned __int128 cur = (unsigned __int128)t[4 + k] * 977u + carry;
            p[k] = (uint64_t)cur; carry = cur >> 64;
        }
        p[4] = (uint64_t)carry;       // < 2^10
        uint64_t s0 =  t[4] << 32;                       // t_hi << 32 (5 limbs)
        uint64_t s1 = (t[5] << 32) | (t[4] >> 32);
        uint64_t s2 = (t[6] << 32) | (t[5] >> 32);
        uint64_t s3 = (t[7] << 32) | (t[6] >> 32);
        uint64_t s4 =  t[7] >> 32;
        unsigned __int128 c = 0;                          // m = p + (t_hi << 32)
        c += (unsigned __int128)p[0] + s0; m[0] = (uint64_t)c; c >>= 64;
        c += (unsigned __int128)p[1] + s1; m[1] = (uint64_t)c; c >>= 64;
        c += (unsigned __int128)p[2] + s2; m[2] = (uint64_t)c; c >>= 64;
        c += (unsigned __int128)p[3] + s3; m[3] = (uint64_t)c; c >>= 64;
        m[4] = p[4] + s4 + (uint64_t)c;   // < 2^33
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
