// 8-lane SIMD field arithmetic for secp256k1, radix-2^52 (5 limbs x 8 lanes).
//
// Stage C.0a: representation, Int<->limbs conversion, and add/sub/neg/normalize.
// These use only AVX-512F/DQ (no IFMA) so they build with the existing flags.
// The IFMA multiply/square (C.0b) will be added once this foundation is verified
// by --selftest-ifma. Field elements are stored Structure-of-Arrays: n[k] holds
// limb k of all 8 lanes (lane j == element j). A normalized limb is < 2^52
// (n[4] < 2^48); operations here produce "weakly normalized" results whose value
// is congruent mod p and whose limbs stay well within 64 bits.
//
// p = 2^256 - 2^32 - 977,  2^256 == 0x1000003D1 (mod p).

#ifndef IFMA_FIELD_H
#define IFMA_FIELD_H

#include <immintrin.h>
#include <cstdint>
#include "Int.h"

namespace ifma {

static const uint64_t MASK52 = 0xFFFFFFFFFFFFFULL;
static const uint64_t MASK48 = 0x0FFFFFFFFFFFFULL;
static const uint64_t R256   = 0x1000003D1ULL;       // 2^256 mod p
static const uint64_t RR260  = 0x1000003D10ULL;      // 2^260 mod p (== R256 << 4)

// 2*p in radix-2^52 (a value == 0 mod p, large enough to keep sub/neg positive).
static const uint64_t TWO_P[5] = {
    0x1FFFFDFFFFF85EULL, // 2 * 0xFFFFEFFFFFC2F
    0x1FFFFFFFFFFFFEULL, // 2 * 0xFFFFFFFFFFFFF
    0x1FFFFFFFFFFFFEULL,
    0x1FFFFFFFFFFFFEULL,
    0x1FFFFFFFFFFFEULL    // 2 * 0xFFFFFFFFFFFF
};

struct FieldVec8 {
    __m512i n[5];
};

// 256-bit Int (bits64[0..3]) -> 5 radix-2^52 limbs.
static inline void int_to_limbs(const Int &a, uint64_t out[5]) {
    const uint64_t w0 = a.bits64[0], w1 = a.bits64[1], w2 = a.bits64[2], w3 = a.bits64[3];
    out[0] =  w0                        & MASK52;
    out[1] = ((w0 >> 52) | (w1 << 12))  & MASK52;
    out[2] = ((w1 >> 40) | (w2 << 24))  & MASK52;
    out[3] = ((w2 >> 28) | (w3 << 36))  & MASK52;
    out[4] =  (w3 >> 16);
}

// Pack 8 Ints (a[0..7]) into SoA lanes (lane j == a[j]).
static inline FieldVec8 load8(const Int *a) {
    uint64_t L[8][5];
    for (int j = 0; j < 8; j++) int_to_limbs(a[j], L[j]);
    FieldVec8 r;
    for (int k = 0; k < 5; k++) {
        r.n[k] = _mm512_set_epi64((long long)L[7][k], (long long)L[6][k],
                                  (long long)L[5][k], (long long)L[4][k],
                                  (long long)L[3][k], (long long)L[2][k],
                                  (long long)L[1][k], (long long)L[0][k]);
    }
    return r;
}

// Unpack to out[8][5] (out[j][k] == lane j, limb k).
static inline void store8(const FieldVec8 &v, uint64_t out[8][5]) {
    uint64_t tmp[5][8];
    for (int k = 0; k < 5; k++)
        _mm512_storeu_si512((void *)tmp[k], v.n[k]);
    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 5; k++)
            out[j][k] = tmp[k][j];
}

static inline FieldVec8 add(const FieldVec8 &a, const FieldVec8 &b) {
    FieldVec8 r;
    for (int k = 0; k < 5; k++) r.n[k] = _mm512_add_epi64(a.n[k], b.n[k]);
    return r;
}

// a - b (mod p), via a + 2p - b so every limb stays non-negative.
static inline FieldVec8 sub(const FieldVec8 &a, const FieldVec8 &b) {
    FieldVec8 r;
    for (int k = 0; k < 5; k++) {
        __m512i t = _mm512_add_epi64(a.n[k], _mm512_set1_epi64((long long)TWO_P[k]));
        r.n[k] = _mm512_sub_epi64(t, b.n[k]);
    }
    return r;
}

// -a (mod p) == 2p - a.
static inline FieldVec8 neg(const FieldVec8 &a) {
    FieldVec8 r;
    for (int k = 0; k < 5; k++)
        r.n[k] = _mm512_sub_epi64(_mm512_set1_epi64((long long)TWO_P[k]), a.n[k]);
    return r;
}

// Carry-propagate to limbs < 2^52 and fold the >=2^256 overflow back via R256.
// Result is congruent mod p (may still be >= p; canonicalization comes later).
static inline void normalize_weak(FieldVec8 &v) {
    const __m512i M52 = _mm512_set1_epi64((long long)MASK52);
    const __m512i M48 = _mm512_set1_epi64((long long)MASK48);
    const __m512i R   = _mm512_set1_epi64((long long)R256);
    __m512i c;

    c = _mm512_srli_epi64(v.n[0], 52); v.n[0] = _mm512_and_si512(v.n[0], M52); v.n[1] = _mm512_add_epi64(v.n[1], c);
    c = _mm512_srli_epi64(v.n[1], 52); v.n[1] = _mm512_and_si512(v.n[1], M52); v.n[2] = _mm512_add_epi64(v.n[2], c);
    c = _mm512_srli_epi64(v.n[2], 52); v.n[2] = _mm512_and_si512(v.n[2], M52); v.n[3] = _mm512_add_epi64(v.n[3], c);
    c = _mm512_srli_epi64(v.n[3], 52); v.n[3] = _mm512_and_si512(v.n[3], M52); v.n[4] = _mm512_add_epi64(v.n[4], c);

    // Fold bits >= 2^256 (everything above n[4]'s low 48 bits) into n[0] via R.
    __m512i over = _mm512_srli_epi64(v.n[4], 48);
    v.n[4] = _mm512_and_si512(v.n[4], M48);
    v.n[0] = _mm512_add_epi64(v.n[0], _mm512_mullo_epi64(over, R)); // AVX-512DQ

    // Second carry pass (over*R is small, so one pass settles n[0..4]).
    c = _mm512_srli_epi64(v.n[0], 52); v.n[0] = _mm512_and_si512(v.n[0], M52); v.n[1] = _mm512_add_epi64(v.n[1], c);
    c = _mm512_srli_epi64(v.n[1], 52); v.n[1] = _mm512_and_si512(v.n[1], M52); v.n[2] = _mm512_add_epi64(v.n[2], c);
    c = _mm512_srli_epi64(v.n[2], 52); v.n[2] = _mm512_and_si512(v.n[2], M52); v.n[3] = _mm512_add_epi64(v.n[3], c);
    c = _mm512_srli_epi64(v.n[3], 52); v.n[3] = _mm512_and_si512(v.n[3], M52); v.n[4] = _mm512_add_epi64(v.n[4], c);
}

// a * b (mod p), 8-lane, AVX-512 IFMA. Inputs must be weakly normalized
// (each limb < 2^52, top limb small). Output is congruent mod p with limbs
// < 2^55 (not fully canonical -- callers normalize when needed). Requires
// -mavx512ifma. p reduction uses 2^260 == RR260 (limb-5 boundary).
static inline FieldVec8 mul(const FieldVec8 &a, const FieldVec8 &b) {
    const __m512i M52 = _mm512_set1_epi64((long long)MASK52);
    const __m512i RR  = _mm512_set1_epi64((long long)RR260);
    const __m512i Z   = _mm512_setzero_si512();

    // Phase 1: schoolbook 5x5 -> 10 columns (each accumulates <=9 terms < 2^52).
    __m512i d[10];
    for (int i = 0; i < 10; i++) d[i] = Z;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            d[i + j]     = _mm512_madd52lo_epu64(d[i + j],     a.n[i], b.n[j]);
            d[i + j + 1] = _mm512_madd52hi_epu64(d[i + j + 1], a.n[i], b.n[j]);
        }
    }

    // Phase 2: carry-propagate so every d[k] < 2^52 (required as fold multiplicands).
    __m512i c;
    c = _mm512_srli_epi64(d[0], 52); d[0] = _mm512_and_si512(d[0], M52); d[1] = _mm512_add_epi64(d[1], c);
    c = _mm512_srli_epi64(d[1], 52); d[1] = _mm512_and_si512(d[1], M52); d[2] = _mm512_add_epi64(d[2], c);
    c = _mm512_srli_epi64(d[2], 52); d[2] = _mm512_and_si512(d[2], M52); d[3] = _mm512_add_epi64(d[3], c);
    c = _mm512_srli_epi64(d[3], 52); d[3] = _mm512_and_si512(d[3], M52); d[4] = _mm512_add_epi64(d[4], c);
    c = _mm512_srli_epi64(d[4], 52); d[4] = _mm512_and_si512(d[4], M52); d[5] = _mm512_add_epi64(d[5], c);
    c = _mm512_srli_epi64(d[5], 52); d[5] = _mm512_and_si512(d[5], M52); d[6] = _mm512_add_epi64(d[6], c);
    c = _mm512_srli_epi64(d[6], 52); d[6] = _mm512_and_si512(d[6], M52); d[7] = _mm512_add_epi64(d[7], c);
    c = _mm512_srli_epi64(d[7], 52); d[7] = _mm512_and_si512(d[7], M52); d[8] = _mm512_add_epi64(d[8], c);
    c = _mm512_srli_epi64(d[8], 52); d[8] = _mm512_and_si512(d[8], M52); d[9] = _mm512_add_epi64(d[9], c);

    // Phase 3: fold high limbs d[5..9] into low via 2^260 == RR.
    //   d[5+m] contributes lo52(d[5+m]*RR) to column m and hi52 to column m+1.
    __m512i r0 = d[0], r1 = d[1], r2 = d[2], r3 = d[3], r4 = d[4], r5 = Z;
    r0 = _mm512_madd52lo_epu64(r0, d[5], RR); r1 = _mm512_madd52hi_epu64(r1, d[5], RR);
    r1 = _mm512_madd52lo_epu64(r1, d[6], RR); r2 = _mm512_madd52hi_epu64(r2, d[6], RR);
    r2 = _mm512_madd52lo_epu64(r2, d[7], RR); r3 = _mm512_madd52hi_epu64(r3, d[7], RR);
    r3 = _mm512_madd52lo_epu64(r3, d[8], RR); r4 = _mm512_madd52hi_epu64(r4, d[8], RR);
    r4 = _mm512_madd52lo_epu64(r4, d[9], RR); r5 = _mm512_madd52hi_epu64(r5, d[9], RR);

    // Phase 4: fold r5 (weight 2^260) back into r0/r1.
    r0 = _mm512_madd52lo_epu64(r0, r5, RR);
    r1 = _mm512_madd52hi_epu64(r1, r5, RR);

    FieldVec8 r;
    r.n[0] = r0; r.n[1] = r1; r.n[2] = r2; r.n[3] = r3; r.n[4] = r4;
    normalize_weak(r);   // limbs < 2^52 so the result can feed another mul/sqr
    return r;
}

// a^2 (mod p). First-correct version: just mul(a, a). (A dedicated squaring that
// halves the cross-product count is a later optimization.)
static inline FieldVec8 sqr(const FieldVec8 &a) { return mul(a, a); }

// Broadcast one Int field element to all 8 lanes.
static inline FieldVec8 broadcast(const Int &a) {
    uint64_t L[5]; int_to_limbs(a, L);
    FieldVec8 r;
    for (int k = 0; k < 5; k++) r.n[k] = _mm512_set1_epi64((long long)L[k]);
    return r;
}

// Compute 8 EC points  P = base (+/-) G  in affine coords, lane-wise:
//   spx,spy = base.x, base.y (broadcast to all lanes)
//   gx,gy   = G.x, G.y for 8 different points G (per lane)
//   inv     = 1/(G.x - base.x) per lane (precomputed batch inverse)
//   plus    = true for base+G, false for base-G ( -G == (G.x, -G.y) )
// Mirrors the scalar slope formula used by the search loop. All add/sub/neg
// outputs are re-normalized before feeding a mul (mul/sqr self-normalize).
static inline void gen8(const FieldVec8 &spx, const FieldVec8 &spy,
                        const FieldVec8 &gx, const FieldVec8 &gy,
                        const FieldVec8 &inv, bool plus,
                        FieldVec8 &rx, FieldVec8 &ry) {
    FieldVec8 dy;
    if (plus) {
        dy = sub(gy, spy);                 // dy = G.y - base.y
    } else {
        FieldVec8 ng = neg(gy); normalize_weak(ng);
        dy = sub(ng, spy);                 // dy = -G.y - base.y
    }
    normalize_weak(dy);

    FieldVec8 s  = mul(dy, inv);           // s = dy / (G.x - base.x)
    FieldVec8 p2 = sqr(s);                 // s^2

    rx = sub(p2, spx);
    rx = sub(rx, gx);                      // rx = s^2 - base.x - G.x
    normalize_weak(rx);

    FieldVec8 t = sub(gx, rx);             // G.x - rx
    normalize_weak(t);
    FieldVec8 ts = mul(t, s);              // s * (G.x - rx)

    if (plus) ry = sub(ts, gy);           // ry = s*(G.x-rx) - G.y
    else      ry = add(ts, gy);           // ry = s*(G.x-rx) + G.y
    normalize_weak(ry);
}

} // namespace ifma

#endif // IFMA_FIELD_H
