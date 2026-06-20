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

} // namespace ifma

#endif // IFMA_FIELD_H
