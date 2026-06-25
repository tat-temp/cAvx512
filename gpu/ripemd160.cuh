// ripemd160.cuh -- canonical RIPEMD-160 of a single 32-byte message (a SHA-256
// digest) on the device, fused with the SHA hand-off. The 32-byte input is the
// big-endian SHA digest; RIPEMD reads little-endian words, so each SHA state word
// is byte-swapped into the message schedule (no intermediate digest store).
//
// out[0..4] are the five RIPEMD words; the 20-byte hash160 is out[i] written
// little-endian, so out[i] compares directly against a little-endian read of the
// target's bytes (see search.cuh).
#pragma once
#include <stdint.h>

namespace ripemd160 {

// message-word selection and rotation tables (left line l*, right line r*)
__constant__ uint8_t RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__constant__ uint8_t RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__constant__ uint8_t SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__constant__ uint8_t SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

__device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
// nonlinear functions, selected by block index 0..4.
__device__ __forceinline__ uint32_t f(int idx, uint32_t x, uint32_t y, uint32_t z) {
    switch (idx) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        default:return x ^ (y | ~z);
    }
}
__device__ __forceinline__ uint32_t KL(int idx) {
    const uint32_t k[5] = {0x00000000u,0x5A827999u,0x6ED9EBA1u,0x8F1BBCDCu,0xA953FD4Eu};
    return k[idx];
}
__device__ __forceinline__ uint32_t KR(int idx) {
    const uint32_t k[5] = {0x50A28BE6u,0x5C4DD124u,0x6D703EF3u,0x7A6D76E9u,0x00000000u};
    return k[idx];
}

// Fuse: X[0..7] = byteswap(SHA words); X[8]=0x80 pad; X[14]=256-bit length.
__device__ __forceinline__ void ripemd160_from_sha(const uint32_t H[8], uint32_t out[5]) {
    uint32_t X[16];
#pragma unroll
    for (int i = 0; i < 8; i++) X[i] = __byte_perm(H[i], 0, 0x0123);  // big-endian -> little
    X[8]  = 0x00000080u;
    X[9]  = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;          // message length in bits (low word, little-endian)
    X[15] = 0;

    const uint32_t h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu,
                   h3 = 0x10325476u, h4 = 0xC3D2E1F0u;
    uint32_t al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    uint32_t ar = h0, br = h1, cr = h2, dr = h3, er = h4;

#pragma unroll
    for (int j = 0; j < 80; j++) {
        int blk = j >> 4;
        uint32_t t;
        t  = rotl(al + f(blk, bl, cl, dl) + X[RL[j]] + KL(blk), SL[j]) + el;
        al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = t;

        t  = rotl(ar + f(4 - blk, br, cr, dr) + X[RR[j]] + KR(blk), SR[j]) + er;
        ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = t;
    }
    uint32_t tmp = h1 + cl + dr;
    out[1] = h2 + dl + er;
    out[2] = h3 + el + ar;
    out[3] = h4 + al + br;
    out[4] = h0 + bl + cr;
    out[0] = tmp;
}

} // namespace ripemd160
