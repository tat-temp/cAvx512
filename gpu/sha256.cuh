// sha256.cuh -- canonical SHA-256 of a single 33-byte message (a compressed
// public key) on the device. One 512-bit block; the padding (0x80, 264-bit length
// big-endian) is folded in as compile-time schedule words. Uses a rolling 16-word
// message schedule to keep register pressure low. Output H[0..7] are the digest
// state words as big-endian 32-bit integers (H[0] = first 4 digest bytes).
#pragma once
#include <stdint.h>

namespace sha256 {

__constant__ uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 compression over the 16 initial schedule words (rolling window; w is
// overwritten). Produces the 8 digest words (big-endian integers).
__device__ __forceinline__ void compress16(uint32_t w[16], uint32_t H[8]) {
    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint32_t s0 = rotr(w[(i-15)&15], 7) ^ rotr(w[(i-15)&15], 18) ^ (w[(i-15)&15] >> 3);
            uint32_t s1 = rotr(w[(i-2)&15], 17) ^ rotr(w[(i-2)&15], 19) ^ (w[(i-2)&15] >> 10);
            wi = w[i&15] + s0 + w[(i-7)&15] + s1;
            w[i&15] = wi;
        }
        uint32_t S1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch  = (e & f) ^ (~e & g);
        uint32_t t1  = h + S1 + ch + K[i] + wi;
        uint32_t S0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    H[0] = a + 0x6a09e667; H[1] = b + 0xbb67ae85; H[2] = c + 0x3c6ef372; H[3] = d + 0xa54ff53a;
    H[4] = e + 0x510e527f; H[5] = f + 0x9b05688c; H[6] = g + 0x1f83d9ab; H[7] = h + 0x5be0cd19;
}

// msg = 33 bytes. Produces the 8 digest words.
__device__ __forceinline__ void sha256_33(const uint8_t msg[33], uint32_t H[8]) {
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = ((uint32_t)msg[4*i]   << 24) | ((uint32_t)msg[4*i+1] << 16) |
               ((uint32_t)msg[4*i+2] << 8)  |  (uint32_t)msg[4*i+3];
    w[8]  = ((uint32_t)msg[32] << 24) | (0x80u << 16);
    w[9]  = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 264;
    compress16(w, H);
}

// Build the 16 SHA words straight from x's 4 limbs + the parity bit, for the
// compressed key (0x02|parity) || BE(x) -- no 33-byte array round-trip. wp[] are the
// big-endian 32-bit halves of x (high then low of v3..v0; the aligned message words);
// the 1-byte 0x02/0x03 prefix shifts every word right by one byte.
__device__ __forceinline__ void sha256_x(const uint64_t x[4], unsigned parity, uint32_t H[8]) {
    uint32_t wp[8];
    wp[0] = (uint32_t)(x[3] >> 32); wp[1] = (uint32_t)x[3];
    wp[2] = (uint32_t)(x[2] >> 32); wp[3] = (uint32_t)x[2];
    wp[4] = (uint32_t)(x[1] >> 32); wp[5] = (uint32_t)x[1];
    wp[6] = (uint32_t)(x[0] >> 32); wp[7] = (uint32_t)x[0];
    uint32_t w[16];
    w[0] = ((0x02u | (parity & 1u)) << 24) | (wp[0] >> 8);
#pragma unroll
    for (int i = 1; i < 8; i++) w[i] = (wp[i-1] << 24) | (wp[i] >> 8);
    w[8]  = (wp[7] << 24) | 0x00800000u;     // last x byte, then 0x80
    w[9]  = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 264;
    compress16(w, H);
}

} // namespace sha256
