// hash160.cuh -- hash160(compressed pubkey) = RIPEMD160(SHA256(33-byte key)) on
// the device, producing the 5 RIPEMD words (compare directly to the target words).
#pragma once
#include "field.cuh"
#include "ec.cuh"
#include "sha256.cuh"
#include "ripemd160.cuh"

namespace h160 {

// Serialize an affine point to its 33-byte compressed form: 0x02/0x03 || BE(x).
__device__ __forceinline__ void compressed(const ec::PointA &P, uint8_t out[33]) {
    out[0] = 0x02 | (uint8_t)fe::fe_is_odd(P.y);
    fe::fe_to_bytes_be(P.x, &out[1]);
}

// hash160 of an affine pubkey -> out[0..4] (little-endian words). Feeds x's limbs
// straight into SHA (sha256_x), skipping the 33-byte compressed-key array.
__device__ __forceinline__ void hash160_point(const ec::PointA &P, uint32_t out[5]) {
    uint32_t H[8];
    sha256::sha256_x(P.x.v, fe::fe_is_odd(P.y), H);
    ripemd160::ripemd160_from_sha(H, out);
}

// hash160 directly from x bytes (big-endian) + y parity, avoiding a PointA copy.
__device__ __forceinline__ void hash160_xparity(const uint8_t xbe[32], unsigned yodd,
                                                uint32_t out[5]) {
    uint8_t msg[33];
    msg[0] = 0x02 | (uint8_t)(yodd & 1u);
#pragma unroll
    for (int i = 0; i < 32; i++) msg[1 + i] = xbe[i];
    uint32_t H[8];
    sha256::sha256_33(msg, H);
    ripemd160::ripemd160_from_sha(H, out);
}

} // namespace h160
