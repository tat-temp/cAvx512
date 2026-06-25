// host_hash.h -- intrinsic-free host helpers for the GPU tool: a scalar SHA-256,
// base58 encode/decode, P2PKH address <-> hash160, and WIF. Replaces the AVX-using
// p2pkh_decoder for the self-contained build. Used off the hot path (target decode,
// found-key output, address round-trip verification), so clarity over speed.
#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace hosth {

// --- scalar SHA-256 (arbitrary length) --------------------------------------
struct SHA256 {
    static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    static std::vector<uint8_t> hash(const std::vector<uint8_t> &msg) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t H[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                         0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        std::vector<uint8_t> m = msg;
        uint64_t bitlen = (uint64_t)msg.size() * 8;
        m.push_back(0x80);
        while (m.size() % 64 != 56) m.push_back(0x00);
        for (int i = 7; i >= 0; i--) m.push_back((uint8_t)(bitlen >> (8 * i)));

        for (size_t off = 0; off < m.size(); off += 64) {
            uint32_t w[64];
            for (int i = 0; i < 16; i++)
                w[i] = ((uint32_t)m[off+4*i] << 24) | ((uint32_t)m[off+4*i+1] << 16) |
                       ((uint32_t)m[off+4*i+2] << 8) | (uint32_t)m[off+4*i+3];
            for (int i = 16; i < 64; i++) {
                uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
                uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
                w[i] = w[i-16] + s0 + w[i-7] + s1;
            }
            uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
            for (int i = 0; i < 64; i++) {
                uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
                uint32_t ch = (e&f)^(~e&g);
                uint32_t t1 = h + S1 + ch + K[i] + w[i];
                uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
                uint32_t maj = (a&b)^(a&c)^(b&c);
                uint32_t t2 = S0 + maj;
                h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
            }
            H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
        }
        std::vector<uint8_t> out(32);
        for (int i = 0; i < 8; i++) {
            out[4*i]   = (uint8_t)(H[i] >> 24); out[4*i+1] = (uint8_t)(H[i] >> 16);
            out[4*i+2] = (uint8_t)(H[i] >> 8);  out[4*i+3] = (uint8_t)(H[i]);
        }
        return out;
    }
};

static inline std::vector<uint8_t> sha256d(const std::vector<uint8_t> &d) {
    return SHA256::hash(SHA256::hash(d));
}

// --- base58 (port of p2pkh_decoder, intrinsic-free) -------------------------
static const std::string B58 =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static inline std::vector<uint8_t> base58_decode(const std::string &input) {
    std::vector<uint8_t> bytes(1, 0);
    for (char ch : input) {
        size_t idx = B58.find(ch);
        if (idx == std::string::npos) throw std::invalid_argument("invalid base58 char");
        int carry = (int)idx;
        for (size_t i = 0; i < bytes.size(); i++) {
            carry += 58 * bytes[i];
            bytes[i] = carry & 0xFF;
            carry >>= 8;
        }
        while (carry > 0) { bytes.push_back(carry & 0xFF); carry >>= 8; }
    }
    size_t zeros = 0;
    for (char ch : input) { if (ch == '1') zeros++; else break; }
    std::reverse(bytes.begin(), bytes.end());
    while (!bytes.empty() && bytes[0] == 0) bytes.erase(bytes.begin());
    std::vector<uint8_t> out(zeros, 0x00);
    out.insert(out.end(), bytes.begin(), bytes.end());
    return out;
}

static inline std::string base58_encode(const std::vector<uint8_t> &input) {
    std::vector<int> digits(1, 0);
    for (uint8_t byte : input) {
        int carry = byte;
        for (size_t i = 0; i < digits.size(); i++) {
            carry += digits[i] * 256;
            digits[i] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) { digits.push_back(carry % 58); carry /= 58; }
    }
    size_t zeros = 0;
    for (uint8_t byte : input) { if (byte == 0x00) zeros++; else break; }
    std::string s(zeros, '1');
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) s += B58[*it];
    return s;
}

// P2PKH Base58Check address -> 20-byte hash160 (verifies version + checksum).
static inline std::vector<uint8_t> address_to_hash160(const std::string &addr) {
    std::vector<uint8_t> dec = base58_decode(addr);
    if (dec.size() != 25) throw std::invalid_argument("bad decoded length");
    std::vector<uint8_t> data(dec.begin(), dec.begin() + 21);
    std::vector<uint8_t> checksum(dec.begin() + 21, dec.end());
    std::vector<uint8_t> chk = sha256d(data);
    if (!std::equal(chk.begin(), chk.begin() + 4, checksum.begin()))
        throw std::invalid_argument("bad checksum");
    if (data[0] != 0x00) throw std::invalid_argument("not a mainnet P2PKH address");
    return std::vector<uint8_t>(data.begin() + 1, data.end());
}

static inline std::string hash160_to_address(const uint8_t h160[20]) {
    std::vector<uint8_t> payload(1, 0x00);
    payload.insert(payload.end(), h160, h160 + 20);
    std::vector<uint8_t> chk = sha256d(payload);
    payload.insert(payload.end(), chk.begin(), chk.begin() + 4);
    return base58_encode(payload);
}

// WIF (compressed) from a 32-byte big-endian private key.
static inline std::string to_wif(const uint8_t priv[32], bool compressed) {
    std::vector<uint8_t> w(1, 0x80);
    w.insert(w.end(), priv, priv + 32);
    if (compressed) w.push_back(0x01);
    std::vector<uint8_t> chk = sha256d(w);
    w.insert(w.end(), chk.begin(), chk.begin() + 4);
    return base58_encode(w);
}

} // namespace hosth
