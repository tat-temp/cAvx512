// u256.h -- tiny host-side 256-bit unsigned integer (little-endian, 4x64). Just
// enough for the search: parse/format hex private keys, add the per-launch /
// per-thread / in-batch offsets to rangeStart to reconstruct a found key, compute
// the range size, and compare against the range end. No intrinsics -- replaces the
// AVX-leaning Int for the GPU tool's host side.
#pragma once
#include <stdint.h>
#include <string>
#include <stdexcept>
#include <cctype>

struct u256 {
    uint64_t v[4];   // v[0] = least significant

    u256() { v[0] = v[1] = v[2] = v[3] = 0; }
    explicit u256(uint64_t x) { v[0] = x; v[1] = v[2] = v[3] = 0; }

    static u256 from_hex(const std::string &hexIn) {
        std::string hex = hexIn;
        if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
            hex = hex.substr(2);
        if (hex.empty()) hex = "0";
        if (hex.size() > 64) throw std::invalid_argument("range value exceeds 256 bits");
        u256 r;
        // Parse right-to-left in 16-hex-digit (64-bit) chunks.
        int limb = 0;
        for (int end = (int)hex.size(); end > 0 && limb < 4; end -= 16, limb++) {
            int start = end - 16; if (start < 0) start = 0;
            uint64_t w = 0;
            for (int i = start; i < end; i++) {
                char c = hex[i];
                uint64_t d;
                if (c >= '0' && c <= '9') d = c - '0';
                else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
                else throw std::invalid_argument("invalid hex digit in range");
                w = (w << 4) | d;
            }
            r.v[limb] = w;
        }
        return r;
    }

    // 64-char zero-padded uppercase hex (matches the CPU tool's key output).
    std::string to_hex64() const {
        static const char *H = "0123456789ABCDEF";
        std::string s(64, '0');
        for (int limb = 0; limb < 4; limb++) {
            uint64_t w = v[limb];
            int base = (3 - limb) * 16;
            for (int nib = 0; nib < 16; nib++)
                s[base + nib] = H[(w >> (60 - 4 * nib)) & 0xF];
        }
        return s;
    }

    // 32-byte big-endian (out[0] = most significant).
    void to_bytes_be(uint8_t out[32]) const {
        for (int limb = 0; limb < 4; limb++) {
            uint64_t w = v[3 - limb];
            for (int b = 0; b < 8; b++) out[limb * 8 + b] = (uint8_t)(w >> (56 - 8 * b));
        }
    }
};

static inline u256 add(const u256 &a, const u256 &b) {
    u256 r; uint64_t carry = 0;
    for (int k = 0; k < 4; k++) {
        uint64_t s = a.v[k] + b.v[k];
        uint64_t c = (s < a.v[k]);
        s += carry; c += (s < carry);
        r.v[k] = s; carry = c;
    }
    return r;
}

static inline u256 add_u64(const u256 &a, uint64_t x) {
    u256 b(x);
    return add(a, b);
}

// a * m (m small; result truncated to 256 bits, fine for in-range offsets).
static inline u256 mul_u32(const u256 &a, uint32_t m) {
    u256 r; unsigned __int128 carry = 0;
    for (int k = 0; k < 4; k++) {
        unsigned __int128 cur = (unsigned __int128)a.v[k] * m + carry;
        r.v[k] = (uint64_t)cur; carry = cur >> 64;
    }
    return r;
}

// a / m (m small), quotient only (used to split the range across GPUs).
static inline u256 div_u32(const u256 &a, uint32_t m) {
    u256 q; unsigned __int128 rem = 0;
    for (int k = 3; k >= 0; k--) {
        unsigned __int128 cur = (rem << 64) | a.v[k];
        q.v[k] = (uint64_t)(cur / m); rem = cur % m;
    }
    return q;
}

// a - b (assumes a >= b).
static inline u256 sub(const u256 &a, const u256 &b) {
    u256 r; uint64_t borrow = 0;
    for (int k = 0; k < 4; k++) {
        uint64_t d = a.v[k] - b.v[k];
        uint64_t br = (a.v[k] < b.v[k]);
        uint64_t d2 = d - borrow; br += (d < borrow);
        r.v[k] = d2; borrow = br;
    }
    return r;
}

// -1 if a<b, 0 if a==b, 1 if a>b.
static inline int cmp(const u256 &a, const u256 &b) {
    for (int k = 3; k >= 0; k--) {
        if (a.v[k] < b.v[k]) return -1;
        if (a.v[k] > b.v[k]) return 1;
    }
    return 0;
}

// Approximate magnitude as a long double (for progress %, like the CPU tool).
static inline long double to_long_double(const u256 &a) {
    long double r = 0.0L;
    for (int k = 3; k >= 0; k--) r = r * 18446744073709551616.0L + (long double)a.v[k];
    return r;
}
