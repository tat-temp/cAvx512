#include "sha256_avx512.h"
#include <immintrin.h>
#include <stdint.h>
#include <cstring>

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 alignas(64)
#endif

#define ROR(x,n) _mm512_or_epi32(_mm512_srli_epi32((x),(n)), _mm512_slli_epi32((x),(32-(n))))
#define SHR(x,n) _mm512_srli_epi32((x),(n))
#define Ch(x,y,z)  _mm512_xor_epi32(_mm512_and_epi32((x),(y)), _mm512_andnot_si512((x),(z)))
#define Maj(x,y,z) _mm512_or_epi32(_mm512_and_epi32((x),(y)), _mm512_and_epi32((z),_mm512_or_epi32((x),(y))))
#define S0(x) _mm512_xor_epi32(ROR((x),2),  _mm512_xor_epi32(ROR((x),13), ROR((x),22)))
#define S1(x) _mm512_xor_epi32(ROR((x),6),  _mm512_xor_epi32(ROR((x),11), ROR((x),25)))
#define s0(x) _mm512_xor_epi32(ROR((x),7),  _mm512_xor_epi32(ROR((x),18), SHR((x),3)))
#define s1(x) _mm512_xor_epi32(ROR((x),17), _mm512_xor_epi32(ROR((x),19), SHR((x),10)))

static const uint32_t K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
  0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
  0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
  0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
  0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
  0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
  0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
  0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
  0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Compile-time SHA-256 primitives, used to fold the constant parts of a fixed
// 33-byte-message block (see sha256_compress_msg33). The compiler evaluates
// these, so the folded constants provably match the runtime kernel.
namespace {
constexpr uint32_t crotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
constexpr uint32_t cBigS0(uint32_t x) { return crotr(x,2) ^ crotr(x,13) ^ crotr(x,22); }
constexpr uint32_t cBigS1(uint32_t x) { return crotr(x,6) ^ crotr(x,11) ^ crotr(x,25); }
constexpr uint32_t cSmS1(uint32_t x)  { return crotr(x,17) ^ crotr(x,19) ^ (x >> 10); }
constexpr uint32_t cCh(uint32_t x, uint32_t y, uint32_t z)  { return (x & y) ^ (~x & z); }
constexpr uint32_t cMaj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
constexpr uint32_t IV0=0x6a09e667u, IV1=0xbb67ae85u, IV2=0x3c6ef372u, IV3=0xa54ff53au,
                   IV4=0x510e527fu, IV5=0x9b05688cu, IV6=0x1f83d9abu, IV7=0x5be0cd19u;
// Round 0 reads only the constant IV + W[0]: T1 == ROUND0_T1C + W[0], T2 == ROUND0_T2C.
constexpr uint32_t ROUND0_T1C  = IV7 + cBigS1(IV4) + cCh(IV4,IV5,IV6) + 0x428a2f98u; // + K[0]
constexpr uint32_t ROUND0_T2C  = cBigS0(IV0) + cMaj(IV0,IV1,IV2);
constexpr uint32_t MSG33_W15   = 0x108u;             // bit length 264 of a 33-byte message
constexpr uint32_t MSG33_S1W15 = cSmS1(MSG33_W15);   // s1(W[15]) is constant
} // anonymous namespace

static inline void Initialize(__m512i* s) {
  static const uint32_t iv[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
  };
  for (int i = 0; i < 8; i++) {
    s[i] = _mm512_set1_epi32(iv[i]);
  }
}

#define ROUND(a,b,c,d,e,f,g,h,Kt,Wt,T1,T2) \
  T1 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(h,S1(e)),Ch(e,f,g)),Kt),Wt); \
  T2 = _mm512_add_epi32(S0(a),Maj(a,b,c)); \
  h = g; g = f; f = e; e = _mm512_add_epi32(d,T1); \
  d = c; c = b; b = a; a = _mm512_add_epi32(T1,T2);

// Message schedule expansion + 64 compression rounds, given the 16 already-built
// big-endian schedule words W16[0..15] (lane i == word of block i). Folds the
// result into state. Shared by both the byte-gather and packed (transposed)
// front ends so there is a single compression implementation.
static inline void sha256_compress(__m512i* state, const __m512i* W16) {
  __m512i a = state[0], b = state[1], c = state[2], d = state[3];
  __m512i e = state[4], f = state[5], g = state[6], h = state[7];
  __m512i W[64], T1, T2;

  for (int t = 0; t < 16; t++) W[t] = W16[t];

  for (int t = 16; t < 64; t++) {
    W[t] = _mm512_add_epi32(
             _mm512_add_epi32(s1(W[t-2]), W[t-7]),
             _mm512_add_epi32(s0(W[t-15]),W[t-16])
           );
  }

  for (int t = 0; t < 64; t++) {
    __m512i Kt = _mm512_set1_epi32(K[t]);
    ROUND(a,b,c,d,e,f,g,h,Kt,W[t],T1,T2);
  }

  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

// Compression specialized for a single fixed-shape 33-byte-message block (the
// only thing this tool hashes): schedule words W[9..14] are 0 and W[15] is the
// constant bit length 264, so the early schedule terms collapse and round 0
// (constant IV + W[0]) is folded. W16[0..8] are the data words; W16[9..15] are
// ignored. state must hold the standard IV on entry (single-block hash). Result
// is identical to sha256_compress for such a block -- cross-checked by --selftest.
static inline void sha256_compress_msg33(__m512i* state, const __m512i* W16) {
  __m512i W[64], T1, T2;
  for (int t = 0; t < 9; t++) W[t] = W16[t];
  for (int t = 9; t < 15; t++) W[t] = _mm512_setzero_si512();
  W[15] = _mm512_set1_epi32((int)MSG33_W15);

  // Early expansion with the zero/constant schedule words folded out.
  const __m512i s1w15 = _mm512_set1_epi32((int)MSG33_S1W15);
  W[16] = _mm512_add_epi32(s0(W[1]), W[0]);                              // s1(W14)=0, W9=0
  W[17] = _mm512_add_epi32(_mm512_add_epi32(s1w15, s0(W[2])), W[1]);     // s1(W15)=const, W10=0
  W[18] = _mm512_add_epi32(_mm512_add_epi32(s1(W[16]), s0(W[3])), W[2]); // W11=0
  W[19] = _mm512_add_epi32(_mm512_add_epi32(s1(W[17]), s0(W[4])), W[3]); // W12=0
  W[20] = _mm512_add_epi32(_mm512_add_epi32(s1(W[18]), s0(W[5])), W[4]); // W13=0
  W[21] = _mm512_add_epi32(_mm512_add_epi32(s1(W[19]), s0(W[6])), W[5]); // W14=0
  W[22] = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(s1(W[20]), W[15]), s0(W[7])), W[6]);
  for (int t = 23; t < 64; t++) {
    W[t] = _mm512_add_epi32(
             _mm512_add_epi32(s1(W[t-2]), W[t-7]),
             _mm512_add_epi32(s0(W[t-15]),W[t-16])
           );
  }

  // Round 0 folded from the constant IV (only W[0] is data).
  __m512i a = _mm512_add_epi32(_mm512_set1_epi32((int)(ROUND0_T1C + ROUND0_T2C)), W[0]);
  __m512i b = _mm512_set1_epi32((int)IV0);
  __m512i c = _mm512_set1_epi32((int)IV1);
  __m512i d = _mm512_set1_epi32((int)IV2);
  __m512i e = _mm512_add_epi32(_mm512_set1_epi32((int)(IV3 + ROUND0_T1C)), W[0]);
  __m512i f = _mm512_set1_epi32((int)IV4);
  __m512i g = _mm512_set1_epi32((int)IV5);
  __m512i h = _mm512_set1_epi32((int)IV6);

  for (int t = 1; t < 64; t++) {
    __m512i Kt = _mm512_set1_epi32(K[t]);
    ROUND(a,b,c,d,e,f,g,h,Kt,W[t],T1,T2);
  }

  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

// Generic front end: build the 16 schedule words from 16 (possibly discontiguous)
// input pointers via a scalar big-endian gather, then compress. Unchanged path;
// kept as the independent reference the self-test cross-checks against.
static void Transform(__m512i* state, const uint8_t* data[16]) {
  __m512i W[16];

  for (int t = 0; t < 16; t++) {
    uint32_t tmp[16];
    for (int i = 0; i < 16; i++) {
      const uint8_t* p = data[i] + t*4;
      tmp[i] = (uint32_t(p[0]) << 24) |
               (uint32_t(p[1]) << 16) |
               (uint32_t(p[2]) <<  8) |
               (uint32_t(p[3]) <<  0);
    }
    W[t] = _mm512_setr_epi32(
      tmp[0], tmp[1], tmp[2], tmp[3],
      tmp[4], tmp[5], tmp[6], tmp[7],
      tmp[8], tmp[9], tmp[10],tmp[11],
      tmp[12],tmp[13],tmp[14],tmp[15]
    );
  }

  sha256_compress(state, W);
}

// Packed front end: build the 16 schedule words from 16 CONTIGUOUS 64-byte blocks
// (block k at base + k*64) using an in-register 16x16 dword transpose, replacing
// the scalar 16x16 byte gather. Each row is byte-reversed (big-endian message
// words) then the rows are transposed so W[t] lane k == word t of block k.
static inline void transpose16x16_bswap(const uint8_t* base, __m512i W[16]) {
  // Reverse the 4 bytes within every dword (little-endian load -> big-endian word).
  const __m512i BSW = _mm512_set_epi8(
      12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
      12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
      12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
      12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3);

  __m512i A[16];
  for (int k = 0; k < 16; k++) {
    __m512i row = _mm512_loadu_si512((const void*)(base + (long)k * 64));
    A[k] = _mm512_shuffle_epi8(row, BSW);
  }

  // Stage 1: interleave dwords of adjacent rows.
  __m512i b[16];
  for (int k = 0; k < 8; k++) {
    b[2*k]   = _mm512_unpacklo_epi32(A[2*k], A[2*k+1]);
    b[2*k+1] = _mm512_unpackhi_epi32(A[2*k], A[2*k+1]);
  }

  // Stage 2: interleave qwords (within each 128-bit lane).
  __m512i c[16];
  for (int k = 0; k < 4; k++) {
    __m512i x0 = b[4*k+0], x1 = b[4*k+1], x2 = b[4*k+2], x3 = b[4*k+3];
    c[4*k+0] = _mm512_unpacklo_epi64(x0, x2);
    c[4*k+1] = _mm512_unpackhi_epi64(x0, x2);
    c[4*k+2] = _mm512_unpacklo_epi64(x1, x3);
    c[4*k+3] = _mm512_unpackhi_epi64(x1, x3);
  }

  // Stages 3-4: assemble full columns across 128-bit lanes. For residue r the
  // quad (c[r], c[r+4], c[r+8], c[r+12]) yields output words r, r+4, r+8, r+12.
  for (int r = 0; r < 4; r++) {
    __m512i p0 = c[r], p1 = c[r+4], p2 = c[r+8], p3 = c[r+12];
    __m512i q0 = _mm512_shuffle_i32x4(p0, p1, 0x88);
    __m512i q1 = _mm512_shuffle_i32x4(p0, p1, 0xdd);
    __m512i q2 = _mm512_shuffle_i32x4(p2, p3, 0x88);
    __m512i q3 = _mm512_shuffle_i32x4(p2, p3, 0xdd);
    W[r + 0]  = _mm512_shuffle_i32x4(q0, q2, 0x88);
    W[r + 8]  = _mm512_shuffle_i32x4(q0, q2, 0xdd);
    W[r + 4]  = _mm512_shuffle_i32x4(q1, q3, 0x88);
    W[r + 12] = _mm512_shuffle_i32x4(q1, q3, 0xdd);
  }
}

// Transpose the 8 state lanes back to 16 big-endian 32-byte digests.
static inline void store_digests(const __m512i* s, unsigned char* const H[16]) {
  ALIGN64 uint32_t tmp[8][16];
  for (int i = 0; i < 8; i++) {
    _mm512_store_si512((__m512i*)tmp[i], s[i]);
  }
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      uint32_t w = tmp[j][i];
#ifdef _MSC_VER
      w = _byteswap_ulong(w);
#else
      w = __builtin_bswap32(w);
#endif
      std::memcpy(H[i] + j*4, &w, 4);
    }
  }
}

extern "C" void sha256_avx512_16B(
  const uint8_t* d0,  const uint8_t* d1,  const uint8_t* d2,  const uint8_t* d3,
  const uint8_t* d4,  const uint8_t* d5,  const uint8_t* d6,  const uint8_t* d7,
  const uint8_t* d8,  const uint8_t* d9,  const uint8_t* d10, const uint8_t* d11,
  const uint8_t* d12, const uint8_t* d13, const uint8_t* d14, const uint8_t* d15,
  unsigned char* h0,  unsigned char* h1,  unsigned char* h2,  unsigned char* h3,
  unsigned char* h4,  unsigned char* h5,  unsigned char* h6,  unsigned char* h7,
  unsigned char* h8,  unsigned char* h9,  unsigned char* h10, unsigned char* h11,
  unsigned char* h12, unsigned char* h13, unsigned char* h14, unsigned char* h15
) {
  __m512i s[8];
  Initialize(s);
  const uint8_t* data[16] = {
    d0,d1,d2,d3,d4,d5,d6,d7,
    d8,d9,d10,d11,d12,d13,d14,d15
  };
  Transform(s, data);

  unsigned char* H[16] = {
    h0,h1,h2,h3,h4,h5,h6,h7,
    h8,h9,h10,h11,h12,h13,h14,h15
  };
  store_digests(s, H);
}

// Like sha256_avx512_16B but for 16 contiguous 64-byte input blocks (block k at
// base + k*64). Uses the in-register transpose; outputs are the same 16 digest
// pointers. Used by the production block path (hash16Blocks).
extern "C" void sha256_avx512_16B_packed(
  const uint8_t* base,
  unsigned char* h0,  unsigned char* h1,  unsigned char* h2,  unsigned char* h3,
  unsigned char* h4,  unsigned char* h5,  unsigned char* h6,  unsigned char* h7,
  unsigned char* h8,  unsigned char* h9,  unsigned char* h10, unsigned char* h11,
  unsigned char* h12, unsigned char* h13, unsigned char* h14, unsigned char* h15
) {
  __m512i s[8];
  Initialize(s);

  __m512i W[16];
  transpose16x16_bswap(base, W);
  sha256_compress(s, W);

  unsigned char* H[16] = {
    h0,h1,h2,h3,h4,h5,h6,h7,
    h8,h9,h10,h11,h12,h13,h14,h15
  };
  store_digests(s, H);
}

// Like sha256_avx512_16B_packed, but instead of writing 16 byte digests it
// leaves the 8 SHA-256 state words in SoA form (lane k == block k, native
// uint32) in stateOut (8 contiguous __m512i, 64-byte aligned). This skips the
// output transpose so a fused RIPEMD front end can consume the state directly.
extern "C" void sha256_avx512_16B_state(const uint8_t* base, void* stateOut) {
  __m512i s[8];
  Initialize(s);

  __m512i W[16];
  transpose16x16_bswap(base, W);
  sha256_compress_msg33(s, W);   // fixed 33-byte message: constant words folded

  __m512i* out = (__m512i*)stateOut;
  for (int i = 0; i < 8; i++) _mm512_store_si512(out + i, s[i]);
}

// Same as sha256_avx512_16B_state but with the GENERAL compression (no 33-byte
// specialization). Used only as the same-binary A/B control for --bench-hash-blocks
// so the msg33 speedup can be measured without a code-layout confound.
extern "C" void sha256_avx512_16B_state_general(const uint8_t* base, void* stateOut) {
  __m512i s[8];
  Initialize(s);

  __m512i W[16];
  transpose16x16_bswap(base, W);
  sha256_compress(s, W);

  __m512i* out = (__m512i*)stateOut;
  for (int i = 0; i < 8; i++) _mm512_store_si512(out + i, s[i]);
}
