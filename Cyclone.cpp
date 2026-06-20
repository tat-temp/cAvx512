//g++ -std=c++17 -Ofast -ffast-math -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -fipa-modref -flto -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -o Cyclone Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp ripemd160_avx512.cpp sha256_avx512.cpp

//The software is developed for solving Satoshi's puzzles; any use for illegal purposes is strictly prohibited. The author is not responsible for any actions taken by the user when using this software for unlawful activities.
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <array>
#include <utility>
// Adding program modules
#include "p2pkh_decoder.h"
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "sha256_avx512.h"
#include "ripemd160_avx512.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"
#include "ifma_field.h"
#include <cstdlib>

// Reference (non-AVX512) helpers defined in p2pkh_decoder.cpp but not exposed in
// its header. Declared here so the --selftest correctness cross-check can compare
// the AVX-512 hash160 path against this independent implementation.
namespace P2PKHDecoder {
    std::vector<uint8_t> compute_hash160(const std::vector<uint8_t>& data);
    std::vector<uint8_t> compute_sha256(const std::vector<uint8_t>& data);
    std::string base58_encode(const std::vector<uint8_t>& input);
}

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

//------------------------------------------------------------------------------
// Batch size: ±256 public keys (512), hashed in groups of 16 (AVX512).
static constexpr int CPU_GROUP_SIZE = 4096;
static constexpr int POINTS_BATCH_SIZE = 256;
static constexpr int HASH_BATCH_SIZE   = 16;

// Status output and progress saving frequency
static constexpr double statusIntervalSec = 5.0;
static constexpr double saveProgressIntervalSec = 300.0;

static int g_progressSaveCount = 0;
static std::vector<Int> g_threadPrivateKeys;

//------------------------------------------------------------------------------
void saveProgressToFile(const std::string &progressStr)
{
    std::ofstream ofs("progress.txt", std::ios::app);
    if (ofs) {
        ofs << progressStr << "\n";
    } else {
        std::cerr << "Cannot open progress.txt for writing\n";
    }
}

static void writeFoundKey(const std::string& privHex,
                          const std::string& pubHex,
                          const std::string& wif,
                          const std::string& address)
{
    std::ofstream ofs("found_keys.txt", std::ios::app);
    if (!ofs) {
        std::cerr << "Cannot open found_keys.txt for writing\n";
        return;
    }
    ofs << privHex << ' ' << pubHex << ' ' << wif << ' ' << address << '\n';
}

//------------------------------------------------------------------------------
//Converts a HEX string into a large number (a vector of 64-bit words, little-endian).

std::vector<uint64_t> hexToBigNum(const std::string& hex) {
    std::vector<uint64_t> bigNum;
    const size_t len = hex.size();
    bigNum.reserve((len + 15) / 16);
    for (size_t i = 0; i < len; i += 16) {
        size_t start = (len >= 16 + i) ? len - 16 - i : 0;
        size_t partLen = (len >= 16 + i) ? 16 : (len - i);
        uint64_t value = std::stoull(hex.substr(start, partLen), nullptr, 16);
        bigNum.push_back(value);
    }
    return bigNum;
}

//Reverse conversion to a HEX string (with correct leading zeros within blocks).

std::string bigNumToHex(const std::vector<uint64_t>& num) {
    std::ostringstream oss;
    for (auto it = num.rbegin(); it != num.rend(); ++it) {
         if (it != num.rbegin())
            oss << std::setw(16) << std::setfill('0');
        oss << std::hex << *it;
    }
    return oss.str();
}

std::vector<uint64_t> singleElementVector(uint64_t val) {
    return { val };
}

std::vector<uint64_t> bigNumAdd(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> sum;
    sum.reserve(std::max(a.size(), b.size()) + 1);
    uint64_t carry = 0;
    for (size_t i = 0, sz = std::max(a.size(), b.size()); i < sz; ++i) {
        uint64_t x = (i < a.size()) ? a[i] : 0ULL;
        uint64_t y = (i < b.size()) ? b[i] : 0ULL;
        __uint128_t s = ( __uint128_t )x + ( __uint128_t )y + carry;
        carry = (uint64_t)(s >> 64);
        sum.push_back((uint64_t)s);
    }
    if (carry) sum.push_back(carry);
    return sum;
}

std::vector<uint64_t> bigNumSubtract(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> diff = a;
    uint64_t borrow = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        uint64_t subtrahend = b[i];
        if (diff[i] < subtrahend + borrow) {
            diff[i] = diff[i] + (~0ULL) - subtrahend - borrow + 1ULL; // eqv diff[i] = diff[i] - subtrahend - borrow
            borrow = 1ULL;
        } else {
            diff[i] -= (subtrahend + borrow);
            borrow = 0ULL;
        }
    }
    
    for (size_t i = b.size(); i < diff.size() && borrow; ++i) {
        if (diff[i] == 0ULL) {
            diff[i] = ~0ULL;
        } else {
            diff[i] -= 1ULL;
            borrow = 0ULL;
        }
    }
    // delete leading zeros
    while (!diff.empty() && diff.back() == 0ULL)
        diff.pop_back();
    return diff;
}


std::pair<std::vector<uint64_t>, uint64_t> bigNumDivide(const std::vector<uint64_t>& a, uint64_t divisor) {
    std::vector<uint64_t> quotient(a.size(), 0ULL);
    uint64_t remainder = 0ULL;
    for (int i = (int)a.size() - 1; i >= 0; --i) {
        __uint128_t temp = ((__uint128_t)remainder << 64) | a[i];
        uint64_t q = (uint64_t)(temp / divisor);
        uint64_t r = (uint64_t)(temp % divisor);
        quotient[i] = q;
        remainder   = r;
    }
    while (!quotient.empty() && quotient.back() == 0ULL)
        quotient.pop_back();
    return { quotient, remainder };
}

long double hexStrToLongDouble(const std::string &hex) {
    long double result = 0.0L;
    for (char c : hex) {
        result *= 16.0L;
        if (c >= '0' && c <= '9')
            result += (c - '0');
        else if (c >= 'a' && c <= 'f')
            result += (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            result += (c - 'A' + 10);
    }
    return result;
}

//------------------------------------------------------------------------------
static inline std::string padHexTo64(const std::string &hex) {
    return (hex.size() >= 64) ? hex : std::string(64 - hex.size(), '0') + hex;
}
static inline Int hexToInt(const std::string &hex) {
    Int number;
    char buf[65] = {0};
    std::strncpy(buf, hex.c_str(), 64);
    number.SetBase16(buf);
    return number;
}
static inline std::string intToHex(const Int &value) {
    Int temp;
    temp.Set((Int*)&value);
    return temp.GetBase16();
}
static inline bool intGreater(const Int &a, const Int &b) {
    std::string ha = ((Int&)a).GetBase16();
    std::string hb = ((Int&)b).GetBase16();
    if (ha.size() != hb.size()) return (ha.size() > hb.size());
    return (ha > hb);
}
static inline bool isEven(const Int &number) {
    return ((Int&)number).IsEven();
}

static inline std::string intXToHex64(const Int &x) {
    Int temp;
    temp.Set((Int*)&x);
    std::string hex = temp.GetBase16();
    if (hex.size() < 64)
        hex.insert(0, 64 - hex.size(), '0');
    return hex;
}

static inline std::string pointToCompressedHex(const Point &point) {
    return (isEven(point.y) ? "02" : "03") + intXToHex64(point.x);
}
static inline void pointToCompressedBin(const Point &point, uint8_t outCompressed[33]) {
    outCompressed[0] = isEven(point.y) ? 0x02 : 0x03;
    Int temp;
    temp.Set((Int*)&point.x);
    for (int i = 0; i < 32; i++) {
        outCompressed[1 + i] = (uint8_t)temp.GetByte(31 - i);
    }
}

//------------------------------------------------------------------------------
inline void prepareShaBlock(const uint8_t* dataSrc, size_t dataLen, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, dataLen);
    outBlock[dataLen] = 0x80;
    const uint32_t bitLen = (uint32_t)(dataLen * 8);
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}
inline void prepareRipemdBlock(const uint8_t* dataSrc, uint8_t* outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, 32);
    outBlock[32] = 0x80;
    const uint32_t bitLen = 256;
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >>  8) & 0xFF);
    outBlock[63] = (uint8_t)( bitLen        & 0xFF);
}

// Computing hash160 using avx512 (16 hashes per try)
static void computeHash160BatchBinSingle(int numKeys,
                                         uint8_t pubKeys[][33],
                                         uint8_t hashResults[][20])
{
    std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
    std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
    std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
    std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE> ripemdOutputs;

    const size_t totalBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;
    for (size_t batch = 0; batch < totalBatches; batch++) {
        const size_t batchCount = std::min<size_t>(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            prepareShaBlock(pubKeys[idx], 33, shaInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(shaInputs[i].data(), shaInputs[0].data(), 64);
        }
        const uint8_t* inPtr[HASH_BATCH_SIZE];
        uint8_t* outPtr[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = shaInputs[i].data();
            outPtr[i] = shaOutputs[i].data();
        }
        sha256_avx512_16B(inPtr[0], inPtr[1], inPtr[2], inPtr[3],
                      inPtr[4], inPtr[5], inPtr[6], inPtr[7],
                      inPtr[8], inPtr[9], inPtr[10], inPtr[11],
                      inPtr[12], inPtr[13], inPtr[14], inPtr[15],
                      outPtr[0], outPtr[1], outPtr[2], outPtr[3],
                      outPtr[4], outPtr[5], outPtr[6], outPtr[7],
                      outPtr[8], outPtr[9], outPtr[10], outPtr[11],
                      outPtr[12], outPtr[13], outPtr[14], outPtr[15]);

        for (size_t i = 0; i < batchCount; i++) {
            prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(ripemdInputs[i].data(), ripemdInputs[0].data(), 64);
        }
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i]  = ripemdInputs[i].data();
            outPtr[i] = ripemdOutputs[i].data();
        }
        ripemd160avx512::ripemd160avx512_32(
            (unsigned char*)inPtr[0],
            (unsigned char*)inPtr[1],
            (unsigned char*)inPtr[2],
            (unsigned char*)inPtr[3],
            (unsigned char*)inPtr[4],
            (unsigned char*)inPtr[5],
            (unsigned char*)inPtr[6],
            (unsigned char*)inPtr[7],
            (unsigned char*)inPtr[8],
            (unsigned char*)inPtr[9],
            (unsigned char*)inPtr[10],
            (unsigned char*)inPtr[11],
            (unsigned char*)inPtr[12],
            (unsigned char*)inPtr[13],
            (unsigned char*)inPtr[14],
            (unsigned char*)inPtr[15],
            outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            outPtr[4], outPtr[5], outPtr[6], outPtr[7],
            outPtr[8], outPtr[9], outPtr[10], outPtr[11],
            outPtr[12], outPtr[13], outPtr[14], outPtr[15]
        );
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            std::memcpy(hashResults[idx], ripemdOutputs[i].data(), 20);
        }
    }
}

static void computeHash160BatchBinSingle2(
    Point* p,
    uint8_t outHash[][20])
{
    uint8_t shaIn[HASH_BATCH_SIZE][64];

    std::memset(&shaIn, 0, HASH_BATCH_SIZE * 64);

    for (int i = 0; i < HASH_BATCH_SIZE; i++)
    {
        shaIn[i][0] = isEven(&p[i].y) ? 0x02 : 0x03;
        uint8_t* pSrc = (&p[i].x)->GetBytes();
        std::reverse_copy(pSrc, pSrc + 32, &shaIn[i][1]);

        shaIn[i][33] = 0x80;
        shaIn[i][60] = uint8_t((33 * 8) >> 24);
        shaIn[i][61] = uint8_t((33 * 8) >> 16);
        shaIn[i][62] = uint8_t((33 * 8) >> 8);
        shaIn[i][63] = uint8_t((33 * 8));
    }

    sha256_avx512_16B(shaIn[0], shaIn[1], shaIn[2], shaIn[3],
        shaIn[4], shaIn[5], shaIn[6], shaIn[7],
        shaIn[8], shaIn[9], shaIn[10], shaIn[11],
        shaIn[12], shaIn[13], shaIn[14], shaIn[15],

        shaIn[0], shaIn[1], shaIn[2], shaIn[3],
        shaIn[4], shaIn[5], shaIn[6], shaIn[7],
        shaIn[8], shaIn[9], shaIn[10], shaIn[11],
        shaIn[12], shaIn[13], shaIn[14], shaIn[15]);

    ripemd160avx512::ripemd160avx512_32(
        (unsigned char*)shaIn[0], (unsigned char*)shaIn[1], (unsigned char*)shaIn[2], (unsigned char*)shaIn[3],
        (unsigned char*)shaIn[4], (unsigned char*)shaIn[5], (unsigned char*)shaIn[6], (unsigned char*)shaIn[7],
        (unsigned char*)shaIn[8], (unsigned char*)shaIn[9], (unsigned char*)shaIn[10], (unsigned char*)shaIn[11],
        (unsigned char*)shaIn[12], (unsigned char*)shaIn[13], (unsigned char*)shaIn[14], (unsigned char*)shaIn[15],
        outHash[0], outHash[1], outHash[2], outHash[3],
        outHash[4], outHash[5], outHash[6], outHash[7],
        outHash[8], outHash[9], outHash[10], outHash[11],
        outHash[12], outHash[13], outHash[14], outHash[15]
    );
}

//------------------------------------------------------------------------------
// C.1c block-buffer path: hash straight from pre-built SHA input blocks, with no
// per-point Point round-trip. A group's blocks live in one CPU_GROUP_SIZE x 64
// buffer; the gen phase writes bytes [0..32] (parity + big-endian x) of each,
// initBlocks lays down the constant message padding once, and the in-place SHA
// (which only writes the low 32 bytes) leaves that padding intact across groups.
//------------------------------------------------------------------------------

// One-time formatting of a CPU_GROUP_SIZE x 64 block buffer: zero it, then set
// the 33-byte-message padding (0x80 terminator at [33], bit-length 264 at
// [62..63]). The gen phase overwrites only [0..32]; SHA overwrites only [0..31].
static void initBlocks(uint8_t *blocks) {
    std::memset(blocks, 0, (size_t)CPU_GROUP_SIZE * 64);
    for (int i = 0; i < CPU_GROUP_SIZE; i++) {
        uint8_t *b = blocks + (long)i * 64;
        b[33] = 0x80;
        b[62] = (uint8_t)((33 * 8) >> 8); // 0x01
        b[63] = (uint8_t)((33 * 8));      // 0x08
    }
}

// Write one scalar Point as a compressed-key SHA block (parity + big-endian x at
// bytes [0..32]); padding bytes [33..63] are left to initBlocks. Used for the
// genGroupIFMABlocks scalar tail (the lanes gen8 does not cover).
static inline void storeCompScalar(Point &pt, uint8_t *blk) {
    blk[0] = isEven(&pt.y) ? 0x02 : 0x03;
    uint8_t *xb = (&pt.x)->GetBytes();
    for (int b = 0; b < 32; b++) blk[1 + b] = xb[31 - b]; // little-endian -> big-endian
}

// Hash 16 consecutive 64-byte blocks (at B) to 16 hash160 outputs. NOT in place:
// SHA reads the blocks into a separate 'digest' scratch, and RIPEMD reads/pads
// that scratch. This matters because ripemd160avx512_32 writes its own padding
// into bytes [32..63] of its input -- hashing in place would clobber the SHA
// message padding (0x80 at [33], length at [62..63]) that initBlocks lays down
// once and the buffer relies on persisting across reused groups.
static inline void hash16Blocks(uint8_t *B, uint8_t outHash[][20]) {
    ALIGN64 uint8_t digest[HASH_BATCH_SIZE][64]; // SHA out / RIPEMD in (RIPEMD pads [32..63])
    sha256_avx512_16B(
        B + 0 * 64,  B + 1 * 64,  B + 2 * 64,  B + 3 * 64,
        B + 4 * 64,  B + 5 * 64,  B + 6 * 64,  B + 7 * 64,
        B + 8 * 64,  B + 9 * 64,  B + 10 * 64, B + 11 * 64,
        B + 12 * 64, B + 13 * 64, B + 14 * 64, B + 15 * 64,
        digest[0],  digest[1],  digest[2],  digest[3],
        digest[4],  digest[5],  digest[6],  digest[7],
        digest[8],  digest[9],  digest[10], digest[11],
        digest[12], digest[13], digest[14], digest[15]);
    ripemd160avx512::ripemd160avx512_32(
        digest[0],  digest[1],  digest[2],  digest[3],
        digest[4],  digest[5],  digest[6],  digest[7],
        digest[8],  digest[9],  digest[10], digest[11],
        digest[12], digest[13], digest[14], digest[15],
        outHash[0],  outHash[1],  outHash[2],  outHash[3],
        outHash[4],  outHash[5],  outHash[6],  outHash[7],
        outHash[8],  outHash[9],  outHash[10], outHash[11],
        outHash[12], outHash[13], outHash[14], outHash[15]);
}

//------------------------------------------------------------------------------
// Phase 0 instrumentation: shared building blocks for --bench and --selftest.
//
// processGroupFused (below) is the single generate+hash+compare routine shared by
// the production search loop, --selftest, and --bench, so the harness exercises
// the exact hot path that runs in production.
//------------------------------------------------------------------------------
static void buildGeneratorTable(Secp256K1 &secp, Point *Gn, Point &_2Gn) {
    Point g = secp.G;
    Gn[0] = g;
    g = secp.DoubleDirect(g);
    Gn[1] = g;
    for (int i = 2; i < CPU_GROUP_SIZE / 2; i++) {
        g = secp.AddDirect(g, secp.G);
        Gn[i] = g;
    }
    _2Gn = secp.DoubleDirect(Gn[CPU_GROUP_SIZE / 2 - 1]);
}

// Generate the CPU_GROUP_SIZE public keys for consecutive private keys
// [base, base+CPU_GROUP_SIZE), where startP on entry is the public key of
// (base + CPU_GROUP_SIZE/2), and hash each HASH_BATCH_SIZE-sized chunk to hash160
// immediately -- "fused", so the working set is one 16-point chunk (L1-resident)
// instead of a 4096-point array. Each candidate is compared against target16
// (first 16 bytes) then the full 20-byte targetHash160. Returns the in-group
// index [0, CPU_GROUP_SIZE) of the first match, or -1 if none. startP is advanced
// by CPU_GROUP_SIZE*G on return (using dx[CENTER], computed before inversion from
// the OLD startP). dx must hold CPU_GROUP_SIZE/2+1 entries; grp must be Set() to it.
//
// The per-index point math is identical to the original two-pass generator:
//   idx == CENTER : startP
//   idx >  CENTER : startP + m*G  (m = idx-CENTER), via Gn[m-1], dx[m-1]
//   idx <  CENTER : startP - m*G  (m = CENTER-idx), via Gn[m-1], dx[m-1]
static int processGroupFused(Point &startP, Point *Gn, Point &_2Gn,
                             std::vector<Int> &dx, IntGroup &grp,
                             __m128i target16, const uint8_t *targetHash160) {
    const int CENTER  = CPU_GROUP_SIZE / 2;
    const int hLength = CENTER - 1;
    Int dy, dyn, _s, _p;
    int j;

    // 1. Batch all (Gn[j].x - startP.x) differences, then invert the whole group.
    for (j = 0; j < hLength; j++) {
        dx[j].ModSub(&Gn[j].x, &startP.x);
    }
    dx[j].ModSub(&Gn[j].x, &startP.x);     // dx[CENTER-1]
    dx[j + 1].ModSub(&_2Gn.x, &startP.x);  // dx[CENTER]  (for the next center point)
    grp.ModInv();

    // 2. Generate + hash one HASH_BATCH_SIZE chunk at a time.
    Point chunk[HASH_BATCH_SIZE];
    ALIGN64 uint8_t hashRes[HASH_BATCH_SIZE][20];
    int foundIdx = -1;

    for (int i = 0; i < CPU_GROUP_SIZE && foundIdx < 0; i += HASH_BATCH_SIZE) {
        for (int L = 0; L < HASH_BATCH_SIZE; L++) {
            const int idx = i + L;
            Point &out = chunk[L];
            if (idx == CENTER) {
                out = startP;
                continue;
            }
            if (idx > CENTER) {                 // startP + m*G
                const int m = idx - CENTER;
                dy.ModSub(&Gn[m - 1].y, &startP.y);
                _s.ModMulK1(&dy, &dx[m - 1]);
                _p.ModSquareK1(&_s);
                out.x.Set(&startP.x);
                out.x.ModNeg();
                out.x.ModAdd(&_p);
                out.x.ModSub(&Gn[m - 1].x);
                out.y.ModSub(&Gn[m - 1].x, &out.x);
                out.y.ModMulK1(&_s);
                out.y.ModSub(&Gn[m - 1].y);
            } else {                            // startP - m*G  ( -G = (x, -y) )
                const int m = CENTER - idx;
                dyn.Set(&Gn[m - 1].y);
                dyn.ModNeg();
                dyn.ModSub(&startP.y);
                _s.ModMulK1(&dyn, &dx[m - 1]);
                _p.ModSquareK1(&_s);
                out.x.Set(&startP.x);
                out.x.ModNeg();
                out.x.ModAdd(&_p);
                out.x.ModSub(&Gn[m - 1].x);
                out.y.ModSub(&Gn[m - 1].x, &out.x);
                out.y.ModMulK1(&_s);
                out.y.ModAdd(&Gn[m - 1].y);
            }
        }

        computeHash160BatchBinSingle2(chunk, hashRes);

        for (int L = 0; L < HASH_BATCH_SIZE; L++) {
            __m128i cand = _mm_loadu_si128(reinterpret_cast<const __m128i *>(hashRes[L]));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(cand, target16)) == 0xFFFF &&
                std::memcmp(hashRes[L], targetHash160, 20) == 0) {
                foundIdx = i + L;
                break;
            }
        }
    }

    // 3. Advance startP by CPU_GROUP_SIZE*G (reads the OLD startP via dx[CENTER]).
    Point pp = startP;
    dy.ModSub(&_2Gn.y, &pp.y);
    _s.ModMulK1(&dy, &dx[hLength + 1]);
    _p.ModSquareK1(&_s);
    pp.x.ModNeg();
    pp.x.ModAdd(&_p);
    pp.x.ModSub(&_2Gn.x);
    pp.y.ModSub(&_2Gn.x, &pp.x);
    pp.y.ModMulK1(&_s);
    pp.y.ModSub(&_2Gn.y);
    startP = pp;

    return foundIdx;
}

// Build a mainnet P2PKH Base58Check address from a 20-byte hash160.
static std::string hash160ToAddress(const std::vector<uint8_t> &h160) {
    std::vector<uint8_t> payload;
    payload.reserve(25);
    payload.push_back(0x00); // version byte
    payload.insert(payload.end(), h160.begin(), h160.end());
    std::vector<uint8_t> c1 = P2PKHDecoder::compute_sha256(payload);
    std::vector<uint8_t> c2 = P2PKHDecoder::compute_sha256(c1);
    payload.insert(payload.end(), c2.begin(), c2.begin() + 4);
    return P2PKHDecoder::base58_encode(payload);
}

// Defined further below (need the IFMA helpers from ifma_field.h); forward-declared
// so the self-test can drive the exact production search path.
static void genGroupIFMA(Point &startP, Point *Gn, Point &_2Gn,
                         std::vector<Int> &dx, IntGroup &grp, Point *pts);
static void genGroupIFMABlocks(Point &startP, Point *Gn, Point &_2Gn,
                               std::vector<Int> &dx, IntGroup &grp, uint8_t *blocks);
static int processGroupIFMA(Point &startP, Point *Gn, Point &_2Gn,
                            std::vector<Int> &dx, IntGroup &grp, uint8_t *blocks,
                            __m128i target16, const uint8_t *targetHash160);

//------------------------------------------------------------------------------
// 0.2 Correctness self-test. Returns 0 on success, 1 on any failure.
//------------------------------------------------------------------------------
static int runSelfTest(Secp256K1 &secp) {
    std::cout << "================= SELF TEST =================\n";
    int failures = 0;

    // --- Part A: AVX-512 hash160 vs the independent reference path -------------
    std::vector<Int> testKeys;
    for (uint32_t i = 1; i <= 40; ++i) { Int k; k.SetInt32(i); testKeys.push_back(k); }
    const char *bigHex[] = {
        "123456789ABCDEF",
        "DEADBEEFCAFEBABE",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140", // order-1
        "8000000000000000000000000000000000000000000000000000000000000000",
        "1111111111111111111111111111111111111111111111111111111111111111",
        "00000000000000000000000000000000000000007FFFFFFFFFFFFFFFFFFFFFFF"
    };
    for (const char *h : bigHex) testKeys.push_back(hexToInt(h));

    int hashChecked = 0, hashFail = 0;
    const size_t nKeys = testKeys.size();
    for (size_t base = 0; base < nKeys; base += HASH_BATCH_SIZE) {
        Point batch[HASH_BATCH_SIZE];
        const size_t cnt = std::min<size_t>(HASH_BATCH_SIZE, nKeys - base);
        for (size_t i = 0; i < HASH_BATCH_SIZE; ++i) {
            Int k; k.Set(&testKeys[base + (i < cnt ? i : cnt - 1)]);
            batch[i] = secp.ComputePublicKey(&k);
        }
        ALIGN64 uint8_t out[HASH_BATCH_SIZE][20];
        computeHash160BatchBinSingle2(batch, out);
        for (size_t i = 0; i < cnt; ++i) {
            uint8_t comp[33];
            pointToCompressedBin(batch[i], comp);
            std::vector<uint8_t> ref =
                P2PKHDecoder::compute_hash160(std::vector<uint8_t>(comp, comp + 33));
            ++hashChecked;
            if (std::memcmp(out[i], ref.data(), 20) != 0) {
                ++hashFail;
                Int kk; kk.Set(&testKeys[base + i]);
                std::cout << "  [FAIL] hash160 mismatch for key "
                          << padHexTo64(intToHex(kk)) << "\n";
            }
        }
    }
    std::cout << "Hash160 cross-check : " << (hashChecked - hashFail) << "/"
              << hashChecked << " match\n";
    if (hashFail) ++failures;

    // --- Part B: end-to-end batched search finds a known in-range key ----------
    Int k = hexToInt("ABCDEF");
    Point Pk = secp.ComputePublicKey(&k);
    uint8_t compk[33];
    pointToCompressedBin(Pk, compk);
    std::vector<uint8_t> targetH =
        P2PKHDecoder::compute_hash160(std::vector<uint8_t>(compk, compk + 33));

    const std::string addr = hash160ToAddress(targetH);
    bool decodeOk = false;
    try { decodeOk = (P2PKHDecoder::getHash160(addr) == targetH); }
    catch (...) { decodeOk = false; }
    std::cout << "Address round-trip  : " << (decodeOk ? "OK" : "FAIL")
              << " (" << addr << ")\n";
    if (!decodeOk) ++failures;

    Int w;      w.SetInt32((uint32_t)CPU_GROUP_SIZE * 2);
    Int startK; startK.Set(&k); startK.Sub(&w);
    Int endK;   endK.Set(&k);   endK.Add(&w);

    std::vector<Point> Gn(CPU_GROUP_SIZE / 2);
    Point _2Gn;
    buildGeneratorTable(secp, Gn.data(), _2Gn);

    Int priv;   priv.Set(&startK);
    Int half;   half.SetInt32(CPU_GROUP_SIZE / 2);
    Int center; center.Set(&priv); center.Add(&half);
    Point startP = secp.ComputePublicKey(&center);

    std::vector<Int> dx(CPU_GROUP_SIZE / 2 + 1);
    IntGroup grp(CPU_GROUP_SIZE / 2 + 1);
    grp.Set(dx.data());
    std::vector<uint8_t> blocks((size_t)CPU_GROUP_SIZE * 64); // SHA-block scratch
    initBlocks(blocks.data());
    __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(targetH.data()));

    bool found = false;
    std::string foundPrivHex, foundWif;
    while (!found) {
        if (priv.IsGreater(&endK)) break;
        int idx = processGroupIFMA(startP, Gn.data(), _2Gn, dx, grp, blocks.data(),
                                   target16, targetH.data());
        if (idx >= 0) {
            Int mPriv; mPriv.Set(&priv);
            Int off;   off.SetInt32((uint32_t)idx);
            mPriv.Add(&off);
            foundPrivHex = padHexTo64(intToHex(mPriv));
            foundWif = P2PKHDecoder::compute_wif(foundPrivHex, true);
            found = true;
        }
        priv.Add((uint64_t)CPU_GROUP_SIZE);
    }

    const std::string expectHex = padHexTo64(intToHex(k));
    const std::string expectWif = P2PKHDecoder::compute_wif(expectHex, true);
    const bool e2eOk = found && (foundPrivHex == expectHex) && (foundWif == expectWif);
    std::cout << "End-to-end search   : " << (e2eOk ? "OK" : "FAIL");
    if (found) std::cout << " (found priv " << foundPrivHex << ")";
    else       std::cout << " (key not found in window)";
    std::cout << "\n";
    if (!e2eOk) ++failures;

    std::cout << "============================================\n";
    std::cout << (failures == 0 ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    return failures == 0 ? 0 : 1;
}

//------------------------------------------------------------------------------
// C.0a self-test: validate the SIMD radix-2^52 field foundation (Int<->limbs
// conversion, SoA pack/unpack, add/sub/neg, normalize) against the scalar Int
// field ops. No IFMA yet -- AVX-512F/DQ only. Run: --selftest-ifma
//------------------------------------------------------------------------------
static uint64_t ifma_rng_state = 0x0123456789ABCDEFULL;
static inline uint64_t ifma_next_rand() {            // splitmix64 (deterministic)
    ifma_rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = ifma_rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// A random field element in [0, p).
static Int ifma_randFE() {
    Int v;
    for (int i = 0; i < NB64BLOCK; i++) v.bits64[i] = 0;
    v.bits64[0] = ifma_next_rand();
    v.bits64[1] = ifma_next_rand();
    v.bits64[2] = ifma_next_rand();
    v.bits64[3] = ifma_next_rand() & 0x7FFFFFFFFFFFFFFFULL; // < 2^255 < p (no Mod needed)
    return v;
}

// Canonicalize redundant radix-2^52 limbs to a < p value, entirely in scalar
// (carry -> fold 2^256==R -> conditional subtract p). Deliberately avoids
// Int::ShiftL / Int::Mod so the checker is independent of those, and never feeds
// a >= 2^256 value to Int (which only supports <=256-bit operands).
static Int ifma_limbsToInt(const uint64_t limbs[5]) {
    uint64_t n[5];
    __uint128_t c = (__uint128_t)limbs[0];           n[0] = (uint64_t)c & ifma::MASK52; c >>= 52;
    c += limbs[1];                                    n[1] = (uint64_t)c & ifma::MASK52; c >>= 52;
    c += limbs[2];                                    n[2] = (uint64_t)c & ifma::MASK52; c >>= 52;
    c += limbs[3];                                    n[3] = (uint64_t)c & ifma::MASK52; c >>= 52;
    c += limbs[4];                                    n[4] = (uint64_t)c; // may exceed 48 bits

    // Fold everything >= 2^256 back via R until the value is < 2^256.
    while (n[4] >> 48) {
        uint64_t over = n[4] >> 48; n[4] &= ifma::MASK48;
        __uint128_t d = (__uint128_t)n[0] + (__uint128_t)over * ifma::R256;
        n[0] = (uint64_t)d & ifma::MASK52; d >>= 52;
        d += n[1]; n[1] = (uint64_t)d & ifma::MASK52; d >>= 52;
        d += n[2]; n[2] = (uint64_t)d & ifma::MASK52; d >>= 52;
        d += n[3]; n[3] = (uint64_t)d & ifma::MASK52; d >>= 52;
        d += n[4]; n[4] = (uint64_t)d;
    }

    // Pack 5x52 -> 4x64 (value now < 2^256).
    uint64_t w0 =  n[0]        | (n[1] << 52);
    uint64_t w1 = (n[1] >> 12) | (n[2] << 40);
    uint64_t w2 = (n[2] >> 24) | (n[3] << 28);
    uint64_t w3 = (n[3] >> 36) | (n[4] << 16);

    // Conditional subtract p. value < 2^256 < 2p, and [p, 2^256) is exactly the
    // set with w1==w2==w3==~0 and w0 >= 0xFFFFFFFEFFFFFC2F, so one subtract suffices.
    const uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL, PHI = 0xFFFFFFFFFFFFFFFFULL;
    if (w3 == PHI && w2 == PHI && w1 == PHI && w0 >= P0) { w0 -= P0; w1 = 0; w2 = 0; w3 = 0; }

    Int v; for (int i = 0; i < NB64BLOCK; i++) v.bits64[i] = 0;
    v.bits64[0] = w0; v.bits64[1] = w1; v.bits64[2] = w2; v.bits64[3] = w3;
    return v;
}

// Scatter 8 lane results (rx,ry) into 8 Points at base, base+step, ... base+7*step.
static inline void storePts8(const ifma::FieldVec8 &rx, const ifma::FieldVec8 &ry,
                             Point *base, int step) {
    uint64_t ox[8][5], oy[8][5];
    ifma::store8(rx, ox);
    ifma::store8(ry, oy);
    for (int l = 0; l < 8; l++) {
        Point *d = base + (long)l * step;
        d->x = ifma_limbsToInt(ox[l]);
        d->y = ifma_limbsToInt(oy[l]);
    }
}

// IFMA group generator: fill pts[0..CPU_GROUP_SIZE-1] with pubkeys for the keys
// [base, base+CPU_GROUP_SIZE) (startP on entry = pubkey of base+CENTER), using
// gen8 (8 plus + 8 minus per batch) for the bulk and scalar for the tail. dx is
// computed+inverted here; startP is advanced by CPU_GROUP_SIZE*G on return.
static void genGroupIFMA(Point &startP, Point *Gn, Point &_2Gn,
                         std::vector<Int> &dx, IntGroup &grp, Point *pts) {
    const int CENTER  = CPU_GROUP_SIZE / 2;
    const int hLength = CENTER - 1;
    int j;

    for (j = 0; j < hLength; j++) dx[j].ModSub(&Gn[j].x, &startP.x);
    dx[j].ModSub(&Gn[j].x, &startP.x);     // dx[hLength]
    dx[j + 1].ModSub(&_2Gn.x, &startP.x);  // dx[CENTER] (for the advance)
    grp.ModInv();

    ifma::FieldVec8 SPX = ifma::broadcast(startP.x);
    ifma::FieldVec8 SPY = ifma::broadcast(startP.y);
    pts[CENTER] = startP;

    Int gx[8], gy[8], iv[8];
    for (j = 0; j + 8 <= hLength; j += 8) {
        for (int l = 0; l < 8; l++) {
            gx[l].Set(&Gn[j + l].x); gy[l].Set(&Gn[j + l].y); iv[l].Set(&dx[j + l]);
        }
        ifma::FieldVec8 GX = ifma::load8(gx), GY = ifma::load8(gy), IV = ifma::load8(iv);
        ifma::FieldVec8 RX, RY;
        ifma::gen8(SPX, SPY, GX, GY, IV, true,  RX, RY);
        storePts8(RX, RY, &pts[CENTER + j + 1], +1);     // pts[CENTER+j+1 .. +8]
        ifma::gen8(SPX, SPY, GX, GY, IV, false, RX, RY);
        storePts8(RX, RY, &pts[CENTER - j - 1], -1);     // pts[CENTER-j-1 .. -8]
    }

    // Scalar tail (leftover j's, pts[0], and the startP advance).
    auto scalarPoint = [&](int gi, bool plus, Point &out) {
        Int dyv, s, p2;
        if (plus) dyv.ModSub(&Gn[gi].y, &startP.y);
        else { dyv.Set(&Gn[gi].y); dyv.ModNeg(); dyv.ModSub(&startP.y); }
        s.ModMulK1(&dyv, &dx[gi]);
        p2.ModSquareK1(&s);
        out.x.Set(&startP.x); out.x.ModNeg(); out.x.ModAdd(&p2); out.x.ModSub(&Gn[gi].x);
        out.y.ModSub(&Gn[gi].x, &out.x); out.y.ModMulK1(&s);
        if (plus) out.y.ModSub(&Gn[gi].y); else out.y.ModAdd(&Gn[gi].y);
    };
    for (; j < hLength; j++) {
        scalarPoint(j, true,  pts[CENTER + j + 1]);
        scalarPoint(j, false, pts[CENTER - j - 1]);
    }
    scalarPoint(hLength, false, pts[0]);  // startP - Gn[hLength] == pubkey(base)

    Int dyv, s, p2; Point pp = startP;
    dyv.ModSub(&_2Gn.y, &pp.y);
    s.ModMulK1(&dyv, &dx[hLength + 1]);
    p2.ModSquareK1(&s);
    pp.x.ModNeg(); pp.x.ModAdd(&p2); pp.x.ModSub(&_2Gn.x);
    pp.y.ModSub(&_2Gn.x, &pp.x); pp.y.ModMulK1(&s); pp.y.ModSub(&_2Gn.y);
    startP = pp;
}

// Like genGroupIFMA, but writes each public key directly as a 33-byte compressed
// SHA block (via to_compressed8 for the 8-lane bulk, storeCompScalar for the
// tail) into the caller's pre-formatted CPU_GROUP_SIZE x 64 buffer -- no Point
// round-trip, no GetBytes() in the hash phase. Index mapping matches genGroupIFMA
// exactly. startP is advanced by CPU_GROUP_SIZE*G on return.
static void genGroupIFMABlocks(Point &startP, Point *Gn, Point &_2Gn,
                               std::vector<Int> &dx, IntGroup &grp, uint8_t *blocks) {
    const int CENTER  = CPU_GROUP_SIZE / 2;
    const int hLength = CENTER - 1;
    int j;

    for (j = 0; j < hLength; j++) dx[j].ModSub(&Gn[j].x, &startP.x);
    dx[j].ModSub(&Gn[j].x, &startP.x);     // dx[hLength]
    dx[j + 1].ModSub(&_2Gn.x, &startP.x);  // dx[CENTER] (for the advance)
    grp.ModInv();

    ifma::FieldVec8 SPX = ifma::broadcast(startP.x);
    ifma::FieldVec8 SPY = ifma::broadcast(startP.y);
    storeCompScalar(startP, blocks + (long)CENTER * 64);

    Int gx[8], gy[8], iv[8];
    for (j = 0; j + 8 <= hLength; j += 8) {
        for (int l = 0; l < 8; l++) {
            gx[l].Set(&Gn[j + l].x); gy[l].Set(&Gn[j + l].y); iv[l].Set(&dx[j + l]);
        }
        ifma::FieldVec8 GX = ifma::load8(gx), GY = ifma::load8(gy), IV = ifma::load8(iv);
        ifma::FieldVec8 RX, RY;
        ifma::gen8(SPX, SPY, GX, GY, IV, true,  RX, RY);
        ifma::to_compressed8(RX, RY, blocks + (long)(CENTER + j + 1) * 64, 64, +1);
        ifma::gen8(SPX, SPY, GX, GY, IV, false, RX, RY);
        ifma::to_compressed8(RX, RY, blocks + (long)(CENTER - j - 1) * 64, 64, -1);
    }

    // Scalar tail (leftover j's, block 0, and the startP advance).
    auto scalarPoint = [&](int gi, bool plus, Point &out) {
        Int dyv, s, p2;
        if (plus) dyv.ModSub(&Gn[gi].y, &startP.y);
        else { dyv.Set(&Gn[gi].y); dyv.ModNeg(); dyv.ModSub(&startP.y); }
        s.ModMulK1(&dyv, &dx[gi]);
        p2.ModSquareK1(&s);
        out.x.Set(&startP.x); out.x.ModNeg(); out.x.ModAdd(&p2); out.x.ModSub(&Gn[gi].x);
        out.y.ModSub(&Gn[gi].x, &out.x); out.y.ModMulK1(&s);
        if (plus) out.y.ModSub(&Gn[gi].y); else out.y.ModAdd(&Gn[gi].y);
    };
    Point tmp;
    for (; j < hLength; j++) {
        scalarPoint(j, true,  tmp); storeCompScalar(tmp, blocks + (long)(CENTER + j + 1) * 64);
        scalarPoint(j, false, tmp); storeCompScalar(tmp, blocks + (long)(CENTER - j - 1) * 64);
    }
    scalarPoint(hLength, false, tmp); storeCompScalar(tmp, blocks + 0); // pubkey(base) -> block 0

    Int dyv, s, p2; Point pp = startP;
    dyv.ModSub(&_2Gn.y, &pp.y);
    s.ModMulK1(&dyv, &dx[hLength + 1]);
    p2.ModSquareK1(&s);
    pp.x.ModNeg(); pp.x.ModAdd(&p2); pp.x.ModSub(&_2Gn.x);
    pp.y.ModSub(&_2Gn.x, &pp.x); pp.y.ModMulK1(&s); pp.y.ModSub(&_2Gn.y);
    startP = pp;
}

// Production IFMA search path: generate the whole 4096-point group straight into
// compressed SHA blocks (genGroupIFMABlocks), then hash + compare each
// HASH_BATCH_SIZE chunk in place. Same contract as processGroupFused (returns the
// in-group index of the first match or -1, advances startP by CPU_GROUP_SIZE*G),
// but the per-point EC math runs on AVX-512 IFMA and there is no Point round-trip.
// blocks is caller-owned scratch (CPU_GROUP_SIZE*64 bytes), formatted once by
// initBlocks and reused across calls.
static int processGroupIFMA(Point &startP, Point *Gn, Point &_2Gn,
                            std::vector<Int> &dx, IntGroup &grp, uint8_t *blocks,
                            __m128i target16, const uint8_t *targetHash160) {
    genGroupIFMABlocks(startP, Gn, _2Gn, dx, grp, blocks);

    ALIGN64 uint8_t hashRes[HASH_BATCH_SIZE][20];
    for (int i = 0; i < CPU_GROUP_SIZE; i += HASH_BATCH_SIZE) {
        hash16Blocks(blocks + (long)i * 64, hashRes);
        for (int L = 0; L < HASH_BATCH_SIZE; L++) {
            __m128i cand = _mm_loadu_si128(reinterpret_cast<const __m128i *>(hashRes[L]));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(cand, target16)) == 0xFFFF &&
                std::memcmp(hashRes[L], targetHash160, 20) == 0) {
                return i + L;
            }
        }
    }
    return -1;
}

static int runSelfTestIFMA() {
    Secp256K1 secp; secp.Init();   // sets up Int::P (the secp256k1 field prime)

    std::cout << "============== IFMA FIELD SELFTEST ==============\n";
    const int BATCHES = 4000;      // 8 lanes -> 32k random vectors per op
    int fRound = 0, fSoA = 0, fAdd = 0, fSub = 0, fNeg = 0, fNorm = 0, fMul = 0, fSqr = 0;

    for (int b = 0; b < BATCHES; b++) {
        Int a[8], bb[8];
        for (int j = 0; j < 8; j++) { a[j] = ifma_randFE(); bb[j] = ifma_randFE(); }

        // 0) scalar conversion round-trip
        for (int j = 0; j < 8; j++) {
            uint64_t L[5]; ifma::int_to_limbs(a[j], L);
            Int got = ifma_limbsToInt(L);
            if (!got.IsEqual(&a[j])) fRound++;
        }

        ifma::FieldVec8 A = ifma::load8(a);
        ifma::FieldVec8 B = ifma::load8(bb);

        // 0b) SoA pack/unpack round-trip
        uint64_t outA[8][5]; ifma::store8(A, outA);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(outA[j]);
            if (!got.IsEqual(&a[j])) fSoA++;
        }

        uint64_t o[8][5];

        ifma::FieldVec8 C = ifma::add(A, B); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.Set(&a[j]); exp.ModAdd(&bb[j]);
            if (!got.IsEqual(&exp)) fAdd++;
        }

        C = ifma::sub(A, B); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.Set(&a[j]); exp.ModSub(&bb[j]);
            if (!got.IsEqual(&exp)) fSub++;
        }

        C = ifma::neg(A); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.Set(&a[j]); exp.ModNeg();
            if (!got.IsEqual(&exp)) fNeg++;
        }

        C = ifma::add(A, B); ifma::normalize_weak(C); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.Set(&a[j]); exp.ModAdd(&bb[j]);
            if (!got.IsEqual(&exp)) fNorm++;
        }

        C = ifma::mul(A, B); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.ModMulK1(&a[j], &bb[j]);
            if (!got.IsEqual(&exp)) fMul++;
        }

        C = ifma::sqr(A); ifma::store8(C, o);
        for (int j = 0; j < 8; j++) {
            Int got = ifma_limbsToInt(o[j]);
            Int exp; exp.ModSquareK1(&a[j]);
            if (!got.IsEqual(&exp)) fSqr++;
        }
    }

    // gen8: 8-lane EC point add (base +/- G), validated against scalar AddDirect.
    // fCompGen also cross-checks to_compressed8 on the SAME gen8 outputs (which
    // hit reduce8's fold path) against the scalar ifma_limbsToInt + BE + parity.
    int fGenP = 0, fGenM = 0, fCompGen = 0;
    auto checkCompGen = [&](const ifma::FieldVec8 &RX, const ifma::FieldVec8 &RY,
                            uint64_t ox[8][5], uint64_t oy[8][5]) {
        uint8_t gotc[8][33];
        ifma::to_compressed8(RX, RY, &gotc[0][0], 33, 1);
        for (int k = 0; k < 8; k++) {
            Int cx = ifma_limbsToInt(ox[k]), cy = ifma_limbsToInt(oy[k]);
            uint8_t ref[33];
            ref[0] = isEven(&cy) ? 0x02 : 0x03;
            uint8_t *xb = cx.GetBytes();
            for (int b2 = 0; b2 < 32; b2++) ref[1 + b2] = xb[31 - b2];
            if (std::memcmp(gotc[k], ref, 33) != 0) fCompGen++;
        }
    };
    for (int b = 0; b < 500; b++) {
        Int kb; for (int i = 0; i < NB64BLOCK; i++) kb.bits64[i] = 0;
        kb.bits64[0] = ifma_next_rand() | 1ULL; kb.bits64[1] = ifma_next_rand();
        Point P = secp.ComputePublicKey(&kb);

        Int gx[8], gy[8], inv[8];
        Point G[8];
        for (int k = 0; k < 8; k++) {
            Int kk; for (int i = 0; i < NB64BLOCK; i++) kk.bits64[i] = 0;
            kk.bits64[0] = ifma_next_rand() | 1ULL; kk.bits64[1] = ifma_next_rand();
            G[k] = secp.ComputePublicKey(&kk);
            gx[k].Set(&G[k].x); gy[k].Set(&G[k].y);
            Int d; d.ModSub(&G[k].x, &P.x); d.ModInv(); inv[k].Set(&d);
        }
        ifma::FieldVec8 SPX = ifma::broadcast(P.x), SPY = ifma::broadcast(P.y);
        ifma::FieldVec8 GX = ifma::load8(gx), GY = ifma::load8(gy), INV = ifma::load8(inv);
        ifma::FieldVec8 RX, RY;
        uint64_t ox[8][5], oy[8][5];

        ifma::gen8(SPX, SPY, GX, GY, INV, true, RX, RY);
        ifma::store8(RX, ox); ifma::store8(RY, oy);
        for (int k = 0; k < 8; k++) {
            Int gotx = ifma_limbsToInt(ox[k]), goty = ifma_limbsToInt(oy[k]);
            Point e = secp.AddDirect(P, G[k]);
            if (!gotx.IsEqual(&e.x) || !goty.IsEqual(&e.y)) fGenP++;
        }
        checkCompGen(RX, RY, ox, oy);

        ifma::gen8(SPX, SPY, GX, GY, INV, false, RX, RY);
        ifma::store8(RX, ox); ifma::store8(RY, oy);
        for (int k = 0; k < 8; k++) {
            Int gotx = ifma_limbsToInt(ox[k]), goty = ifma_limbsToInt(oy[k]);
            Point negG; negG.x.Set(&G[k].x); negG.y.Set(&G[k].y); negG.y.ModNeg(); negG.z.SetInt32(1);
            Point e = secp.AddDirect(P, negG);
            if (!gotx.IsEqual(&e.x) || !goty.IsEqual(&e.y)) fGenM++;
        }
        checkCompGen(RX, RY, ox, oy);
    }

    // genGroupIFMA: whole 4096-point group validated against ComputePublicKey.
    int fGroup = 0;
    {
        std::vector<Point> Gn(CPU_GROUP_SIZE / 2); Point _2Gn;
        buildGeneratorTable(secp, Gn.data(), _2Gn);
        Int base; for (int i = 0; i < NB64BLOCK; i++) base.bits64[i] = 0;
        base.bits64[0] = 0x100000ULL;
        Int half; half.SetInt32(CPU_GROUP_SIZE / 2);
        Int center; center.Set(&base); center.Add(&half);
        Point startP = secp.ComputePublicKey(&center);
        std::vector<Int> dx(CPU_GROUP_SIZE / 2 + 1);
        IntGroup grp(CPU_GROUP_SIZE / 2 + 1); grp.Set(dx.data());
        std::vector<Point> pts(CPU_GROUP_SIZE);
        genGroupIFMA(startP, Gn.data(), _2Gn, dx, grp, pts.data());
        for (int idx = 0; idx < CPU_GROUP_SIZE; idx++) {
            Int k; k.Set(&base); Int off; off.SetInt32((uint32_t)idx); k.Add(&off);
            Point e = secp.ComputePublicKey(&k);
            if (!pts[idx].x.IsEqual(&e.x) || !pts[idx].y.IsEqual(&e.y)) fGroup++;
        }
    }

    // to_compressed8 (C.1c): SoA -> 33-byte compressed pubkey, vs scalar reference
    // (ifma_limbsToInt canonicalize + big-endian x + y parity). ifma_limbsToInt is
    // the reference, so crafted >= p inputs validate reduce8's subtract path too.
    int fComp = 0;
    {
        auto checkComp = [&](Int *xs, Int *ys) {
            ifma::FieldVec8 X = ifma::load8(xs), Y = ifma::load8(ys);
            uint8_t got[8][33];
            ifma::to_compressed8(X, Y, &got[0][0], 33, 1);
            for (int l = 0; l < 8; l++) {
                uint64_t xl[5], yl[5];
                ifma::int_to_limbs(xs[l], xl); ifma::int_to_limbs(ys[l], yl);
                Int cx = ifma_limbsToInt(xl), cy = ifma_limbsToInt(yl);
                uint8_t ref[33];
                ref[0] = isEven(&cy) ? 0x02 : 0x03;
                uint8_t *xb = cx.GetBytes();
                for (int b = 0; b < 32; b++) ref[1 + b] = xb[31 - b]; // little-endian -> big-endian
                if (std::memcmp(got[l], ref, 33) != 0) fComp++;
            }
        };
        for (int b = 0; b < 4000; b++) {
            Int xs[8], ys[8];
            for (int j = 0; j < 8; j++) { xs[j] = ifma_randFE(); ys[j] = ifma_randFE(); }
            checkComp(xs, ys);
        }
        // Edge values around the [p, 2^256) reduction window.
        auto setW = [](Int &v, uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {
            for (int i = 0; i < NB64BLOCK; i++) v.bits64[i] = 0;
            v.bits64[0] = w0; v.bits64[1] = w1; v.bits64[2] = w2; v.bits64[3] = w3;
        };
        const uint64_t PH = 0xFFFFFFFFFFFFFFFFULL;
        Int edge[8];
        setW(edge[0], 0xFFFFFFFEFFFFFC2FULL, PH, PH, PH);                  // p        -> 0
        setW(edge[1], 0xFFFFFFFEFFFFFC30ULL, PH, PH, PH);                  // p+1      -> 1
        setW(edge[2], PH, PH, PH, PH);                                     // 2^256-1  -> R256-1
        setW(edge[3], 0xFFFFFFFEFFFFFC2EULL, PH, PH, PH);                  // p-1      (unchanged)
        setW(edge[4], 0, 0, 0, 0);                                         // 0
        setW(edge[5], 1, 0, 0, 0);                                         // 1
        setW(edge[6], 0xFFFFFFFEFFFFFC2FULL, PH, PH, 0x7FFFFFFFFFFFFFFFULL); // < p   (no subtract)
        setW(edge[7], 0xFFFFFFFF00000000ULL, PH, PH, PH);                  // window   -> w0-P0
        checkComp(edge, edge);
    }

    // Block path (genGroupIFMABlocks + hash16Blocks) vs Point path (genGroupIFMA
    // + computeHash160BatchBinSingle2) across MULTIPLE groups, reusing one block
    // buffer exactly as the production search does. This replicates Part B's
    // multi-group scenario, so it catches advance / buffer-reuse bugs that a
    // single-group check misses. fBlocks = hash mismatches; fAdv = advance drift.
    int fBlocks = 0, fAdv = 0;
    {
        const int NGROUPS = 6;
        std::vector<Point> Gn(CPU_GROUP_SIZE / 2); Point _2Gn;
        buildGeneratorTable(secp, Gn.data(), _2Gn);
        Int base; for (int i = 0; i < NB64BLOCK; i++) base.bits64[i] = 0;
        base.bits64[0] = 0x100000ULL;
        Int half; half.SetInt32(CPU_GROUP_SIZE / 2);
        Int center; center.Set(&base); center.Add(&half);

        Point sp1 = secp.ComputePublicKey(&center);   // point path startP
        Point sp2 = sp1;                               // block path startP (same start)
        std::vector<Int> dx1(CPU_GROUP_SIZE / 2 + 1), dx2(CPU_GROUP_SIZE / 2 + 1);
        IntGroup grp1(CPU_GROUP_SIZE / 2 + 1); grp1.Set(dx1.data());
        IntGroup grp2(CPU_GROUP_SIZE / 2 + 1); grp2.Set(dx2.data());
        std::vector<Point> pts(CPU_GROUP_SIZE);
        std::vector<uint8_t> blocks((size_t)CPU_GROUP_SIZE * 64);
        initBlocks(blocks.data());                     // ONCE -- buffer is reused below
        std::vector<uint8_t> hashA((size_t)CPU_GROUP_SIZE * 20);
        ALIGN64 uint8_t hr[HASH_BATCH_SIZE][20];

        for (int g = 0; g < NGROUPS; g++) {
            genGroupIFMA(sp1, Gn.data(), _2Gn, dx1, grp1, pts.data());
            for (int i = 0; i < CPU_GROUP_SIZE; i += HASH_BATCH_SIZE) {
                computeHash160BatchBinSingle2(&pts[i], hr);
                for (int L = 0; L < HASH_BATCH_SIZE; L++) std::memcpy(&hashA[(size_t)(i + L) * 20], hr[L], 20);
            }
            genGroupIFMABlocks(sp2, Gn.data(), _2Gn, dx2, grp2, blocks.data());
            for (int i = 0; i < CPU_GROUP_SIZE; i += HASH_BATCH_SIZE) {
                hash16Blocks(blocks.data() + (long)i * 64, hr);
                for (int L = 0; L < HASH_BATCH_SIZE; L++)
                    if (std::memcmp(&hashA[(size_t)(i + L) * 20], hr[L], 20) != 0) fBlocks++;
            }
            if (!sp1.x.IsEqual(&sp2.x) || !sp1.y.IsEqual(&sp2.y)) fAdv++; // advance drift
        }
    }

    auto line = [](const char *name, int fails) {
        std::cout << name << " : " << (fails == 0 ? "OK" : "FAIL");
        if (fails) std::cout << " (" << fails << " mismatches)";
        std::cout << "\n";
    };
    line("conversion round-trip", fRound);
    line("SoA pack/unpack      ", fSoA);
    line("add                  ", fAdd);
    line("sub                  ", fSub);
    line("neg                  ", fNeg);
    line("normalize            ", fNorm);
    line("mul (IFMA)           ", fMul);
    line("sqr (IFMA)           ", fSqr);
    line("gen8 plus            ", fGenP);
    line("gen8 minus           ", fGenM);
    line("to_compressed8 (gen8)", fCompGen);
    line("genGroupIFMA         ", fGroup);
    line("to_compressed8       ", fComp);
    line("block path vs point  ", fBlocks);
    line("block advance drift  ", fAdv);

    int failures = fRound + fSoA + fAdd + fSub + fNeg + fNorm + fMul + fSqr
                 + fGenP + fGenM + fCompGen + fGroup + fComp + fBlocks + fAdv;
    std::cout << "================================================\n";
    std::cout << (failures == 0 ? "IFMA FIELD SELFTEST PASSED\n" : "IFMA FIELD SELFTEST FAILED\n");
    return failures == 0 ? 0 : 1;
}

// Point generation ONLY (no hashing) -- for --bench-gen, to attribute time
// between the scalar EC math and the AVX-512 hashing. Mirrors processGroupFused's
// generation exactly, but instead of hashing returns a checksum of the generated
// coordinates so the work cannot be optimized away. Advances startP by GRP*G.
static uint64_t processGroupGenOnly(Point &startP, Point *Gn, Point &_2Gn,
                                    std::vector<Int> &dx, IntGroup &grp) {
    const int CENTER  = CPU_GROUP_SIZE / 2;
    const int hLength = CENTER - 1;
    Int dy, dyn, _s, _p;
    int j;

    for (j = 0; j < hLength; j++) {
        dx[j].ModSub(&Gn[j].x, &startP.x);
    }
    dx[j].ModSub(&Gn[j].x, &startP.x);
    dx[j + 1].ModSub(&_2Gn.x, &startP.x);
    grp.ModInv();

    Point chunk[HASH_BATCH_SIZE];
    uint64_t sink = 0;

    for (int i = 0; i < CPU_GROUP_SIZE; i += HASH_BATCH_SIZE) {
        for (int L = 0; L < HASH_BATCH_SIZE; L++) {
            const int idx = i + L;
            Point &out = chunk[L];
            if (idx == CENTER) {
                out = startP;
                continue;
            }
            if (idx > CENTER) {
                const int m = idx - CENTER;
                dy.ModSub(&Gn[m - 1].y, &startP.y);
                _s.ModMulK1(&dy, &dx[m - 1]);
                _p.ModSquareK1(&_s);
                out.x.Set(&startP.x);
                out.x.ModNeg();
                out.x.ModAdd(&_p);
                out.x.ModSub(&Gn[m - 1].x);
                out.y.ModSub(&Gn[m - 1].x, &out.x);
                out.y.ModMulK1(&_s);
                out.y.ModSub(&Gn[m - 1].y);
            } else {
                const int m = CENTER - idx;
                dyn.Set(&Gn[m - 1].y);
                dyn.ModNeg();
                dyn.ModSub(&startP.y);
                _s.ModMulK1(&dyn, &dx[m - 1]);
                _p.ModSquareK1(&_s);
                out.x.Set(&startP.x);
                out.x.ModNeg();
                out.x.ModAdd(&_p);
                out.x.ModSub(&Gn[m - 1].x);
                out.y.ModSub(&Gn[m - 1].x, &out.x);
                out.y.ModMulK1(&_s);
                out.y.ModAdd(&Gn[m - 1].y);
            }
        }
        for (int L = 0; L < HASH_BATCH_SIZE; L++) {
            sink ^= chunk[L].x.bits64[0] ^ chunk[L].y.bits64[0];
        }
    }

    Point pp = startP;
    dy.ModSub(&_2Gn.y, &pp.y);
    _s.ModMulK1(&dy, &dx[hLength + 1]);
    _p.ModSquareK1(&_s);
    pp.x.ModNeg();
    pp.x.ModAdd(&_p);
    pp.x.ModSub(&_2Gn.x);
    pp.y.ModSub(&_2Gn.x, &pp.x);
    pp.y.ModMulK1(&_s);
    pp.y.ModSub(&_2Gn.y);
    startP = pp;

    return sink;
}

enum class BenchMode { Full, Gen, Hash, Inv, GenIFMA, GenBlocks, FullIFMA };

//------------------------------------------------------------------------------
// 0.1 Benchmark mode: run a fixed number of batches/thread with no I/O and no
// early exit, then report throughput. Run several times and take the median; use
// the optional thread argument (e.g. "--bench 2000 1") for the single-thread
// number. Modes:
//   Full -> point generation + hash160 (the real hot path)
//   Gen  -> point generation only      (isolates the scalar EC math)
//   Hash -> hash160 only               (isolates the AVX-512 hashing)
// 1/rate_Full ~= 1/rate_Gen + 1/rate_Hash, so the time split is
//   gen fraction  = rate_Full / rate_Gen,  hash fraction = rate_Full / rate_Hash.
//------------------------------------------------------------------------------
static void runBenchmark(Secp256K1 &secp, long long batchesPerThread, int threads,
                         BenchMode mode) {
    if (threads <= 0) threads = omp_get_num_procs();

    // Read-only generator table, built once outside the timed region and shared.
    std::vector<Point> Gn(CPU_GROUP_SIZE / 2);
    Point _2Gn;
    buildGeneratorTable(secp, Gn.data(), _2Gn);

    // A target that essentially never matches, so the loop never early-exits.
    uint8_t dummy[20];
    std::memset(dummy, 0xAB, sizeof(dummy));
    __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dummy));

    unsigned long long sink = 0; // consumes output so it can't be optimized away

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threads) reduction(+:sink)
    {
        const int tid = omp_get_thread_num();

        Int priv;   priv.SetInt32((uint32_t)(tid + 1) * 1000003u); // distinct per thread
        Int half;   half.SetInt32(CPU_GROUP_SIZE / 2);
        Int center; center.Set(&priv); center.Add(&half);
        Point startP = secp.ComputePublicKey(&center);

        std::vector<Int> dx(CPU_GROUP_SIZE / 2 + 1);
        IntGroup grp(CPU_GROUP_SIZE / 2 + 1);
        grp.Set(dx.data());

        if (mode == BenchMode::Hash) {
            // Hash a fixed (but valid, distinct) chunk repeatedly -- hash cost is
            // data-independent, so the same number of hashes/compares as Full.
            Point chunk[HASH_BATCH_SIZE];
            for (int L = 0; L < HASH_BATCH_SIZE; L++) {
                Int kk; kk.SetInt32((uint32_t)(tid + 1) * 1000003u + (uint32_t)L + 1u);
                chunk[L] = secp.ComputePublicKey(&kk);
            }
            ALIGN64 uint8_t hashRes[HASH_BATCH_SIZE][20];
            for (long long b = 0; b < batchesPerThread; ++b) {
                for (int i = 0; i < CPU_GROUP_SIZE; i += HASH_BATCH_SIZE) {
                    computeHash160BatchBinSingle2(chunk, hashRes);
                    for (int L = 0; L < HASH_BATCH_SIZE; L++) {
                        __m128i cand = _mm_loadu_si128(reinterpret_cast<const __m128i *>(hashRes[L]));
                        sink += (unsigned long long)hashRes[L][0];
                        if (_mm_movemask_epi8(_mm_cmpeq_epi8(cand, target16)) == 0xFFFF) sink += 1;
                    }
                }
            }
        } else if (mode == BenchMode::Gen) {
            for (long long b = 0; b < batchesPerThread; ++b) {
                sink += processGroupGenOnly(startP, Gn.data(), _2Gn, dx, grp);
            }
        } else if (mode == BenchMode::GenIFMA) {
            std::vector<Point> pts(CPU_GROUP_SIZE);
            for (long long b = 0; b < batchesPerThread; ++b) {
                genGroupIFMA(startP, Gn.data(), _2Gn, dx, grp, pts.data());
                sink += pts[0].x.bits64[0] ^ pts[CPU_GROUP_SIZE - 1].y.bits64[0];
            }
        } else if (mode == BenchMode::GenBlocks) {
            // Block-path gen ONLY (genGroupIFMABlocks, no hashing) -- isolates the
            // production gen cost so the gen/hash split of --bench-ifma is visible.
            std::vector<uint8_t> blocks((size_t)CPU_GROUP_SIZE * 64);
            initBlocks(blocks.data());
            for (long long b = 0; b < batchesPerThread; ++b) {
                genGroupIFMABlocks(startP, Gn.data(), _2Gn, dx, grp, blocks.data());
                uint64_t acc = 0;          // read spread-out blocks so gen can't be elided
                for (int q = 0; q < CPU_GROUP_SIZE; q += 256) acc ^= blocks[(size_t)q * 64];
                sink += acc;
            }
        } else if (mode == BenchMode::Inv) {
            // Batch inversion (+ its dx setup) only. ModInv overwrites dx in place,
            // so refill the differences each iteration, invert, then consume them.
            const int CENTER  = CPU_GROUP_SIZE / 2;
            const int hLength = CENTER - 1;
            for (long long b = 0; b < batchesPerThread; ++b) {
                int j;
                for (j = 0; j < hLength; j++) dx[j].ModSub(&Gn[j].x, &startP.x);
                dx[j].ModSub(&Gn[j].x, &startP.x);
                dx[j + 1].ModSub(&_2Gn.x, &startP.x);
                grp.ModInv();
                uint64_t acc = 0;
                for (int k = 0; k <= CENTER; k++) acc ^= dx[k].bits64[0];
                sink += acc;
            }
        } else if (mode == BenchMode::FullIFMA) {
            std::vector<uint8_t> blocks((size_t)CPU_GROUP_SIZE * 64);
            initBlocks(blocks.data());
            for (long long b = 0; b < batchesPerThread; ++b) {
                int idx = processGroupIFMA(startP, Gn.data(), _2Gn, dx, grp,
                                           blocks.data(), target16, dummy);
                if (idx >= 0) sink += 1; // never matches; keeps the work live
            }
        } else { // Full
            for (long long b = 0; b < batchesPerThread; ++b) {
                int idx = processGroupFused(startP, Gn.data(), _2Gn, dx, grp, target16, dummy);
                if (idx >= 0) sink += 1; // never matches; keeps the work live
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const unsigned long long totalKeys =
        (unsigned long long)threads * (unsigned long long)batchesPerThread * CPU_GROUP_SIZE;
    const double mkeys = secs > 0.0 ? (totalKeys / secs / 1e6) : 0.0;
    const char *modeName = mode == BenchMode::Gen      ? "GEN  (scalar point-gen only)"
                         : mode == BenchMode::GenIFMA  ? "GEN-IFMA (8-lane point-gen only)"
                         : mode == BenchMode::GenBlocks ? "GEN-BLOCKS (block-path gen only)"
                         : mode == BenchMode::Hash     ? "HASH (hash160 only)"
                         : mode == BenchMode::Inv      ? "INV  (batch inversion only)"
                         : mode == BenchMode::FullIFMA ? "FULL-IFMA (8-lane gen + hash)"
                                                       : "FULL (gen + hash)";

    std::cout << "================= BENCHMARK =================\n";
    std::cout << "Mode             : " << modeName << "\n";
    std::cout << "Threads          : " << threads << "\n";
    std::cout << "Batches/thread   : " << batchesPerThread
              << "  (group size " << CPU_GROUP_SIZE << ")\n";
    std::cout << "Keys processed   : " << totalKeys << "\n";
    std::cout << "Elapsed          : " << std::fixed << std::setprecision(3) << secs << " s\n";
    std::cout << "Throughput       : " << std::fixed << std::setprecision(2) << mkeys
              << " Mkeys/s\n";
    std::cout << "(checksum sink   : " << sink << ")\n";
    std::cout << "============================================\n";
}

//------------------------------------------------------------------------------
static void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " -a <Base58_P2PKH> -r <START:END>\n";
    std::cerr << "       " << programName << " --selftest            (correctness checks)\n";
    std::cerr << "       " << programName << " --selftest-ifma       (SIMD field-arith checks)\n";
    std::cerr << "       " << programName << " --bench [batches] [threads]       (full gen+hash)\n";
    std::cerr << "       " << programName << " --bench-ifma [batches] [threads]  (8-lane gen+hash)\n";
    std::cerr << "       " << programName << " --bench-gen [batches] [threads]   (scalar point-gen)\n";
    std::cerr << "       " << programName << " --bench-gen-ifma [batches] [thr]  (8-lane point-gen)\n";
    std::cerr << "       " << programName << " --bench-gen-blocks [batches][thr]  (block-path gen only)\n";
    std::cerr << "       " << programName << " --bench-hash [batches] [threads]  (hash160 only)\n";
    std::cerr << "       " << programName << " --bench-inv [batches] [threads]   (batch inversion only)\n";
}

static std::string formatElapsedTime(double seconds) {
    int hrs = (int)seconds / 3600;
    int mins = ((int)seconds % 3600) / 60;
    int secs = (int)seconds % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

//------------------------------------------------------------------------------
static void printStatsBlock(int numCPUs, const std::string &targetAddr,
                            const std::string &rangeStr, double mkeysPerSec,
                            unsigned long long totalChecked, double elapsedTime,
                            int progressSaves, long double progressPercent)
{
    static bool firstPrint = true;
    if (!firstPrint) {
        std::cout << "\033[9A";
    } else {
        firstPrint = false;
    }
    std::cout << "================= WORK IN PROGRESS =================\n";
    std::cout << "Target Address: " << targetAddr << "\n";
    std::cout << "CPU Threads   : " << numCPUs << "\n";
    std::cout << "Mkeys/s       : " << std::fixed << std::setprecision(2) << mkeysPerSec << "\n";
    std::cout << "Total Checked : " << totalChecked << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(elapsedTime) << "\n";
    std::cout << "Range         : " << rangeStr << "\n";
    std::cout << "Progress      : " << std::fixed << std::setprecision(4) << progressPercent << " %\n";
    std::cout << "Progress Save : " << progressSaves << "\n";
    std::cout.flush();
}

//------------------------------------------------------------------------------
struct ThreadRange {
    std::string startHex;
    std::string endHex;
};

static std::vector<ThreadRange> g_threadRanges;

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Phase 0 instrumentation entry points, handled before normal arg parsing.
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--selftest")) {
            Secp256K1 secp; secp.Init();
            return runSelfTest(secp);
        }
        if (!std::strcmp(argv[i], "--selftest-ifma")) {
            return runSelfTestIFMA();
        }
        if (!std::strcmp(argv[i], "--bench") ||
            !std::strcmp(argv[i], "--bench-gen") ||
            !std::strcmp(argv[i], "--bench-hash") ||
            !std::strcmp(argv[i], "--bench-inv") ||
            !std::strcmp(argv[i], "--bench-gen-ifma") ||
            !std::strcmp(argv[i], "--bench-gen-blocks") ||
            !std::strcmp(argv[i], "--bench-ifma")) {
            BenchMode mode = BenchMode::Full;
            if (!std::strcmp(argv[i], "--bench-gen"))      mode = BenchMode::Gen;
            if (!std::strcmp(argv[i], "--bench-gen-ifma")) mode = BenchMode::GenIFMA;
            if (!std::strcmp(argv[i], "--bench-gen-blocks")) mode = BenchMode::GenBlocks;
            if (!std::strcmp(argv[i], "--bench-ifma"))     mode = BenchMode::FullIFMA;
            if (!std::strcmp(argv[i], "--bench-hash"))     mode = BenchMode::Hash;
            if (!std::strcmp(argv[i], "--bench-inv"))      mode = BenchMode::Inv;
            long long batches = 1000;
            int benchThreads = 0; // 0 => all available procs
            if (i + 1 < argc) { long long v = std::strtoll(argv[i + 1], nullptr, 10); if (v > 0) batches = v; }
            if (i + 2 < argc) { int t = std::atoi(argv[i + 2]); if (t > 0) benchThreads = t; }
            Secp256K1 secp; secp.Init();
            runBenchmark(secp, batches, benchThreads, mode);
            return 0;
        }
    }

    bool addressProvided = false, rangeProvided = false;
    std::string targetAddress, rangeInput;
    std::vector<uint8_t> targetHash160;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-a") && i + 1 < argc) {
            targetAddress = argv[++i];
            addressProvided = true;
            try {
                targetHash160 = P2PKHDecoder::getHash160(targetAddress);
                if (targetHash160.size() != 20)
                    throw std::invalid_argument("Invalid hash160 length.");
            } catch (const std::exception &ex) {
                std::cerr << "Error parsing address: " << ex.what() << "\n";
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            rangeInput = argv[++i];
            rangeProvided = true;
        } else {
            std::cerr << "Unknown parameter: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (!addressProvided || !rangeProvided) {
        std::cerr << "Both -a <Base58_P2PKH> and -r <START:END> are required!\n";
        printUsage(argv[0]);
        return 1;
    }

    const size_t colonPos = rangeInput.find(':');
    if (colonPos == std::string::npos) {
        std::cerr << "Invalid range format. Use <START:END> in HEX.\n";
        return 1;
    }
    const std::string rangeStartHex = rangeInput.substr(0, colonPos);
    const std::string rangeEndHex   = rangeInput.substr(colonPos + 1);

    auto rangeStart = hexToBigNum(rangeStartHex);
    auto rangeEnd   = hexToBigNum(rangeEndHex);

    bool validRange = false;
    if (rangeStart.size() < rangeEnd.size()) {
        validRange = true;
    } else if (rangeStart.size() > rangeEnd.size()) {
        validRange = false;
    } else {
        validRange = true;
        for (int i = (int)rangeStart.size() - 1; i >= 0; --i) {
            if (rangeStart[i] < rangeEnd[i]) {
                break;
            } else if (rangeStart[i] > rangeEnd[i]) {
                validRange = false;
                break;
            }
        }
    }
    if (!validRange) {
        std::cerr << "Range start must be <= range end.\n";
        return 1;
    }

    auto rangeSize = bigNumSubtract(rangeEnd, rangeStart);
    rangeSize = bigNumAdd(rangeSize, singleElementVector(1ULL));

    const std::string rangeSizeHex = bigNumToHex(rangeSize);
    
    const long double totalRangeLD = hexStrToLongDouble(rangeSizeHex);

    const int numCPUs = omp_get_num_procs();

    g_threadPrivateKeys.resize(numCPUs, Int((uint64_t)0));

    auto [chunkSize, remainder] = bigNumDivide(rangeSize, (uint64_t)numCPUs);
    g_threadRanges.resize(numCPUs);

    std::vector<uint64_t> currentStart = rangeStart;
    for (int t = 0; t < numCPUs; t++) {
        auto currentEnd = bigNumAdd(currentStart, chunkSize);
        if (t < (int)remainder) {
            currentEnd = bigNumAdd(currentEnd, singleElementVector(1ULL));
        }
        currentEnd = bigNumSubtract(currentEnd, singleElementVector(1ULL));

        g_threadRanges[t].startHex = bigNumToHex(currentStart);
        g_threadRanges[t].endHex   = bigNumToHex(currentEnd);

        currentStart = bigNumAdd(currentEnd, singleElementVector(1ULL));
    }
    const std::string displayRange = g_threadRanges.front().startHex + ":" + g_threadRanges.back().endHex;

    unsigned long long globalComparedCount = 0ULL;
    double globalElapsedTime = 0.0;
    double mkeysPerSec       = 0.0;

    const auto tStart = std::chrono::high_resolution_clock::now();
    auto lastStatusTime = tStart;
    auto lastSaveTime   = tStart;

    bool matchFound            = false;
    std::string foundPrivateKeyHex, foundPublicKeyHex, foundWIF;

    Secp256K1 secp;
    secp.Init();

    // PARRALEL COMPUTING BLOCK
    #pragma omp parallel num_threads(numCPUs) \
      shared(globalComparedCount, globalElapsedTime, mkeysPerSec, matchFound, \
             foundPrivateKeyHex, foundPublicKeyHex, foundWIF, \
             tStart, lastStatusTime, lastSaveTime, g_progressSaveCount, \
             g_threadPrivateKeys)
    {
        const int threadId = omp_get_thread_num();

        Int priv = hexToInt(g_threadRanges[threadId].startHex);
        Int threadRangeEnd = hexToInt(g_threadRanges[threadId].endHex);

        Int halfGroupSize; halfGroupSize.SetInt32(CPU_GROUP_SIZE / 2);
        Int privStartMiddleGroup = priv; privStartMiddleGroup.Add(&halfGroupSize);

        Point startP = secp.ComputePublicKey(&privStartMiddleGroup);

        std::vector<Int> dx(CPU_GROUP_SIZE / 2 + 1);
        IntGroup grp(CPU_GROUP_SIZE / 2 + 1);
        grp.Set(dx.data());

        Point Gn[CPU_GROUP_SIZE / 2];
        Point _2Gn;
        buildGeneratorTable(secp, Gn, _2Gn);

        // SHA-block scratch for the IFMA generator (reused every group; the
        // constant message padding is laid down once here).
        std::vector<uint8_t> blocks((size_t)CPU_GROUP_SIZE * 64);
        initBlocks(blocks.data());

        //#pragma omp critical
        {
            g_threadPrivateKeys[threadId].Set((Int*)&priv);
        }

        // Target hash160: first 16 bytes preloaded for the fast SIMD pre-compare.
        __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(targetHash160.data()));

        // main
        while (!matchFound) {
            if (priv.IsGreater((Int*)&threadRangeEnd)) break;

            int idx = processGroupIFMA(startP, Gn, _2Gn, dx, grp, blocks.data(),
                                       target16, targetHash160.data());
            if (idx >= 0) {
#pragma omp critical(full_match)
                {
                    if (!matchFound) {
                        matchFound = true;
                        Int mPriv = priv;
                        Int off; off.SetInt32(idx);
                        mPriv.Add(&off);

                        foundPrivateKeyHex = padHexTo64(intToHex(mPriv));
                        Point fp = secp.ComputePublicKey(&mPriv);
                        foundPublicKeyHex = pointToCompressedHex(fp);
                        foundWIF = P2PKHDecoder::compute_wif(foundPrivateKeyHex, true);
                    }
                }
#pragma omp cancel parallel
            }

            {
                priv.Add((uint64_t)CPU_GROUP_SIZE);
            }

#pragma omp atomic
            globalComparedCount += CPU_GROUP_SIZE;

            if (threadId == 0)
            {
                auto now = std::chrono::high_resolution_clock::now();

                if (std::chrono::duration<double>(now - lastStatusTime).count()
                    >= statusIntervalSec)
                {
#pragma omp critical
                    {
                        globalElapsedTime =
                            std::chrono::duration<double>(now - tStart).count();

                        mkeysPerSec = globalComparedCount / globalElapsedTime / 1e6;
                        long double progressPercent = totalRangeLD > 0.0L
                            ? (globalComparedCount / totalRangeLD * 100.0L)
                            : 0.0L;

                        printStatsBlock(numCPUs, targetAddress, displayRange,
                            mkeysPerSec, globalComparedCount,
                            globalElapsedTime, g_progressSaveCount,
                            progressPercent);

                        lastStatusTime = now;
                    }
                }

                if (std::chrono::duration<double>(now - lastSaveTime).count()
                    >= saveProgressIntervalSec)
                {
                    if (threadId == 0) {
#pragma omp critical
                        {
                            g_progressSaveCount++;
                            std::ostringstream oss;
                            oss << "Progress Save #" << g_progressSaveCount << " at "
                                << std::chrono::duration<double>(now - tStart).count() << " sec: "
                                << "TotalChecked=" << globalComparedCount << ", "
                                << "ElapsedTime=" << formatElapsedTime(globalElapsedTime) << ", "
                                << "Mkeys/s=" << std::fixed << std::setprecision(2) << mkeysPerSec << "\n";
                            for (int k = 0; k < numCPUs; k++) {
                                Int t1;
                                t1.Set((Int*)&g_threadPrivateKeys[k]);
                                oss << "Thread Key " << k << ": " << padHexTo64(intToHex(t1)) << "\n";
                            }
                            saveProgressToFile(oss.str());
                            lastSaveTime = now;
                        }
                    }
                }
            }

            //#pragma omp critical
            {
                g_threadPrivateKeys[threadId].Set((Int*)&priv);
            }
        } // while(true)

    } // end of parralel section

    // Main results
    auto tEnd = std::chrono::high_resolution_clock::now();
    globalElapsedTime = std::chrono::duration<double>(tEnd - tStart).count();

    if (!matchFound) {
        mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
        std::cout << "\nNo match found.\n";
        std::cout << "Total Checked : " << globalComparedCount << "\n";
        std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
        std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
        return 0;
    }
    
    writeFoundKey(foundPrivateKeyHex, foundPublicKeyHex, foundWIF, targetAddress);  

    // If the key was found
    std::cout << "================== FOUND MATCH! ==================\n";
    std::cout << "Private Key   : " << foundPrivateKeyHex << "\n";
    std::cout << "Public Key    : " << foundPublicKeyHex << "\n";
    std::cout << "WIF           : " << foundWIF << "\n";
    std::cout << "P2PKH Address : " << targetAddress << "\n";
    std::cout << "Total Checked : " << globalComparedCount << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
    std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
    return 0;
}
