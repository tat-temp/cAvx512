//g++ -std=c++17 -Ofast -ffast-math -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -mavx512f -mavx512vl -mavx512bw -mavx512dq -fipa-modref -flto -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -o Cyclone Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp ripemd160_avx512.cpp sha256_avx512.cpp

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
    __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(targetH.data()));

    bool found = false;
    std::string foundPrivHex, foundWif;
    while (!found) {
        if (priv.IsGreater(&endK)) break;
        int idx = processGroupFused(startP, Gn.data(), _2Gn, dx, grp, target16, targetH.data());
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
// 0.1 Benchmark mode: run a fixed number of batches/thread of the real hot path
// (point generation + hash160) with no I/O and no early exit, then report
// throughput. Run several times and take the median; use the optional thread
// argument (e.g. "--bench 2000 1") for the single-thread number.
//------------------------------------------------------------------------------
static void runBenchmark(Secp256K1 &secp, long long batchesPerThread, int threads) {
    if (threads <= 0) threads = omp_get_num_procs();

    // Read-only generator table, built once outside the timed region and shared.
    std::vector<Point> Gn(CPU_GROUP_SIZE / 2);
    Point _2Gn;
    buildGeneratorTable(secp, Gn.data(), _2Gn);

    // A target that essentially never matches, so the loop never early-exits.
    uint8_t dummy[20];
    std::memset(dummy, 0xAB, sizeof(dummy));
    __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dummy));

    unsigned long long sink = 0; // consumes hash output so it can't be optimized away

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

        for (long long b = 0; b < batchesPerThread; ++b) {
            int idx = processGroupFused(startP, Gn.data(), _2Gn, dx, grp, target16, dummy);
            if (idx >= 0) sink += 1; // never matches; keeps the hashing live
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const unsigned long long totalKeys =
        (unsigned long long)threads * (unsigned long long)batchesPerThread * CPU_GROUP_SIZE;
    const double mkeys = secs > 0.0 ? (totalKeys / secs / 1e6) : 0.0;

    std::cout << "================= BENCHMARK =================\n";
    std::cout << "Threads          : " << threads << "\n";
    std::cout << "Batches/thread   : " << batchesPerThread
              << "  (group size " << CPU_GROUP_SIZE << ")\n";
    std::cout << "Keys hashed      : " << totalKeys << "\n";
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
    std::cerr << "       " << programName << " --bench [batches] [threads]  (throughput)\n";
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
        if (!std::strcmp(argv[i], "--bench")) {
            long long batches = 1000;
            int benchThreads = 0; // 0 => all available procs
            if (i + 1 < argc) { long long v = std::strtoll(argv[i + 1], nullptr, 10); if (v > 0) batches = v; }
            if (i + 2 < argc) { int t = std::atoi(argv[i + 2]); if (t > 0) benchThreads = t; }
            Secp256K1 secp; secp.Init();
            runBenchmark(secp, batches, benchThreads);
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

        //#pragma omp critical
        {
            g_threadPrivateKeys[threadId].Set((Int*)&priv);
        }

        // Target hash160: first 16 bytes preloaded for the fast SIMD pre-compare.
        __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(targetHash160.data()));

        // main
        while (!matchFound) {
            if (priv.IsGreater((Int*)&threadRangeEnd)) break;

            int idx = processGroupFused(startP, Gn, _2Gn, dx, grp, target16, targetHash160.data());
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
