// cyclone_gpu.cu -- CUDA secp256k1 P2PKH puzzle searcher (RTX 5090 / sm_120).
//
// Self-contained GPU port of Cyclone: all elliptic-curve math, SHA-256 and
// RIPEMD-160 run on the device (field.cuh / ec.cuh / hash160.cuh); the host only
// decodes the target address, sizes the launch, reconstructs a found key, and
// prints/saves it (u256.h / host_hash.h).
//
//   cyclone_gpu -a <addr> -r <START:END> --grid i,j --slices N    search
//   cyclone_gpu --selftest                                        correctness
//   cyclone_gpu --bench [launches] [--grid i,j] [--slices N]      throughput
//
// Build (WSL2/Linux, CUDA 13.2):
//   nvcc -O3 -arch=sm_120 -o cyclone_gpu cyclone_gpu.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>

#include "field.cuh"
#include "ec.cuh"
#include "hash160.cuh"
#include "search.cuh"
#include "u256.h"
#include "host_hash.h"

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,            \
                     __FILE__, __LINE__, cudaGetErrorString(_e));              \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// ============================================================================
// Host big-integer oracle (independent of the device fast-fold reduction): used
// only by --selftest to cross-check field arithmetic. Reduces mod p by binary
// long division, so a bug in the device fold/carry chains shows up as a mismatch.
// ============================================================================
namespace oracle {

static const uint64_t P[4]   = {FE_P0, FE_P1, FE_P2, FE_P3};
static const uint64_t PM2[4] = {0xFFFFFFFEFFFFFC2DULL, FE_P1, FE_P2, FE_P3}; // p-2

static bool ge4(const uint64_t a[4], const uint64_t b[4]) {
    for (int k = 3; k >= 0; k--) { if (a[k] != b[k]) return a[k] > b[k]; }
    return true;
}
static void sub4(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow = 0;
    for (int k = 0; k < 4; k++) {
        uint64_t d = a[k] - b[k]; uint64_t br = (a[k] < b[k]);
        uint64_t d2 = d - borrow; br += (d < borrow);
        r[k] = d2; borrow = br;
    }
}
// canonicalize a 4-limb value into [0,p)
static void canon(uint64_t a[4]) { if (ge4(a, P)) sub4(a, P, a); }

static void addmod(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    for (int k = 0; k < 4; k++) {
        uint64_t s = a[k] + b[k]; uint64_t c = (s < a[k]);
        s += carry; c += (s < carry); r[k] = s; carry = c;
    }
    // value (carry:r) < 2p: subtract p if carry or r>=p
    if (carry || ge4(r, P)) sub4(r, P, r);
}
static void submod(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow = 0;
    for (int k = 0; k < 4; k++) {
        uint64_t d = a[k] - b[k]; uint64_t br = (a[k] < b[k]);
        uint64_t d2 = d - borrow; br += (d < borrow);
        r[k] = d2; borrow = br;
    }
    if (borrow) { uint64_t c = 0;             // add p back
        for (int k = 0; k < 4; k++) {
            uint64_t s = r[k] + P[k]; uint64_t cc = (s < r[k]);
            s += c; cc += (s < c); r[k] = s; c = cc;
        }
    }
}
static void mul512(const uint64_t a[4], const uint64_t b[4], uint64_t r[8]) {
    for (int k = 0; k < 8; k++) r[k] = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 cur = (unsigned __int128)a[i] * b[j] + r[i + j] + (uint64_t)carry;
            r[i + j] = (uint64_t)cur;
            carry = cur >> 64;
        }
        r[i + 4] += (uint64_t)carry;
    }
}
// 512-bit product mod p via bit-by-bit long division (independent of the fold).
static void mod512(const uint64_t prod[8], uint64_t out[4]) {
    uint64_t rem[5] = {0, 0, 0, 0, 0};
    const uint64_t p5[5] = {P[0], P[1], P[2], P[3], 0};
    for (int bit = 511; bit >= 0; bit--) {
        // rem <<= 1
        uint64_t carry = 0;
        for (int k = 0; k < 5; k++) { uint64_t nc = rem[k] >> 63; rem[k] = (rem[k] << 1) | carry; carry = nc; }
        rem[0] |= (prod[bit >> 6] >> (bit & 63)) & 1ULL;
        // if rem >= p, subtract
        bool ge = rem[4] != 0;
        if (!ge) { for (int k = 4; k >= 0; k--) { if (rem[k] != p5[k]) { ge = rem[k] > p5[k]; break; } if (k == 0) ge = true; } }
        if (ge) {
            uint64_t borrow = 0;
            for (int k = 0; k < 5; k++) {
                uint64_t d = rem[k] - p5[k]; uint64_t br = (rem[k] < p5[k]);
                uint64_t d2 = d - borrow; br += (d < borrow);
                rem[k] = d2; borrow = br;
            }
        }
    }
    for (int k = 0; k < 4; k++) out[k] = rem[k];
}
static void mulmod(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t prod[8]; mul512(a, b, prod); mod512(prod, r);
}
static void invmod(const uint64_t a[4], uint64_t r[4]) {
    uint64_t result[4] = {1, 0, 0, 0};
    uint64_t base[4]; for (int k = 0; k < 4; k++) base[k] = a[k];
    for (int bit = 0; bit < 256; bit++) {
        if ((PM2[bit >> 6] >> (bit & 63)) & 1ULL) mulmod(result, base, result);
        mulmod(base, base, base);
    }
    for (int k = 0; k < 4; k++) r[k] = result[k];
}

} // namespace oracle

// ============================================================================
// Device test kernels (selftest only)
// ============================================================================
__global__ void k_test_field(const uint64_t *A, const uint64_t *B, int N,
                             uint64_t *Radd, uint64_t *Rsub, uint64_t *Rmul,
                             uint64_t *Rsqr, uint64_t *Rinv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fe::Elem a, b;
#pragma unroll
    for (int k = 0; k < 4; k++) { a.v[k] = A[i*4+k]; b.v[k] = B[i*4+k]; }
    fe::Elem r;
    r = fe::fe_add(a, b); for (int k=0;k<4;k++) Radd[i*4+k]=r.v[k];
    r = fe::fe_sub(a, b); for (int k=0;k<4;k++) Rsub[i*4+k]=r.v[k];
    r = fe::fe_mul(a, b); for (int k=0;k<4;k++) Rmul[i*4+k]=r.v[k];
    r = fe::fe_sqr(a);    for (int k=0;k<4;k++) Rsqr[i*4+k]=r.v[k];
    r = fe::fe_inv(a);    for (int k=0;k<4;k++) Rinv[i*4+k]=r.v[k];
}

// P = scalar*G ; verify y^2 == x^3 + 7 (mod p). Independent of any external vector.
__global__ void k_oncurve(const uint64_t *scalars, int N, int *ok) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t s[4]; for (int k=0;k<4;k++) s[k]=scalars[i*4+k];
    ec::PointA G = ec::ec_gen();
    ec::PointA P = ec::ec_scalarmul(s, G);
    fe::Elem lhs = fe::fe_sqr(P.y);
    fe::Elem rhs = fe::fe_add(fe::fe_mul(fe::fe_sqr(P.x), P.x), fe::fe_set_u64(7));
    ok[i] = fe::fe_eq(lhs, rhs) ? 1 : 0;
}

// hash160 of (k*G) -> outH[5] (LE words) and outPub[33].
__global__ void k_hash_known(uint64_t k, uint32_t *outH, unsigned char *outPub) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    ec::PointA G = ec::ec_gen();
    ec::PointA P = ec::ec_scalarmul_u64(k, G);
    h160::hash160_point(P, outH);
    h160::compressed(P, outPub);
}

// ============================================================================
// Search context: device buffers + launch config
// ============================================================================
struct Config {
    uint64_t i, j, slices;
    unsigned numBlocks;
    uint64_t halfI, K;
    uint64_t totalThreads;
    uint64_t perThreadKeys;   // i * slices
    uint64_t perLaunch;       // totalThreads * perThreadKeys
};

static Config make_config(uint64_t i, uint64_t j, uint64_t slices) {
    if (i < 2 || (i & (i - 1)) != 0) throw std::invalid_argument("--grid i must be a power of 2 >= 2");
    if (j < 1) throw std::invalid_argument("--grid j (threads/block) must be >= 1");
    if (slices < 1) throw std::invalid_argument("--slices must be >= 1");

    Config c{};
    c.i = i; c.j = j; c.slices = slices;
    c.halfI = i / 2;
    c.K = c.halfI + 1;
    c.perThreadKeys = i * slices;

    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, search::k_step<true>, (int)j, 0));
    if (blocksPerSM < 1) blocksPerSM = 1;
    unsigned numBlocks = (unsigned)blocksPerSM * (unsigned)prop.multiProcessorCount;
    if (const char *e = std::getenv("CYCLONE_BLOCKS")) { int v = std::atoi(e); if (v > 0) numBlocks = (unsigned)v; }

    // Cap by available memory for the per-thread Montgomery prefix scratch
    // (K * totalThreads field elements) plus the per-thread center points.
    size_t freeB = 0, totalB = 0; CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    double budget = (double)freeB * 0.85;
    double perThread = (double)c.K * sizeof(fe::Elem) + sizeof(ec::PointA);
    uint64_t maxThreads = (uint64_t)(budget / perThread);
    uint64_t want = (uint64_t)numBlocks * j;
    if (want > maxThreads) {
        unsigned maxBlocks = (unsigned)(maxThreads / j);
        if (maxBlocks < 1) throw std::invalid_argument("batch size i too large for device memory at this --grid j");
        std::fprintf(stderr, "[note] reducing blocks %u -> %u to fit prefix scratch in VRAM\n", numBlocks, maxBlocks);
        numBlocks = maxBlocks;
    }
    c.numBlocks = numBlocks;
    c.totalThreads = (uint64_t)numBlocks * j;
    if (c.perThreadKeys != 0 && c.totalThreads > (UINT64_MAX / c.perThreadKeys))
        throw std::invalid_argument("keys-per-launch overflows 64-bit; reduce --grid/--slices");
    c.perLaunch = c.totalThreads * c.perThreadKeys;
    return c;
}

struct SearchCtx {
    Config cfg;
    ec::PointA *d_Gn = nullptr;
    ec::PointA *d_special = nullptr;
    ec::PointA *d_centers = nullptr;
    fe::Elem   *d_prefix = nullptr;
    search::FoundResult *d_res = nullptr;

    void alloc() {
        CUDA_CHECK(cudaMalloc(&d_Gn, cfg.halfI * sizeof(ec::PointA)));
        CUDA_CHECK(cudaMalloc(&d_special, 2 * sizeof(ec::PointA)));
        CUDA_CHECK(cudaMalloc(&d_centers, cfg.totalThreads * sizeof(ec::PointA)));
        CUDA_CHECK(cudaMalloc(&d_prefix, cfg.K * cfg.totalThreads * sizeof(fe::Elem)));
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(search::FoundResult)));
    }
    void free_all() {
        cudaFree(d_Gn); cudaFree(d_special); cudaFree(d_centers);
        cudaFree(d_prefix); cudaFree(d_res);
    }

    // Build Gn / special points and init each thread's center for rangeStart.
    void build(const u256 &rangeStart) {
        uint64_t jump = cfg.perLaunch - cfg.perThreadKeys;   // (totalThreads-1)*perThreadKeys
        unsigned gnThreads = (unsigned)((cfg.halfI < 2) ? 2 : cfg.halfI);
        unsigned gnBlocks = (gnThreads + 255) / 256;
        search::k_build_gn<<<gnBlocks, 256>>>(d_Gn, (unsigned)cfg.halfI, d_special,
                                      cfg.i, jump, 0, 0, 0);
        CUDA_CHECK(cudaGetLastError());

        unsigned initBlocks = (unsigned)((cfg.totalThreads + cfg.j - 1) / cfg.j);
        search::k_init_points<<<initBlocks, (unsigned)cfg.j>>>(d_centers, (unsigned)cfg.totalThreads,
            rangeStart.v[0], rangeStart.v[1], rangeStart.v[2], rangeStart.v[3],
            cfg.perThreadKeys, cfg.halfI);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void reset_found() { CUDA_CHECK(cudaMemset(d_res, 0, sizeof(search::FoundResult))); }

    void launch_step(bool genOnly = false) {
        if (genOnly)
            search::k_step<false><<<cfg.numBlocks, (unsigned)cfg.j>>>(d_centers, d_Gn, d_special, d_prefix,
                (unsigned)cfg.totalThreads, (unsigned)cfg.halfI, (unsigned)cfg.slices, d_res);
        else
            search::k_step<true><<<cfg.numBlocks, (unsigned)cfg.j>>>(d_centers, d_Gn, d_special, d_prefix,
                (unsigned)cfg.totalThreads, (unsigned)cfg.halfI, (unsigned)cfg.slices, d_res);
        CUDA_CHECK(cudaGetLastError());
    }
};

static void set_target_words(const uint8_t h160[20]) {
    uint32_t words[5];
    for (int t = 0; t < 5; t++)
        words[t] = (uint32_t)h160[4*t] | ((uint32_t)h160[4*t+1] << 8) |
                   ((uint32_t)h160[4*t+2] << 16) | ((uint32_t)h160[4*t+3] << 24);
    CUDA_CHECK(cudaMemcpyToSymbol(search::c_target, words, sizeof(words)));
}

static void h160_words_to_bytes(const uint32_t w[5], uint8_t out[20]) {
    for (int t = 0; t < 5; t++) {
        out[4*t]   = (uint8_t)(w[t]);       out[4*t+1] = (uint8_t)(w[t] >> 8);
        out[4*t+2] = (uint8_t)(w[t] >> 16); out[4*t+3] = (uint8_t)(w[t] >> 24);
    }
}

// ============================================================================
// Status / output helpers
// ============================================================================
static std::string fmt_elapsed(double seconds) {
    int hrs = (int)seconds / 3600, mins = ((int)seconds % 3600) / 60, secs = (int)seconds % 60;
    char buf[16]; std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hrs, mins, secs);
    return std::string(buf);
}

static std::string bytes_to_hex(const uint8_t *b, int n) {
    static const char *H = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (int i = 0; i < n; i++) { s += H[(b[i] >> 4) & 0xF]; s += H[b[i] & 0xF]; }
    return s;
}

// ============================================================================
// Self-test
// ============================================================================
static uint64_t sm_state = 0x243F6A8885A308D3ULL;
static uint64_t splitmix64() {
    sm_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = sm_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int run_search(const Config &cfg, const u256 &rangeStart, const u256 &rangeSize,
                      const std::string &targetAddr, const uint8_t targetH160[20],
                      bool quiet, u256 *foundKeyOut);

static int run_selftest() {
    std::cout << "============== GPU SELF TEST ==============\n";
    int failures = 0;

    // --- field arithmetic vs the independent host oracle ----------------------
    const int N = 8192;
    std::vector<uint64_t> A(N*4), B(N*4);
    for (int i = 0; i < N; i++) {
        uint64_t a[4], b[4];
        do { for (int k=0;k<4;k++) a[k]=splitmix64(); oracle::canon(a); } while ((a[0]|a[1]|a[2]|a[3])==0);
        for (int k=0;k<4;k++) b[k]=splitmix64(); oracle::canon(b);
        for (int k=0;k<4;k++){ A[i*4+k]=a[k]; B[i*4+k]=b[k]; }
    }
    uint64_t *dA,*dB,*dAdd,*dSub,*dMul,*dSqr,*dInv;
    size_t sz = (size_t)N*4*sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&dA,sz)); CUDA_CHECK(cudaMalloc(&dB,sz));
    CUDA_CHECK(cudaMalloc(&dAdd,sz)); CUDA_CHECK(cudaMalloc(&dSub,sz));
    CUDA_CHECK(cudaMalloc(&dMul,sz)); CUDA_CHECK(cudaMalloc(&dSqr,sz)); CUDA_CHECK(cudaMalloc(&dInv,sz));
    CUDA_CHECK(cudaMemcpy(dA,A.data(),sz,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,B.data(),sz,cudaMemcpyHostToDevice));
    k_test_field<<<(N+255)/256,256>>>(dA,dB,N,dAdd,dSub,dMul,dSqr,dInv);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> Radd(N*4),Rsub(N*4),Rmul(N*4),Rsqr(N*4),Rinv(N*4);
    CUDA_CHECK(cudaMemcpy(Radd.data(),dAdd,sz,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Rsub.data(),dSub,sz,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Rmul.data(),dMul,sz,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Rsqr.data(),dSqr,sz,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Rinv.data(),dInv,sz,cudaMemcpyDeviceToHost));

    int fAdd=0,fSub=0,fMul=0,fSqr=0,fInv=0;
    auto eq4=[](const uint64_t*x,const uint64_t*y){return x[0]==y[0]&&x[1]==y[1]&&x[2]==y[2]&&x[3]==y[3];};
    for (int i=0;i<N;i++){
        const uint64_t *a=&A[i*4], *b=&B[i*4];
        uint64_t e[4];
        oracle::addmod(a,b,e); if(!eq4(e,&Radd[i*4])) fAdd++;
        oracle::submod(a,b,e); if(!eq4(e,&Rsub[i*4])) fSub++;
        oracle::mulmod(a,b,e); if(!eq4(e,&Rmul[i*4])) fMul++;
        oracle::mulmod(a,a,e); if(!eq4(e,&Rsqr[i*4])) fSqr++;
        oracle::invmod(a,e);   if(!eq4(e,&Rinv[i*4])) fInv++;
    }
    cudaFree(dA);cudaFree(dB);cudaFree(dAdd);cudaFree(dSub);cudaFree(dMul);cudaFree(dSqr);cudaFree(dInv);
    auto line=[&](const char*n,int f){ std::cout<<n<<" : "<<(f==0?"OK":"FAIL"); if(f)std::cout<<" ("<<f<<" mismatches)"; std::cout<<"\n"; if(f)failures++; };
    line("field add          ",fAdd);
    line("field sub          ",fSub);
    line("field mul          ",fMul);
    line("field sqr          ",fSqr);
    line("field inv (Fermat) ",fInv);

    // --- EC scalar mult on-curve invariant (y^2 == x^3 + 7) -------------------
    const int M = 4096;
    std::vector<uint64_t> S(M*4);
    for (int i=0;i<M;i++){ for(int k=0;k<4;k++) S[i*4+k]=splitmix64(); S[i*4]|=1ULL; }
    uint64_t *dS; int *dOk; std::vector<int> Ok(M);
    CUDA_CHECK(cudaMalloc(&dS,(size_t)M*4*sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dOk,M*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dS,S.data(),(size_t)M*4*sizeof(uint64_t),cudaMemcpyHostToDevice));
    k_oncurve<<<(M+255)/256,256>>>(dS,M,dOk);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Ok.data(),dOk,M*sizeof(int),cudaMemcpyDeviceToHost));
    cudaFree(dS); cudaFree(dOk);
    int fEC=0; for(int i=0;i<M;i++) if(!Ok[i]) fEC++;
    line("EC scalarmul curve ",fEC);

    // --- hash160(0xABCDEF * G) vs the known address ---------------------------
    const std::string knownAddr = "1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L";
    std::vector<uint8_t> tgt;
    int fAddr = 0;
    try { tgt = hosth::address_to_hash160(knownAddr); } catch (...) { tgt.clear(); }
    uint32_t *dH; unsigned char *dPub; uint32_t hH[5]; unsigned char hPub[33];
    CUDA_CHECK(cudaMalloc(&dH,5*sizeof(uint32_t))); CUDA_CHECK(cudaMalloc(&dPub,33));
    k_hash_known<<<1,1>>>(0xABCDEFULL, dH, dPub);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hH,dH,5*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPub,dPub,33,cudaMemcpyDeviceToHost));
    cudaFree(dH); cudaFree(dPub);
    uint8_t gotH[20]; h160_words_to_bytes(hH, gotH);
    if (tgt.size()!=20 || std::memcmp(gotH, tgt.data(), 20)!=0) fAddr=1;
    std::cout << "hash160 known-vec  : " << (fAddr?"FAIL":"OK")
              << " (" << bytes_to_hex(gotH,20) << ")\n";
    if (fAddr) failures++;

    // --- end-to-end search finds 0xABCDEF over 100000:FFFFFF ------------------
    int fE2E = 0;
    if (tgt.size()==20) {
        try {
            Config c = make_config(256, 64, 4);
            u256 rs = u256::from_hex("100000");
            u256 re = u256::from_hex("FFFFFF");
            u256 size = add_u64(sub(re, rs), 1);
            u256 foundKey;
            int rc = run_search(c, rs, size, knownAddr, tgt.data(), /*quiet=*/true, &foundKey);
            u256 expect = u256::from_hex("ABCDEF");
            fE2E = (rc==1 && cmp(foundKey, expect)==0) ? 0 : 1;
            std::cout << "end-to-end search  : " << (fE2E?"FAIL":"OK");
            if (rc==1) std::cout << " (found " << foundKey.to_hex64().substr(58) << ")";
            std::cout << "\n";
        } catch (const std::exception &ex) { fE2E=1; std::cout << "end-to-end search  : FAIL ("<<ex.what()<<")\n"; }
    } else { fE2E=1; std::cout << "end-to-end search  : SKIP (address decode failed)\n"; }
    if (fE2E) failures++;

    std::cout << "==========================================\n";
    std::cout << (failures==0 ? "GPU SELFTEST PASSED\n" : "GPU SELFTEST FAILED\n");
    return failures==0 ? 0 : 1;
}

// ============================================================================
// Search driver (shared by production and the end-to-end self-test)
// ============================================================================
static int run_search(const Config &cfg, const u256 &rangeStart, const u256 &rangeSize,
                      const std::string &targetAddr, const uint8_t targetH160[20],
                      bool quiet, u256 *foundKeyOut) {
    set_target_words(targetH160);
    SearchCtx ctx; ctx.cfg = cfg; ctx.alloc(); ctx.build(rangeStart); ctx.reset_found();

    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    if (!quiet) {
        std::cout << "GPU            : " << prop.name << " (" << prop.multiProcessorCount << " SMs)\n";
        std::cout << "Grid           : " << cfg.numBlocks << " blocks x " << cfg.j
                  << " threads,  i=" << cfg.i << "  slices=" << cfg.slices << "\n";
        std::cout << "Keys/launch    : " << cfg.perLaunch << "\n\n";
    }

    u256 launchBase;                         // offset of the current launch from rangeStart
    auto t0 = std::chrono::high_resolution_clock::now();
    auto lastStatus = t0, lastSave = t0;
    long double sizeLD = to_long_double(rangeSize);
    int result = 0;

    while (cmp(launchBase, rangeSize) < 0) {
        ctx.launch_step();
        CUDA_CHECK(cudaDeviceSynchronize());

        search::FoundResult res;
        CUDA_CHECK(cudaMemcpy(&res, ctx.d_res, sizeof(res), cudaMemcpyDeviceToHost));
        if (res.found) {
            uint64_t within = (uint64_t)res.gid * cfg.perThreadKeys
                            + (uint64_t)res.slice * cfg.i + res.idx;
            u256 key = add_u64(add(rangeStart, launchBase), within);
            if (foundKeyOut) *foundKeyOut = key;
            if (!quiet) {
                uint8_t mb[20]; h160_words_to_bytes(res.h160, mb);
                std::string priv = key.to_hex64();
                uint8_t pk[32]; key.to_bytes_be(pk);
                std::string wif = hosth::to_wif(pk, true);
                std::string pub = bytes_to_hex(res.pub, 33);
                std::string addr = hosth::hash160_to_address(mb);
                std::cout << "\n================== FOUND MATCH! ==================\n";
                std::cout << "Private Key   : " << priv << "\n";
                std::cout << "Public Key    : " << pub << "\n";
                std::cout << "WIF           : " << wif << "\n";
                std::cout << "P2PKH Address : " << addr << "\n";
                std::cout << "Verify        : " << (addr==targetAddr ? "OK (address matches)" : "MISMATCH") << "\n";
                std::ofstream ofs("found_keys.txt", std::ios::app);
                if (ofs) ofs << priv << ' ' << pub << ' ' << wif << ' ' << addr << "\n";
            }
            result = 1;
            break;
        }

        launchBase = add_u64(launchBase, cfg.perLaunch);

        if (!quiet) {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - lastStatus).count() >= 5.0) {
                double el = std::chrono::duration<double>(now - t0).count();
                long double doneLD = to_long_double(launchBase);
                double mk = el > 0 ? (double)(doneLD / 1e6 / el) : 0.0;
                long double pct = sizeLD > 0 ? (doneLD / sizeLD * 100.0L) : 0.0L;
                std::printf("\rMkeys/s %.2f | checked %.0Lf | %s | %.4Lf%%      ",
                            mk, doneLD, fmt_elapsed(el).c_str(), pct);
                std::fflush(stdout);
                lastStatus = now;
            }
            if (std::chrono::duration<double>(now - lastSave).count() >= 300.0) {
                std::ofstream ofs("progress.txt", std::ios::app);
                if (ofs) ofs << "checked=" << launchBase.to_hex64() << "\n";
                lastSave = now;
            }
        }
    }

    if (!quiet && !result) std::cout << "\nNo match found in range.\n";
    ctx.free_all();
    return result;
}

// ============================================================================
// Benchmark
// ============================================================================
static void run_bench(uint64_t i, uint64_t j, uint64_t slices, int launches, bool genOnly) {
    Config cfg = make_config(i, j, slices);
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    uint8_t dummy[20]; std::memset(dummy, 0xAB, 20);
    set_target_words(dummy);
    SearchCtx ctx; ctx.cfg = cfg; ctx.alloc();
    u256 rs = u256::from_hex("8000000000000000");   // arbitrary high start
    ctx.build(rs); ctx.reset_found();

    ctx.launch_step(genOnly); CUDA_CHECK(cudaDeviceSynchronize());   // warm
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int l = 0; l < launches; l++) ctx.launch_step(genOnly);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    long double keys = (long double)cfg.perLaunch * launches;
    double mk = secs > 0 ? (double)(keys / 1e6 / secs) : 0.0;

    std::cout << "================= GPU BENCHMARK =================\n";
    std::cout << "GPU            : " << prop.name << "\n";
    std::cout << "Mode           : " << (genOnly ? "GEN-ONLY (EC + Montgomery, no hash)"
                                                 : "FULL (gen + hash160)") << "\n";
    std::cout << "Grid           : " << cfg.numBlocks << " x " << cfg.j
              << "  i=" << cfg.i << "  slices=" << cfg.slices << "\n";
    std::cout << "Keys/launch    : " << cfg.perLaunch << "\n";
    std::cout << "Launches       : " << launches << "\n";
    std::cout << "Elapsed        : " << secs << " s\n";
    std::cout << "Throughput     : " << mk << " Mkeys/s\n";
    std::cout << "================================================\n";
    ctx.free_all();
}

// ============================================================================
// main
// ============================================================================
static void usage(const char *p) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s -a <Base58_P2PKH> -r <START:END> --grid i,j --slices N\n"
        "  %s --selftest\n"
        "  %s --bench [launches] [--grid i,j] [--slices N]      full pipeline\n"
        "  %s --bench-gen [launches] [--grid i,j] [--slices N]  EC+Montgomery only (no hash)\n"
        "    --grid i,j : i = keys per batch per thread (power of 2), j = threads per block\n"
        "    --slices N : batches per thread per kernel launch\n", p, p, p, p);
}

static bool parse_grid(const char *s, uint64_t &i, uint64_t &j) {
    const char *comma = std::strchr(s, ',');
    if (!comma) return false;
    i = std::strtoull(s, nullptr, 10);
    j = std::strtoull(comma + 1, nullptr, 10);
    return i > 0 && j > 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    for (int a = 1; a < argc; a++) if (!std::strcmp(argv[a], "--selftest")) return run_selftest();

    bool bench = false, benchGen = false;
    int launches = 10;
    uint64_t gi = 512, gj = 256, slices = 16;   // bench defaults
    std::string addr, range;
    bool haveAddr = false, haveRange = false, haveGrid = false, haveSlices = false;

    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        if (s == "--bench" || s == "--bench-gen") {
            bench = true;
            benchGen = (s == "--bench-gen");
            if (a + 1 < argc && argv[a+1][0] != '-') launches = std::atoi(argv[++a]);
        } else if ((s == "-a" || s == "--address") && a + 1 < argc) {
            addr = argv[++a]; haveAddr = true;
        } else if ((s == "-r" || s == "--range") && a + 1 < argc) {
            range = argv[++a]; haveRange = true;
        } else if (s == "--grid" && a + 1 < argc) {
            if (!parse_grid(argv[++a], gi, gj)) { std::fprintf(stderr, "bad --grid\n"); return 1; }
            haveGrid = true;
        } else if (s == "--slices" && a + 1 < argc) {
            slices = std::strtoull(argv[++a], nullptr, 10); haveSlices = true;
        } else {
            std::fprintf(stderr, "Unknown/!incomplete argument: %s\n", argv[a]); usage(argv[0]); return 1;
        }
    }

    try {
        if (bench) { run_bench(gi, gj, slices, launches, benchGen); return 0; }

        if (!haveAddr || !haveRange || !haveGrid || !haveSlices) {
            std::fprintf(stderr, "search needs -a, -r, --grid i,j and --slices N\n");
            usage(argv[0]); return 1;
        }
        size_t colon = range.find(':');
        if (colon == std::string::npos) { std::fprintf(stderr, "range must be START:END (hex)\n"); return 1; }
        u256 rs = u256::from_hex(range.substr(0, colon));
        u256 re = u256::from_hex(range.substr(colon + 1));
        if (cmp(rs, re) > 0) { std::fprintf(stderr, "range start must be <= end\n"); return 1; }
        u256 size = add_u64(sub(re, rs), 1);

        std::vector<uint8_t> h160 = hosth::address_to_hash160(addr);
        if (h160.size() != 20) { std::fprintf(stderr, "address decode failed\n"); return 1; }

        Config cfg = make_config(gi, gj, slices);
        std::cout << "================= CYCLONE GPU =================\n";
        std::cout << "Target Address : " << addr << "\n";
        std::cout << "Range          : " << rs.to_hex64() << " : " << re.to_hex64() << "\n";
        int rc = run_search(cfg, rs, size, addr, h160.data(), /*quiet=*/false, nullptr);
        return rc == 1 ? 0 : 2;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
