// search.cuh -- the search kernels.
//
//   k_init_points  : once, scalar-mults each thread's slice-0 center key to a point.
//   k_build_gn     : once, fills the generator table Gn[k]=(k+1)*G and the special
//                    points iG = i*G (per-slice advance) and J (inter-launch jump).
//   k_step         : the hot loop. Each thread runs `slices` batches; each batch is
//                    the symmetric batch-addition trick (center +/- k*G) over i keys
//                    with ONE Montgomery inversion (the i/2 generator x-differences
//                    plus the advance difference share a single fe_inv), hashing each
//                    candidate to hash160 and comparing the target. The center
//                    advances by i*G per slice (folded into that batch's inverse) and
//                    by the precomputed jump J at kernel end, so persistent points
//                    stride perLaunch keys per launch with no re-scalar-mult.
//
// Index mapping mirrors the CPU genGroupIFMABlocks: for a batch whose center holds
// private key `base + i/2`, in-batch index idx in [0,i) holds key `base + idx`:
//   idx == i/2          : the center point
//   idx  < i/2 (minus)  : center - (i/2 - idx)*G
//   idx  > i/2 (plus)   : center + (idx - i/2)*G
#pragma once
#include "field.cuh"
#include "ec.cuh"
#include "hash160.cuh"

namespace search {

struct FoundResult {
    unsigned int  found;       // 0 until a match (set via atomicCAS)
    unsigned int  gid;         // winning global thread id
    unsigned int  slice;       // slice index within the launch
    unsigned int  idx;         // in-batch index [0, i)
    unsigned char pub[33];     // compressed public key of the match
    unsigned int  h160[5];     // matched hash160 (little-endian words)
};

// Target hash160 as 5 little-endian words (set from host via cudaMemcpyToSymbol).
__constant__ uint32_t c_target[5];

__device__ __forceinline__ void check_candidate(const ec::PointA &P, unsigned idx,
                                                unsigned slice, unsigned gid,
                                                FoundResult *res) {
    uint32_t h[5];
    h160::hash160_point(P, h);
    if (h[0] == c_target[0] && h[1] == c_target[1] && h[2] == c_target[2] &&
        h[3] == c_target[3] && h[4] == c_target[4]) {
        if (atomicCAS(&res->found, 0u, 1u) == 0u) {
            res->gid = gid; res->slice = slice; res->idx = idx;
            h160::compressed(P, res->pub);
#pragma unroll
            for (int t = 0; t < 5; t++) res->h160[t] = h[t];
        }
    }
}

// center[g] = (rangeStart + g*perThreadKeys + halfI) * G
__global__ void k_init_points(ec::PointA *centers, unsigned totalThreads,
                              uint64_t rs0, uint64_t rs1, uint64_t rs2, uint64_t rs3,
                              uint64_t perThreadKeys, uint64_t halfI) {
    unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= totalThreads) return;
    uint64_t off = (uint64_t)g * perThreadKeys + halfI;
    uint64_t sc[4];
    uint64_t s = rs0 + off; uint64_t c = (s < rs0); sc[0] = s;
    s = rs1 + c; c = (s < rs1); sc[1] = s;
    s = rs2 + c; c = (s < rs2); sc[2] = s;
    s = rs3 + c;                 sc[3] = s;
    ec::PointA G = ec::ec_gen();
    centers[g] = ec::ec_scalarmul(sc, G);
}

// Gn[k] = (k+1)*G for k in [0,halfI); special[0] = i*G; special[1] = jump*G.
__global__ void k_build_gn(ec::PointA *Gn, unsigned halfI, ec::PointA *special,
                           uint64_t iVal, uint64_t j0, uint64_t j1, uint64_t j2, uint64_t j3) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    ec::PointA G = ec::ec_gen();
    if (k < halfI) Gn[k] = ec::ec_scalarmul_u64((uint64_t)k + 1, G);
    if (k == 0) special[0] = ec::ec_scalarmul_u64(iVal, G);
    if (k == 1) { uint64_t js[4] = {j0, j1, j2, j3}; special[1] = ec::ec_scalarmul(js, G); }
}

__global__ void k_step(ec::PointA *centers, const ec::PointA *__restrict__ Gn,
                       const ec::PointA *__restrict__ special, fe::Elem *prefix,
                       unsigned totalThreads, unsigned halfI, unsigned slices,
                       FoundResult *res) {
    unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= totalThreads) return;
    const unsigned K = halfI + 1;
    ec::PointA C = centers[g];
    const fe::Elem iGx = special[0].x;
    const fe::Elem iGy = special[0].y;

    for (unsigned s = 0; s < slices; s++) {
        if (res->found) break;
        const fe::Elem Cx = C.x, Cy = C.y;

        // forward: prefix[k] = product_{0..k} (Gn[k].x - Cx), advance diff at k=halfI.
        fe::Elem acc = fe::fe_sub(Gn[0].x, Cx);
        prefix[g] = acc;
        for (unsigned k = 1; k < K; k++) {
            fe::Elem gx = (k < halfI) ? Gn[k].x : iGx;
            fe::Elem dk = fe::fe_sub(gx, Cx);
            acc = fe::fe_mul(acc, dk);
            prefix[(uint64_t)k * totalThreads + g] = acc;
        }
        fe::Elem inv = fe::fe_inv(acc);   // 1 / product over all K diffs

        // center point (idx = halfI)
        check_candidate(C, halfI, s, g, res);

        // backward: peel each inverse out, generate +/- points / advance.
        ec::PointA newC = C;
        for (unsigned k = K - 1; k >= 1; k--) {
            bool isAdv = (k == halfI);
            fe::Elem Gx = isAdv ? iGx : Gn[k].x;
            fe::Elem Gy = isAdv ? iGy : Gn[k].y;
            fe::Elem dk = fe::fe_sub(Gx, Cx);
            fe::Elem prefkm1 = prefix[(uint64_t)(k - 1) * totalThreads + g];
            fe::Elem invk = fe::fe_mul(prefkm1, inv);   // 1/dk
            inv = fe::fe_mul(inv, dk);                  // peel
            if (isAdv) {
                newC = ec::ec_batch_add(Cx, Cy, Gx, Gy, invk, true);   // center + i*G
            } else {
                ec::PointA m = ec::ec_batch_add(Cx, Cy, Gx, Gy, invk, false);
                check_candidate(m, halfI - (k + 1), s, g, res);        // minus
                if (k + 1 <= halfI - 1) {                              // plus (if in range)
                    ec::PointA p = ec::ec_batch_add(Cx, Cy, Gx, Gy, invk, true);
                    check_candidate(p, halfI + (k + 1), s, g, res);
                }
            }
        }
        // k = 0 (Gn[0] = 1*G): inv now holds 1/d0.
        {
            ec::PointA m = ec::ec_batch_add(Cx, Cy, Gn[0].x, Gn[0].y, inv, false);
            check_candidate(m, halfI - 1, s, g, res);                  // idx = halfI-1
            if (halfI >= 2) {
                ec::PointA p = ec::ec_batch_add(Cx, Cy, Gn[0].x, Gn[0].y, inv, true);
                check_candidate(p, halfI + 1, s, g, res);              // idx = halfI+1
            }
        }
        C = newC;   // advance to next slice's center
    }

    // inter-launch jump so the next launch continues at +perLaunch keys.
    C = ec::ec_add_direct(C, special[1]);
    centers[g] = C;
}

} // namespace search
