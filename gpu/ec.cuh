// ec.cuh -- secp256k1 elliptic-curve ops on the device.
//
// Two layers:
//   * Jacobian double-and-add scalar multiplication (ec_scalarmul) -- used once at
//     startup to derive each thread's center point and to build the generator table.
//     No per-step inversion; one final inversion converts to affine.
//   * Affine "batch add" (ec_batch_add) -- the hot path. Given a precomputed inverse
//     1/(Gx - Cx) it adds (or subtracts) a generator multiple to the center point
//     using the exact slope formula of SECP256K1.cpp::AddDirect, so the GPU group
//     generator mirrors the CPU processGroupFused / genGroupIFMABlocks math.
#pragma once
#include "field.cuh"

namespace ec {

struct PointA { fe::Elem x, y; };               // affine
struct JacP   { fe::Elem X, Y, Z; bool inf; };  // Jacobian (x=X/Z^2, y=Y/Z^3)

// secp256k1 generator G (limbs little-endian; see SECP256K1.cpp Init()).
__device__ __forceinline__ PointA ec_gen() {
    PointA g;
    g.x.v[0] = 0x59F2815B16F81798ULL; g.x.v[1] = 0x029BFCDB2DCE28D9ULL;
    g.x.v[2] = 0x55A06295CE870B07ULL; g.x.v[3] = 0x79BE667EF9DCBBACULL;
    g.y.v[0] = 0x9C47D08FFB10D4B8ULL; g.y.v[1] = 0xFD17B448A6855419ULL;
    g.y.v[2] = 0x5DA4FBFC0E1108A8ULL; g.y.v[3] = 0x483ADA7726A3C465ULL;
    return g;
}

__device__ __forceinline__ fe::Elem fe_dbl(const fe::Elem &a) { return fe::fe_add(a, a); }
__device__ __forceinline__ fe::Elem fe_mul3(const fe::Elem &a) { return fe::fe_add(fe_dbl(a), a); }
__device__ __forceinline__ fe::Elem fe_mul4(const fe::Elem &a) { return fe_dbl(fe_dbl(a)); }
__device__ __forceinline__ fe::Elem fe_mul8(const fe::Elem &a) { return fe_dbl(fe_mul4(a)); }

// --- Jacobian arithmetic (a = 0) --------------------------------------------

__device__ __forceinline__ JacP jac_infinity() {
    JacP r;
    r.X = fe::fe_set_u64(1); r.Y = fe::fe_set_u64(1); r.Z = fe::fe_set_u64(0);
    r.inf = true;
    return r;
}

// dbl-2009-l
__device__ __forceinline__ JacP jac_double(const JacP &P) {
    if (P.inf) return P;
    fe::Elem A = fe::fe_sqr(P.X);
    fe::Elem B = fe::fe_sqr(P.Y);
    fe::Elem C = fe::fe_sqr(B);
    fe::Elem t = fe::fe_add(P.X, B);
    t = fe::fe_sqr(t);
    t = fe::fe_sub(t, A);
    t = fe::fe_sub(t, C);
    fe::Elem D = fe_dbl(t);                       // 2*((X+B)^2 - A - C)
    fe::Elem E = fe_mul3(A);                      // 3*A
    fe::Elem F = fe::fe_sqr(E);
    JacP R;
    R.X = fe::fe_sub(F, fe_dbl(D));               // F - 2D
    R.Y = fe::fe_sub(fe::fe_mul(E, fe::fe_sub(D, R.X)), fe_mul8(C)); // E*(D-X3) - 8C
    R.Z = fe::fe_mul(fe_dbl(P.Y), P.Z);          // 2*Y*Z
    R.inf = false;
    return R;
}

// madd-2007-bl: Jacobian P + affine Q (Z2 = 1).
__device__ __forceinline__ JacP jac_add_affine(const JacP &P, const PointA &Q) {
    if (P.inf) { JacP r; r.X = Q.x; r.Y = Q.y; r.Z = fe::fe_set_u64(1); r.inf = false; return r; }
    fe::Elem Z1Z1 = fe::fe_sqr(P.Z);
    fe::Elem U2 = fe::fe_mul(Q.x, Z1Z1);
    fe::Elem S2 = fe::fe_mul(fe::fe_mul(Q.y, P.Z), Z1Z1);   // Y2 * Z1^3
    fe::Elem H  = fe::fe_sub(U2, P.X);
    fe::Elem rr = fe::fe_sub(S2, P.Y);
    if (fe::fe_is_zero(H)) {
        if (fe::fe_is_zero(rr)) return jac_double(P);       // P == Q
        return jac_infinity();                              // P == -Q
    }
    fe::Elem HH = fe::fe_sqr(H);
    fe::Elem I  = fe_mul4(HH);
    fe::Elem J  = fe::fe_mul(H, I);
    fe::Elem r  = fe_dbl(rr);                                // 2*(S2 - Y1)
    fe::Elem V  = fe::fe_mul(P.X, I);
    JacP R;
    R.X = fe::fe_sub(fe::fe_sub(fe::fe_sqr(r), J), fe_dbl(V));            // r^2 - J - 2V
    R.Y = fe::fe_sub(fe::fe_mul(r, fe::fe_sub(V, R.X)),
                     fe::fe_mul(fe_dbl(P.Y), J));                        // r*(V-X3) - 2*Y1*J
    R.Z = fe_dbl(fe::fe_mul(P.Z, H));                                    // 2*Z1*H
    R.inf = false;
    return R;
}

__device__ __forceinline__ PointA jac_to_affine(const JacP &P) {
    PointA r;
    if (P.inf) { r.x = fe::fe_set_u64(0); r.y = fe::fe_set_u64(0); return r; }
    fe::Elem zi  = fe::fe_inv(P.Z);
    fe::Elem zi2 = fe::fe_sqr(zi);
    fe::Elem zi3 = fe::fe_mul(zi2, zi);
    r.x = fe::fe_mul(P.X, zi2);
    r.y = fe::fe_mul(P.Y, zi3);
    return r;
}

// scalar * base, scalar = 4 little-endian 64-bit limbs. Returns affine.
__device__ __forceinline__ PointA ec_scalarmul(const uint64_t scalar[4], const PointA &base) {
    JacP acc = jac_infinity();
#pragma unroll 1
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t w = scalar[limb];
#pragma unroll 1
        for (int bit = 63; bit >= 0; bit--) {
            acc = jac_double(acc);
            if ((w >> bit) & 1ULL) acc = jac_add_affine(acc, base);
        }
    }
    return jac_to_affine(acc);
}

__device__ __forceinline__ PointA ec_scalarmul_u64(uint64_t k, const PointA &base) {
    uint64_t s[4] = {k, 0, 0, 0};
    return ec_scalarmul(s, base);
}

// --- affine hot path --------------------------------------------------------

// Center C + (plus ? +Gpt : -Gpt), given inv = 1/(Gpt.x - C.x). Matches AddDirect:
//   s = (Gy' - Cy) * inv ; rx = s^2 - Cx - Gx ; ry = s*(Gx - rx) - Gy'
// where Gy' = Gpt.y for plus, -Gpt.y for minus (x-difference, hence inv, is identical).
__device__ __forceinline__ PointA ec_batch_add(const fe::Elem &Cx, const fe::Elem &Cy,
                                               const fe::Elem &Gx, const fe::Elem &Gy,
                                               const fe::Elem &inv, bool plus) {
    fe::Elem gy = plus ? Gy : fe::fe_neg(Gy);
    fe::Elem dy = fe::fe_sub(gy, Cy);
    fe::Elem s  = fe::fe_mul(dy, inv);
    fe::Elem p  = fe::fe_sqr(s);
    PointA r;
    r.x = fe::fe_sub(fe::fe_sub(p, Cx), Gx);
    r.y = fe::fe_sub(fe::fe_mul(s, fe::fe_sub(Gx, r.x)), gy);
    return r;
}

// General affine add P1 + P2 with an internal inversion (used once per launch for
// the inter-launch center jump; not on the per-key hot path).
__device__ __forceinline__ PointA ec_add_direct(const PointA &P1, const PointA &P2) {
    fe::Elem dy = fe::fe_sub(P2.y, P1.y);
    fe::Elem dx = fe::fe_sub(P2.x, P1.x);
    fe::Elem inv = fe::fe_inv(dx);
    fe::Elem s  = fe::fe_mul(dy, inv);
    fe::Elem p  = fe::fe_sqr(s);
    PointA r;
    r.x = fe::fe_sub(fe::fe_sub(p, P1.x), P2.x);
    r.y = fe::fe_sub(fe::fe_mul(s, fe::fe_sub(P2.x, r.x)), P2.y);
    return r;
}

} // namespace ec
