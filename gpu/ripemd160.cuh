// ripemd160.cuh -- canonical RIPEMD-160 of a single 32-byte message (a SHA-256
// digest) on the device, fused with the SHA hand-off. The 32-byte input is the
// big-endian SHA digest; RIPEMD reads little-endian words, so each SHA state word
// is byte-swapped into the message schedule (no intermediate digest store).
//
// The 80 rounds of each line are FULLY UNROLLED with compile-time message-word
// indices and rotation amounts, so the X[16] schedule stays in registers instead
// of local memory (a large GPU win -- dynamic X[RL[j]] indexing would otherwise
// force X to local). Constants/order are the canonical RIPEMD-160 tables.
//
// out[0..4] are the five RIPEMD words; the 20-byte hash160 is out[i] written
// little-endian, so out[i] compares directly against a little-endian read of the
// target's bytes (see search.cuh).
#pragma once
#include <stdint.h>

namespace ripemd160 {

__device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// nonlinear functions
#define RMD_F(x,y,z) ((x) ^ (y) ^ (z))
#define RMD_G(x,y,z) (((x) & (y)) | (~(x) & (z)))
#define RMD_H(x,y,z) (((x) | ~(y)) ^ (z))
#define RMD_I(x,y,z) (((x) & (z)) | ((y) & ~(z)))
#define RMD_J(x,y,z) ((x) ^ ((y) | ~(z)))

// left-line round macros (per-block constant embedded)
#define LFF(a,b,c,d,e,x,s) { a += RMD_F(b,c,d) + (x);              a = rotl(a,s) + e; c = rotl(c,10); }
#define LGG(a,b,c,d,e,x,s) { a += RMD_G(b,c,d) + (x) + 0x5a827999u; a = rotl(a,s) + e; c = rotl(c,10); }
#define LHH(a,b,c,d,e,x,s) { a += RMD_H(b,c,d) + (x) + 0x6ed9eba1u; a = rotl(a,s) + e; c = rotl(c,10); }
#define LII(a,b,c,d,e,x,s) { a += RMD_I(b,c,d) + (x) + 0x8f1bbcdcu; a = rotl(a,s) + e; c = rotl(c,10); }
#define LJJ(a,b,c,d,e,x,s) { a += RMD_J(b,c,d) + (x) + 0xa953fd4eu; a = rotl(a,s) + e; c = rotl(c,10); }
// right-line round macros (reverse functions, primed constants)
#define RJJ(a,b,c,d,e,x,s) { a += RMD_J(b,c,d) + (x) + 0x50a28be6u; a = rotl(a,s) + e; c = rotl(c,10); }
#define RII(a,b,c,d,e,x,s) { a += RMD_I(b,c,d) + (x) + 0x5c4dd124u; a = rotl(a,s) + e; c = rotl(c,10); }
#define RHH(a,b,c,d,e,x,s) { a += RMD_H(b,c,d) + (x) + 0x6d703ef3u; a = rotl(a,s) + e; c = rotl(c,10); }
#define RGG(a,b,c,d,e,x,s) { a += RMD_G(b,c,d) + (x) + 0x7a6d76e9u; a = rotl(a,s) + e; c = rotl(c,10); }
#define RFF(a,b,c,d,e,x,s) { a += RMD_F(b,c,d) + (x);              a = rotl(a,s) + e; c = rotl(c,10); }

// Fuse: X[0..7] = byteswap(SHA words); X[8]=0x80 pad; X[14]=256-bit length.
__device__ __forceinline__ void ripemd160_from_sha(const uint32_t H[8], uint32_t out[5]) {
    uint32_t X0  = __byte_perm(H[0], 0, 0x0123), X1 = __byte_perm(H[1], 0, 0x0123);
    uint32_t X2  = __byte_perm(H[2], 0, 0x0123), X3 = __byte_perm(H[3], 0, 0x0123);
    uint32_t X4  = __byte_perm(H[4], 0, 0x0123), X5 = __byte_perm(H[5], 0, 0x0123);
    uint32_t X6  = __byte_perm(H[6], 0, 0x0123), X7 = __byte_perm(H[7], 0, 0x0123);
    uint32_t X8  = 0x00000080u;
    uint32_t X9  = 0, X10 = 0, X11 = 0, X12 = 0, X13 = 0;
    uint32_t X14 = 256u;          // message length in bits (low word, little-endian)
    uint32_t X15 = 0;

    const uint32_t h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu,
                   h3 = 0x10325476u, h4 = 0xC3D2E1F0u;
    uint32_t aa = h0, bb = h1, cc = h2, dd = h3, ee = h4;
    uint32_t aaa = h0, bbb = h1, ccc = h2, ddd = h3, eee = h4;

    // ---- left line ----
    LFF(aa,bb,cc,dd,ee, X0,11) LFF(ee,aa,bb,cc,dd, X1,14) LFF(dd,ee,aa,bb,cc, X2,15) LFF(cc,dd,ee,aa,bb, X3,12)
    LFF(bb,cc,dd,ee,aa, X4, 5) LFF(aa,bb,cc,dd,ee, X5, 8) LFF(ee,aa,bb,cc,dd, X6, 7) LFF(dd,ee,aa,bb,cc, X7, 9)
    LFF(cc,dd,ee,aa,bb, X8,11) LFF(bb,cc,dd,ee,aa, X9,13) LFF(aa,bb,cc,dd,ee, X10,14) LFF(ee,aa,bb,cc,dd, X11,15)
    LFF(dd,ee,aa,bb,cc, X12,6) LFF(cc,dd,ee,aa,bb, X13,7) LFF(bb,cc,dd,ee,aa, X14,9) LFF(aa,bb,cc,dd,ee, X15,8)

    LGG(ee,aa,bb,cc,dd, X7, 7) LGG(dd,ee,aa,bb,cc, X4, 6) LGG(cc,dd,ee,aa,bb, X13,8) LGG(bb,cc,dd,ee,aa, X1,13)
    LGG(aa,bb,cc,dd,ee, X10,11) LGG(ee,aa,bb,cc,dd, X6, 9) LGG(dd,ee,aa,bb,cc, X15,7) LGG(cc,dd,ee,aa,bb, X3,15)
    LGG(bb,cc,dd,ee,aa, X12,7) LGG(aa,bb,cc,dd,ee, X0,12) LGG(ee,aa,bb,cc,dd, X9,15) LGG(dd,ee,aa,bb,cc, X5, 9)
    LGG(cc,dd,ee,aa,bb, X2,11) LGG(bb,cc,dd,ee,aa, X14,7) LGG(aa,bb,cc,dd,ee, X11,13) LGG(ee,aa,bb,cc,dd, X8,12)

    LHH(dd,ee,aa,bb,cc, X3,11) LHH(cc,dd,ee,aa,bb, X10,13) LHH(bb,cc,dd,ee,aa, X14,6) LHH(aa,bb,cc,dd,ee, X4, 7)
    LHH(ee,aa,bb,cc,dd, X9,14) LHH(dd,ee,aa,bb,cc, X15,9) LHH(cc,dd,ee,aa,bb, X8,13) LHH(bb,cc,dd,ee,aa, X1,15)
    LHH(aa,bb,cc,dd,ee, X2,14) LHH(ee,aa,bb,cc,dd, X7, 8) LHH(dd,ee,aa,bb,cc, X0,13) LHH(cc,dd,ee,aa,bb, X6, 6)
    LHH(bb,cc,dd,ee,aa, X13,5) LHH(aa,bb,cc,dd,ee, X11,12) LHH(ee,aa,bb,cc,dd, X5, 7) LHH(dd,ee,aa,bb,cc, X12,5)

    LII(cc,dd,ee,aa,bb, X1,11) LII(bb,cc,dd,ee,aa, X9,12) LII(aa,bb,cc,dd,ee, X11,14) LII(ee,aa,bb,cc,dd, X10,15)
    LII(dd,ee,aa,bb,cc, X0,14) LII(cc,dd,ee,aa,bb, X8,15) LII(bb,cc,dd,ee,aa, X12,9) LII(aa,bb,cc,dd,ee, X4, 8)
    LII(ee,aa,bb,cc,dd, X13,9) LII(dd,ee,aa,bb,cc, X3,14) LII(cc,dd,ee,aa,bb, X7, 5) LII(bb,cc,dd,ee,aa, X15,6)
    LII(aa,bb,cc,dd,ee, X14,8) LII(ee,aa,bb,cc,dd, X5, 6) LII(dd,ee,aa,bb,cc, X6, 5) LII(cc,dd,ee,aa,bb, X2,12)

    LJJ(bb,cc,dd,ee,aa, X4, 9) LJJ(aa,bb,cc,dd,ee, X0,15) LJJ(ee,aa,bb,cc,dd, X5, 5) LJJ(dd,ee,aa,bb,cc, X9,11)
    LJJ(cc,dd,ee,aa,bb, X7, 6) LJJ(bb,cc,dd,ee,aa, X12,8) LJJ(aa,bb,cc,dd,ee, X2,13) LJJ(ee,aa,bb,cc,dd, X10,12)
    LJJ(dd,ee,aa,bb,cc, X14,5) LJJ(cc,dd,ee,aa,bb, X1,12) LJJ(bb,cc,dd,ee,aa, X3,13) LJJ(aa,bb,cc,dd,ee, X8,14)
    LJJ(ee,aa,bb,cc,dd, X11,11) LJJ(dd,ee,aa,bb,cc, X6, 8) LJJ(cc,dd,ee,aa,bb, X15,5) LJJ(bb,cc,dd,ee,aa, X13,6)

    // ---- right line ----
    RJJ(aaa,bbb,ccc,ddd,eee, X5, 8) RJJ(eee,aaa,bbb,ccc,ddd, X14,9) RJJ(ddd,eee,aaa,bbb,ccc, X7, 9) RJJ(ccc,ddd,eee,aaa,bbb, X0,11)
    RJJ(bbb,ccc,ddd,eee,aaa, X9,13) RJJ(aaa,bbb,ccc,ddd,eee, X2,15) RJJ(eee,aaa,bbb,ccc,ddd, X11,15) RJJ(ddd,eee,aaa,bbb,ccc, X4, 5)
    RJJ(ccc,ddd,eee,aaa,bbb, X13,7) RJJ(bbb,ccc,ddd,eee,aaa, X6, 7) RJJ(aaa,bbb,ccc,ddd,eee, X15,8) RJJ(eee,aaa,bbb,ccc,ddd, X8,11)
    RJJ(ddd,eee,aaa,bbb,ccc, X1,14) RJJ(ccc,ddd,eee,aaa,bbb, X10,14) RJJ(bbb,ccc,ddd,eee,aaa, X3,12) RJJ(aaa,bbb,ccc,ddd,eee, X12,6)

    RII(eee,aaa,bbb,ccc,ddd, X6, 9) RII(ddd,eee,aaa,bbb,ccc, X11,13) RII(ccc,ddd,eee,aaa,bbb, X3,15) RII(bbb,ccc,ddd,eee,aaa, X7, 7)
    RII(aaa,bbb,ccc,ddd,eee, X0,12) RII(eee,aaa,bbb,ccc,ddd, X13,8) RII(ddd,eee,aaa,bbb,ccc, X5, 9) RII(ccc,ddd,eee,aaa,bbb, X10,11)
    RII(bbb,ccc,ddd,eee,aaa, X14,7) RII(aaa,bbb,ccc,ddd,eee, X15,7) RII(eee,aaa,bbb,ccc,ddd, X8,12) RII(ddd,eee,aaa,bbb,ccc, X12,7)
    RII(ccc,ddd,eee,aaa,bbb, X4, 6) RII(bbb,ccc,ddd,eee,aaa, X9,15) RII(aaa,bbb,ccc,ddd,eee, X1,13) RII(eee,aaa,bbb,ccc,ddd, X2,11)

    RHH(ddd,eee,aaa,bbb,ccc, X15,9) RHH(ccc,ddd,eee,aaa,bbb, X5, 7) RHH(bbb,ccc,ddd,eee,aaa, X1,15) RHH(aaa,bbb,ccc,ddd,eee, X3,11)
    RHH(eee,aaa,bbb,ccc,ddd, X7, 8) RHH(ddd,eee,aaa,bbb,ccc, X14,6) RHH(ccc,ddd,eee,aaa,bbb, X6, 6) RHH(bbb,ccc,ddd,eee,aaa, X9,14)
    RHH(aaa,bbb,ccc,ddd,eee, X11,12) RHH(eee,aaa,bbb,ccc,ddd, X8,13) RHH(ddd,eee,aaa,bbb,ccc, X12,5) RHH(ccc,ddd,eee,aaa,bbb, X2,14)
    RHH(bbb,ccc,ddd,eee,aaa, X10,13) RHH(aaa,bbb,ccc,ddd,eee, X0,13) RHH(eee,aaa,bbb,ccc,ddd, X4, 7) RHH(ddd,eee,aaa,bbb,ccc, X13,5)

    RGG(ccc,ddd,eee,aaa,bbb, X8,15) RGG(bbb,ccc,ddd,eee,aaa, X6, 5) RGG(aaa,bbb,ccc,ddd,eee, X4, 8) RGG(eee,aaa,bbb,ccc,ddd, X1,11)
    RGG(ddd,eee,aaa,bbb,ccc, X3,14) RGG(ccc,ddd,eee,aaa,bbb, X11,14) RGG(bbb,ccc,ddd,eee,aaa, X15,6) RGG(aaa,bbb,ccc,ddd,eee, X0,14)
    RGG(eee,aaa,bbb,ccc,ddd, X5, 6) RGG(ddd,eee,aaa,bbb,ccc, X12,9) RGG(ccc,ddd,eee,aaa,bbb, X2,12) RGG(bbb,ccc,ddd,eee,aaa, X13,9)
    RGG(aaa,bbb,ccc,ddd,eee, X9,12) RGG(eee,aaa,bbb,ccc,ddd, X7, 5) RGG(ddd,eee,aaa,bbb,ccc, X10,15) RGG(ccc,ddd,eee,aaa,bbb, X14,8)

    RFF(bbb,ccc,ddd,eee,aaa, X12,8) RFF(aaa,bbb,ccc,ddd,eee, X15,5) RFF(eee,aaa,bbb,ccc,ddd, X10,12) RFF(ddd,eee,aaa,bbb,ccc, X4, 9)
    RFF(ccc,ddd,eee,aaa,bbb, X1,12) RFF(bbb,ccc,ddd,eee,aaa, X5, 5) RFF(aaa,bbb,ccc,ddd,eee, X8,14) RFF(eee,aaa,bbb,ccc,ddd, X7, 6)
    RFF(ddd,eee,aaa,bbb,ccc, X6, 8) RFF(ccc,ddd,eee,aaa,bbb, X2,13) RFF(bbb,ccc,ddd,eee,aaa, X13,6) RFF(aaa,bbb,ccc,ddd,eee, X14,5)
    RFF(eee,aaa,bbb,ccc,ddd, X0,15) RFF(ddd,eee,aaa,bbb,ccc, X3,13) RFF(ccc,ddd,eee,aaa,bbb, X9,11) RFF(bbb,ccc,ddd,eee,aaa, X11,11)

    // combine
    uint32_t t = h1 + cc + ddd;
    out[1] = h2 + dd + eee;
    out[2] = h3 + ee + aaa;
    out[3] = h4 + aa + bbb;
    out[4] = h0 + bb + ccc;
    out[0] = t;
}

#undef RMD_F
#undef RMD_G
#undef RMD_H
#undef RMD_I
#undef RMD_J
#undef LFF
#undef LGG
#undef LHH
#undef LII
#undef LJJ
#undef RJJ
#undef RII
#undef RHH
#undef RGG
#undef RFF

} // namespace ripemd160
