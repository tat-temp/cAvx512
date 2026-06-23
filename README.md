# Cyclone

Multi-threaded, AVX-512 **secp256k1** key-search for Bitcoin P2PKH puzzles. Given a target
P2PKH address and a hexadecimal private-key range, Cyclone enumerates the range, derives each
public key, hashes it to `hash160`, and compares against the target.

The hot path uses **8-lane AVX-512 IFMA** field arithmetic for point generation (batch-inversion
"addition trick", one modular inverse per 4096-key group) and **AVX-512** SHA-256 + RIPEMD-160 for
hashing. On a representative AVX-512/IFMA CPU this runs at **~11.8 MKeys/s single-thread** and
**~134 MKeys/s** across all cores — about **1.84x / 1.91x** the original scalar implementation.

> The software is developed for solving Satoshi's puzzles; any use for illegal purposes is strictly
> prohibited. The author is not responsible for any actions taken by the user when using this
> software for unlawful activities.

---

## Requirements

- **x86-64 CPU with AVX-512-IFMA** plus AVX-512 F/VL/BW/DQ, and AVX2/BMI2/ADX.
  IFMA ships on Intel Ice Lake / Tiger Lake / Rocket Lake / Sapphire Rapids and newer, and on
  AMD Zen 4 / Zen 5. The binary is compiled with `-mavx512ifma`, so it will fault with *illegal
  instruction* on a CPU without IFMA.
  - Check on Linux: `grep -o 'avx512_ifma\|avx512ifma' /proc/cpuinfo | head -1`
- **GCC** with OpenMP and C++17 (`g++` 10+ recommended).

## Build

The full build command is kept at the top of `Cyclone.cpp`:

```bash
g++ -std=c++17 -Ofast -ffast-math -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -fipa-modref -flto -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -o Cyclone Cyclone.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp Point.cpp ripemd160_avx2.cpp p2pkh_decoder.cpp sha256_avx2.cpp ripemd160_avx512.cpp sha256_avx512.cpp
```

Produces a single `./Cyclone` executable that handles search, self-test, and benchmark modes.

## Usage

```
Cyclone -a <Base58_P2PKH> -r <START:END>          search a range for an address
Cyclone --selftest                                correctness checks
Cyclone --selftest-ifma                           SIMD field-arithmetic checks
Cyclone --bench            [batches] [threads]    full pipeline, scalar point-gen
Cyclone --bench-ifma       [batches] [threads]    full pipeline, 8-lane IFMA point-gen
Cyclone --bench-gen        [batches] [threads]    scalar point-gen only
Cyclone --bench-gen-ifma   [batches] [threads]    8-lane point-gen only (Point output)
Cyclone --bench-gen-blocks [batches] [threads]    8-lane point-gen only (SHA-block output)
Cyclone --bench-hash       [batches] [threads]    hash160 only
Cyclone --bench-inv        [batches] [threads]    batch inversion only
```

- `START:END` are **hexadecimal** private keys (inclusive range).
- `[batches]` is groups-of-4096 keys **per thread** (default `1000`); `[threads]` defaults to `0`
  = all logical CPUs. Run a benchmark a few times and take the median; use `... <batches> 1` for a
  clean single-thread number.

| mode | isolates |
|---|---|
| `--bench` / `--bench-ifma` | the whole gen+hash pipeline (scalar vs IFMA gen) |
| `--bench-gen` / `--bench-gen-ifma` / `--bench-gen-blocks` | point generation only |
| `--bench-hash` | AVX-512 SHA-256 + RIPEMD-160 only |
| `--bench-inv` | the per-group batch modular inversion only |

The benches share the exact production code paths, and `1/rate_full ~= 1/rate_gen + 1/rate_hash`,
so the modes let you attribute time precisely.

## Self-tests

Both self-tests are self-contained (no network, deterministic) and exit non-zero on any failure.

### `--selftest` — end-to-end correctness

```
$ ./Cyclone --selftest
================= SELF TEST =================
Hash160 cross-check : 46/46 match
Address round-trip  : OK (1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L)
End-to-end search   : OK (found priv 0000000000000000000000000000000000000000000000000000000000abcdef)
============================================
SELFTEST PASSED
```

Cross-checks the AVX-512 `hash160` against an independent reference, round-trips a P2PKH address,
and runs the **production search path** against a known in-range key.

### `--selftest-ifma` — SIMD field arithmetic

```
$ ./Cyclone --selftest-ifma
============== IFMA FIELD SELFTEST ==============
conversion round-trip : OK
SoA pack/unpack       : OK
add                   : OK
sub                   : OK
neg                   : OK
normalize             : OK
mul (IFMA)            : OK
sqr (IFMA)            : OK
gen8 plus             : OK
gen8 minus            : OK
to_compressed8 (gen8) : OK
genGroupIFMA          : OK
to_compressed8        : OK
block path vs point   : OK
block advance drift   : OK
================================================
IFMA FIELD SELFTEST PASSED
```

Validates every 8-lane SIMD field/curve op against a scalar reference (`ModMulK1`, `ModSquareK1`,
`AddDirect`, `ComputePublicKey`), and compares the full block pipeline (`genGroupIFMABlocks` +
`hash16Blocks`) against the validated Point pipeline across multiple reused-buffer groups.

## Benchmarks

```
$ ./Cyclone --bench-ifma 30000 1
================= BENCHMARK =================
Mode             : FULL-IFMA (8-lane gen + hash)
Threads          : 1
Batches/thread   : 30000  (group size 4096)
Keys processed   : 122880000
Elapsed          : 10.450 s
Throughput       : 11.76 Mkeys/s
(checksum sink   : ...)
============================================
```

Measured on the development CPU (median of several runs):

| mode | 1 thread | all threads |
|---|---:|---:|
| `--bench` (scalar full) | 6.38 | 70.34 |
| **`--bench-ifma` (IFMA full)** | **11.76** | **134.32** |
| `--bench-gen-blocks` (gen only) | 23.81 | 264.93 |
| derived hash160 (`1/full − 1/gen`) | ~23.2 | ~272 |

After the IFMA work the pipeline is **balanced ~50/50** between point generation and hashing.

## Production search

```bash
./Cyclone -a 1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L -r 100000:FFFFFF
```

While running it prints a live status block:

```
================= WORK IN PROGRESS =================
Target Address: 1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L
CPU Threads   : 24
Mkeys/s       : 134.21
Total Checked : 8388608
Elapsed Time  : 00:00:00
Range         : 0000...100000:0000...FFFFFF
Progress      : 51.2000 %
Progress Save : 0
```

On success it prints the key and also saves it to disk:

```
================== FOUND MATCH! ==================
Private Key   : 0000000000000000000000000000000000000000000000000000000000ABCDEF
Public Key    : <33-byte compressed public key, hex>
WIF           : <wallet import format>
P2PKH Address : 1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L
Total Checked : ...
Elapsed Time  : 00:00:00
Speed         : 134.21 Mkeys/s
```

(The example range `100000:FFFFFF` contains the key `0xABCDEF` for that address, so it is a quick,
self-verifying sample.) Search is split evenly across threads; the work auto-saves progress
periodically so a long run can be monitored.

## How it works

- **Batch addition trick** — each group of `CPU_GROUP_SIZE = 4096` consecutive keys is generated
  from one center point using a single modular inverse (`IntGroup::ModInv`) over all the group's
  x-differences, instead of one inverse per key.
- **8-lane IFMA field arithmetic** (`ifma_field.h`) — secp256k1 `Fp` in radix-2^52 (5 limbs x 8
  lanes, SoA), with `_mm512_madd52lo/hi_epu64` multiply, a dedicated squaring, and the EC slope
  formula in `gen8`. Points are emitted straight into SHA input blocks (`to_compressed8`), skipping
  any per-point `Int` round-trip.
- **AVX-512 hashing** — 16-way SHA-256 then RIPEMD-160 (`hash16Blocks`) over the prebuilt blocks.
- **Correctness** is guarded entirely by `--selftest` / `--selftest-ifma`; the IFMA path is checked
  op-by-op and end-to-end against the scalar reference before every release.

## Credits

Builds on the secp256k1 / `Int` / hashing primitives from the VanitySearch / BSGS lineage
(Jean-Luc Pons), under GPLv3.
