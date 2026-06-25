# Cyclone GPU

CUDA port of Cyclone's **secp256k1** Bitcoin P2PKH address search, for NVIDIA
**Blackwell** (RTX 5090, `sm_120`) under CUDA 13.2. Given a target P2PKH address and
a hexadecimal private-key range, every GPU thread walks a run of consecutive keys
with the symmetric batch-addition trick (center ± k·G, one Montgomery inversion per
group), derives each compressed public key, computes `hash160 = RIPEMD160(SHA256(pubkey))`,
and compares it to the target.

It is **self-contained** — all elliptic-curve math, SHA-256 and RIPEMD-160 run on
the device; the host only decodes the address, partitions the range, and reconstructs
a found key. The AVX-512 CPU program in the parent directory is unaffected.

On a single **RTX 5090** it runs at **~7.4 GKeys/s** (full pipeline), and scales
~linearly across multiple GPUs.

> Developed for solving Satoshi's puzzles; any use for illegal purposes is strictly
> prohibited. The author is not responsible for unlawful use.

---

## Requirements

- NVIDIA GPU; tuned for Blackwell (`sm_120`). For other archs set `ARCH` (below).
- **CUDA Toolkit 13.x** (`nvcc`) with a C++17 host compiler (g++), on **WSL2 / Linux**.
- A POSIX host (uses `std::thread` for the per-GPU workers; build adds `-pthread`).

## Build

```bash
cd gpu
bash build.sh          # nvcc -O3 -arch=sm_120 -Xptxas -v -Xcompiler -pthread -o cyclone_gpu cyclone_gpu.cu
# or: make
```

For a different GPU: `ARCH=sm_90 make` (or `ARCH=sm_89 bash build.sh`, etc.).
`build.sh` passes `-Xptxas -v` so you can see per-kernel register usage.

## Verify first

```bash
./cyclone_gpu --selftest
```

Must print `GPU SELFTEST PASSED`. It validates, bottom-up:

- **field** add/sub/mul/sqr/inv against an independent host big-integer oracle that
  reduces mod p by long division (so the device fast-reduction is genuinely cross-checked);
- **EC** scalar-multiplication outputs satisfy the curve equation `y² = x³ + 7`;
- **hash160** of `0xABCDEF·G` equals the hash160 of the known address
  `1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L`;
- a real **end-to-end search** over `100000:FFFFFF` finds `0xABCDEF`.

## Usage

```
cyclone_gpu -a <Base58_P2PKH> -r <START:END> --grid i,j --slices N [--gpus a,b,...]
cyclone_gpu --selftest
cyclone_gpu --bench [launches] [--grid i,j] [--slices N] [--gpus d]
```

- `-a, --address` — target mainnet P2PKH (compressed) Base58 address.
- `-r START:END` — inclusive private-key range, **hex**.
- `--grid i,j` — **i** = keys per batch per thread (a power of 2; the per-thread group
  size, like the CPU's `CPU_GROUP_SIZE`), **j** = threads per block.
- `--slices N` — batches each thread runs per kernel launch (amortizes launch/setup cost).
- `--gpus a,b,...` — device ids to use (default: **all** visible GPUs).

The **block count is auto-sized per GPU** to fill it (occupancy × SM count); override
with `CYCLONE_BLOCKS=<n>`. Keys per launch per GPU = `blocks × j × i × slices`.

Self-verifying sample (same key/address as the CPU README):

```bash
./cyclone_gpu -a 1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L -r 100000:FFFFFF --grid 512,256 --slices 16
```

On success it prints the private key, compressed public key, WIF, and address (with
an address round-trip check), and appends to `found_keys.txt`. A live status line
shows aggregate Mkeys/s, keys checked, elapsed, and progress %; `progress.txt` is
appended periodically.

### Multiple GPUs

The range is split into one contiguous sub-range per GPU, each driven by its own host
thread (its own context, buffers, and constant memory). A shared atomic flag stops all
GPUs the instant any one finds the key; progress is aggregated into a single status
line. There is no cross-device coordination on the hot path, so scaling across
identical GPUs is ~linear.

```bash
./cyclone_gpu -a <addr> -r <START:END> --grid 512,256 --slices 16            # all GPUs
./cyclone_gpu -a <addr> -r <START:END> --grid 512,256 --slices 16 --gpus 0,2  # only GPUs 0 and 2
```

## Tuning

```bash
./cyclone_gpu --bench     20 --grid 512,256 --slices 16   # full pipeline
./cyclone_gpu --bench-gen 20 --grid 512,256 --slices 16   # EC + Montgomery only (no hash)
```

`--bench` / `--bench-gen` are permission-free time attribution: comparing them gives
the field-math vs hash split (≈ `1/full = 1/gen + 1/hash`). On the 5090 the work is
roughly **53% field math / 47% hash160**, and `j = 256` (1 block/SM) is the occupancy
sweet spot — more blocks/SM measured *slower* (cache/register-file contention).

- **i** trades inversion amortization vs memory: the per-thread Montgomery prefix
  scratch is `(i/2+1)` field elements in global memory. `i ≤ 1024` also lets the
  generator table live in constant memory (faster broadcast reads). Good starting
  points: `--grid 512,256` or `--grid 1024,128`.
- **slices** ≥ 8–16 keeps the GPU busy between host syncs.

## Performance

Single RTX 5090, `--grid 512,256 --slices 16`, full pipeline. The port was optimized
in measured steps (each `--selftest`-validated; numbers are Mkeys/s):

| step | Mkeys/s | cumulative |
|---|---:|---:|
| initial GPU port | 2407 | 1.00× |
| dedicated squaring + `fe_inv` out-of-line | 4146 | 1.72× |
| de-inline per-point gen/hash (kill 255-reg spill) | 5319 | 2.21× |
| fully-unrolled RIPEMD-160 (schedule in registers) | 6233 | 2.59× |
| `unsigned __int128` carry chains | 6918 | 2.87× |
| branchless squaring | 7010 | 2.91× |
| direct x-limbs → SHA words (skip byte round-trip) | 7219 | 3.00× |
| generator table in `__constant__` memory | **7384** | **3.07×** |

Two experiments were tried and **reverted** after measuring a regression: an 8×32-limb
field multiply and a hand strength-reduced `×C` constant multiply — in both cases
nvcc's `__int128` codegen was already faster. The takeaway: nvcc's *arithmetic*
codegen is optimal here; the wins came from *data movement* (register-resident
schedules, constant-memory broadcast, skipped byte round-trips) and *occupancy/spill*
fixes.

## How it works

- **Batch-addition trick + Montgomery inversion** — each thread generates a group of
  `i` consecutive keys from one center point using a single field inversion over all
  the group's x-differences (Montgomery's simultaneous-inversion trick), instead of one
  inverse per key. The ± symmetric pair shares each inverse. The center advances by
  `i·G` per slice (folded into that batch's inversion) and by a precomputed jump
  between launches, so persistent points stride without re-scalar-multiplying.
- **Field arithmetic** (`field.cuh`) — Fp mod `p = 2²⁵⁶−2³²−977` in 4×64-bit limbs,
  schoolbook multiply via `__int128` carry chains, the secp256k1 two-pass fast
  reduction, and a Fermat (`a^(p−2)`) inverse using the libsecp256k1 addition chain.
- **EC** (`ec.cuh`) — Jacobian double-and-add for the one-time setup (start points and
  the generator table), affine batch-add on the hot path (matching the CPU `AddDirect`).
- **Hashing** (`sha256.cuh` / `ripemd160.cuh` / `hash160.cuh`) — canonical single-block
  SHA-256 then RIPEMD-160, both fully unrolled with register-resident schedules; SHA
  words are built straight from the x limbs, and the SHA→RIPEMD hand-off is a byteswap
  (no intermediate digest store).
- **Correctness** is guarded entirely by `--selftest` (device ops vs an independent
  host oracle, the curve equation, a known address vector, and an end-to-end search).

## Files

| file | role |
|---|---|
| `field.cuh` | Fp arithmetic mod p (4×64 limbs, fast reduce, Fermat inverse) |
| `ec.cuh` | Jacobian scalar-mult (setup) + affine batch-add (hot path) |
| `sha256.cuh`, `ripemd160.cuh`, `hash160.cuh` | device hashing (canonical, single block) |
| `search.cuh` | init / generator-table / step kernels + atomic result |
| `u256.h`, `host_hash.h` | intrinsic-free host 256-bit int, SHA-256, base58, WIF |
| `cyclone_gpu.cu` | host driver, multi-GPU orchestration, self-test, benchmark |
| `build.sh`, `Makefile` | build (`-arch=sm_120 -Xcompiler -pthread`) |

## Scope

This searches by **address** (hash160), which is fundamentally a brute-force scan —
you must hash every candidate, so there is no sub-linear shortcut. If instead the
target's **public key** is known, the discrete-log can be solved in ~√range with
Pollard's Kangaroo / BSGS — a different algorithm, not this tool.

Not implemented: GLV endomorphism (doesn't help a contiguous range), and large-`i`
shared-memory tiling of the batch inversion.

## Credits

Builds on the secp256k1 / `Int` / hashing lineage from VanitySearch / BSGS
(Jean-Luc Pons), under GPLv3.
