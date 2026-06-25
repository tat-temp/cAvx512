# Cyclone GPU

CUDA port of Cyclone's secp256k1 P2PKH puzzle search, for NVIDIA **Blackwell**
(RTX 5090, `sm_120`) under CUDA 13.2. Self-contained: all elliptic-curve math,
SHA-256 and RIPEMD-160 run on the GPU. The CPU AVX-512 program in the parent
directory is unaffected.

Given a target P2PKH address and a hex private-key range, every thread walks a run
of consecutive keys with the symmetric batch-addition trick (center ± k·G, one
Montgomery inversion per batch), hashes each public key to `hash160`, and compares
the target.

## Build

WSL2 / Linux with the CUDA 13.2 toolkit (`nvcc` + `g++`):

```bash
cd gpu
bash build.sh          # nvcc -O3 -arch=sm_120 -o cyclone_gpu cyclone_gpu.cu
# or: make
```

For a different GPU set the arch, e.g. `ARCH=sm_90 make`.

## Verify first

```bash
./cyclone_gpu --selftest
```

This must print `GPU SELFTEST PASSED`. It checks, bottom-up:

- **field** add/sub/mul/sqr/inv against an independent host big-integer oracle
  (which reduces mod p by long division, so the device fast-reduction is genuinely
  cross-checked);
- **EC** scalar multiplication outputs satisfy the curve equation `y² = x³ + 7`;
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
- `-r START:END` — inclusive private-key range in **hex**.
- `--grid i,j` — **i** = keys per batch per thread (a power of 2; the per-thread
  group size, like the CPU's `CPU_GROUP_SIZE`); **j** = threads per block.
- `--slices N` — batches each thread runs per kernel launch (amortizes launch and
  setup cost).
- `--gpus a,b,...` — device ids to use (default: **all** visible GPUs). The range is
  split into one contiguous sub-range per GPU, each driven by its own host thread;
  progress is aggregated into one status line and the first GPU to hit the key wins.
  For `--bench`, only the first listed id is used.

The **number of blocks is auto-sized** per GPU to fill it (occupancy × SM count).
Override with `CYCLONE_BLOCKS=<n>`. Keys processed per launch per GPU =
`blocks × j × i × slices`. Scaling across identical GPUs is ~linear (independent
sub-ranges, no cross-device coordination).

Self-verifying sample (same key/address as the CPU README):

```bash
./cyclone_gpu -a 1NSkWRJ4e1eMCoLxwM2nid9F31fbx7d96L -r 100000:FFFFFF --grid 512,256 --slices 16
```

On success it prints the private key, compressed public key, WIF, address (with an
address round-trip check), and appends to `found_keys.txt`.

## Tuning

```bash
./cyclone_gpu --bench 20 --grid 512,256 --slices 16
```

- **i (batch size)** trades inversion amortization against memory/bandwidth: the
  per-thread Montgomery prefix scratch is `(i/2+1)` field elements per thread in
  global memory. Larger `i` ⇒ fewer inversions but more VRAM and traffic. Good
  starting points on a 5090: `--grid 512,256` or `--grid 1024,128`.
- **j (threads/block)** affects occupancy; 128–256 is typical.
- **slices** ≥ 8–16 keeps the GPU busy between host syncs.

If `i × threads` would exceed VRAM the tool auto-reduces the block count and prints
a note.

## Layout

| file | role |
|---|---|
| `field.cuh` | Fp arithmetic mod p = 2²⁵⁶−2³²−977 (4×64 limbs, fast reduction, Fermat inverse) |
| `ec.cuh` | Jacobian scalar-mult (setup) + affine batch-add (hot path) |
| `sha256.cuh`, `ripemd160.cuh`, `hash160.cuh` | device hashing (canonical, single block) |
| `search.cuh` | init / generator-table / step kernels + result struct |
| `u256.h`, `host_hash.h` | intrinsic-free host 256-bit int, SHA-256, base58, WIF |
| `cyclone_gpu.cu` | host driver, self-test, benchmark |

## Not in this version

Multi-GPU, GLV endomorphism (doesn't help a contiguous range), a dedicated squaring
path, and shared-memory tiling of the batch inversion for large `i` — all noted as
follow-up optimizations once the baseline is confirmed correct on hardware.
