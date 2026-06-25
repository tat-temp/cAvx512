#!/usr/bin/env bash
# Build the CUDA searcher for an RTX 5090 (Blackwell, sm_120) with CUDA 13.2.
# WSL2 / Linux, nvcc + g++ host compiler.
set -euo pipefail

ARCH="${ARCH:-sm_120}"
OUT="${OUT:-cyclone_gpu}"

# -Xptxas -v prints per-kernel register / local-memory usage (occupancy diagnostic).
# -Xcompiler -pthread for the per-GPU host threads (multi-GPU search).
echo "nvcc -O3 -arch=${ARCH} -Xptxas -v -Xcompiler -pthread -o ${OUT} cyclone_gpu.cu"
nvcc -O3 -arch="${ARCH}" -Xptxas -v -Xcompiler -pthread -o "${OUT}" cyclone_gpu.cu

echo "built ./${OUT}  --  run ./${OUT} --selftest"
