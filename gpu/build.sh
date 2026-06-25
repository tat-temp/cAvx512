#!/usr/bin/env bash
# Build the CUDA searcher for an RTX 5090 (Blackwell, sm_120) with CUDA 13.2.
# WSL2 / Linux, nvcc + g++ host compiler.
set -euo pipefail

ARCH="${ARCH:-sm_120}"
OUT="${OUT:-cyclone_gpu}"

echo "nvcc -O3 -arch=${ARCH} -o ${OUT} cyclone_gpu.cu"
nvcc -O3 -arch="${ARCH}" -o "${OUT}" cyclone_gpu.cu

echo "built ./${OUT}  --  run ./${OUT} --selftest"
