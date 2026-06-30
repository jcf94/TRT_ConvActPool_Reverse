#!/usr/bin/env bash
# Build + run a Blackwell (sm_120) ConvActPool benchmark on the 5090 host.
#
# Usage (on the remote RTX 5090 machine, after copying src/ over):
#   ./build_sm120.sh bench_stem_sm120_v2.cu --warmup 500 --iters 20000
#
# The repo's main CMake build targets sm_86 / CUDA 11.8; the Blackwell sources
# are built standalone with CUDA 12.8's nvcc for sm_120.
set -euo pipefail

CUDA="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${CUDA}/bin/nvcc"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="$1"; shift || true
[ -f "$SRC" ] || SRC="${HERE}/${SRC}"
NAME="$(basename "$SRC" .cu)"
OUT="/tmp/${NAME}"

"$NVCC" -arch=sm_120 --use_fast_math -O3 -lineinfo "$SRC" -o "$OUT"
"$OUT" "$@"
