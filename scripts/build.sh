#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/usr/local/cuda-11.8/bin:${PATH}

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "${ROOT_DIR}/build" -j"$(nproc)"
