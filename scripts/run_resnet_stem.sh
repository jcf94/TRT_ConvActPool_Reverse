#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/bench_resnet_stem"
OUT="${ROOT_DIR}/results/resnet_stem.csv"

mkdir -p "${ROOT_DIR}/results"
"${BIN}" --iters 1000 --warmup 100 --csv | tee "${OUT}"
