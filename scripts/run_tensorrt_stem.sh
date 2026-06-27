#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"
RESULT_DIR="${ROOT_DIR}/results"
mkdir -p "${MODEL_DIR}" "${RESULT_DIR}"

if ! python3 -c "import onnx" >/dev/null 2>&1; then
  python3 -m pip install onnx
fi

python3 "${ROOT_DIR}/scripts/make_resnet_stem_onnx.py"

if command -v trtexec >/dev/null 2>&1; then
  trtexec --onnx="${MODEL_DIR}/resnet_stem.onnx" --int8 --dump-profile \
    > "${RESULT_DIR}/trt_resnet_stem_int8_profile.txt" 2>&1

  trtexec --onnx="${MODEL_DIR}/resnet_stem_block_qdq.onnx" --int8 --dump-profile \
    > "${RESULT_DIR}/trt_resnet_stem_block_qdq_int8_profile.txt" 2>&1

  grep -E "myl|Conv|Relu|MaxPool|Total Host Walltime|GPU Compute Time|mean" \
    "${RESULT_DIR}"/trt_*_profile.txt || true
else
  python3 "${ROOT_DIR}/scripts/run_tensorrt_python.py" \
    --onnx="${MODEL_DIR}/resnet_stem.onnx" --int8 \
    > "${RESULT_DIR}/trt_python_plain.csv"
  python3 "${ROOT_DIR}/scripts/run_tensorrt_python.py" \
    --onnx="${MODEL_DIR}/resnet_stem_block_qdq.onnx" --int8 --explicit-qdq \
    > "${RESULT_DIR}/trt_python_qdq.csv"
  tail -20 "${RESULT_DIR}/trt_python_plain.csv"
  tail -20 "${RESULT_DIR}/trt_python_qdq.csv"
fi
