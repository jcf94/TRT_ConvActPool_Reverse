#!/usr/bin/env python3
"""Fully-quantized INT8 QDQ ResNet-stem ONNX for TensorRT 11 (Blackwell / sm_120).

TensorRT 11 removed the implicit-quantization (calibrator + INT8 builder flag)
path, so INT8 must be expressed entirely with Q/DQ nodes. To make TRT run the
convolution itself in INT8 (and fuse Conv->ReLU->MaxPool), every conv operand
must be quantized:

  input  --Q--DQ--+
  weight (int8) --DQ--+--> Conv --> ReLU --Q--DQ--> MaxPool --> output

Weights are stored as an int8 initializer with a per-output-channel DQ (axis 0),
mirroring how TRT's INT8 ResNet stem is normally fed.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 56, 56])

    rng = np.random.default_rng(1234)
    w_float = rng.normal(0, 0.05, size=(64, 3, 7, 7)).astype(np.float32)
    # Per-output-channel symmetric int8 weight quantization (axis 0).
    w_amax = np.max(np.abs(w_float).reshape(64, -1), axis=1)
    w_amax = np.maximum(w_amax, 1e-8)
    w_scale = (w_amax / 127.0).astype(np.float32)
    w_int8 = np.clip(np.round(w_float / w_scale.reshape(64, 1, 1, 1)), -127, 127).astype(np.int8)
    w_zp = np.zeros((64,), dtype=np.int8)

    act_scale = np.array(1.0 / 127.0, dtype=np.float32)  # per-tensor activation scale
    act_zp = np.array(0, dtype=np.int8)

    inits = [
        numpy_helper.from_array(w_int8, "w_int8"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(w_zp, "w_zp"),
        numpy_helper.from_array(act_scale, "in_scale"),
        numpy_helper.from_array(act_zp, "in_zp"),
        numpy_helper.from_array(act_scale, "post_scale"),
        numpy_helper.from_array(act_zp, "post_zp"),
        numpy_helper.from_array(np.zeros((64,), dtype=np.float32), "conv_b"),
    ]

    nodes = [
        helper.make_node("QuantizeLinear", ["input", "in_scale", "in_zp"], ["in_q"]),
        helper.make_node("DequantizeLinear", ["in_q", "in_scale", "in_zp"], ["in_dq"]),
        helper.make_node("DequantizeLinear", ["w_int8", "w_scale", "w_zp"], ["w_dq"], axis=0),
        helper.make_node(
            "Conv",
            ["in_dq", "w_dq", "conv_b"],
            ["conv_out"],
            pads=[3, 3, 3, 3],
            strides=[2, 2],
            kernel_shape=[7, 7],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("QuantizeLinear", ["relu_out", "post_scale", "post_zp"], ["post_q"]),
        helper.make_node("DequantizeLinear", ["post_q", "post_scale", "post_zp"], ["post_dq"]),
        helper.make_node(
            "MaxPool",
            ["post_dq"],
            ["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        ),
    ]

    graph = helper.make_graph(nodes, "resnet_stem_int8_qdq", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    out = MODEL_DIR / "resnet_stem_int8_qdq.onnx"
    onnx.save(model, out)
    print(out)


if __name__ == "__main__":
    main()
