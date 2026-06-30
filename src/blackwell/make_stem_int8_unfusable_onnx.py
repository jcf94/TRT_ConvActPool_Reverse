#!/usr/bin/env python3
"""Un-fusable INT8 graphs to isolate TensorRT's standalone INT8 conv cost.

The fully-fused graph (make_stem_int8_qdq_onnx.py) lets TRT pick the fused
sm80_trt_conv_act_pool_v3 cask kernel. To measure how expensive the *convolution
alone* is on Blackwell, we deny TRT the Conv->ReLU->MaxPool fusion two ways:

  conv_only : input--Q--DQ--+--> Conv --> ReLU --Q--DQ--> output   (no MaxPool)
  no_fuse   : same as the full stem but the ReLU output fans out to a second
              graph output, so the activation must be materialized and the pool
              can no longer be folded into the conv epilogue.

Both keep identical conv weights/scales to the fused model for a fair compare.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def common_inits():
    rng = np.random.default_rng(1234)
    w_float = rng.normal(0, 0.05, size=(64, 3, 7, 7)).astype(np.float32)
    w_amax = np.maximum(np.max(np.abs(w_float).reshape(64, -1), axis=1), 1e-8)
    w_scale = (w_amax / 127.0).astype(np.float32)
    w_int8 = np.clip(np.round(w_float / w_scale.reshape(64, 1, 1, 1)), -127, 127).astype(np.int8)
    act_scale = np.array(1.0 / 127.0, dtype=np.float32)
    act_zp = np.array(0, dtype=np.int8)
    return [
        numpy_helper.from_array(w_int8, "w_int8"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(np.zeros((64,), dtype=np.int8), "w_zp"),
        numpy_helper.from_array(act_scale, "in_scale"),
        numpy_helper.from_array(act_zp, "in_zp"),
        numpy_helper.from_array(act_scale, "post_scale"),
        numpy_helper.from_array(act_zp, "post_zp"),
        numpy_helper.from_array(np.zeros((64,), dtype=np.float32), "conv_b"),
    ]


def conv_prefix():
    return [
        helper.make_node("QuantizeLinear", ["input", "in_scale", "in_zp"], ["in_q"]),
        helper.make_node("DequantizeLinear", ["in_q", "in_scale", "in_zp"], ["in_dq"]),
        helper.make_node("DequantizeLinear", ["w_int8", "w_scale", "w_zp"], ["w_dq"], axis=0),
        helper.make_node("Conv", ["in_dq", "w_dq", "conv_b"], ["conv_out"],
                         pads=[3, 3, 3, 3], strides=[2, 2], kernel_shape=[7, 7]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("QuantizeLinear", ["relu_out", "post_scale", "post_zp"], ["post_q"]),
        helper.make_node("DequantizeLinear", ["post_q", "post_scale", "post_zp"], ["post_dq"]),
    ]


def make_conv_only():
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 112, 112])
    nodes = conv_prefix()
    nodes.append(helper.make_node("Identity", ["post_dq"], ["output"]))
    return y, nodes


def make_no_fuse():
    # ReLU activation is also exported -> cannot be folded into the conv/pool epilogue.
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 56, 56])
    y2 = helper.make_tensor_value_info("act_out", TensorProto.FLOAT, [1, 64, 112, 112])
    nodes = conv_prefix()
    nodes.append(helper.make_node("Identity", ["post_dq"], ["act_out"]))
    nodes.append(helper.make_node("MaxPool", ["post_dq"], ["output"],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]))
    return [y, y2], nodes


def save(stem, outs, nodes):
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    outs = outs if isinstance(outs, list) else [outs]
    graph = helper.make_graph(nodes, stem, [x], outs, initializer=common_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    out = MODEL_DIR / f"{stem}.onnx"
    onnx.save(model, out)
    print(out)


def main():
    y, n = make_conv_only()
    save("resnet_stem_int8_conv_only", y, n)
    ys, n = make_no_fuse()
    save("resnet_stem_int8_no_fuse", ys, n)


if __name__ == "__main__":
    main()
