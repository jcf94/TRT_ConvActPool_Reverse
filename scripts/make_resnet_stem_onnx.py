#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def tensor(name, array):
    return numpy_helper.from_array(array, name=name)


def make_model(path: Path, with_qdq: bool) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 56, 56])

    rng = np.random.default_rng(1234)
    weight = rng.normal(0, 0.05, size=(64, 3, 7, 7)).astype(np.float32)
    bias = np.zeros((64,), dtype=np.float32)
    scale = np.array([1.0 / 127.0], dtype=np.float32)
    zp = np.array([0], dtype=np.int8)

    initializers = [
        tensor("conv_w", weight),
        tensor("conv_b", bias),
        tensor("qdq_scale", scale),
        tensor("qdq_zp", zp),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv_w", "conv_b"],
            ["conv_out"],
            pads=[3, 3, 3, 3],
            strides=[2, 2],
            kernel_shape=[7, 7],
        )
    ]
    relu_input = "conv_out"
    if with_qdq:
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    ["conv_out", "qdq_scale", "qdq_zp"],
                    ["conv_q"],
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["conv_q", "qdq_scale", "qdq_zp"],
                    ["conv_dq"],
                ),
            ]
        )
        relu_input = "conv_dq"
    nodes.extend(
        [
            helper.make_node("Relu", [relu_input], ["relu_out"]),
            helper.make_node(
                "MaxPool",
                ["relu_out"],
                ["output"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
        ]
    )

    graph = helper.make_graph(nodes, path.stem, [x], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    make_model(MODEL_DIR / "resnet_stem.onnx", with_qdq=False)
    make_model(MODEL_DIR / "resnet_stem_block_qdq.onnx", with_qdq=True)
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
