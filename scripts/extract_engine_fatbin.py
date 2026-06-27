#!/usr/bin/env python3
"""Extract embedded CUDA fatbin payloads from a TensorRT engine blob.

TensorRT 10.10 stores the selected CASK kernel image directly in the serialized
engine. cuobjdump does not recognize the whole engine file as a CUDA object, but
it can disassemble the embedded fatbin if we cut from the fatbin magic offset.
The output may still contain trailing engine bytes; cuobjdump normally prints a
fatal "Invalid fatbin header" after it has already emitted useful SASS.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FATBIN_MAGIC_LE = b"\x50\xed\x55\xba"  # 0xBA55ED50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path, help="Serialized TensorRT engine")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/engine_elf/engine_fatbin.bin"),
        help="Output fatbin path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.engine.read_bytes()
    offset = data.find(FATBIN_MAGIC_LE)
    if offset < 0:
        raise SystemExit(f"fatbin magic 0xBA55ED50 not found in {args.engine}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data[offset:])
    print(f"engine={args.engine}")
    print(f"fatbin_offset={offset}")
    print(f"fatbin_tail_bytes={len(data) - offset}")
    print(f"output={args.output}")
    print("next:")
    print(f"  cuobjdump -lelf {args.output}")
    print(f"  cuobjdump -sass -arch sm_86 {args.output} > results/trt_conv_act_pool_sm86.sass")
    print(
        f"  cuobjdump -res-usage -arch sm_86 {args.output} "
        "> results/trt_conv_act_pool_sm86_res_usage.txt"
    )


if __name__ == "__main__":
    main()
