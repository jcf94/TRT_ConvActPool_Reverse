#!/usr/bin/env python3
import ctypes.util
import shutil

try:
    import tensorrt as trt
    print(f"python_tensorrt={trt.__version__}")
except Exception as exc:
    print(f"python_tensorrt=missing ({exc!r})")

print(f"trtexec={shutil.which('trtexec')}")
print(f"libnvinfer={ctypes.util.find_library('nvinfer')}")
