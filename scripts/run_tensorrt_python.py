#!/usr/bin/env python3
import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda


ROOT = Path(__file__).resolve().parents[1]


def ck(ret):
    err = ret[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(err)
    if len(ret) == 1:
        return None
    return ret[1] if len(ret) == 2 else ret[1:]


def create_context(dev):
    try:
        return ck(cuda.cuCtxCreate(None, 0, dev))
    except TypeError:
        return ck(cuda.cuCtxCreate(0, dev))


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.records = {}

    def report_layer_time(self, layer_name, ms):
        total, count = self.records.get(layer_name, (0.0, 0))
        self.records[layer_name] = (total + float(ms), count + 1)


class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shape, cache_file):
        super().__init__()
        self.cache_file = Path(cache_file)
        self.done = False
        self.host = np.random.default_rng(1234).normal(0, 1, size=input_shape).astype(np.float32)
        self.nbytes = self.host.nbytes
        self.ptr = ck(cuda.cuMemAlloc(self.nbytes))
        ck(cuda.cuMemcpyHtoD(self.ptr, self.host.ctypes.data, self.nbytes))

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.done:
            return None
        self.done = True
        return [int(self.ptr)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)

    def free(self):
        if getattr(self, "ptr", None):
            ck(cuda.cuMemFree(self.ptr))
            self.ptr = None


def dtype_nbytes(dtype):
    return np.dtype(trt.nptype(dtype)).itemsize


def volume(shape):
    out = 1
    for v in shape:
        out *= int(v)
    return out


def set_dynamic_ranges(network):
    for i in range(network.num_inputs):
        network.get_input(i).dynamic_range = (-1.0, 1.0)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            t = layer.get_output(j)
            if t is not None:
                t.dynamic_range = (-1.0, 1.0)


def force_int8_layers(network):
    for i in range(network.num_inputs):
        network.get_input(i).dynamic_range = (-1.0, 1.0)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        try:
            layer.precision = trt.int8
        except Exception:
            pass
        for j in range(layer.num_outputs):
            t = layer.get_output(j)
            if t is None:
                continue
            t.dynamic_range = (-1.0, 1.0)
            try:
                layer.set_output_type(j, trt.int8)
            except Exception:
                pass
    for i in range(network.num_outputs):
        out = network.get_output(i)
        out.dynamic_range = (-1.0, 1.0)
        try:
            out.dtype = trt.int8
        except Exception:
            pass


def build_engine(onnx_path, int8, explicit_qdq, save_engine=None):
    ck(cuda.cuInit(0))
    dev = ck(cuda.cuDeviceGet(0))
    build_ctx = create_context(dev)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    data = Path(onnx_path).read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"failed to parse {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    try:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except Exception:
        pass
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if explicit_qdq:
            calibrator = None
        else:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            force_int8_layers(network)
            input_shape = tuple(network.get_input(0).shape)
            calibrator = RandomCalibrator(input_shape, ROOT / "results" / (Path(onnx_path).stem + ".calib"))
            config.int8_calibrator = calibrator
    else:
        calibrator = None
    serialized = builder.build_serialized_network(network, config)
    if calibrator is not None:
        calibrator.free()
    if serialized is None:
        ck(cuda.cuCtxDestroy(build_ctx))
        raise RuntimeError("TensorRT build failed")
    if save_engine:
        Path(save_engine).parent.mkdir(parents=True, exist_ok=True)
        Path(save_engine).write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine, build_ctx


def load_engine(engine_path):
    ck(cuda.cuInit(0))
    dev = ck(cuda.cuDeviceGet(0))
    ctx = create_context(dev)
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        ck(cuda.cuCtxDestroy(ctx))
        raise RuntimeError(f"failed to load engine: {engine_path}")
    return engine, ctx


def tensor_shape(context, engine, name):
    shape = tuple(context.get_tensor_shape(name))
    if any(v < 0 for v in shape):
        shape = tuple(engine.get_tensor_shape(name))
    return shape


def run_engine(engine, warmup, iters, ctx):
    stream = ck(cuda.cuStreamCreate(0))
    context = engine.create_execution_context()
    profiler = Profiler()
    context.profiler = profiler

    allocations = []
    host_inputs = []
    for i in range(engine.num_io_tensors):
      name = engine.get_tensor_name(i)
      shape = tensor_shape(context, engine, name)
      dtype = engine.get_tensor_dtype(name)
      nbytes = volume(shape) * dtype_nbytes(dtype)
      ptr = ck(cuda.cuMemAlloc(nbytes))
      allocations.append(ptr)
      context.set_tensor_address(name, int(ptr))
      if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
          arr = np.random.default_rng(1234).normal(0, 1, size=shape).astype(trt.nptype(dtype))
          host_inputs.append(arr)
          ck(cuda.cuMemcpyHtoDAsync(ptr, arr.ctypes.data, nbytes, stream))

    for _ in range(warmup):
        context.execute_async_v3(stream_handle=int(stream))
    ck(cuda.cuStreamSynchronize(stream))
    profiler.records.clear()

    start = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v3(stream_handle=int(stream))
    ck(cuda.cuStreamSynchronize(stream))
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    try:
        context.report_to_profiler()
    except Exception:
        pass

    for ptr in allocations:
        ck(cuda.cuMemFree(ptr))
    ck(cuda.cuStreamDestroy(stream))
    records = [
        (name, total / count, count)
        for name, (total, count) in sorted(profiler.records.items())
    ]
    return total_ms, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--explicit-qdq", action="store_true")
    ap.add_argument("--save-engine")
    ap.add_argument("--load-engine")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()
    if not args.load_engine and not args.onnx:
        ap.error("--onnx is required unless --load-engine is used")

    libs = Path(trt.__file__).resolve().parent.parent / "tensorrt_libs"
    os.environ["LD_LIBRARY_PATH"] = f"{libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    if args.load_engine:
        engine, ctx = load_engine(args.load_engine)
    else:
        engine, ctx = build_engine(args.onnx, args.int8, args.explicit_qdq,
                                   args.save_engine)
    total_ms, records = run_engine(engine, args.warmup, args.iters, ctx)
    print(f"engine_mean_ms,{total_ms:.6f}")
    print("layer,mean_ms,count")
    for name, ms, count in records:
        print(f"{name},{ms:.6f},{count}")


if __name__ == "__main__":
    main()
