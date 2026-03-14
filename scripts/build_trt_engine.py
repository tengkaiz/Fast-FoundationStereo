"""将 ONNX 模型转换为 TensorRT 引擎文件 (.engine)"""
import argparse
import os
import tensorrt as trt


def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    # 分配最大 4GB 显存用于构建
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building engine for {onnx_path} ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved to {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, required=True, help="Directory containing feature_runner.onnx and post_runner.onnx")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 precision")
    args = parser.parse_args()

    onnx_dir = args.onnx_dir
    fp16 = not args.no_fp16

    for name in ["feature_runner", "post_runner"]:
        onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
        engine_path = os.path.join(onnx_dir, f"{name}.engine")
        if not os.path.exists(onnx_path):
            print(f"Warning: {onnx_path} not found, skipping")
            continue
        build_engine(onnx_path, engine_path, fp16=fp16)

    print("Done!")
