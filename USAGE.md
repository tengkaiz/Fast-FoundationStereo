# Usage

## Environment

```bash
conda create -n ffs python=3.12 && conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements_trt.txt  # optional, for TensorRT
```

## Weights

Download from [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and place under `weights/`.

| Checkpoint | PyTorch (ms) | TRT (ms) | Notes |
|-----------|-------------|---------|-------|
| `23-36-37` | 49.4 | 23.4 | Best accuracy |
| `20-26-39` | 43.6 | 19.4 | Balanced |
| `20-30-48` | 38.4 | 16.6 | Fastest |

> Profiled on RTX 3090 Ti, 480×640, `valid_iters=8`.

## Demo

```bash
python scripts/run_demo.py \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --left_file assets/left.png --right_file assets/right.png \
    --intrinsic_file assets/K.txt --out_dir output/demo/ \
    --get_pc 1 --no_viz 1
```

## TensorRT

```bash
# Export ONNX
python scripts/make_onnx.py \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/23-36-37/ --height 480 --width 640

# Build engine (GPU-specific, rebuild when switching GPU)
python scripts/build_trt_engine.py --onnx_dir output/23-36-37/

# Inference
python scripts/run_demo_tensorrt.py \
    --onnx_dir output/23-36-37/ \
    --left_file assets/left.png --right_file assets/right.png \
    --intrinsic_file assets/K.txt --out_dir output/demo_trt/
```

## ROS Node

Requires ROS Noetic.

```bash
bash scripts/run_ros_depth.sh                     # TRT + 23-36-37
bash scripts/run_ros_depth.sh 20-30-48             # TRT + fastest
bash scripts/run_ros_depth.sh 23-36-37 pytorch     # PyTorch backend
```

Published topics:

| Topic | Type |
|-------|------|
| `/camera/depth/image_raw` | Depth image (default uint16 mm) |
| `/camera/depth/camera_info` | Camera intrinsics |
| `/camera/depth/disparity` | Raw disparity (float32) |
| `/camera/depth/points` | RGB point cloud |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--valid_iters` | 8 | GRU iterations (4–8, fewer = faster) |
| `--max_disp` | 192 | Max disparity in pixels |
| `--scale` | 1.0 | Image scale factor (PyTorch only) |
| `--zfar` / `--zmin` | 10.0 / 0.1 | Valid depth range (meters) |
| `--denoise_cloud` | 1 | Bilateral filter + isolated point removal |
| `--edge_filter` | 1 | Edge flying pixel removal |
| `--edge_threshold` | 0.05 | Edge detection threshold (smaller = more aggressive) |
| `--depth_registration` | 1 | Register depth from left IR to RGB frame |
| `--depth_format` | `16uc1_mm` | `16uc1_mm` (uint16 mm) or `32fc1_m` (float32 m) |

## Intrinsic File Format (K.txt)

```
fx 0 cx 0 fy cy 0 0 1
baseline_in_meters
```

Not needed for ROS node (reads from `camera_info` topics).

## What's Added Beyond Upstream

`ros_stereo_depth.py` extends the original demo scripts with:

- **Dual backend**: switch between PyTorch and TensorRT via `--backend`
- **ROS integration**: subscribes to stereo topics, publishes depth / disparity / point cloud in standard ROS message types
- **Depth post-processing pipeline**:
  - Bilateral filtering + isolated point removal (`--denoise_cloud`)
  - Edge flying pixel removal via depth gradient analysis (`--edge_filter`)
  - Depth range clamping (`--zfar`, `--zmin`)
- **Depth registration**: GPU-accelerated reprojection from left IR to RGB camera frame using TF transforms and `scatter_reduce` z-buffer (`--depth_registration`)
- **Auto camera info**: reads intrinsics and baseline from `camera_info` topics, no manual K.txt needed
- **Latency-optimized**: worker thread with frame skipping, padder caching, pre-computed GPU tensors for registration
