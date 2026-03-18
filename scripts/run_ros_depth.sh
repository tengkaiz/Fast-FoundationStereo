#!/bin/bash
# 一键启动 FoundationStereo ROS 深度推理节点
# 用法：bash scripts/run_ros_depth.sh [checkpoint] [backend]
#   checkpoint: 23-36-37 (默认,最高精度) | 20-26-39 (均衡) | 20-30-48 (最快)
#   backend:    tensorrt (默认) | pytorch
#
# 示例：
#   bash scripts/run_ros_depth.sh                      # TRT + 23-36-37
#   bash scripts/run_ros_depth.sh 20-26-39              # TRT + 20-26-39
#   bash scripts/run_ros_depth.sh 23-36-37 pytorch      # PyTorch + 23-36-37
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT="${1:-23-36-37}"
BACKEND="${2:-tensorrt}"

# 激活 conda 环境
source /opt/ros/noetic/setup.bash

cd "$PROJECT_DIR"

if [ "$BACKEND" = "tensorrt" ]; then
    ONNX_DIR="$PROJECT_DIR/output/$CHECKPOINT"
    FEATURE_ENGINE="$ONNX_DIR/feature_runner.engine"
    POST_ENGINE="$ONNX_DIR/post_runner.engine"

    if [ ! -f "$FEATURE_ENGINE" ] || [ ! -f "$POST_ENGINE" ]; then
        echo "[ERROR] TRT engine 不存在: $ONNX_DIR"
        echo "请先构建 engine:"
        echo "  python scripts/make_onnx.py --model_dir weights/$CHECKPOINT/model_best_bp2_serialize.pth --save_path output/$CHECKPOINT/ --height 480 --width 640"
        echo "  python scripts/build_trt_engine.py --onnx_dir output/$CHECKPOINT/"
        exit 1
    fi

    echo "========================================="
    echo " FoundationStereo ROS Depth Node"
    echo " Backend:    TensorRT"
    echo " Checkpoint: $CHECKPOINT"
    echo " Engine:     $ONNX_DIR"
    echo " Topics:     /camera/depth/image_raw"
    echo "             /camera/depth/camera_info"
    echo "             /camera/depth/points"
    echo "========================================="

    python scripts/ros_stereo_depth.py \
        --backend tensorrt \
        --onnx_dir "$ONNX_DIR" \
        --image_size 480 640

elif [ "$BACKEND" = "pytorch" ]; then
    MODEL_DIR="$PROJECT_DIR/weights/$CHECKPOINT/model_best_bp2_serialize.pth"

    if [ ! -f "$MODEL_DIR" ]; then
        echo "[ERROR] 模型文件不存在: $MODEL_DIR"
        exit 1
    fi

    echo "========================================="
    echo " FoundationStereo ROS Depth Node"
    echo " Backend:    PyTorch"
    echo " Checkpoint: $CHECKPOINT"
    echo " Model:      $MODEL_DIR"
    echo " Topics:     /camera/depth/image_raw"
    echo "             /camera/depth/camera_info"
    echo "             /camera/depth/points"
    echo "========================================="

    python scripts/ros_stereo_depth.py \
        --backend pytorch \
        --model_dir "$MODEL_DIR"

else
    echo "[ERROR] 不支持的 backend: $BACKEND (选择 tensorrt 或 pytorch)"
    exit 1
fi
