#!/bin/bash
# 自动化 benchmark：不同模型 × 不同后端 × 不同 valid_iters
# 前提：相机数据流已开启（roslaunch orbbec_camera gemini_330_series.launch）
# 用法：bash scripts/benchmark_ros.sh

set -e
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ZS_VLN
source /opt/ros/noetic/setup.bash

# 清理残留进程
pkill -f "ros_stereo_depth" 2>/dev/null || true
sleep 2

SCRIPT="scripts/ros_stereo_depth.py"
RESULT_FILE="docs/benchmark_results.txt"
WARMUP_SEC=30       # 等待预热稳定的秒数（TRT 首次加载需要较长时间）
COLLECT_SEC=20      # 采集数据的秒数
WEIGHTS_DIR="weights"
OUTPUT_DIR="output"

mkdir -p docs

echo "============================================" > "$RESULT_FILE"
echo "Fast-FoundationStereo Benchmark Results"       >> "$RESULT_FILE"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"            >> "$RESULT_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$RESULT_FILE"
echo "Camera: Orbbec Gemini 330, 480x640, 15fps IR"   >> "$RESULT_FILE"
echo "============================================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

run_benchmark() {
    local label="$1"
    shift
    local cmd="python $SCRIPT $@ --depth_registration 0"

    echo ""
    echo "========================================"
    echo "Running: $label"
    echo "Command: $cmd"
    echo "========================================"

    # 启动推理进程，输出重定向到临时文件
    local tmplog=$(mktemp /tmp/bench_XXXXXX.log)
    $cmd > "$tmplog" 2>&1 &
    local pid=$!

    # 挂一个订阅者触发推理（节点检测到有订阅才会跑）
    rostopic hz /foundation_stereo/depth/image_raw > /dev/null 2>&1 &
    local sub_pid=$!

    # 等待预热
    echo "  Warming up ${WARMUP_SEC}s ..."
    sleep $WARMUP_SEC

    # 清空临时文件，只采集稳定阶段的日志
    > "$tmplog"
    echo "  Collecting data ${COLLECT_SEC}s ..."
    sleep $COLLECT_SEC

    # 停止订阅者和推理进程
    kill $sub_pid 2>/dev/null || true
    kill $pid 2>/dev/null || true
    wait $sub_pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 2

    # 提取最后几条统计日志（--text 防止 ANSI 颜色码导致 grep 误判为二进制）
    local stats=$(grep --text "\[统计\]" "$tmplog" | tail -5)

    echo "" >> "$RESULT_FILE"
    echo "--- $label ---" >> "$RESULT_FILE"
    echo "CMD: $cmd" >> "$RESULT_FILE"
    if [ -z "$stats" ]; then
        echo "  [WARNING] 未采集到统计数据" >> "$RESULT_FILE"
    else
        echo "$stats" >> "$RESULT_FILE"
        # 提取最后一条的关键数值
        local last_line=$(echo "$stats" | tail -1)
        echo "" >> "$RESULT_FILE"
        echo "  摘要: $(echo "$last_line" | sed 's/.*\[统计\] //')" >> "$RESULT_FILE"
    fi

    rm -f "$tmplog"
    echo "  Done: $label"
}

# ===================== TensorRT 后端 =====================
echo ""
echo "########################################"
echo "#        TensorRT Backend Tests        #"
echo "########################################"

for model in 23-36-37 20-26-39 20-30-48; do
    run_benchmark \
        "TRT | $model | iters=8" \
        --backend tensorrt \
        --onnx_dir "$OUTPUT_DIR/$model" \
        --image_size 480 640
done

# ===================== PyTorch 后端 =====================
echo ""
echo "########################################"
echo "#        PyTorch Backend Tests         #"
echo "########################################"

for model in 23-36-37 20-26-39 20-30-48; do
    for iters in 8 4; do
        run_benchmark \
            "PyTorch | $model | iters=$iters" \
            --backend pytorch \
            --model_dir "$WEIGHTS_DIR/$model/model_best_bp2_serialize.pth" \
            --valid_iters $iters
    done
done

# ===================== 汇总 =====================
echo "" >> "$RESULT_FILE"
echo "============================================" >> "$RESULT_FILE"
echo "Benchmark completed at $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULT_FILE"
echo "============================================" >> "$RESULT_FILE"

echo ""
echo "========================================"
echo "All benchmarks done! Results saved to: $RESULT_FILE"
echo "========================================"
cat "$RESULT_FILE"
