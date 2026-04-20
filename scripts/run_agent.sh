#!/usr/bin/env bash
set -euo pipefail

# ===== 可修改参数 =====
# VIDEO_PATH: 输入视频路径（必填，支持 mp4）
VIDEO_NAME="DJI_test"
VIDEO_PATH="data/${VIDEO_NAME}.MP4"

# FPS: 抽帧率（单位：帧/秒）
# 例如 0.5 表示每 2 秒抽 1 帧；越大越密，分析成本越高
FPS="0.5"

# SIM_THRESHOLD: 去重阈值（0~1）
# 越高表示“更严格去重”（保留更少帧），推荐 0.88~0.95
SIM_THRESHOLD="0.92"

# MODEL: 多模态分析模型名称（OpenAI 兼容）
# 常用：gpt-4o, gpt-4o-mini
MODEL="gpt-4o"

# OUTPUT_DIR: 输出目录
# 将生成 frames/, frame_results/, events.json, memory_store.json
mkdir -p output
OUTPUT_DIR="output/${VIDEO_NAME}"

# 可选：OpenAI 兼容网关
# export OPENAI_API_KEY="your_key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"

# ===== 执行 =====
python -m memurban.app.main \
  --video "${VIDEO_PATH}" \
  --fps "${FPS}" \
  --sim-threshold "${SIM_THRESHOLD}" \
  --model "${MODEL}" \
  --output "${OUTPUT_DIR}"
