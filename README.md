# MemUrban

[中文](#中文说明) | [English](#english)

MemUrban is a research-oriented prototype for transforming first-person urban or campus videos into structured spatial-temporal behavioral memory. The current pipeline preserves the original frame-based video processing workflow while extending it with full-audio transcription, event-level multimodal alignment, hierarchical memory construction, and a placeholder skill-evolution module.

MemUrban 是一个面向研究原型的项目，用于将第一人称城市/校园视频转化为结构化的时空行为记忆。当前实现保留了原有的基于关键帧的视频处理主链，同时扩展了整段音频转写、事件级多模态对齐、分层记忆建模，以及一个用于未来自进化的 skill 占位模块。

## 中文说明

### 1. 项目目标

本项目关注以下问题：

- 如何将连续视频流压缩为结构化的时空事件序列。
- 如何把视觉帧与整段音频转写对齐，提升行为解释的一致性。
- 如何将事件序列进一步组织为短期记忆、长期记忆、语义记忆与个体画像。
- 如何为未来的 agent 自进化机制预留技能反馈与演化接口。

### 2. 方法概览

当前 pipeline 如下：

1. 使用 FFmpeg 抽帧。
2. 使用 OpenCV HSV 直方图进行相邻帧去重。
3. 提取整段视频音频，并通过 `whisper-1` 进行转写。
4. 在帧级分析时注入与时间窗对齐的音频文本。
5. 将帧级结果聚合为事件级时空行为记录。
6. 将事件写入分层 memory，并生成实体记忆与 persona 画像。
7. 为后续 skill 自进化保留 feedback 和 proposal 机制。

### 3. 项目结构

```text
memurban/
  app/        应用与 CLI 入口
  pipeline/   抽帧、音频转写、帧分析、事件聚合
  memory/     分层记忆、实体记忆、向量检索
  skills/     skill 注册、反馈、演化提案占位模块
scripts/      运行脚本与开发辅助脚本
tests/        最小验证脚本
main.py       兼容入口
memory_agent.py 兼容导入入口
```

### 4. 核心模块

#### 4.1 Pipeline

- [memurban/pipeline/frame_extractor.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/pipeline/frame_extractor.py:1)
  负责抽帧与时间戳对齐。
- [memurban/pipeline/audio_transcriber.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/pipeline/audio_transcriber.py:1)
  负责音频提取、Whisper 转写与事件时间窗对齐。
- [memurban/pipeline/frame_analyzer.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/pipeline/frame_analyzer.py:1)
  负责多模态帧分析。
- [memurban/pipeline/event_builder.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/pipeline/event_builder.py:1)
  负责帧级结果到事件级记录的聚合。

#### 4.2 Memory

- [memurban/memory/agent.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/memory/agent.py:1)
  统一编排短期记忆、长期记忆、语义记忆、实体记忆与 persona。
- [memurban/memory/stores.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/memory/stores.py:1)
  负责长期记忆持久化、增量加载与去重。
- [memurban/memory/entities.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/memory/entities.py:1)
  负责地点实体、互动对象与路线模式聚合。
- [memurban/memory/vector_index.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/memory/vector_index.py:1)
  提供本地 Numpy 余弦相似度向量索引。

#### 4.3 Skills

- [memurban/skills/models.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/skills/models.py:1)
  定义 `SkillCard`、`SkillFeedback`、`SkillMutationProposal`。
- [memurban/skills/manager.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/skills/manager.py:1)
  管理 skill 注册表与 JSON 持久化。
- [memurban/skills/evolution.py](/Users/wenwang/Documents/WorkTable/Agent/MemUrban/memurban/skills/evolution.py:1)
  根据 memory 或 decision 反馈生成 skill 演化提案。

### 5. 当前能力

- 保留原始关键帧处理与事件聚合逻辑。
- 支持整段视频音频转写与帧级时间窗注入。
- 支持事件级视觉-音频证据融合。
- 支持短期记忆、长期记忆、语义记忆、实体记忆与 persona 画像。
- 支持长期记忆持久化、增量加载与去重。
- 支持 skill 自进化的占位式 feedback/proposal 流程。

### 6. 安装

```bash
pip install -r requirements.txt
```

系统依赖：

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`

可选环境变量：

```bash
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 7. 运行

推荐入口：

```bash
python -m memurban.app.main \
  --video path/to/campus_walk.mp4 \
  --fps 0.5 \
  --sim-threshold 0.92 \
  --model gpt-4o \
  --whisper-model whisper-1 \
  --language zh \
  --output output
```

兼容入口仍可使用：

```bash
python main.py --video path/to/campus_walk.mp4
```

脚本入口：

```bash
bash scripts/run_agent.sh
```

如果 `output/memory_store.json` 已存在，主流程会先增量加载历史 memory，再合并本次视频事件。

### 8. 输出

- `output/audio/audio_transcript.json`: 全视频音频转写结果
- `output/frames/all/`: 全部抽帧
- `output/frames/selected/`: 去重后关键帧
- `output/frame_results/*.json`: 单帧分析结果
- `output/events.json`: 事件级时空行为序列
- `output/memory_store.json`: memory 导出结果

`memory_store.json` 当前包含：

- `short_term_memory`
- `long_term_memory`
- `semantic_memory`
- `entity_memory`
- `persona_profile`

### 9. 最小验证

```bash
python tests/test_memory_core.py
python tests/test_skill_core.py
```

### 10. 当前局限

- 向量检索当前为本地确定性 hashing 方案，适合原型验证，不等同于生产级语义 embedding。
- `skills` 模块目前只作为占位，不参与 agent 的实时执行。
- 视频主链执行仍依赖本地安装 `opencv-python`、`ffmpeg` 以及可选的 OpenAI SDK。

### 11. 安全说明

- 项目不应提交真实 API key。
- `.gitignore` 已排除 `data/`、`output/`、`.env*`、缓存文件等内容。
- 示例脚本中的 API 配置均通过环境变量读取。

## English

### 1. Objective

MemUrban aims to study how first-person videos can be transformed into structured spatial-temporal behavioral memory for downstream agent reasoning. The repository focuses on:

- compressing continuous video streams into structured event sequences;
- aligning visual frames with full-audio transcription;
- organizing events into hierarchical memory layers;
- reserving a future interface for self-evolving skills.

### 2. Method Overview

The current processing pipeline is:

1. Extract frames with FFmpeg.
2. Deduplicate neighboring frames with OpenCV HSV histogram similarity.
3. Extract the full audio track and transcribe it with `whisper-1`.
4. Inject time-aligned transcript windows into frame-level multimodal analysis.
5. Merge frame-level results into event-level spatial-temporal records.
6. Write events into hierarchical memory with entity memory and persona profiling.
7. Reserve a skill feedback and proposal loop for future self-evolution.

### 3. Repository Layout

```text
memurban/
  app/        application and CLI entrypoints
  pipeline/   frame extraction, transcription, frame analysis, event building
  memory/     hierarchical memory, entity memory, vector retrieval
  skills/     placeholder modules for skill evolution
scripts/      runnable helper scripts
tests/        minimal verification scripts
main.py       backward-compatible CLI shim
memory_agent.py backward-compatible import shim
```

### 4. Features

- Multimodal video processing with frame-level and audio-level alignment.
- Event construction from deduplicated visual observations.
- Hierarchical memory with short-term, long-term, semantic, entity, and persona layers.
- Local vector retrieval with incremental persistence and reload.
- Placeholder skill evolution workflow driven by memory or decision feedback.

### 5. Installation

```bash
pip install -r requirements.txt
```

System dependency:

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`

Optional environment variables:

```bash
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 6. Usage

Recommended entrypoint:

```bash
python -m memurban.app.main \
  --video path/to/campus_walk.mp4 \
  --fps 0.5 \
  --sim-threshold 0.92 \
  --model gpt-4o \
  --whisper-model whisper-1 \
  --language zh \
  --output output
```

Backward-compatible entrypoint:

```bash
python main.py --video path/to/campus_walk.mp4
```

Helper script:

```bash
bash scripts/run_agent.sh
```

If `output/memory_store.json` already exists, the application incrementally reloads the previous memory state before merging the current run.

### 7. Outputs

- `output/audio/audio_transcript.json`
- `output/frames/all/`
- `output/frames/selected/`
- `output/frame_results/*.json`
- `output/events.json`
- `output/memory_store.json`

### 8. Minimal Verification

```bash
python tests/test_memory_core.py
python tests/test_skill_core.py
```

### 9. Limitations

- The current vector retrieval is a local deterministic hashing baseline rather than a production semantic embedding stack.
- The `skills` subsystem is intentionally a placeholder and is not yet connected to runtime agent execution.
- End-to-end video execution still depends on locally available `opencv-python`, `ffmpeg`, and optionally the OpenAI SDK.

### 10. Security Notes

- Do not commit real API keys.
- The repository ignores `data/`, `output/`, `.env*`, caches, and temporary artifacts.
- Example scripts read API configuration from environment variables only.
