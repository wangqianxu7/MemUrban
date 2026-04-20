# 时空行为视频记忆 Agent

基于 `video_parsing_guide.jsx` 的实现：

- 抽帧（FFmpeg）
- 智能去重（OpenCV HSV 直方图）
- 全视频音频提取与 `whisper-1` 转写
- 音频文本与视频帧/事件时间窗对齐
- 多模态帧分析（OpenAI 兼容接口，失败自动回退）
- 帧级结果聚合为事件序列 `events.json`
- 写入时空行为记忆 Agent（短期记忆 + 长期情节记忆 + 语义规则 + 个体风格画像）

## 安装

```bash
pip install -r requirements.txt
```

系统需要 `ffmpeg`：

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`

可选环境变量：

```bash
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
```

## 运行

```bash
python main.py \
  --video path/to/campus_walk.mp4 \
  --fps 0.5 \
  --sim-threshold 0.92 \
  --model gpt-4o \
  --whisper-model whisper-1 \
  --language zh \
  --output output
```

如果 `output/memory_store.json` 已存在，主流程会先自动增量加载历史长期记忆，再把本次新事件合并进去。

## 输出

- `output/audio/audio_transcript.json`: 全视频音频转写结果
- `output/frames/all/`: 全部抽帧
- `output/frames/selected/`: 去重后关键帧
- `output/frame_results/*.json`: 单帧分析结果
- `output/events.json`: 含音频证据的事件级时空行为序列
- `output/memory_store.json`: 分层 memory 与 persona 画像结果

## 当前重构后的 Agent 能力

- 保留原有的视频抽帧、去重、视觉理解和事件聚合流程
- 新增使用 `whisper-1` 对整段视频音频转写，并按时间窗注入帧分析
- 事件层同时保留视觉证据与音频证据，便于做更高一致性的行为解释
- `memory_store.json` 现在包含：
  - `short_term_memory`: 最近事件缓冲
  - `long_term_memory`: 完整情节记忆与检索摘要
  - `semantic_memory`: 从多段事件反思得到的稳定规则
  - `entity_memory`: 地点实体、互动对象实体、路线模式实体
  - `persona_profile`: 个体移动风格、社交风格、环境偏好、决策风格等

## memory_core 模块结构

- `memory_core/embeddings.py`: 本地确定性文本向量化
- `memory_core/vector_index.py`: Numpy 余弦相似度向量索引
- `memory_core/stores.py`: 短期、长期、语义 memory store
- `memory_core/entities.py`: 地点/互动/路线模式实体记忆
- `memory_core/persona.py`: 个体风格推断
- `memory_core/agent.py`: 统一的 agent 编排接口

当前向量检索为纯本地实现，不依赖外部向量数据库，适合先跑通原型。`SpatialTemporalBehaviorAgent.query_related_memories_with_scores()` 可返回相似度分数和 metadata，便于后续接入实体过滤或长期存储后端。

## 新增能力

- `query_entity_memory("places" | "people" | "route_patterns")`: 查询实体记忆
- `save_long_term_memory(path)`: 将长期记忆和本地向量索引状态持久化到磁盘
- `load_long_term_memory(path, merge=True)`: 从长期记忆文件增量加载事件并自动去重
- `load_memory_export(path, merge=True)`: 从完整 `memory_store.json` 增量恢复长期记忆、短期记忆和派生记忆

这套持久化方案当前采用 JSON 文件，特点是简单、可检查、方便增量恢复。后续如果要接到 SQLite、Chroma 或 pgvector，可以直接替换 `LongTermMemoryStore` 的磁盘层，不需要改上层 agent 接口。

## skill_core 占位模块

项目中新增了一个尚未接入主执行链的 `skill_core/`，用于未来实现“基于 memory / decision 反馈的自进化技能系统”。

- `skill_core/models.py`: skill、反馈、演化提案的数据结构
- `skill_core/manager.py`: skill 注册表与 JSON 持久化
- `skill_core/evolution.py`: 基于反馈生成“新增/完善 skill”提案的占位引擎

当前阶段它只负责：

- 记录草案型 `SkillCard`
- 接收来自 memory 或 decision 的反馈信号
- 生成 `SkillMutationProposal`
- 保存到本地 JSON 供后续人工审阅或接入自动执行

当前阶段它不会被 `main.py` 或 `SpatialTemporalBehaviorAgent` 自动调用。
