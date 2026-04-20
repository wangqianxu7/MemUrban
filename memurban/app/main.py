#!/usr/bin/env python3
"""Video parsing guide based spatial-temporal behavior memory agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--fps", type=float, default=0.5, help="抽帧率（帧/秒）")
    p.add_argument("--sim-threshold", type=float, default=0.92, help="去重阈值")
    p.add_argument("--model", type=str, default="gpt-4o", help="视觉分析模型")
    p.add_argument("--whisper-model", type=str, default="whisper-1", help="音频转写模型")
    p.add_argument("--language", type=str, default="zh", help="音频语言提示")
    p.add_argument("--output", type=str, default="output", help="输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from ..memory import SpatialTemporalBehaviorAgent
    from ..pipeline.audio_transcriber import attach_transcript_to_events, load_or_create_transcript, parse_segments
    from ..pipeline.event_builder import merge_frames_to_events
    from ..pipeline.frame_analyzer import analyze_all_frames_sync, build_client
    from ..pipeline.frame_extractor import extract_and_deduplicate

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mem_path = out / "memory_store.json"

    print("=" * 60)
    print("Step1 抽帧 + 去重")
    all_frames, selected = extract_and_deduplicate(
        video_path=args.video,
        output_dir=str(out / "frames"),
        fps=args.fps,
        similarity_threshold=args.sim_threshold,
    )
    print(f"全部帧: {len(all_frames)}, 去重后: {len(selected)}")

    print("=" * 60)
    print("Step2 全视频音频转写")
    client = build_client()
    transcript_payload = load_or_create_transcript(
        video_path=args.video,
        output_dir=str(out / "audio"),
        client=client,
        model=args.whisper_model,
        language=args.language,
    )
    transcript_segments = parse_segments(transcript_payload)
    print(f"音频片段数: {len(transcript_segments)}")

    print("=" * 60)
    print("Step3 多模态帧分析")
    frame_results = analyze_all_frames_sync(
        selected_frames=selected,
        output_dir=str(out / "frame_results"),
        model=args.model,
        transcript_segments=transcript_segments,
    )
    print(f"帧分析结果: {len(frame_results)}")

    print("=" * 60)
    print("Step4 事件聚合与音频对齐")
    events = attach_transcript_to_events(merge_frames_to_events(frame_results), transcript_segments)
    events_path = out / "events.json"
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"事件数: {len(events)} -> {events_path}")

    print("=" * 60)
    print("Step5 写入时空行为记忆 Agent")
    agent = SpatialTemporalBehaviorAgent(perceptual_buffer_size=5)
    if mem_path.exists():
        loaded_count = agent.load_memory_export(str(mem_path), merge=True)
        print(f"检测到历史记忆，已增量加载: {loaded_count} 条")
    agent.ingest_events(events)
    agent.export_memory(str(mem_path))
    print(f"记忆已保存: {mem_path}")

    # quick retrieval demo
    sample_query = "绿荫路径 社交 决策点"
    top = agent.query_related_memories(sample_query, top_k=3)
    print(f"检索示例 query='{sample_query}' -> {len(top)} 条")


if __name__ == "__main__":
    main()
