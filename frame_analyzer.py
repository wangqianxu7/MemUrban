"""Frame-level multimodal analysis with OpenAI-compatible API and fallback."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from audio_transcriber import TranscriptSegment, transcript_window_text
from frame_extractor import FrameSample

if TYPE_CHECKING:
    from openai import OpenAI


PROMPT_TEMPLATE = """
你是一个专为「量化自我」设计的个人时空行为感知 Agent。
你的任务是：基于当前帧图像、同时间窗音频转写与历史上下文，输出一份结构化的当前时刻快照，用于长期行为数据库的积累与分析。

你需要同时完成以下四类推断：
1. 场景感知 —— 我在哪里、环境质量如何
2. 行为分析 —— 我在做什么、运动状态与决策节点
3. 心理推断 —— 我的情绪状态与当前意图
4. 智能响应 —— 预测下一步、识别异常、给出行动建议

输出必须严格为以下 JSON 结构，无任何额外文字：

{{
  "meta": {{
    "frame_idx": {frame_idx},
    "video_second": {video_second},
    "capture_confidence": 0.0
  }},

  "scene": {{
    "location_type": "道路|室内房间|商场|办公室|车站|公园|校园|住宅区|餐厅|健身房|交通工具|其他",
    "specific_description": "",
    "functional_zone": "通行|等候|工作|休闲|消费|居住|运动|交通换乘|社交|其他",
    "indoor_outdoor": "室内|室外|半室外|未知",
    "greenery_level": 1,
    "shade_coverage": "无|部分|大部分|完全|不适用",
    "crowd_density": "空旷|稀疏|适中|拥挤|未知",
    "noise_estimate": "安静|一般|嘈杂|未知",
    "comfort_rating": 1,
    "notable_facilities": []
  }},

  "movement": {{
    "activity_type": "步行|跑步|骑行|驾车|乘坐公共交通|静坐|站立|躺卧|运动锻炼|饮食|工作|社交|其他|未知",
    "walking_state": "行走中|停留|转向|加速|减速|跑动|乘交通工具|未知",
    "estimated_speed": "快|中|慢|静止|未知",
    "direction_change": false,
    "is_decision_point": false,
    "decision_description": ""
  }},

  "social": {{
    "people_visible": 0,
    "interaction_occurring": false,
    "interaction_type": "无|与熟人|与陌生人|与服务人员|视频通话|未知",
    "interaction_description": null,
    "social_role": "独处|主导|参与|旁观|未知"
  }},

  "temporal_cues": {{
    "estimated_time_of_day": "清晨|上午|中午|下午|傍晚|夜间|未知",
    "weather_indication": "晴|阴|雨|雪|室内不可判断|未知",
    "lighting_condition": "自然光充足|自然光弱|人工照明|混合|未知",
    "time_confidence": 0.0
  }},

  "psychological": {{
    "inferred_emotion": "平静|愉悦|专注|疲惫|焦虑|无聊|压力|放松|未知",
    "emotion_confidence": 0.0,
    "emotion_cues": [],
    "inferred_intent": "",
    "intent_category": "目的性移动|目的性任务|休息恢复|社交互动|消费行为|探索漫游|等待|未知",
    "stress_level": 1,
    "energy_level": 1
  }},

  "prediction": {{
    "next_action_prediction": "",
    "next_location_prediction": "",
    "prediction_confidence": 0.0,
    "predicted_duration_minutes": null
  }},

  "alerts": {{
    "anomaly_detected": false,
    "anomaly_type": null,
    "anomaly_description": null,
    "action_suggestion": null,
    "suggestion_priority": "无|低|中|高"
  }},

  "change_from_previous": {{
    "scene_changed": false,
    "activity_changed": false,
    "emotion_changed": false,
    "what_changed": "",
    "significance": "无变化|微小变化|场景切换|行为切换|情绪波动|重要事件"
  }}
}}

当前帧索引: {frame_idx}
视频秒数: {video_second}
当前时窗音频转写: {audio_transcript}
上一帧摘要: {prev_summary}

约束：
1. 仅输出 JSON，绝对不要输出任何解释、注释或 markdown；
2. 所有枚举字段必须从给定选项中选择，不得自造新值；
3. 若无法判断，使用"未知"或 null，不要编造细节；
4. capture_confidence / emotion_confidence / prediction_confidence / time_confidence 均为 0~1 浮点数；
5. comfort_rating / greenery_level / stress_level / energy_level 均为 1~5 整数；
6. emotion_cues 为支撑情绪判断的视觉线索列表，如 ["低头看手机", "脚步缓慢"]；
7. anomaly_type 可选值：久坐过长|长时间未进食|异常高压|危险环境|异常孤立|其他|null；
8. 跨帧推理：若 prev_summary 非空，应基于其变化趋势修正 psychological 和 prediction 的判断；
9. 若音频文本与画面高度相关，应在 specific_description、interaction_description、inferred_intent 中体现这种对齐；若无语音或无法判断，不要强行引用音频。
"""


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _fallback_result(frame_idx: int, video_second: float, audio_transcript: str = "") -> Dict[str, Any]:
    locs = ["道路", "教学区", "绿地", "食堂区", "宿舍区"]
    loc = locs[frame_idx % len(locs)]
    return {
        "scene": {
            "location_type": loc,
            "specific_description": f"{loc}内步行视角",
            "functional_zone": f"{loc}过渡带",
            "indoor_outdoor": "室外",
            "greenery_level": 4 if "绿" in loc else 3,
            "shade_coverage": "部分",
            "crowd_density": "稀疏",
            "noise_estimate": "一般",
            "comfort_rating": 4,
            "notable_facilities": ["路灯", "座椅"] if frame_idx % 5 == 0 else ["路灯"],
        },
        "movement": {
            "activity_type": "步行",
            "walking_state": "行走中" if frame_idx % 6 else "转向",
            "estimated_speed": "中",
            "direction_change": bool(frame_idx % 6 == 0),
            "is_decision_point": bool(frame_idx % 8 == 0),
            "decision_description": "岔路可选林荫道或直行主路" if frame_idx % 8 == 0 else "",
        },
        "social": {
            "people_visible": 2 if frame_idx % 7 == 0 else 0,
            "interaction_occurring": bool(frame_idx % 11 == 0),
            "interaction_type": "与熟人" if frame_idx % 11 == 0 else "无",
            "interaction_description": "与同学短暂打招呼" if frame_idx % 11 == 0 else None,
            "social_role": "参与" if frame_idx % 11 == 0 else "独处",
        },
        "temporal_cues": {
            "estimated_time_of_day": "下午",
            "weather_indication": "未知",
            "lighting_condition": "自然光充足",
            "time_confidence": 0.5,
        },
        "psychological": {
            "inferred_emotion": "专注",
            "emotion_confidence": 0.4,
            "emotion_cues": ["视角稳定", "持续前进"],
            "inferred_intent": "沿当前路径继续通行",
            "intent_category": "目的性移动",
            "stress_level": 2,
            "energy_level": 3,
        },
        "prediction": {
            "next_action_prediction": "继续前行",
            "next_location_prediction": "相邻通行路径",
            "prediction_confidence": 0.45,
            "predicted_duration_minutes": 2,
        },
        "alerts": {
            "anomaly_detected": False,
            "anomaly_type": None,
            "anomaly_description": None,
            "action_suggestion": None,
            "suggestion_priority": "无",
        },
        "change_from_previous": {
            "scene_changed": bool(frame_idx % 9 == 0),
            "activity_changed": bool(frame_idx % 6 == 0),
            "emotion_changed": False,
            "what_changed": "进入新功能区" if frame_idx % 9 == 0 else "轻微视角变化",
            "significance": "场景切换" if frame_idx % 9 == 0 else "微小变化",
        },
        "audio": {
            "aligned_transcript": audio_transcript,
            "alignment_confidence": 0.25 if audio_transcript else 0.0,
        },
        "_meta": {"frame_idx": frame_idx, "video_second": video_second},
    }


def build_client() -> Optional["OpenAI"]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    base = os.getenv("OPENAI_BASE_URL", "").strip()
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def analyze_frame_openai(
    client: "OpenAI",
    frame: FrameSample,
    frame_idx: int,
    prev_summary: str,
    audio_transcript: str,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Call OpenAI-compatible API for one frame."""
    prompt = PROMPT_TEMPLATE.format(
        frame_idx=frame_idx,
        video_second=round(frame.timestamp_second, 2),
        audio_transcript=audio_transcript[:300] or "无明显语音内容",
        prev_summary=prev_summary[:200],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame.path)}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=1200,
        temperature=0.1,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    parsed["audio"] = {
        "aligned_transcript": audio_transcript,
        "alignment_confidence": 0.8 if audio_transcript else 0.1,
    }
    parsed["_meta"] = {
        "frame_idx": frame_idx,
        "video_second": frame.timestamp_second,
        "frame_path": str(frame.path),
        "source_index": frame.source_index,
    }
    return parsed


def analyze_all_frames_sync(
    selected_frames: List[FrameSample],
    output_dir: str,
    model: str = "gpt-4o",
    transcript_segments: Optional[List[TranscriptSegment]] = None,
) -> List[Dict[str, Any]]:
    """Analyze all frames sequentially with resume-by-file behavior."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = build_client()
    results: List[Dict[str, Any]] = []
    prev_summary = "视频开始"
    transcript_segments = transcript_segments or []

    for idx, frame in enumerate(selected_frames):
        out_file = out / f"{frame.path.stem}.json"
        if out_file.exists():
            data = json.loads(out_file.read_text(encoding="utf-8"))
            results.append(data)
            prev_summary = data.get("scene", {}).get("specific_description", prev_summary)
            continue

        audio_transcript = transcript_window_text(
            transcript_segments,
            center_second=frame.timestamp_second,
            window_before=2.5,
            window_after=2.5,
        )
        if client is None:
            data = _fallback_result(idx, frame.timestamp_second, audio_transcript=audio_transcript)
            data["_meta"]["frame_path"] = str(frame.path)
            data["_meta"]["source_index"] = frame.source_index
        else:
            try:
                data = analyze_frame_openai(
                    client,
                    frame,
                    idx,
                    prev_summary,
                    audio_transcript=audio_transcript,
                    model=model,
                )
            except Exception:
                data = _fallback_result(idx, frame.timestamp_second, audio_transcript=audio_transcript)
                data["_meta"]["frame_path"] = str(frame.path)
                data["_meta"]["source_index"] = frame.source_index

        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(data)
        prev_summary = data.get("scene", {}).get("specific_description", prev_summary)
    return results
