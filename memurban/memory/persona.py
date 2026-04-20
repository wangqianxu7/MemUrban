"""Persona inference from episodic memory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


def _top_counter(values: List[str], default: str = "未知") -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return default
    return Counter(cleaned).most_common(1)[0][0]


@dataclass
class PersonaProfile:
    """Long-lived style profile inferred from event history."""

    mobility_style: str = "未知"
    social_style: str = "未知"
    environmental_preference: str = "未知"
    decision_style: str = "未知"
    speech_style: str = "未知"
    stress_baseline: float = 3.0
    energy_baseline: float = 3.0
    dominant_emotion: str = "未知"
    preferred_locations: List[str] = field(default_factory=list)


class PersonaInferer:
    """Infer persona-level traits from event sequences."""

    def infer(self, events: List[Dict[str, Any]]) -> PersonaProfile:
        if not events:
            return PersonaProfile()

        actions = [event.get("primary_action", "") for event in events]
        emotions = [event.get("affective_state", {}).get("emotion", "") for event in events]
        locations = [event.get("location_type", "") for event in events]
        transcripts = [event.get("audio_evidence", {}).get("transcript", "") for event in events]
        greenery = [event.get("spatial_features", {}).get("greenery_level", 3) for event in events]
        stress = [event.get("affective_state", {}).get("stress_level", 3) for event in events]
        energy = [event.get("affective_state", {}).get("energy_level", 3) for event in events]
        decision_ratio = sum(1 for event in events if event.get("decision", {}).get("was_decision_point")) / len(events)
        social_ratio = sum(1 for event in events if event.get("social_interaction", {}).get("occurred")) / len(events)

        return PersonaProfile(
            mobility_style=self._classify_mobility_style(actions),
            social_style="社交开放" if social_ratio >= 0.3 else "低频社交",
            environmental_preference="偏好绿化环境" if float(np.mean(greenery)) >= 3.5 else "环境偏好不显著",
            decision_style="探索式决策" if decision_ratio >= 0.3 else "惯性路径决策",
            speech_style=self._classify_speech_style(transcripts),
            stress_baseline=round(float(np.mean(stress)), 2),
            energy_baseline=round(float(np.mean(energy)), 2),
            dominant_emotion=_top_counter(emotions),
            preferred_locations=[loc for loc, _ in Counter(locations).most_common(3)],
        )

    def _classify_mobility_style(self, actions: List[str]) -> str:
        dominant_action = _top_counter(actions)
        if dominant_action in {"行走中", "转向"}:
            return "持续步行型"
        if dominant_action in {"停留", "静止"}:
            return "停留观察型"
        return "混合移动型"

    def _classify_speech_style(self, transcripts: List[str]) -> str:
        total_text = " ".join(text for text in transcripts if text).strip()
        if not total_text:
            return "少语音线索"
        if any(token in total_text for token in ["你好", "谢谢", "再见"]):
            return "礼貌互动型"
        if len(total_text) >= 60:
            return "持续语言表达型"
        return "低密度表达型"
