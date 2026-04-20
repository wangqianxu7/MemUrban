"""Agent orchestration across short-term, long-term, semantic, and persona memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .entities import EntityMemoryStore
from .persona import PersonaInferer, PersonaProfile
from .stores import LongTermMemoryStore, SemanticMemoryStore, ShortTermMemoryStore


@dataclass
class SpatialTemporalBehaviorAgent:
    """High-level agent API kept compatible with the previous single-file version."""

    perceptual_buffer_size: int = 8
    short_term_memory: ShortTermMemoryStore = field(init=False)
    long_term_memory: LongTermMemoryStore = field(default_factory=LongTermMemoryStore)
    semantic_memory: SemanticMemoryStore = field(default_factory=SemanticMemoryStore)
    entity_memory: EntityMemoryStore = field(default_factory=EntityMemoryStore)
    persona_profile: PersonaProfile = field(default_factory=PersonaProfile)
    persona_inferer: PersonaInferer = field(default_factory=PersonaInferer)

    def __post_init__(self) -> None:
        self.short_term_memory = ShortTermMemoryStore(capacity=self.perceptual_buffer_size)

    def ingest_events(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            self.short_term_memory.push(event)
            added = self.long_term_memory.add_event(
                event=event,
                summary=self._build_event_summary(event),
                metadata=self._build_event_metadata(event),
            )
            if added:
                self.entity_memory.ingest_event(event)
        self._reflect_semantic_memory()
        self.persona_profile = self.persona_inferer.infer(self.long_term_memory.events)

    def _build_event_summary(self, event: Dict[str, Any]) -> str:
        tags: List[str] = []
        if event.get("social_interaction", {}).get("occurred"):
            tags.append("社交互动")
        if event.get("decision", {}).get("was_decision_point"):
            tags.append("决策点")
        audio_text = event.get("audio_evidence", {}).get("transcript", "")
        if audio_text:
            tags.append("音频对齐")

        return (
            f"{event.get('location_type', '')} "
            f"{event.get('functional_zone', '')} "
            f"{event.get('primary_action', '')} "
            f"{event.get('action_detail', '')} "
            f"{' '.join(tags)} "
            f"{event.get('audio_evidence', {}).get('transcript', '')} "
            f"{event.get('affective_state', {}).get('emotion', '')} "
            f"{event.get('affective_state', {}).get('intent', '')}"
        ).strip()

    def _build_event_metadata(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "location_type": event.get("location_type"),
            "functional_zone": event.get("functional_zone"),
            "primary_action": event.get("primary_action"),
            "has_social_interaction": bool(event.get("social_interaction", {}).get("occurred")),
            "has_decision_point": bool(event.get("decision", {}).get("was_decision_point")),
        }

    def _reflect_semantic_memory(self) -> None:
        events = self.long_term_memory.events
        if not events:
            return

        greenery_levels = [event.get("spatial_features", {}).get("greenery_level", 3) for event in events]
        decision_events = [event for event in events if event.get("decision", {}).get("was_decision_point")]
        social_events = [event for event in events if event.get("social_interaction", {}).get("occurred")]
        transcript_events = [event for event in events if event.get("audio_evidence", {}).get("transcript")]
        avg_greenery = float(np.mean(greenery_levels))

        rules: List[str] = []
        if avg_greenery >= 3.5:
            rules.append("个体持续偏向绿化程度较高的通行环境。")
        if len(decision_events) >= max(2, len(events) // 4):
            rules.append("个体在空间岔路口存在稳定的路径选择行为。")
        if len(social_events) >= max(1, len(events) // 5):
            rules.append("个体的社交接触主要出现在过渡空间和高人流区域。")
        if len(transcript_events) >= max(1, len(events) // 3):
            rules.append("音频线索能够稳定补充视觉帧中的意图与互动判断。")
        if not rules:
            rules.append("整体行为以稳定通行为主，异常事件较少。")

        self.semantic_memory.rules = rules
        self.semantic_memory.concepts = {
            "avg_greenery": round(avg_greenery, 2),
            "decision_event_ratio": round(len(decision_events) / len(events), 2),
            "social_event_ratio": round(len(social_events) / len(events), 2),
            "audio_supported_ratio": round(len(transcript_events) / len(events), 2),
        }

    def query_related_memories(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        hits = self.long_term_memory.query(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )
        return [hit.item for hit in hits]

    def query_related_memories_with_scores(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        hits = self.long_term_memory.query(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )
        return [
            {
                "score": round(hit.score, 4),
                "metadata": hit.metadata,
                "event": hit.item,
            }
            for hit in hits
        ]

    def query_entity_memory(self, entity_type: str, name: Optional[str] = None) -> Dict[str, Any]:
        buckets = {
            "places": self.entity_memory.places,
            "people": self.entity_memory.people,
            "route_patterns": self.entity_memory.route_patterns,
        }
        bucket = buckets.get(entity_type, {})
        if name is None:
            return bucket
        return bucket.get(name, {})

    def query_entity_memory_with_scores(
        self,
        query: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        hits = self.entity_memory.search(
            query=query,
            top_k=top_k,
            entity_type=entity_type,
            min_score=min_score,
        )
        return [
            {
                "score": round(hit.score, 4),
                "metadata": hit.metadata,
                "entity": hit.item,
            }
            for hit in hits
        ]

    def save_long_term_memory(self, path: str) -> None:
        self.long_term_memory.save_to_disk(path)

    def load_long_term_memory(self, path: str, merge: bool = True) -> int:
        loaded = self.long_term_memory.load_from_disk(path, merge=merge)
        self._rebuild_derived_memory()
        return loaded

    def load_memory_export(self, path: str, merge: bool = True) -> int:
        target = Path(path)
        if not target.exists():
            return 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        loaded = self.long_term_memory.load_state(payload.get("long_term_memory", {}), merge=merge)
        if not merge:
            self.short_term_memory = ShortTermMemoryStore(capacity=self.perceptual_buffer_size)
        recent_events = payload.get("short_term_memory", {}).get("recent_events", [])
        for event in recent_events[-self.perceptual_buffer_size :]:
            self.short_term_memory.push(event)
        self._rebuild_derived_memory()
        return loaded

    def _rebuild_derived_memory(self) -> None:
        self.entity_memory = EntityMemoryStore()
        for event in self.long_term_memory.events:
            self.entity_memory.ingest_event(event)
        self.entity_memory.rebuild_index()
        self._reflect_semantic_memory()
        self.persona_profile = self.persona_inferer.infer(self.long_term_memory.events)

    def export_memory(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "short_term_memory": {
                "capacity": self.short_term_memory.capacity,
                "recent_events": self.short_term_memory.recent_events(),
            },
            "long_term_memory": {
                **self.long_term_memory.export_state(),
            },
            "semantic_memory": asdict(self.semantic_memory),
            "entity_memory": self.entity_memory.export_state(),
            "persona_profile": asdict(self.persona_profile),
            "stats": {
                "episodic_count": len(self.long_term_memory.events),
                "semantic_rule_count": len(self.semantic_memory.rules),
                "recent_window_size": len(self.short_term_memory.recent_events()),
                "vector_index_size": len(self.long_term_memory.vector_index),
                "place_entity_count": len(self.entity_memory.places),
                "people_entity_count": len(self.entity_memory.people),
                "route_pattern_count": len(self.entity_memory.route_patterns),
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
