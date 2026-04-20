"""Entity memory extraction and aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .vector_index import InMemoryVectorIndex, SearchHit


@dataclass
class EntityMemoryStore:
    """Stores normalized place, interaction, and route-pattern entities."""

    places: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    people: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    route_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vector_index: InMemoryVectorIndex = field(default_factory=InMemoryVectorIndex)

    def ingest_event(self, event: Dict[str, Any]) -> None:
        self._ingest_place(event)
        self._ingest_people(event)
        self._ingest_route_pattern(event)
        self.rebuild_index()

    def _ingest_place(self, event: Dict[str, Any]) -> None:
        location = event.get("location_type") or "未知地点"
        place = self.places.setdefault(
            location,
            {
                "name": location,
                "visit_count": 0,
                "functional_zones": Counter(),
                "dominant_actions": Counter(),
                "greenery_levels": [],
                "event_ids": [],
            },
        )
        place["visit_count"] += 1
        place["functional_zones"][event.get("functional_zone", "未知")] += 1
        place["dominant_actions"][event.get("primary_action", "未知")] += 1
        place["greenery_levels"].append(event.get("spatial_features", {}).get("greenery_level", 3))
        place["event_ids"].append(event.get("event_id"))

    def _ingest_people(self, event: Dict[str, Any]) -> None:
        social = event.get("social_interaction", {})
        if not social.get("occurred"):
            return
        role = social.get("type") or "未识别互动对象"
        person = self.people.setdefault(
            role,
            {
                "label": role,
                "interaction_count": 0,
                "descriptions": Counter(),
                "locations": Counter(),
                "event_ids": [],
            },
        )
        person["interaction_count"] += 1
        if social.get("description"):
            person["descriptions"][social["description"]] += 1
        person["locations"][event.get("location_type", "未知地点")] += 1
        person["event_ids"].append(event.get("event_id"))

    def _ingest_route_pattern(self, event: Dict[str, Any]) -> None:
        decision = event.get("decision", {})
        key = "稳定通行"
        if decision.get("was_decision_point"):
            key = "决策路径"
        elif event.get("primary_action") in {"转向", "行走中"}:
            key = "移动路径"

        pattern = self.route_patterns.setdefault(
            key,
            {
                "name": key,
                "count": 0,
                "locations": Counter(),
                "actions": Counter(),
                "decision_descriptions": Counter(),
                "event_ids": [],
            },
        )
        pattern["count"] += 1
        pattern["locations"][event.get("location_type", "未知地点")] += 1
        pattern["actions"][event.get("primary_action", "未知")] += 1
        if decision.get("description"):
            pattern["decision_descriptions"][decision["description"]] += 1
        pattern["event_ids"].append(event.get("event_id"))

    def export_state(self) -> Dict[str, Any]:
        return {
            "places": self._serialize_bucket(self.places),
            "people": self._serialize_bucket(self.people),
            "route_patterns": self._serialize_bucket(self.route_patterns),
            "vector_index": self.vector_index.export_state(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        self.places = self._deserialize_bucket(payload.get("places", {}), greenery=True)
        self.people = self._deserialize_bucket(payload.get("people", {}))
        self.route_patterns = self._deserialize_bucket(payload.get("route_patterns", {}))
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self.vector_index = InMemoryVectorIndex(embedder=self.vector_index.embedder)
        for entity_type, bucket in (
            ("places", self.places),
            ("people", self.people),
            ("route_patterns", self.route_patterns),
        ):
            for name, entity in bucket.items():
                self.vector_index.add(
                    text=self._entity_summary(entity_type, name, entity),
                    item=entity,
                    metadata={"entity_type": entity_type, "name": name},
                )

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchHit]:
        metadata_filter = {"entity_type": entity_type} if entity_type else None
        return self.vector_index.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

    def _entity_summary(self, entity_type: str, name: str, entity: Dict[str, Any]) -> str:
        if entity_type == "places":
            return (
                f"地点 {name} "
                f"访问 {entity.get('visit_count', 0)} 次 "
                f"功能区 {' '.join(entity.get('functional_zones', {}).keys())} "
                f"动作 {' '.join(entity.get('dominant_actions', {}).keys())}"
            )
        if entity_type == "people":
            return (
                f"互动对象 {name} "
                f"互动 {entity.get('interaction_count', 0)} 次 "
                f"位置 {' '.join(entity.get('locations', {}).keys())}"
            )
        return (
            f"路线模式 {name} "
            f"出现 {entity.get('count', 0)} 次 "
            f"位置 {' '.join(entity.get('locations', {}).keys())} "
            f"动作 {' '.join(entity.get('actions', {}).keys())}"
        )

    def _serialize_bucket(self, bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        serialized: Dict[str, Dict[str, Any]] = {}
        for key, value in bucket.items():
            serialized[key] = {
                inner_key: dict(inner_value) if isinstance(inner_value, Counter) else inner_value
                for inner_key, inner_value in value.items()
            }
        return serialized

    def _deserialize_bucket(self, bucket: Dict[str, Dict[str, Any]], greenery: bool = False) -> Dict[str, Dict[str, Any]]:
        restored: Dict[str, Dict[str, Any]] = {}
        for key, value in bucket.items():
            restored_value: Dict[str, Any] = {}
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, dict):
                    restored_value[inner_key] = Counter(inner_value)
                else:
                    restored_value[inner_key] = inner_value
            if greenery and "greenery_levels" not in restored_value:
                restored_value["greenery_levels"] = []
            restored[key] = restored_value
        return restored
