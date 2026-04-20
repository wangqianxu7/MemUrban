"""Structured memory stores used by the agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .vector_index import InMemoryVectorIndex, SearchHit


@dataclass
class ShortTermMemoryStore:
    """Recent event buffer for local reasoning."""

    capacity: int = 8
    buffer: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        self.buffer = deque(self.buffer, maxlen=self.capacity)

    def push(self, event: Dict[str, Any]) -> None:
        self.buffer.append(event)

    def recent_events(self) -> List[Dict[str, Any]]:
        return list(self.buffer)


@dataclass
class LongTermMemoryStore:
    """Persistent episodic memory with vector retrieval."""

    vector_index: InMemoryVectorIndex = field(default_factory=InMemoryVectorIndex)
    events: List[Dict[str, Any]] = field(default_factory=list)
    event_summaries: List[str] = field(default_factory=list)
    metadata_records: List[Dict[str, Any]] = field(default_factory=list)
    seen_event_keys: set[str] = field(default_factory=set)

    def add_event(
        self,
        event: Dict[str, Any],
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> bool:
        event_key = self._event_key(event, metadata)
        if deduplicate and event_key in self.seen_event_keys:
            return False

        metadata = metadata or {}
        self.events.append(event)
        self.event_summaries.append(summary)
        self.vector_index.add(text=summary, item=event, metadata=metadata)
        self.metadata_records.append(metadata)
        self.seen_event_keys.add(event_key)
        return True

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        return self.vector_index.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

    def export_state(self) -> Dict[str, Any]:
        return {
            "episodic_events": self.events,
            "event_summaries": self.event_summaries,
            "metadata_records": self.metadata_records,
            "vector_index": self.vector_index.export_state(),
        }

    def save_to_disk(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_from_disk(self, path: str, merge: bool = True) -> int:
        target = Path(path)
        if not target.exists():
            return 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        return self.load_state(payload, merge=merge)

    def load_state(self, payload: Dict[str, Any], merge: bool = True) -> int:
        if not merge:
            self.vector_index = InMemoryVectorIndex(embedder=self.vector_index.embedder)
            self.events = []
            self.event_summaries = []
            self.metadata_records = []
            self.seen_event_keys = set()

        count = 0
        for idx, event in enumerate(payload.get("episodic_events", [])):
            summaries = payload.get("event_summaries", [])
            metadata_records = payload.get("metadata_records", [])
            summary = summaries[idx] if idx < len(summaries) else ""
            metadata = metadata_records[idx] if idx < len(metadata_records) else None
            added = self.add_event(event=event, summary=summary, metadata=metadata, deduplicate=merge)
            if added:
                count += 1
        return count

    def _event_key(self, event: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> str:
        event_id = event.get("event_id")
        location = event.get("location_type", "")
        action = event.get("primary_action", "")
        start_second = event.get("start_second", "")
        meta_event_id = (metadata or {}).get("event_id", "")
        return f"{event_id}|{meta_event_id}|{start_second}|{location}|{action}"


@dataclass
class SemanticMemoryStore:
    """Stable abstractions reflected from long-term events."""

    rules: List[str] = field(default_factory=list)
    concepts: Dict[str, Any] = field(default_factory=dict)
