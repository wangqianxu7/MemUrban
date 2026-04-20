"""Minimal local verification for memory_core vector retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from memory_core import SpatialTemporalBehaviorAgent


def build_event(event_id: int, location: str, action: str, transcript: str, emotion: str) -> dict:
    return {
        "event_id": event_id,
        "location_type": location,
        "functional_zone": "通行",
        "primary_action": action,
        "action_detail": f"{location}中的{action}",
        "spatial_features": {"greenery_level": 4 if "绿" in location else 2},
        "social_interaction": {"occurred": "打招呼" in transcript},
        "decision": {"was_decision_point": "岔路" in transcript},
        "audio_evidence": {"transcript": transcript},
        "affective_state": {
            "emotion": emotion,
            "intent": "前往下一个地点",
            "stress_level": 2,
            "energy_level": 3,
        },
    }


def main() -> None:
    agent = SpatialTemporalBehaviorAgent(perceptual_buffer_size=3)
    events = [
        build_event(0, "校园绿道", "行走中", "沿着林荫路继续往前走", "平静"),
        build_event(1, "教学楼门口", "停留", "在岔路口停一下再决定去哪边", "专注"),
        build_event(2, "食堂外通道", "转向", "和同学打招呼然后转向食堂", "愉悦"),
    ]
    agent.ingest_events(events)

    retrieval = agent.query_related_memories_with_scores("绿荫路径 决策点 社交", top_k=2)
    print(json.dumps({"retrieval": retrieval}, ensure_ascii=False, indent=2))

    print(json.dumps({"entities": agent.query_entity_memory("places")}, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "entity_search": agent.query_entity_memory_with_scores(
                    "绿道 通行 地点",
                    entity_type="places",
                    top_k=2,
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        vector_state = temp_root / "long_term_memory.json"
        export_state = temp_root / "memory_store.json"

        agent.save_long_term_memory(str(vector_state))
        agent.export_memory(str(export_state))

        restored = SpatialTemporalBehaviorAgent(perceptual_buffer_size=3)
        loaded_count = restored.load_long_term_memory(str(vector_state), merge=True)
        merged_again = restored.load_memory_export(str(export_state), merge=True)

        print(
            json.dumps(
                {
                    "loaded_count": loaded_count,
                    "merged_again": merged_again,
                    "restored_stats": {
                        "episodic_count": len(restored.long_term_memory.events),
                        "place_entity_count": len(restored.entity_memory.places),
                        "route_pattern_count": len(restored.entity_memory.route_patterns),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
