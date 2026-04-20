"""Frame-level results -> event-level spatial-temporal behavior records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np


def _most_common(values: List[Any]) -> Any:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def finalize_event(raw_event: Dict[str, Any]) -> Dict[str, Any]:
    """Build memory-ready event schema from grouped frames."""
    frames = raw_event["frames"]
    scene_list = [f.get("scene", {}) for f in frames]
    social_frames = [f for f in frames if f.get("social", {}).get("interaction_occurring")]
    decision_frames = [f for f in frames if f.get("movement", {}).get("is_decision_point")]
    avg_comfort = float(np.mean([scene.get("comfort_rating", 3) for scene in scene_list]))

    return {
        "event_id": raw_event["event_id"],
        "start_second": raw_event["start_second"],
        "end_second": raw_event["end_second"],
        "duration_seconds": raw_event["end_second"] - raw_event["start_second"],
        "location_type": raw_event["location_type"],
        "functional_zone": raw_event["functional_zone"],
        "spatial_features": {
            "greenery_level": float(np.mean([scene.get("greenery_level", 3) for scene in scene_list])),
            "shade_coverage": _most_common([scene.get("shade_coverage") for scene in scene_list]),
            "crowd_density": _most_common([scene.get("crowd_density") for scene in scene_list]),
            "facilities": sorted({x for scene in scene_list for x in scene.get("notable_facilities", [])}),
        },
        "primary_action": _most_common([f.get("movement", {}).get("walking_state") for f in frames]),
        "action_detail": frames[len(frames) // 2].get("scene", {}).get("specific_description", ""),
        "social_interaction": {
            "occurred": len(social_frames) > 0,
            "type": social_frames[0].get("social", {}).get("interaction_type") if social_frames else None,
            "description": social_frames[0].get("social", {}).get("interaction_description") if social_frames else None,
        },
        "decision": {
            "was_decision_point": len(decision_frames) > 0,
            "description": decision_frames[0].get("movement", {}).get("decision_description") if decision_frames else None,
        },
        "environment_rating": {
            "comfort": round(avg_comfort, 2),
            "greenery": round(float(np.mean([scene.get("greenery_level", 3) for scene in scene_list])), 2),
            "crowd_density": _most_common([scene.get("crowd_density") for scene in scene_list]),
        },
        "audio_evidence": {
            "transcript": " ".join(
                f.get("audio", {}).get("aligned_transcript", "").strip()
                for f in frames
                if f.get("audio", {}).get("aligned_transcript")
            ).strip(),
            "alignment_confidence": round(
                float(
                    np.mean(
                        [f.get("audio", {}).get("alignment_confidence", 0.0) for f in frames]
                    )
                ),
                2,
            ),
        },
        "affective_state": {
            "emotion": _most_common([f.get("psychological", {}).get("inferred_emotion") for f in frames]),
            "intent": _most_common([f.get("psychological", {}).get("inferred_intent") for f in frames]),
            "stress_level": round(
                float(np.mean([f.get("psychological", {}).get("stress_level", 3) for f in frames])),
                2,
            ),
            "energy_level": round(
                float(np.mean([f.get("psychological", {}).get("energy_level", 3) for f in frames])),
                2,
            ),
        },
        "frame_count": len(frames),
    }


def merge_frames_to_events(frame_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split on scene/movement/social/decision changes and produce events."""
    events: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for f in frame_results:
        if "_error" in f:
            continue
        scene = f.get("scene", {})
        move = f.get("movement", {})
        meta = f.get("_meta", {})
        sec = float(meta.get("video_second", 0.0))

        should_split = False
        if current is None:
            should_split = True
        else:
            if scene.get("location_type") != current["location_type"]:
                should_split = True
            elif move.get("walking_state") != current["last_walking_state"]:
                should_split = True
            elif f.get("social", {}).get("interaction_occurring"):
                should_split = True
            elif move.get("is_decision_point"):
                should_split = True

        if should_split:
            if current is not None:
                current["end_second"] = sec
                events.append(finalize_event(current))
            current = {
                "event_id": len(events),
                "start_second": sec,
                "end_second": sec,
                "location_type": scene.get("location_type"),
                "functional_zone": scene.get("functional_zone"),
                "last_walking_state": move.get("walking_state"),
                "frames": [f],
            }
        else:
            current["frames"].append(f)
            current["last_walking_state"] = move.get("walking_state")
            current["end_second"] = sec

    if current is not None:
        events.append(finalize_event(current))
    return events
