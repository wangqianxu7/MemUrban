"""Minimal verification for placeholder skill evolution module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memurban.skills import SkillEvolutionEngine, SkillManager


def main() -> None:
    engine = SkillEvolutionEngine()
    engine.bootstrap_placeholder_skills()
    engine.ingest_memory_feedback(
        memory_id="memory-001",
        summary="长期记忆中的地点摘要过粗，缺少路径偏好刻画。",
        suggested_skill_id="memory_reflection",
        evidence={"entity_type": "places"},
    )
    engine.ingest_decision_feedback(
        decision_id="decision-001",
        summary="岔路口选择解释不足，未给出可审阅的备选策略。",
        suggested_skill_id="decision_repair",
        evidence={"decision_type": "path-selection"},
    )
    proposals = engine.generate_proposals()
    print(json.dumps({"proposal_count": len(proposals)}, ensure_ascii=False, indent=2))

    with TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "skill_registry.json"
        engine.manager.save(str(target))
        restored = SkillManager.load(str(target))
        print(
            json.dumps(
                {
                    "skills": [skill.skill_id for skill in restored.list_skills()],
                    "proposal_titles": [proposal.title for proposal in restored.list_proposals()],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
