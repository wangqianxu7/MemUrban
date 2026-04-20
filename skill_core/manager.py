"""Skill registry and persistence layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import SkillCard, SkillFeedback, SkillMutationProposal


@dataclass
class SkillManager:
    """Manages draft skills, feedback logs, and mutation proposals."""

    skills: Dict[str, SkillCard] = field(default_factory=dict)
    feedback_log: List[SkillFeedback] = field(default_factory=list)
    proposals: List[SkillMutationProposal] = field(default_factory=list)

    def upsert_skill(self, skill: SkillCard) -> None:
        self.skills[skill.skill_id] = skill

    def add_feedback(self, feedback: SkillFeedback) -> None:
        self.feedback_log.append(feedback)

    def add_proposal(self, proposal: SkillMutationProposal) -> None:
        self.proposals.append(proposal)

    def get_skill(self, skill_id: str) -> Optional[SkillCard]:
        return self.skills.get(skill_id)

    def list_skills(self) -> List[SkillCard]:
        return list(self.skills.values())

    def list_proposals(self, status: Optional[str] = None) -> List[SkillMutationProposal]:
        if status is None:
            return list(self.proposals)
        return [proposal for proposal in self.proposals if proposal.status == status]

    def export_state(self) -> Dict[str, object]:
        return {
            "skills": {skill_id: skill.to_dict() for skill_id, skill in self.skills.items()},
            "feedback_log": [feedback.to_dict() for feedback in self.feedback_log],
            "proposals": [proposal.to_dict() for proposal in self.proposals],
        }

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_state(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SkillManager":
        target = Path(path)
        manager = cls()
        if not target.exists():
            return manager

        payload = json.loads(target.read_text(encoding="utf-8"))
        for skill_id, skill in payload.get("skills", {}).items():
            manager.skills[skill_id] = SkillCard(**skill)
        for feedback in payload.get("feedback_log", []):
            manager.feedback_log.append(SkillFeedback(**feedback))
        for proposal in payload.get("proposals", []):
            manager.proposals.append(SkillMutationProposal(**proposal))
        return manager
