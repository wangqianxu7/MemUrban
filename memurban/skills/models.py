"""Data models for skill definitions and evolution proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillCard:
    """Serializable placeholder definition of a future executable skill."""

    skill_id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    trigger_domains: List[str] = field(default_factory=list)
    capability_tags: List[str] = field(default_factory=list)
    draft_instruction: str = ""
    status: str = "draft"
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillFeedback:
    """Feedback event from memory or agent decision traces."""

    feedback_id: str
    source_type: str
    source_id: str
    summary: str
    signal_type: str
    priority: str = "medium"
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_skill_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillMutationProposal:
    """Mutation or creation proposal generated from accumulated feedback."""

    proposal_id: str
    proposal_type: str
    target_skill_id: Optional[str]
    title: str
    rationale: str
    change_summary: str
    expected_impact: str = ""
    status: str = "proposed"
    backing_feedback_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
