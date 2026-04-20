"""Placeholder self-evolution engine for future skill refinement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .manager import SkillManager
from .models import SkillCard, SkillFeedback, SkillMutationProposal


@dataclass
class SkillEvolutionEngine:
    """Generate draft skill mutation proposals from feedback signals."""

    manager: SkillManager = field(default_factory=SkillManager)

    def bootstrap_placeholder_skills(self) -> None:
        """Create a few draft placeholder skills if the registry is empty."""
        if self.manager.skills:
            return

        self.manager.upsert_skill(
            SkillCard(
                skill_id="memory_reflection",
                name="Memory Reflection",
                description="基于长期记忆反思结果，优化事件摘要、实体抽取和规则归纳。",
                trigger_domains=["memory"],
                capability_tags=["reflection", "summarization", "rule-extraction"],
                draft_instruction="分析 memory 输出中的薄弱点，为后续改进提供结构化建议。",
                source="bootstrap",
            )
        )
        self.manager.upsert_skill(
            SkillCard(
                skill_id="decision_repair",
                name="Decision Repair",
                description="根据 agent 决策失误或不确定性高的轨迹，提出决策策略修补建议。",
                trigger_domains=["agent", "decision"],
                capability_tags=["decision", "fallback", "policy"],
                draft_instruction="接收失败决策证据，生成可审阅的策略修补草案。",
                source="bootstrap",
            )
        )

    def ingest_memory_feedback(
        self,
        memory_id: str,
        summary: str,
        signal_type: str = "memory_gap",
        evidence: Optional[Dict[str, object]] = None,
        suggested_skill_id: Optional[str] = None,
    ) -> SkillFeedback:
        feedback = SkillFeedback(
            feedback_id=f"memory-{len(self.manager.feedback_log) + 1}",
            source_type="memory",
            source_id=memory_id,
            summary=summary,
            signal_type=signal_type,
            evidence=evidence or {},
            suggested_skill_id=suggested_skill_id,
        )
        self.manager.add_feedback(feedback)
        return feedback

    def ingest_decision_feedback(
        self,
        decision_id: str,
        summary: str,
        signal_type: str = "decision_gap",
        evidence: Optional[Dict[str, object]] = None,
        suggested_skill_id: Optional[str] = None,
    ) -> SkillFeedback:
        feedback = SkillFeedback(
            feedback_id=f"decision-{len(self.manager.feedback_log) + 1}",
            source_type="decision",
            source_id=decision_id,
            summary=summary,
            signal_type=signal_type,
            evidence=evidence or {},
            suggested_skill_id=suggested_skill_id,
        )
        self.manager.add_feedback(feedback)
        return feedback

    def generate_proposals(self) -> List[SkillMutationProposal]:
        """Generate draft proposals from feedback clusters."""
        proposals: List[SkillMutationProposal] = []
        grouped_feedback: Dict[str, List[SkillFeedback]] = {}
        for feedback in self.manager.feedback_log:
            cluster_key = feedback.suggested_skill_id or feedback.signal_type
            grouped_feedback.setdefault(cluster_key, []).append(feedback)

        for cluster_key, feedback_items in grouped_feedback.items():
            existing_skill = self.manager.get_skill(cluster_key)
            top_signal = Counter(item.signal_type for item in feedback_items).most_common(1)[0][0]
            proposal = SkillMutationProposal(
                proposal_id=f"proposal-{len(self.manager.proposals) + len(proposals) + 1}",
                proposal_type="refine" if existing_skill else "create",
                target_skill_id=existing_skill.skill_id if existing_skill else None,
                title=f"{'完善' if existing_skill else '新增'}技能: {existing_skill.name if existing_skill else cluster_key}",
                rationale="；".join(item.summary for item in feedback_items[:3]),
                change_summary=self._build_change_summary(existing_skill, top_signal, feedback_items),
                expected_impact="提高记忆反思质量或降低决策失误率。",
                backing_feedback_ids=[item.feedback_id for item in feedback_items],
                metadata={"signal_type": top_signal, "feedback_count": len(feedback_items)},
            )
            proposals.append(proposal)

        for proposal in proposals:
            self.manager.add_proposal(proposal)
        return proposals

    def _build_change_summary(
        self,
        existing_skill: Optional[SkillCard],
        signal_type: str,
        feedback_items: List[SkillFeedback],
    ) -> str:
        if existing_skill is not None:
            return (
                f"针对 {existing_skill.name} 补充 {signal_type} 的处理规则，"
                f"并吸收 {len(feedback_items)} 条反馈中的共性模式。"
            )
        return (
            f"新增一个围绕 {signal_type} 的占位 skill，"
            f"先记录触发域、证据格式和预期修复策略。"
        )
