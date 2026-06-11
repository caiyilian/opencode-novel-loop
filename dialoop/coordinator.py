from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

from .annotations import AnnotationRecord
from .risk import RiskAssessment


AGENT_NAMES = frozenset(
    {
        "labeler",
        "verifier",
        "identity_locator",
        "identity_resolver",
        "normalizer",
        "arbiter",
    }
)
AGENT_RESULT_VERDICTS = frozenset(
    {
        "accept",
        "reject",
        "uncertain",
        "resolved",
        "not_same_person",
        "not_enough_evidence",
    }
)
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})


class VerifierLike(Protocol):
    def verify(self, record: AnnotationRecord, risk: RiskAssessment) -> Any:
        ...


@dataclass(frozen=True)
class AgentSpec:
    agent: str
    prompt_constructor: str
    context_budget_tokens: int

    def __post_init__(self) -> None:
        if self.agent not in AGENT_NAMES:
            raise ValueError(f"unsupported coordinator agent: {self.agent}")
        if self.context_budget_tokens < 0:
            raise ValueError("context_budget_tokens must be 0 or greater")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "prompt_constructor": self.prompt_constructor,
            "context_budget_tokens": self.context_budget_tokens,
        }


DEFAULT_AGENT_SPECS: dict[str, AgentSpec] = {
    "verifier": AgentSpec(
        agent="verifier",
        prompt_constructor="dialoop.verifier.verifier_messages",
        context_budget_tokens=1200,
    ),
    "identity_locator": AgentSpec(
        agent="identity_locator",
        prompt_constructor="deterministic_tool:locate_identity",
        context_budget_tokens=0,
    ),
    "identity_resolver": AgentSpec(
        agent="identity_resolver",
        prompt_constructor="deterministic_tool:resolve_identity",
        context_budget_tokens=0,
    ),
    "normalizer": AgentSpec(
        agent="normalizer",
        prompt_constructor="deterministic_tool:normalize_speaker",
        context_budget_tokens=0,
    ),
    "arbiter": AgentSpec(
        agent="arbiter",
        prompt_constructor="dialoop.coordinator.StructuredArbiterAgent.arbitrate",
        context_budget_tokens=0,
    ),
}


@dataclass(frozen=True)
class AgentResult:
    agent: str
    verdict: str
    recommended_speaker: Optional[str] = None
    evidence_lines: Optional[list[int]] = None
    counter_evidence_lines: Optional[list[int]] = None
    reason: str = ""
    confidence: str = "medium"

    def __post_init__(self) -> None:
        if self.agent not in AGENT_NAMES:
            raise ValueError(f"unsupported coordinator agent: {self.agent}")
        if self.verdict not in AGENT_RESULT_VERDICTS:
            raise ValueError(f"unsupported agent verdict: {self.verdict}")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unsupported confidence: {self.confidence}")
        object.__setattr__(self, "evidence_lines", _line_numbers(self.evidence_lines))
        object.__setattr__(self, "counter_evidence_lines", _line_numbers(self.counter_evidence_lines))
        object.__setattr__(self, "recommended_speaker", _optional_string(self.recommended_speaker))
        object.__setattr__(self, "reason", self.reason.strip() if self.reason.strip() else "No reason recorded.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "verdict": self.verdict,
            "recommended_speaker": self.recommended_speaker,
            "evidence_lines": list(self.evidence_lines or []),
            "counter_evidence_lines": list(self.counter_evidence_lines or []),
            "reason": self.reason,
            "confidence": self.confidence,
        }


class ArbiterLike(Protocol):
    def arbitrate(
        self,
        *,
        labeler_result: AgentResult,
        verifier_result: AgentResult,
        verifier_review: dict[str, Any],
        record: AnnotationRecord,
        risk: RiskAssessment,
    ) -> dict[str, Any]:
        ...


class StructuredArbiterAgent:
    def arbitrate(
        self,
        *,
        labeler_result: AgentResult,
        verifier_result: AgentResult,
        verifier_review: dict[str, Any],
        record: AnnotationRecord,
        risk: RiskAssessment,
    ) -> dict[str, Any]:
        if verifier_result.verdict == "reject":
            return {
                "enabled": True,
                "decision": "reject_labeler",
                "verdict": "reject",
                "recommended_speaker": None,
                "evidence_lines": list(labeler_result.evidence_lines or []),
                "counter_evidence_lines": list(verifier_result.counter_evidence_lines or []),
                "reason": f"Verifier rejected the labeler speaker: {verifier_result.reason}",
                "confidence": verifier_result.confidence,
                "blocks_submission": True,
            }
        if verifier_result.verdict == "uncertain":
            return {
                "enabled": True,
                "decision": "needs_more_evidence",
                "verdict": "uncertain",
                "recommended_speaker": record.speaker,
                "evidence_lines": list(labeler_result.evidence_lines or []),
                "counter_evidence_lines": list(verifier_result.counter_evidence_lines or []),
                "reason": f"Verifier could not fully confirm the labeler evidence: {verifier_result.reason}",
                "confidence": "low",
                "blocks_submission": False,
            }
        fragile_reason = _fragile_high_risk_pass_reason(record, risk, verifier_result)
        if fragile_reason is not None:
            return {
                "enabled": True,
                "decision": "needs_more_evidence",
                "verdict": "uncertain",
                "recommended_speaker": record.speaker,
                "evidence_lines": list(labeler_result.evidence_lines or []),
                "counter_evidence_lines": [],
                "reason": f"High-risk verifier pass needs stronger disambiguation before writing: {fragile_reason}",
                "confidence": "low",
                "blocks_submission": True,
            }
        if risk.needs_verifier and verifier_result.verdict == "accept" and verifier_result.confidence == "low":
            return {
                "enabled": True,
                "decision": "needs_more_evidence",
                "verdict": "uncertain",
                "recommended_speaker": record.speaker,
                "evidence_lines": list(labeler_result.evidence_lines or []),
                "counter_evidence_lines": [],
                "reason": "High-risk verifier pass had low confidence; require stronger evidence before writing.",
                "confidence": "low",
                "blocks_submission": True,
            }
        return {
            "enabled": True,
            "decision": "accept_labeler",
            "verdict": "accept",
            "recommended_speaker": record.speaker,
            "evidence_lines": list(labeler_result.evidence_lines or []),
            "counter_evidence_lines": [],
            "reason": "Verifier did not conflict with the labeler speaker.",
            "confidence": verifier_review.get("confidence", "medium"),
            "blocks_submission": False,
        }


@dataclass(frozen=True)
class CoordinatorTraceEvent:
    step: int
    agent: str
    action: str
    reason: str
    result: Optional[AgentResult] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("trace step must be greater than 0")
        if self.agent not in AGENT_NAMES:
            raise ValueError(f"unsupported coordinator agent: {self.agent}")
        if not self.action.strip():
            raise ValueError("trace action is required")
        if not self.reason.strip():
            raise ValueError("trace reason is required")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step": self.step,
            "agent": self.agent,
            "action": self.action,
            "reason": self.reason,
        }
        if self.result is not None:
            data["result"] = self.result.to_dict()
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class CoordinatorDecision:
    risk: dict[str, Any]
    verifier: Optional[dict[str, Any]]
    arbiter: Optional[dict[str, Any]]
    trace: list[CoordinatorTraceEvent]

    @property
    def blocks_submission(self) -> bool:
        if isinstance(self.arbiter, dict) and self.arbiter.get("blocks_submission") is True:
            return True
        return (
            isinstance(self.verifier, dict)
            and self.verifier.get("enabled") is True
            and self.verifier.get("verdict") == "fail"
        )

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.trace]


class Coordinator:
    def __init__(
        self,
        verifier_agent: Optional[VerifierLike] = None,
        arbiter_agent: Optional[ArbiterLike] = None,
        verifier_mode: str = "off",
        verifier_context_budget: int = 1200,
        agent_specs: Optional[dict[str, AgentSpec]] = None,
    ):
        if verifier_mode not in {"off", "risk", "all"}:
            raise ValueError(f"unsupported verifier_mode: {verifier_mode}")
        self.verifier_agent = verifier_agent
        self.arbiter_agent = arbiter_agent or StructuredArbiterAgent()
        self.verifier_mode = verifier_mode
        self.agent_specs = dict(DEFAULT_AGENT_SPECS if agent_specs is None else agent_specs)
        self.agent_specs["verifier"] = replace(
            self.agent_specs["verifier"],
            context_budget_tokens=verifier_context_budget,
        )

    def review(self, record: AnnotationRecord, risk: RiskAssessment) -> CoordinatorDecision:
        trace: list[CoordinatorTraceEvent] = []
        labeler_result = self._append_labeler_result(trace, record)
        self._append_observed_tool_results(trace, record)
        verifier, verifier_result = self._maybe_call_verifier(trace, record, risk)
        arbiter = self._maybe_call_arbiter(trace, record, risk, labeler_result, verifier, verifier_result)
        return CoordinatorDecision(risk=risk.to_dict(), verifier=verifier, arbiter=arbiter, trace=trace)

    def _append_labeler_result(self, trace: list[CoordinatorTraceEvent], record: AnnotationRecord) -> AgentResult:
        result = AgentResult(
            agent="labeler",
            verdict="accept",
            recommended_speaker=record.speaker,
            evidence_lines=record.evidence_lines,
            reason=record.reason,
            confidence=record.confidence,
        )
        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="labeler",
                action="accepted",
                reason="Labeler submitted a speaker candidate for coordinator review.",
                result=result,
            )
        )
        return result

    def _append_observed_tool_results(self, trace: list[CoordinatorTraceEvent], record: AnnotationRecord) -> None:
        tool_summary = record.tool_summary if isinstance(record.tool_summary, dict) else {}
        observed_tools = (
            ("locate_identity", "identity_locator"),
            ("resolve_identity", "identity_resolver"),
            ("normalize_speaker", "normalizer"),
            ("arbitrate_identity", "arbiter"),
        )
        for tool_name, agent_name in observed_tools:
            calls = tool_summary.get(tool_name)
            if not isinstance(calls, list) or not calls:
                continue
            result = _agent_result_from_tool_summary(agent_name, calls[-1])
            trace.append(
                CoordinatorTraceEvent(
                    step=len(trace) + 1,
                    agent=agent_name,
                    action="observed",
                    reason=(
                        f"Labeler invoked {tool_name} before submit; Coordinator recorded the "
                        "sub-agent protocol result for audit."
                    ),
                    result=result,
                    metadata={
                        "tool_name": tool_name,
                        "tool_calls": len(calls),
                        "agent_spec": self.agent_specs[agent_name].to_dict(),
                    },
                )
            )

    def _maybe_call_verifier(
        self,
        trace: list[CoordinatorTraceEvent],
        record: AnnotationRecord,
        risk: RiskAssessment,
    ) -> tuple[Optional[dict[str, Any]], Optional[AgentResult]]:
        if self.verifier_agent is None or self.verifier_mode == "off":
            trace.append(
                CoordinatorTraceEvent(
                    step=len(trace) + 1,
                    agent="verifier",
                    action="skipped",
                    reason="Verifier is disabled for this run.",
                    metadata={"verifier_mode": self.verifier_mode},
                )
            )
            return None, None
        if self.verifier_mode == "risk" and not risk.needs_verifier:
            trace.append(
                CoordinatorTraceEvent(
                    step=len(trace) + 1,
                    agent="verifier",
                    action="skipped",
                    reason="Risk gate did not require verifier review.",
                    metadata={
                        "verifier_mode": self.verifier_mode,
                        "risk_level": risk.level,
                        "risk_signal_codes": _risk_signal_codes(risk),
                    },
                )
            )
            return None, None

        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="verifier",
                action="called",
                reason=_verifier_call_reason(self.verifier_mode, risk),
                metadata={
                    "verifier_mode": self.verifier_mode,
                    "risk_level": risk.level,
                    "risk_signal_codes": _risk_signal_codes(risk),
                    "agent_spec": self.agent_specs["verifier"].to_dict(),
                },
            )
        )
        review = _review_to_dict(self.verifier_agent.verify(record, risk), risk)
        result = _agent_result_from_verifier_review(review, record)
        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="verifier",
                action=_verifier_trace_action(result.verdict),
                reason=result.reason,
                result=result,
            )
        )
        return review, result

    def _maybe_call_arbiter(
        self,
        trace: list[CoordinatorTraceEvent],
        record: AnnotationRecord,
        risk: RiskAssessment,
        labeler_result: AgentResult,
        verifier: Optional[dict[str, Any]],
        verifier_result: Optional[AgentResult],
    ) -> Optional[dict[str, Any]]:
        if verifier is None or verifier_result is None:
            return None
        if not _needs_arbiter(verifier_result, risk, record):
            return None

        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="arbiter",
                action="called",
                reason=_arbiter_call_reason(verifier_result, risk, record),
                metadata={
                    "agent_spec": self.agent_specs["arbiter"].to_dict(),
                    "verifier_verdict": verifier.get("verdict"),
                    "verifier_confidence": verifier.get("confidence"),
                },
            )
        )
        decision = self.arbiter_agent.arbitrate(
            labeler_result=labeler_result,
            verifier_result=verifier_result,
            verifier_review=verifier,
            record=record,
            risk=risk,
        )
        result = _agent_result_from_arbiter_decision(decision)
        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="arbiter",
                action=_arbiter_trace_action(result.verdict),
                reason=result.reason,
                result=result,
            )
        )
        return decision


def _agent_result_from_verifier_review(review: dict[str, Any], record: AnnotationRecord) -> AgentResult:
    verdict = _optional_string(review.get("verdict")) or "error"
    mapped_verdict = {
        "pass": "accept",
        "fail": "reject",
        "uncertain": "uncertain",
        "error": "uncertain",
    }.get(verdict, "uncertain")
    confidence = _confidence(review.get("confidence"), default="low" if verdict == "error" else "medium")
    return AgentResult(
        agent="verifier",
        verdict=mapped_verdict,
        recommended_speaker=record.speaker if mapped_verdict == "accept" else None,
        evidence_lines=record.evidence_lines if mapped_verdict == "accept" else [],
        counter_evidence_lines=_line_numbers(review.get("counter_evidence_lines")),
        reason=_optional_string(review.get("reason")) or "Verifier did not provide a reason.",
        confidence=confidence,
    )


def _needs_arbiter(verifier_result: AgentResult, risk: RiskAssessment, record: AnnotationRecord) -> bool:
    if verifier_result.verdict in {"reject", "uncertain"}:
        return True
    if not risk.needs_verifier or verifier_result.verdict != "accept":
        return False
    if verifier_result.confidence == "low":
        return True
    return _fragile_high_risk_pass_reason(record, risk, verifier_result) is not None


def _arbiter_call_reason(verifier_result: AgentResult, risk: RiskAssessment, record: AnnotationRecord) -> str:
    if verifier_result.verdict == "accept":
        fragile_reason = _fragile_high_risk_pass_reason(record, risk, verifier_result)
        if fragile_reason is not None:
            return "High-risk Verifier pass lacks enough structured disambiguation for a short dialogue."
        return "High-risk Verifier pass has low confidence and needs structured arbitration."
    return "Verifier result conflicts with or cannot confirm the Labeler result."


def _fragile_high_risk_pass_reason(
    record: AnnotationRecord,
    risk: RiskAssessment,
    verifier_result: AgentResult,
) -> Optional[str]:
    if not risk.needs_verifier or verifier_result.verdict != "accept":
        return None
    signal_codes = set(_risk_signal_codes(risk))
    if "no_rejected_candidates_for_short_dialogue" in signal_codes and not record.rejected_candidates:
        return "short dialogue has no rejected speaker candidates"
    return None


def _agent_result_from_arbiter_decision(decision: dict[str, Any]) -> AgentResult:
    verdict = _optional_string(decision.get("verdict")) or "uncertain"
    if verdict not in {"accept", "reject", "uncertain"}:
        verdict = "uncertain"
    return AgentResult(
        agent="arbiter",
        verdict=verdict,
        recommended_speaker=_optional_string(decision.get("recommended_speaker")),
        evidence_lines=_line_numbers(decision.get("evidence_lines")),
        counter_evidence_lines=_line_numbers(decision.get("counter_evidence_lines")),
        reason=_optional_string(decision.get("reason")) or "Arbiter did not provide a reason.",
        confidence=_confidence(decision.get("confidence"), default="medium"),
    )


def _agent_result_from_tool_summary(agent_name: str, summary: Any) -> Optional[AgentResult]:
    if not isinstance(summary, dict):
        return None
    if agent_name == "identity_locator":
        candidate_count = _int_value(summary.get("candidate_count"), 0)
        return AgentResult(
            agent=agent_name,
            verdict="accept" if candidate_count > 0 else "not_enough_evidence",
            reason=f"Identity locator returned {candidate_count} candidate(s).",
            confidence="medium" if candidate_count > 0 else "low",
        )
    if agent_name == "identity_resolver":
        verdict = _optional_string(summary.get("verdict")) or "not_enough_evidence"
        if verdict not in {"resolved", "not_same_person", "not_enough_evidence"}:
            verdict = "not_enough_evidence"
        return AgentResult(
            agent=agent_name,
            verdict=verdict,
            recommended_speaker=_optional_string(summary.get("recommended_speaker")),
            reason=_optional_string(summary.get("reason")) or f"Identity resolver returned {verdict}.",
            confidence="medium" if verdict == "resolved" else "low",
        )
    if agent_name == "normalizer":
        suggested = _optional_string(summary.get("suggested_display_name"))
        return AgentResult(
            agent=agent_name,
            verdict="resolved" if suggested else "not_enough_evidence",
            recommended_speaker=suggested,
            reason=_optional_string(summary.get("reason")) or "Normalizer returned a display-name suggestion.",
            confidence="medium" if suggested else "low",
        )
    if agent_name == "arbiter":
        speaker = _optional_string(summary.get("recommended_speaker"))
        return AgentResult(
            agent=agent_name,
            verdict="resolved" if speaker else "uncertain",
            recommended_speaker=speaker,
            reason=_optional_string(summary.get("reason")) or "Arbiter returned a conflict decision.",
            confidence="medium" if speaker else "low",
        )
    return None


def _review_to_dict(review: Any, risk: RiskAssessment) -> dict[str, Any]:
    if hasattr(review, "to_dict"):
        data = review.to_dict()
        if isinstance(data, dict):
            return data
    if isinstance(review, dict):
        return dict(review)
    return {
        "enabled": True,
        "verdict": "error",
        "reason": "Verifier returned an unsupported review object.",
        "counter_evidence_lines": [],
        "risk_signal_codes": _risk_signal_codes(risk),
        "confidence": "low",
        "error": type(review).__name__,
    }


def _verifier_call_reason(verifier_mode: str, risk: RiskAssessment) -> str:
    if verifier_mode == "all":
        return "verifier_mode=all requires verifier review for every annotation."
    codes = _risk_signal_codes(risk)
    if codes:
        return f"High-risk annotation requires verifier review: {', '.join(codes)}."
    return "High-risk annotation requires verifier review."


def _verifier_trace_action(verdict: str) -> str:
    if verdict == "accept":
        return "accepted"
    if verdict == "reject":
        return "rejected"
    return "uncertain"


def _arbiter_trace_action(verdict: str) -> str:
    if verdict == "accept":
        return "accepted"
    if verdict == "reject":
        return "rejected"
    return "uncertain"


def _risk_signal_codes(risk: RiskAssessment) -> list[str]:
    return [signal.code for signal in risk.signals]


def _line_numbers(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    seen: set[int] = set()
    lines: list[int] = []
    for item in value:
        if type(item) is not int or item <= 0 or item in seen:
            continue
        seen.add(item)
        lines.append(item)
    return lines


def _optional_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _confidence(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in CONFIDENCE_LEVELS:
        return value.strip().lower()
    return default


def _int_value(value: Any, default: int) -> int:
    if type(value) is int:
        return value
    return default
