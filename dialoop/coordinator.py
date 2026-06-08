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
        prompt_constructor="deterministic_tool:arbitrate_identity",
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
    trace: list[CoordinatorTraceEvent]

    @property
    def blocks_submission(self) -> bool:
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
        verifier_mode: str = "off",
        verifier_context_budget: int = 1200,
        agent_specs: Optional[dict[str, AgentSpec]] = None,
    ):
        if verifier_mode not in {"off", "risk", "all"}:
            raise ValueError(f"unsupported verifier_mode: {verifier_mode}")
        self.verifier_agent = verifier_agent
        self.verifier_mode = verifier_mode
        self.agent_specs = dict(DEFAULT_AGENT_SPECS if agent_specs is None else agent_specs)
        self.agent_specs["verifier"] = replace(
            self.agent_specs["verifier"],
            context_budget_tokens=verifier_context_budget,
        )

    def review(self, record: AnnotationRecord, risk: RiskAssessment) -> CoordinatorDecision:
        trace: list[CoordinatorTraceEvent] = []
        self._append_labeler_result(trace, record)
        self._append_observed_tool_results(trace, record)
        verifier = self._maybe_call_verifier(trace, record, risk)
        return CoordinatorDecision(risk=risk.to_dict(), verifier=verifier, trace=trace)

    def _append_labeler_result(self, trace: list[CoordinatorTraceEvent], record: AnnotationRecord) -> None:
        trace.append(
            CoordinatorTraceEvent(
                step=len(trace) + 1,
                agent="labeler",
                action="accepted",
                reason="Labeler submitted a speaker candidate for coordinator review.",
                result=AgentResult(
                    agent="labeler",
                    verdict="accept",
                    recommended_speaker=record.speaker,
                    evidence_lines=record.evidence_lines,
                    reason=record.reason,
                    confidence=record.confidence,
                ),
            )
        )

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
    ) -> Optional[dict[str, Any]]:
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
            return None
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
            return None

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
        return review


def _agent_result_from_verifier_review(review: dict[str, Any], record: AnnotationRecord) -> AgentResult:
    verdict = _optional_string(review.get("verdict")) or "error"
    mapped_verdict = {
        "pass": "accept",
        "fail": "reject",
        "uncertain": "uncertain",
        "error": "uncertain",
    }.get(verdict, "uncertain")
    confidence = "low" if verdict == "error" else "medium"
    return AgentResult(
        agent="verifier",
        verdict=mapped_verdict,
        recommended_speaker=record.speaker if mapped_verdict == "accept" else None,
        evidence_lines=record.evidence_lines if mapped_verdict == "accept" else [],
        counter_evidence_lines=_line_numbers(review.get("counter_evidence_lines")),
        reason=_optional_string(review.get("reason")) or "Verifier did not provide a reason.",
        confidence=confidence,
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


def _int_value(value: Any, default: int) -> int:
    if type(value) is int:
        return value
    return default
