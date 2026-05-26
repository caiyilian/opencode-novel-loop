from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .annotations import AnnotationRecord
from .model_client import ChatMessage
from .risk import RiskAssessment


VERIFIER_VERDICTS = {"pass", "fail", "uncertain", "error"}


@dataclass(frozen=True)
class VerifierReview:
    enabled: bool
    verdict: str
    reason: str
    counter_evidence_lines: list[int]
    risk_signal_codes: list[str]
    raw: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.verdict not in VERIFIER_VERDICTS:
            raise ValueError(f"unsupported verifier verdict: {self.verdict}")

    @property
    def blocks_submission(self) -> bool:
        return self.enabled and self.verdict == "fail"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": self.enabled,
            "verdict": self.verdict,
            "reason": self.reason,
            "counter_evidence_lines": self.counter_evidence_lines,
            "risk_signal_codes": self.risk_signal_codes,
        }
        if self.raw is not None:
            data["raw"] = self.raw
        if self.error is not None:
            data["error"] = self.error
        return data


class VerifierAgent:
    def __init__(
        self,
        model_client: Any,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.model_client = model_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def verify(self, record: AnnotationRecord, risk: RiskAssessment) -> VerifierReview:
        messages = verifier_messages(record, risk)
        try:
            response = self.model_client.chat(
                messages=messages,
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as error:  # noqa: BLE001 - verifier errors must not break unattended runs.
            return VerifierReview(
                enabled=True,
                verdict="error",
                reason="Verifier model call failed.",
                counter_evidence_lines=[],
                risk_signal_codes=_risk_signal_codes(risk),
                error=str(error),
            )

        raw = response.content
        try:
            payload = json.loads(_extract_json_object(raw))
        except (json.JSONDecodeError, ValueError) as error:
            return VerifierReview(
                enabled=True,
                verdict="error",
                reason="Verifier returned invalid JSON.",
                counter_evidence_lines=[],
                risk_signal_codes=_risk_signal_codes(risk),
                raw=raw,
                error=str(error),
            )

        return review_from_payload(payload, risk, raw=raw)


def verifier_messages(record: AnnotationRecord, risk: RiskAssessment) -> list[ChatMessage]:
    payload = {
        "task": "Verify whether the submitted speaker is supported by the labeler's evidence.",
        "rules": [
            "Do not relabel the dialogue from scratch.",
            "Look only for counter-evidence, turn-order conflicts, addressee/speaker confusion, or insufficient evidence.",
            "Return verdict=fail only when the submitted speaker is not supported.",
            "Return verdict=uncertain when evidence is weak but no clear counter-evidence is found.",
        ],
        "annotation": {
            "index": record.index,
            "line_number": record.line_number,
            "text": record.text,
            "speaker": record.speaker,
            "evidence_lines": record.evidence_lines,
            "reason": record.reason,
            "rejected_candidates": record.rejected_candidates,
            "confidence": record.confidence,
        },
        "risk": risk.to_dict(),
        "required_json_shape": {
            "verdict": "pass | fail | uncertain",
            "reason": "short explanation",
            "counter_evidence_lines": [1],
        },
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the Dialoop Verifier Agent. Your only job is to check whether "
                "the Labeler evidence supports the submitted speaker. Return exactly one JSON object."
            ),
        ),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def review_from_payload(payload: Any, risk: RiskAssessment, raw: Optional[str] = None) -> VerifierReview:
    if not isinstance(payload, dict):
        return VerifierReview(
            enabled=True,
            verdict="error",
            reason="Verifier JSON payload was not an object.",
            counter_evidence_lines=[],
            risk_signal_codes=_risk_signal_codes(risk),
            raw=raw,
        )

    verdict = _verdict(payload.get("verdict"))
    return VerifierReview(
        enabled=True,
        verdict=verdict,
        reason=_string_value(payload.get("reason"), default="Verifier did not provide a reason."),
        counter_evidence_lines=_line_numbers(payload.get("counter_evidence_lines")),
        risk_signal_codes=_risk_signal_codes(risk),
        raw=raw,
    )


def _verdict(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in {"pass", "fail", "uncertain"}:
        return value.strip().lower()
    return "error"


def _string_value(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


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


def _risk_signal_codes(risk: RiskAssessment) -> list[str]:
    return [signal.code for signal in risk.signals]


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("model output does not contain a JSON object")
    return stripped[start : end + 1]
