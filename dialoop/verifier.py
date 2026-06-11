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
    confidence: str = "medium"
    raw: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.verdict not in VERIFIER_VERDICTS:
            raise ValueError(f"unsupported verifier verdict: {self.verdict}")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError(f"unsupported verifier confidence: {self.confidence}")

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
            "confidence": self.confidence,
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
        max_tokens: int = 1200,
        retries: int = 1,
    ):
        self.model_client = model_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = max(0, retries)

    def verify(self, record: AnnotationRecord, risk: RiskAssessment) -> VerifierReview:
        messages = verifier_messages(record, risk)
        last_raw: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(self.retries + 1):
            try:
                response = self.model_client.chat(
                    messages=messages,
                    tools=None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as error:  # noqa: BLE001 - verifier errors must not break unattended runs.
                last_error = str(error)
                if attempt < self.retries:
                    continue
                return _error_review(
                    risk=risk,
                    reason="Verifier model call failed.",
                    raw=last_raw,
                    error=last_error,
                )

            last_raw = response.content
            try:
                payload = json.loads(_extract_json_object(last_raw))
            except (json.JSONDecodeError, ValueError) as error:
                last_error = str(error)
                if attempt < self.retries:
                    messages = verifier_retry_messages(last_raw, last_error)
                    continue
                return _error_review(
                    risk=risk,
                    reason="Verifier returned invalid JSON.",
                    raw=last_raw,
                    error=last_error,
                )

            review = review_from_payload(payload, risk, raw=last_raw)
            if review.verdict != "error" or attempt >= self.retries:
                return review
            last_error = review.error or "verifier JSON did not include a valid verdict"
            messages = verifier_retry_messages(last_raw, last_error)

        return _error_review(
            risk=risk,
            reason="Verifier returned invalid JSON.",
            raw=last_raw,
            error=last_error,
        )


def verifier_messages(record: AnnotationRecord, risk: RiskAssessment) -> list[ChatMessage]:
    payload = {
        "task": "Verify whether the submitted speaker is supported by the labeler's evidence.",
        "rules": [
            "Do not relabel the dialogue from scratch.",
            "Look only for counter-evidence, turn-order conflicts, addressee/speaker confusion, or insufficient evidence.",
            "Return verdict=fail only when the submitted speaker is not supported.",
            "Return verdict=pass only when the evidence directly supports the submitted speaker and no plausible counter-evidence remains.",
            "Return verdict=uncertain when evidence is weak but no clear counter-evidence is found.",
            "For second-person dialogue, verify who is speaking and who is being addressed; do not pass from pronouns alone.",
            "For very short dialogue with no rejected candidates, return uncertain unless at least one plausible alternative speaker was compared.",
            "Turn order, speaking style, or conversational flow alone is not enough for a high-confidence pass on high-risk samples.",
            "Use confidence=high only when the verifier conclusion is strongly supported by explicit evidence.",
            "Keep reason short: at most 80 Chinese characters or one short English sentence.",
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
            "confidence": "high | medium | low",
        },
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the Dialoop Verifier Agent. Your only job is to check whether "
                "the Labeler evidence supports the submitted speaker. Return exactly one compact JSON object. "
                "Do not use Markdown fences."
            ),
        ),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def verifier_retry_messages(raw: Optional[str], error: str) -> list[ChatMessage]:
    previous = raw if raw is not None and raw.strip() else "<empty response>"
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the Dialoop Verifier Agent. Repair your previous verifier response. "
                "Return exactly one compact JSON object and nothing else."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "The previous verifier response was not usable JSON.\n"
                f"Error: {error}\n"
                f"Previous response: {previous[:600]}\n"
                'Return only this shape: {"verdict":"pass|fail|uncertain","reason":"short",'
                '"counter_evidence_lines":[],"confidence":"high|medium|low"}.'
            ),
        ),
    ]


def review_from_payload(payload: Any, risk: RiskAssessment, raw: Optional[str] = None) -> VerifierReview:
    if not isinstance(payload, dict):
        return VerifierReview(
            enabled=True,
            verdict="error",
            reason="Verifier JSON payload was not an object.",
            counter_evidence_lines=[],
            risk_signal_codes=_risk_signal_codes(risk),
            confidence="low",
            raw=raw,
        )

    verdict = _verdict(payload.get("verdict"))
    return VerifierReview(
        enabled=True,
        verdict=verdict,
        reason=_string_value(payload.get("reason"), default="Verifier did not provide a reason."),
        counter_evidence_lines=_line_numbers(payload.get("counter_evidence_lines")),
        risk_signal_codes=_risk_signal_codes(risk),
        confidence=_confidence(payload.get("confidence")),
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


def _confidence(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in {"high", "medium", "low"}:
        return value.strip().lower()
    return "low"


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


def _error_review(
    risk: RiskAssessment,
    reason: str,
    raw: Optional[str] = None,
    error: Optional[str] = None,
) -> VerifierReview:
    return VerifierReview(
        enabled=True,
        verdict="error",
        reason=reason,
        counter_evidence_lines=[],
        risk_signal_codes=_risk_signal_codes(risk),
        confidence="low",
        raw=raw,
        error=error,
    )


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
