from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .annotations import AnnotationRecord, DEFAULT_REASON


RISK_LEVELS = ("low", "medium", "high")
PUNCTUATION_CHARS = set(" \t\r\n.,!?;:'\"()[]{}<>，。！？；：、“”‘’（）【】《》…—-~")
SECOND_PERSON_MARKERS = ("你", "您")


@dataclass(frozen=True)
class RiskSignal:
    code: str
    message: str
    level: str

    def __post_init__(self) -> None:
        if self.level not in RISK_LEVELS:
            raise ValueError(f"unsupported risk level: {self.level}")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "level": self.level,
        }


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    signals: list[RiskSignal]

    def __post_init__(self) -> None:
        if self.level not in RISK_LEVELS:
            raise ValueError(f"unsupported risk level: {self.level}")

    @property
    def needs_verifier(self) -> bool:
        return self.level == "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "needs_verifier": self.needs_verifier,
            "signals": [signal.to_dict() for signal in self.signals],
        }


def assess_annotation_risk(record: AnnotationRecord) -> RiskAssessment:
    signals: list[RiskSignal] = []
    semantic_length = _semantic_length(record.text)
    stripped_text = record.text.strip()

    if record.confidence == "low":
        signals.append(
            RiskSignal(
                code="low_confidence",
                message="Labeler reported low confidence or omitted confidence.",
                level="high",
            )
        )
    elif record.confidence == "medium":
        signals.append(
            RiskSignal(
                code="medium_confidence",
                message="Labeler reported medium confidence.",
                level="medium",
            )
        )

    if record.reason == DEFAULT_REASON:
        signals.append(
            RiskSignal(
                code="fallback_annotation_metadata",
                message="Labeler omitted structured reason/evidence metadata.",
                level="high",
            )
        )

    if not record.evidence_lines:
        signals.append(
            RiskSignal(
                code="missing_evidence_lines",
                message="No evidence line numbers were provided.",
                level="high",
            )
        )

    if semantic_length <= 2:
        signals.append(
            RiskSignal(
                code="very_short_dialogue",
                message="Dialogue is very short and likely depends on turn context.",
                level="high",
            )
        )
    elif semantic_length <= 4:
        signals.append(
            RiskSignal(
                code="short_dialogue",
                message="Dialogue is short and may be ambiguous without neighboring turns.",
                level="medium",
            )
        )

    if _looks_like_silence_or_fragment(stripped_text):
        signals.append(
            RiskSignal(
                code="silence_or_fragment",
                message="Dialogue looks like silence, ellipsis, dash, or an incomplete fragment.",
                level="high",
            )
        )

    if any(marker in stripped_text for marker in SECOND_PERSON_MARKERS):
        signals.append(
            RiskSignal(
                code="second_person_address",
                message="Dialogue contains a generic second-person marker; addressee/speaker confusion is possible.",
                level="high",
            )
        )

    if ("?" in stripped_text or "？" in stripped_text) and semantic_length <= 8:
        signals.append(
            RiskSignal(
                code="short_question",
                message="Short questions often depend on adjacent turn order.",
                level="medium",
            )
        )

    if not record.rejected_candidates and semantic_length <= 4:
        signals.append(
            RiskSignal(
                code="no_rejected_candidates_for_short_dialogue",
                message="No rejected speaker candidates were recorded for a short dialogue.",
                level="medium",
            )
        )

    return RiskAssessment(level=_highest_level(signals), signals=signals)


def _highest_level(signals: list[RiskSignal]) -> str:
    if any(signal.level == "high" for signal in signals):
        return "high"
    if any(signal.level == "medium" for signal in signals):
        return "medium"
    return "low"


def _semantic_length(text: str) -> int:
    return sum(1 for char in text if char not in PUNCTUATION_CHARS)


def _looks_like_silence_or_fragment(text: str) -> bool:
    if not text:
        return True
    semantic_length = _semantic_length(text)
    if semantic_length == 0:
        return True
    return "..." in text or "……" in text or "..." in text.replace("…", ".")
