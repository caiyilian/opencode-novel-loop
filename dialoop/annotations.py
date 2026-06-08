from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


VALID_CONFIDENCE = {"high", "medium", "low"}
DEFAULT_REASON = "No structured reason supplied; fallback annotation recorded from accepted submit_labels."


@dataclass(frozen=True)
class AnnotationRecord:
    index: int
    line_number: int
    text: str
    speaker: str
    evidence_lines: list[int]
    reason: str
    rejected_candidates: list[str]
    confidence: str
    tool_summary: dict[str, Any]
    recovery: Optional[dict[str, Any]] = None
    risk: Optional[dict[str, Any]] = None
    verifier: Optional[dict[str, Any]] = None
    arbiter: Optional[dict[str, Any]] = None
    coordinator_trace: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line_number": self.line_number,
            "text": self.text,
            "speaker": self.speaker,
            "evidence_lines": self.evidence_lines,
            "reason": self.reason,
            "rejected_candidates": self.rejected_candidates,
            "confidence": self.confidence,
            "tool_summary": self.tool_summary,
            "recovery": self.recovery,
            "risk": self.risk,
            "verifier": self.verifier,
            "arbiter": self.arbiter,
            "coordinator_trace": self.coordinator_trace,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def with_review(
        self,
        risk: Optional[dict[str, Any]],
        verifier: Optional[dict[str, Any]] = None,
        arbiter: Optional[dict[str, Any]] = None,
        coordinator_trace: Optional[list[dict[str, Any]]] = None,
    ) -> "AnnotationRecord":
        return AnnotationRecord(
            index=self.index,
            line_number=self.line_number,
            text=self.text,
            speaker=self.speaker,
            evidence_lines=list(self.evidence_lines),
            reason=self.reason,
            rejected_candidates=list(self.rejected_candidates),
            confidence=self.confidence,
            tool_summary=self.tool_summary,
            recovery=self.recovery,
            risk=risk,
            verifier=verifier,
            arbiter=arbiter,
            coordinator_trace=_copy_trace(coordinator_trace),
        )


class AnnotationStore:
    def __init__(self, annotations_path: Path):
        self.annotations_path = annotations_path.expanduser().resolve()

    def append(self, records: list[AnnotationRecord]) -> int:
        if not records:
            return 0

        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.annotations_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(record.to_json_line())
                file.write("\n")
        return len(records)


def build_annotation_records(
    dialogues: list[dict[str, Any]],
    speakers: list[str],
    submit_args: dict[str, Any],
    tool_summary: dict[str, Any],
    recovery: Optional[dict[str, Any]] = None,
) -> list[AnnotationRecord]:
    count = min(len(dialogues), len(speakers))
    evidence_lines = _evidence_lines_by_dialogue(submit_args, count)
    reasons = _strings_by_dialogue(
        submit_args,
        singular_key="reason",
        plural_key="reasons",
        count=count,
        default=DEFAULT_REASON,
    )
    rejected_candidates = _string_lists_by_dialogue(
        submit_args,
        singular_key="rejected_candidates",
        plural_key="rejected_candidates_by_dialogue",
        count=count,
    )
    confidence = _confidence_by_dialogue(submit_args, count)

    records: list[AnnotationRecord] = []
    for index in range(count):
        dialogue = dialogues[index]
        line_number = _int_value(dialogue.get("line_number"), 0)
        record_evidence_lines = evidence_lines[index] or [line_number]
        records.append(
            AnnotationRecord(
                index=_int_value(dialogue.get("index"), index),
                line_number=line_number,
                text=str(dialogue.get("text") or ""),
                speaker=speakers[index],
                evidence_lines=record_evidence_lines,
                reason=reasons[index],
                rejected_candidates=rejected_candidates[index],
                confidence=confidence[index],
                tool_summary=tool_summary,
                recovery=recovery,
            )
        )
    return records


def _evidence_lines_by_dialogue(args: dict[str, Any], count: int) -> list[list[int]]:
    per_dialogue = args.get("evidence_lines_by_dialogue")
    if isinstance(per_dialogue, list):
        values = [_line_numbers(item) for item in per_dialogue[:count]]
        return values + [[] for _ in range(count - len(values))]

    shared = _line_numbers(args.get("evidence_lines"))
    return [list(shared) for _ in range(count)]


def _strings_by_dialogue(
    args: dict[str, Any],
    singular_key: str,
    plural_key: str,
    count: int,
    default: str,
) -> list[str]:
    plural = args.get(plural_key)
    if isinstance(plural, list):
        values = [_string_or_default(item, default) for item in plural[:count]]
        return values + [default for _ in range(count - len(values))]

    shared = _string_or_default(args.get(singular_key), default)
    return [shared for _ in range(count)]


def _string_lists_by_dialogue(
    args: dict[str, Any],
    singular_key: str,
    plural_key: str,
    count: int,
) -> list[list[str]]:
    plural = args.get(plural_key)
    if isinstance(plural, list):
        values = [_string_list(item) for item in plural[:count]]
        return values + [[] for _ in range(count - len(values))]

    shared = _string_list(args.get(singular_key))
    return [list(shared) for _ in range(count)]


def _confidence_by_dialogue(args: dict[str, Any], count: int) -> list[str]:
    plural = args.get("confidences")
    if isinstance(plural, list):
        values = [_confidence(item) for item in plural[:count]]
        return values + ["low" for _ in range(count - len(values))]

    shared = _confidence(args.get("confidence"))
    return [shared for _ in range(count)]


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _confidence(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in VALID_CONFIDENCE:
        return value.strip().lower()
    return "low"


def _int_value(value: Any, default: int) -> int:
    if type(value) is int:
        return value
    return default


def _copy_trace(value: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    return [dict(item) for item in value if isinstance(item, dict)]
