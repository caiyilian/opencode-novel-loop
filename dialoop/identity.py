from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


VALID_CONFIDENCE = {"high", "medium", "low"}
DEFAULT_LOOKAHEAD_LINES = 120
DEFAULT_LOOKAHEAD_ROUNDS = 2
IDENTITY_MARKERS = (
    "\u6211\u53eb",
    "\u6211\u53eb\u505a",
    "\u6211\u53eb\u4f5c",
    "\u5c0f\u7684\u540d\u53eb",
    "\u540d\u53eb",
    "\u540d\u5b57\u53eb",
    "\u540d\u5b57\u662f",
    "\u6211\u662f",
)
NAME_CHARS = r"[\u4e00-\u9fffA-Za-z0-9\u00b7\u2022\u30fb\uff0e\.]{1,20}"
NAME_PATTERNS = (
    re.compile(
        r"(?:"
        r"\u6211\u53eb\u505a|\u6211\u53eb\u4f5c|\u6211\u53eb|"
        r"\u5c0f\u7684\u540d\u53eb|\u5c0f\u7684\u53eb\u505a|\u5c0f\u7684\u53eb\u4f5c|\u5c0f\u7684\u53eb|"
        r"\u5728\u4e0b\u540d\u53eb|\u5728\u4e0b\u53eb|\u672c\u4eba\u540d\u53eb|\u672c\u4eba\u53eb|"
        r"\u4ed6\u53eb|\u5979\u53eb|\u540d\u53eb|\u540d\u5b57\u53eb|\u540d\u5b57\u662f"
        r")("
        + NAME_CHARS
        + r")"
    ),
    re.compile(r"(?:\u6211\u662f|\u5c0f\u7684\u662f|\u5728\u4e0b\u662f|\u672c\u4eba\u662f)(" + NAME_CHARS + ")"),
)
NAME_TRAILING_PUNCTUATION = set(".,!?;: \t\r\n")
NAME_TRAILING_PUNCTUATION.update("\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u201c\u201d\u2018\u2019")
NON_PERSON_NAME_FRAGMENTS = (
    "\u57ce\u9547",
    "\u57ce\u5e02",
    "\u6751\u843d",
    "\u6751\u5b50",
    "\u5730\u65b9",
    "\u6559\u4f1a",
    "\u4fee\u9053\u9662",
    "\u5546\u884c",
)
PERSON_ROLE_PREFIXES = (
    "\u65c5\u884c\u5546\u4eba",
    "\u521a\u5165\u884c\u7684\u65c5\u884c\u5546\u4eba",
    "\u65b0\u624b\u65c5\u884c\u5546\u4eba",
    "\u5546\u4eba",
    "\u884c\u5546",
    "\u9886\u4e3b",
    "\u9a91\u58eb",
    "\u8001\u677f",
)
RANGE_CONTINUITY_TERMS = {
    "\u5c11\u5973",
    "\u5973\u5b69",
    "\u59d1\u5a18",
    "\u5c11\u5e74",
    "\u7537\u5b69",
    "\u5b69\u5b50",
    "\u5c0f\u5b69",
    "\u8001\u4eba",
    "\u8001\u8005",
    "\u7537\u4eba",
    "\u7537\u5b50",
    "\u5973\u4eba",
    "\u5973\u5b50",
    "\u9752\u5e74",
    "\u5e74\u8f7b\u4eba",
    "\u5c0f\u7684",
    "\u54b1",
}
STRICT_MATCH_IDENTITY_TERMS = {
    "\u7537\u5b69",
    "\u5b69\u5b50",
    "\u5c0f\u5b69",
    "\u7537\u4eba",
    "\u7537\u5b50",
    "\u5973\u4eba",
    "\u5973\u5b50",
    "\u9752\u5e74",
    "\u5e74\u8f7b\u4eba",
}


class IdentityValidationError(ValueError):
    """Raised when an identity helper receives invalid input."""


@dataclass(frozen=True)
class IdentityCandidate:
    start_line: int
    end_line: int
    matched_line: int
    reason: str
    suggested_names: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "matched_line": self.matched_line,
            "reason": self.reason,
            "suggested_names": list(self.suggested_names),
            "text": self.text,
        }


@dataclass(frozen=True)
class CharacterRecord:
    id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    evidence_lines: list[int] = field(default_factory=list)
    last_seen_dialogue_index: Optional[int] = None
    last_seen_line_number: Optional[int] = None
    confidence: str = "medium"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "summary": self.summary,
            "evidence_lines": list(self.evidence_lines),
            "last_seen_dialogue_index": self.last_seen_dialogue_index,
            "last_seen_line_number": self.last_seen_line_number,
            "confidence": self.confidence,
        }


class CharacterLibrary:
    def __init__(self) -> None:
        self._records: list[CharacterRecord] = []

    @property
    def records(self) -> list[CharacterRecord]:
        return list(self._records)

    def to_list(self) -> list[dict]:
        return [record.to_dict() for record in self._records]

    def add_or_update(
        self,
        *,
        display_name: str,
        aliases: Optional[list[str]] = None,
        summary: str = "",
        evidence_lines: Optional[list[int]] = None,
        last_seen_dialogue_index: Optional[int] = None,
        last_seen_line_number: Optional[int] = None,
        confidence: str = "medium",
    ) -> CharacterRecord:
        cleaned_display_name = _required_name(display_name, "display_name")
        cleaned_aliases = _unique_strings([*(aliases or []), cleaned_display_name])
        cleaned_evidence_lines = _unique_positive_ints(evidence_lines or [])
        cleaned_confidence = _clean_confidence(confidence)
        existing = self.find(cleaned_display_name)
        if existing is None:
            record = CharacterRecord(
                id=f"char_{len(self._records) + 1:04d}",
                display_name=cleaned_display_name,
                aliases=[alias for alias in cleaned_aliases if alias != cleaned_display_name],
                summary=summary.strip(),
                evidence_lines=cleaned_evidence_lines,
                last_seen_dialogue_index=last_seen_dialogue_index,
                last_seen_line_number=last_seen_line_number,
                confidence=cleaned_confidence,
            )
            self._records.append(record)
            return record

        merged_aliases = _unique_strings([*existing.aliases, *cleaned_aliases])
        merged_evidence_lines = _unique_positive_ints([*existing.evidence_lines, *cleaned_evidence_lines])
        updated = CharacterRecord(
            id=existing.id,
            display_name=existing.display_name,
            aliases=[alias for alias in merged_aliases if alias != existing.display_name],
            summary=summary.strip() or existing.summary,
            evidence_lines=merged_evidence_lines,
            last_seen_dialogue_index=last_seen_dialogue_index
            if last_seen_dialogue_index is not None
            else existing.last_seen_dialogue_index,
            last_seen_line_number=last_seen_line_number
            if last_seen_line_number is not None
            else existing.last_seen_line_number,
            confidence=_highest_confidence(existing.confidence, cleaned_confidence),
        )
        self._records = [updated if record.id == existing.id else record for record in self._records]
        return updated

    def find(self, speaker: str) -> Optional[CharacterRecord]:
        cleaned = speaker.strip()
        if not cleaned:
            return None
        for record in self._records:
            if cleaned == record.display_name or cleaned in record.aliases:
                return record
        return None

    def normalize(self, speaker: str) -> dict:
        cleaned = _required_name(speaker, "speaker")
        record = self.find(cleaned)
        if record is None:
            return {
                "speaker": cleaned,
                "suggested_display_name": None,
                "matched": False,
                "confidence": "low",
                "reason": "no character library entry matched this speaker",
                "record": None,
            }
        if cleaned == record.display_name:
            reason = "speaker already uses the library display_name"
        else:
            reason = "speaker matched a recorded alias; normalizer suggests display_name"
        return {
            "speaker": cleaned,
            "suggested_display_name": record.display_name,
            "matched": cleaned != record.display_name,
            "confidence": record.confidence,
            "reason": reason,
            "record": record.to_dict(),
        }


class IdentityLocatorAgent:
    def __init__(self, dialogue_index: Any) -> None:
        self.dialogue_index = dialogue_index

    def locate(
        self,
        *,
        speaker: str,
        current_line: int,
        search_after_line: Optional[int] = None,
        lookahead_lines: int = DEFAULT_LOOKAHEAD_LINES,
        max_candidates: int = 3,
    ) -> dict:
        cleaned_speaker = _required_name(speaker, "speaker")
        if current_line <= 0:
            raise IdentityValidationError("current_line must be positive")
        if lookahead_lines <= 0:
            raise IdentityValidationError("lookahead_lines must be greater than 0")
        if max_candidates <= 0:
            raise IdentityValidationError("max_candidates must be greater than 0")

        start_line = max(current_line + 1, search_after_line or current_line + 1)
        end_line = min(len(self.dialogue_index.lines), start_line + lookahead_lines - 1)
        candidates: list[IdentityCandidate] = []
        for line_number in range(start_line, end_line + 1):
            line = self.dialogue_index.lines[line_number - 1]
            names = extract_stable_names(line)
            if not names:
                continue
            if cleaned_speaker in STRICT_MATCH_IDENTITY_TERMS and cleaned_speaker not in line:
                continue
            reasons: list[str] = []
            if cleaned_speaker in line:
                reasons.append("matched speaker term")
            reasons.append("matched identity marker")
            candidates.append(
                IdentityCandidate(
                    start_line=max(1, line_number - 2),
                    end_line=min(len(self.dialogue_index.lines), line_number + 2),
                    matched_line=line_number,
                    reason=", ".join(reasons),
                    suggested_names=names,
                    text=line,
                )
            )
            if len(candidates) >= max_candidates:
                break

        return {
            "speaker": cleaned_speaker,
            "current_line": current_line,
            "search_start_line": start_line,
            "search_end_line": end_line,
            "lookahead_lines": lookahead_lines,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "suggest_continue": bool(candidates) and candidates[-1].matched_line < end_line,
        }


class IdentityResolverAgent:
    def __init__(self, dialogue_index: Any) -> None:
        self.dialogue_index = dialogue_index

    def resolve(
        self,
        *,
        speaker: str,
        start_line: int,
        end_line: int,
        current_dialogue: Optional[Any] = None,
    ) -> dict:
        cleaned_speaker = _required_name(speaker, "speaker")
        if start_line <= 0 or end_line <= 0:
            raise IdentityValidationError("line numbers are 1-based and must be positive")
        if end_line < start_line:
            raise IdentityValidationError("end_line must be greater than or equal to start_line")

        bounded_start = min(max(1, start_line), len(self.dialogue_index.lines) + 1)
        bounded_end = min(len(self.dialogue_index.lines), end_line)
        if bounded_start > bounded_end:
            return {
                "speaker": cleaned_speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "candidate range is outside the novel",
            }

        candidate_lines = [
            (line_number, self.dialogue_index.lines[line_number - 1])
            for line_number in range(bounded_start, bounded_end + 1)
        ]
        if cleaned_speaker in RANGE_CONTINUITY_TERMS and not any(
            cleaned_speaker in line for _, line in candidate_lines
        ):
            return {
                "speaker": cleaned_speaker,
                "verdict": "not_same_person",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "candidate range has an identity marker but does not mention the temporary speaker term",
            }

        names_by_line: list[tuple[int, str]] = []
        for line_number, line in candidate_lines:
            for name in extract_stable_names(line):
                names_by_line.append((line_number, name))
        if not names_by_line:
            return {
                "speaker": cleaned_speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "no stable name marker was found in the candidate range",
            }

        counts = Counter(name for _, name in names_by_line)
        recommended = counts.most_common(1)[0][0]
        evidence_lines = _unique_positive_ints([line for line, name in names_by_line if name == recommended])
        current = current_dialogue.to_dict() if current_dialogue is not None else None
        return {
            "speaker": cleaned_speaker,
            "verdict": "resolved",
            "recommended_speaker": recommended,
            "evidence_lines": evidence_lines,
            "reason": "stable name marker found in bounded lookahead range",
            "current_dialogue": current,
        }


class IdentityArbiterAgent:
    def arbitrate(
        self,
        *,
        labeler_speaker: str,
        verifier_verdict: Optional[str] = None,
        resolver_verdict: Optional[str] = None,
        resolver_speaker: Optional[str] = None,
        normalizer_speaker: Optional[str] = None,
    ) -> dict:
        cleaned_labeler = _required_name(labeler_speaker, "labeler_speaker")
        cleaned_verifier = _optional_clean(verifier_verdict)
        cleaned_resolver_verdict = _optional_clean(resolver_verdict)
        cleaned_resolver_speaker = _optional_clean(resolver_speaker)
        cleaned_normalizer_speaker = _optional_clean(normalizer_speaker)

        if cleaned_verifier == "fail":
            return {
                "decision": "needs_manual_review",
                "recommended_speaker": None,
                "reason": "verifier rejected the labeler speaker",
            }
        if cleaned_resolver_verdict == "resolved" and cleaned_resolver_speaker:
            if cleaned_resolver_speaker != cleaned_labeler:
                return {
                    "decision": "use_resolved_identity",
                    "recommended_speaker": cleaned_resolver_speaker,
                    "reason": "resolver found a stable evidence-backed identity different from labeler speaker",
                }
            return {
                "decision": "accept_labeler",
                "recommended_speaker": cleaned_labeler,
                "reason": "resolver confirmed the labeler speaker",
            }
        if cleaned_normalizer_speaker and cleaned_normalizer_speaker != cleaned_labeler:
            return {
                "decision": "use_normalized_display_name",
                "recommended_speaker": cleaned_normalizer_speaker,
                "reason": "normalizer matched an existing character display_name",
            }
        if cleaned_verifier == "uncertain" or cleaned_resolver_verdict == "not_enough_evidence":
            return {
                "decision": "needs_manual_review",
                "recommended_speaker": None,
                "reason": "available agents did not provide enough evidence",
            }
        return {
            "decision": "accept_labeler",
            "recommended_speaker": cleaned_labeler,
            "reason": "no conflicting identity evidence",
        }


def extract_stable_names(text: str) -> list[str]:
    names: list[str] = []
    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(text):
            name = _clean_name(match.group(1))
            if name:
                names.append(name)
    return _unique_strings(names)


def _clean_name(name: str) -> str:
    cleaned = name.strip().strip("".join(NAME_TRAILING_PUNCTUATION))
    if not cleaned:
        return ""
    if any(fragment in cleaned for fragment in NON_PERSON_NAME_FRAGMENTS):
        return ""
    for prefix in PERSON_ROLE_PREFIXES:
        if cleaned == prefix:
            return ""
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
            return cleaned[len(prefix) :].strip().strip("".join(NAME_TRAILING_PUNCTUATION))
    return cleaned


def _required_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def _unique_positive_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if type(value) is not int or value <= 0 or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clean_confidence(value: str) -> str:
    cleaned = value.strip().lower() if isinstance(value, str) else ""
    if cleaned not in VALID_CONFIDENCE:
        raise IdentityValidationError("confidence must be high, medium, or low")
    return cleaned


def _highest_confidence(first: str, second: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return first if order[first] >= order[second] else second


def _optional_clean(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
