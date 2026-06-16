from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from .model_client import ChatMessage


VALID_CONFIDENCE = {"high", "medium", "low"}
DEFAULT_LOOKAHEAD_LINES = 120
DEFAULT_LOOKAHEAD_ROUNDS = 2
DEFAULT_IDENTITY_MODEL_MAX_TOKENS = 1400
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
COORDINATOR_IDENTITY_SPEAKERS = {
    "\u5c11\u5973",
    "\u5973\u5b69",
    "\u59d1\u5a18",
    "\u5c11\u5e74",
    "\u8001\u4eba",
    "\u8001\u8005",
    "\u5c0f\u7684",
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


class IdentityPipelineAgent:
    def __init__(
        self,
        dialogue_index: Any,
        *,
        model_client: Optional[Any] = None,
        lookahead_lines: int = DEFAULT_LOOKAHEAD_LINES,
        lookahead_rounds: int = DEFAULT_LOOKAHEAD_ROUNDS,
        max_candidates: int = 3,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_IDENTITY_MODEL_MAX_TOKENS,
        retries: int = 1,
    ) -> None:
        if lookahead_lines <= 0:
            raise IdentityValidationError("lookahead_lines must be greater than 0")
        if lookahead_rounds < 0:
            raise IdentityValidationError("lookahead_rounds must be 0 or greater")
        if max_candidates <= 0:
            raise IdentityValidationError("max_candidates must be greater than 0")
        self.dialogue_index = dialogue_index
        self.model_client = model_client
        self.lookahead_lines = lookahead_lines
        self.lookahead_rounds = lookahead_rounds
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = max(0, retries)
        self.locator = IdentityLocatorAgent(dialogue_index)
        self.resolver = IdentityResolverAgent(dialogue_index)

    def review(self, record: Any) -> dict[str, Any]:
        if self.model_client is not None:
            return self._model_review(record)
        return self._deterministic_review(record)

    def _deterministic_review(self, record: Any) -> dict[str, Any]:
        speaker = _required_name(record.speaker, "speaker")
        if not should_coordinate_identity_lookup(speaker):
            return {
                "enabled": True,
                "triggered": False,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "speaker is not a coordinator-managed temporary identity",
                "confidence": "low",
                "same_person": None,
                "locator_attempts": [],
                "candidate_ranges": [],
                "resolver": None,
            }
        if self.lookahead_rounds == 0:
            return {
                "enabled": True,
                "triggered": True,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "identity lookahead round limit is zero",
                "confidence": "low",
                "same_person": None,
                "locator_attempts": [],
                "candidate_ranges": [],
                "resolver": None,
            }

        locator_attempts: list[dict[str, Any]] = []
        candidate_ranges: list[dict[str, Any]] = []
        search_after_line: Optional[int] = None
        for round_number in range(1, self.lookahead_rounds + 1):
            located = self.locator.locate(
                speaker=speaker,
                current_line=record.line_number,
                search_after_line=search_after_line,
                lookahead_lines=self.lookahead_lines,
                max_candidates=self.max_candidates,
            )
            located["round"] = round_number
            located["round_limit"] = self.lookahead_rounds
            locator_attempts.append(located)
            candidates = located.get("candidates") if isinstance(located.get("candidates"), list) else []
            if candidates:
                candidate_ranges.extend(candidates[: self.max_candidates])
                break
            search_end = located.get("search_end_line")
            if type(search_end) is not int or search_end >= len(self.dialogue_index.lines):
                break
            search_after_line = search_end + 1

        if not candidate_ranges:
            return {
                "enabled": True,
                "triggered": True,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "identity locator found no bounded candidate identity marker",
                "confidence": "low",
                "same_person": None,
                "locator_attempts": locator_attempts,
                "candidate_ranges": [],
                "resolver": None,
            }

        resolver_reviews: list[dict[str, Any]] = []
        for candidate in candidate_ranges[: self.max_candidates]:
            resolved = self.resolver.resolve(
                speaker=speaker,
                start_line=_int_or_default(candidate.get("start_line"), 0),
                end_line=_int_or_default(candidate.get("end_line"), 0),
            )
            resolver_reviews.append(resolved)
            if resolved.get("verdict") == "resolved" and resolved.get("recommended_speaker"):
                evidence_lines = _unique_positive_ints(resolved.get("evidence_lines", []))
                recommended = _optional_clean(resolved.get("recommended_speaker"))
                return {
                    "enabled": True,
                    "triggered": True,
                    "speaker": speaker,
                    "verdict": "resolved",
                    "recommended_speaker": recommended,
                    "evidence_lines": evidence_lines,
                    "reason": (
                        "identity resolver found the same temporary speaker's stable name "
                        f"in bounded lookahead lines: {', '.join(str(line) for line in evidence_lines) or 'none'}"
                    ),
                    "confidence": "high" if evidence_lines else "medium",
                    "same_person": True,
                    "locator_attempts": locator_attempts,
                    "candidate_ranges": candidate_ranges,
                    "resolver": resolved,
                    "resolver_attempts": resolver_reviews,
                }

        last_review = resolver_reviews[-1] if resolver_reviews else {}
        verdict = _optional_clean(last_review.get("verdict")) or "not_enough_evidence"
        if verdict not in {"not_same_person", "not_enough_evidence"}:
            verdict = "not_enough_evidence"
        return {
            "enabled": True,
            "triggered": True,
            "speaker": speaker,
            "verdict": verdict,
            "recommended_speaker": None,
            "evidence_lines": [],
            "reason": _optional_clean(last_review.get("reason"))
            or "identity resolver did not confirm the candidate as the same person",
            "confidence": "low",
            "same_person": False if verdict == "not_same_person" else None,
            "locator_attempts": locator_attempts,
            "candidate_ranges": candidate_ranges,
            "resolver": last_review or None,
            "resolver_attempts": resolver_reviews,
        }

    def _model_review(self, record: Any) -> dict[str, Any]:
        speaker = _required_name(record.speaker, "speaker")
        if not should_coordinate_identity_lookup(speaker):
            return {
                "enabled": True,
                "triggered": False,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "speaker is not a coordinator-managed temporary identity",
                "confidence": "low",
                "same_person": None,
                "locator_attempts": [],
                "candidate_ranges": [],
                "resolver": None,
                "model_agent": True,
            }
        if self.lookahead_rounds == 0:
            return {
                "enabled": True,
                "triggered": True,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": "identity lookahead round limit is zero",
                "confidence": "low",
                "same_person": None,
                "locator_attempts": [],
                "candidate_ranges": [],
                "resolver": None,
                "model_agent": True,
            }

        locator_attempts: list[dict[str, Any]] = []
        candidate_ranges: list[dict[str, Any]] = []
        search_after_line: Optional[int] = None
        for round_number in range(1, self.lookahead_rounds + 1):
            attempt = self._locate_with_model(record, speaker, round_number, search_after_line)
            locator_attempts.append(attempt)
            candidates = attempt.get("candidates") if isinstance(attempt.get("candidates"), list) else []
            if candidates:
                candidate_ranges.extend(candidates[: self.max_candidates])
                break
            search_end = attempt.get("search_end_line")
            if type(search_end) is not int or search_end >= len(self.dialogue_index.lines):
                break
            search_after_line = search_end + 1

        if not candidate_ranges:
            reason = "identity locator model found no bounded candidate identity marker"
            error = _last_identity_error(locator_attempts)
            if error:
                reason = f"{reason}; last model error: {error}"
            return {
                "enabled": True,
                "triggered": True,
                "speaker": speaker,
                "verdict": "not_enough_evidence",
                "recommended_speaker": None,
                "evidence_lines": [],
                "reason": reason,
                "confidence": "low",
                "same_person": None,
                "locator_attempts": locator_attempts,
                "candidate_ranges": [],
                "resolver": None,
                "model_agent": True,
            }

        resolver_reviews: list[dict[str, Any]] = []
        for candidate in candidate_ranges[: self.max_candidates]:
            resolved = self._resolve_with_model(record, speaker, candidate)
            resolver_reviews.append(resolved)
            if resolved.get("verdict") == "resolved" and resolved.get("recommended_speaker"):
                evidence_lines = _unique_positive_ints(resolved.get("evidence_lines", []))
                recommended = _optional_clean(resolved.get("recommended_speaker"))
                return {
                    "enabled": True,
                    "triggered": True,
                    "speaker": speaker,
                    "verdict": "resolved",
                    "recommended_speaker": recommended,
                    "evidence_lines": evidence_lines,
                    "reason": _optional_clean(resolved.get("reason"))
                    or "identity resolver model confirmed the same person and stable speaker",
                    "confidence": _confidence_or_low(resolved.get("confidence")),
                    "same_person": True,
                    "locator_attempts": locator_attempts,
                    "candidate_ranges": candidate_ranges,
                    "resolver": resolved,
                    "resolver_attempts": resolver_reviews,
                    "model_agent": True,
                }

        last_review = resolver_reviews[-1] if resolver_reviews else {}
        verdict = _optional_clean(last_review.get("verdict")) or "not_enough_evidence"
        if verdict not in {"not_same_person", "not_enough_evidence"}:
            verdict = "not_enough_evidence"
        return {
            "enabled": True,
            "triggered": True,
            "speaker": speaker,
            "verdict": verdict,
            "recommended_speaker": None,
            "evidence_lines": [],
            "reason": _optional_clean(last_review.get("reason"))
            or "identity resolver model did not confirm the candidate as the same person",
            "confidence": "low",
            "same_person": False if verdict == "not_same_person" else None,
            "locator_attempts": locator_attempts,
            "candidate_ranges": candidate_ranges,
            "resolver": last_review or None,
            "resolver_attempts": resolver_reviews,
            "model_agent": True,
        }

    def _locate_with_model(
        self,
        record: Any,
        speaker: str,
        round_number: int,
        search_after_line: Optional[int],
    ) -> dict[str, Any]:
        search_start = max(record.line_number + 1, search_after_line or record.line_number + 1)
        search_end = min(len(self.dialogue_index.lines), search_start + self.lookahead_lines - 1)
        context = _line_context(self.dialogue_index, search_start, search_end)
        payload, error = self._chat_json(identity_locator_messages(record, speaker, search_start, search_end, context))
        candidates = _candidate_ranges_from_payload(
            payload,
            dialogue_index=self.dialogue_index,
            search_start=search_start,
            search_end=search_end,
            max_candidates=self.max_candidates,
        )
        attempt = {
            "speaker": speaker,
            "current_line": record.line_number,
            "search_start_line": search_start,
            "search_end_line": search_end,
            "lookahead_lines": self.lookahead_lines,
            "round": round_number,
            "round_limit": self.lookahead_rounds,
            "candidates": candidates,
            "reason": _optional_clean(payload.get("reason")) if isinstance(payload, dict) else None,
            "model_agent": True,
        }
        if error:
            attempt["error"] = error
        return attempt

    def _resolve_with_model(self, record: Any, speaker: str, candidate: dict[str, Any]) -> dict[str, Any]:
        start_line = _int_or_default(candidate.get("start_line"), 0)
        end_line = _int_or_default(candidate.get("end_line"), 0)
        context = _line_context(self.dialogue_index, start_line, end_line)
        payload, error = self._chat_json(identity_resolver_messages(record, speaker, candidate, context))
        resolved = _resolver_review_from_payload(payload, speaker=speaker, candidate=candidate)
        resolved["model_agent"] = True
        if error:
            resolved["error"] = error
        return resolved

    def _chat_json(self, messages: list[ChatMessage]) -> tuple[dict[str, Any], Optional[str]]:
        last_raw: Optional[str] = None
        last_error: Optional[str] = None
        current_messages = messages
        for attempt in range(self.retries + 1):
            try:
                response = self.model_client.chat(
                    messages=current_messages,
                    tools=None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as error:  # noqa: BLE001 - identity errors must not break unattended runs.
                last_error = str(error)
                if attempt < self.retries:
                    continue
                return {}, last_error

            last_raw = response.content
            try:
                payload = json.loads(_extract_json_object(last_raw))
            except (json.JSONDecodeError, ValueError) as error:
                last_error = str(error)
                if attempt < self.retries:
                    current_messages = identity_retry_messages(last_raw, last_error)
                    continue
                return {}, last_error
            if isinstance(payload, dict):
                return payload, None
            last_error = "identity model JSON payload was not an object"
            if attempt < self.retries:
                current_messages = identity_retry_messages(last_raw, last_error)
                continue
            return {}, last_error
        return {}, last_error or "identity model returned invalid JSON"


def identity_locator_messages(
    record: Any,
    speaker: str,
    search_start: int,
    search_end: int,
    context: str,
) -> list[ChatMessage]:
    payload = {
        "task": "Find bounded later-context ranges that may reveal the stable identity of a temporary speaker.",
        "rules": [
            "Only locate candidate identity-reveal ranges; do not decide the final speaker.",
            "Use the numbered later-context lines only.",
            "Return candidates only when later text appears to reveal the same concrete person's name or stable title.",
            "Do not trigger on pronouns, speech habits, ordinary groups, places, or characters inside a quoted story.",
            "Keep each candidate range small and include the matched evidence line.",
        ],
        "current_dialogue": {
            "index": record.index,
            "line_number": record.line_number,
            "text": record.text,
            "temporary_speaker": speaker,
        },
        "later_context": {
            "start_line": search_start,
            "end_line": search_end,
            "text": context,
        },
        "required_json_shape": {
            "candidates": [
                {
                    "start_line": 1,
                    "end_line": 3,
                    "matched_line": 2,
                    "suggested_names": ["stable speaker name"],
                    "reason": "why this bounded range may reveal the same person's identity",
                }
            ],
            "reason": "short explanation, or why no candidate was found",
        },
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the Dialoop Identity Locator Agent. You have an isolated, bounded context window. "
                "Return exactly one compact JSON object. Do not use Markdown fences."
            ),
        ),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def identity_resolver_messages(
    record: Any,
    speaker: str,
    candidate: dict[str, Any],
    context: str,
) -> list[ChatMessage]:
    payload = {
        "task": "Decide whether a locator candidate range reveals the same person's stable speaker identity.",
        "rules": [
            "Read only the current dialogue metadata and the candidate range.",
            "Return resolved only when the candidate clearly refers to the same concrete person as the temporary speaker.",
            "Return not_same_person when the range is about a different person, a quoted story character, a group, or a role.",
            "Return not_enough_evidence when same-person continuity is ambiguous.",
            "recommended_speaker must be empty unless verdict is resolved.",
            "evidence_lines must cite numbered lines in the candidate range.",
        ],
        "current_dialogue": {
            "index": record.index,
            "line_number": record.line_number,
            "text": record.text,
            "temporary_speaker": speaker,
        },
        "candidate_range": candidate,
        "candidate_text": context,
        "required_json_shape": {
            "verdict": "resolved | not_same_person | not_enough_evidence",
            "same_person": True,
            "recommended_speaker": "stable speaker name or null",
            "evidence_lines": [1],
            "reason": "why this is or is not the same person",
            "confidence": "high | medium | low",
        },
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the Dialoop Identity Resolver Agent. You independently judge same-person identity "
                "from a bounded candidate range. Return exactly one compact JSON object. Do not use Markdown fences."
            ),
        ),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    ]


def identity_retry_messages(raw: Optional[str], error: str) -> list[ChatMessage]:
    previous = raw if raw is not None and raw.strip() else "<empty response>"
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a Dialoop identity sub-agent. Repair your previous response. "
                "Return exactly one compact JSON object and nothing else."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "The previous identity response was not usable JSON.\n"
                f"Error: {error}\n"
                f"Previous response: {previous[:600]}\n"
                'Return only a valid JSON object matching the requested shape.'
            ),
        ),
    ]


def _line_context(dialogue_index: Any, start_line: int, end_line: int) -> str:
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return ""
    bounded_start = min(max(1, start_line), len(dialogue_index.lines) + 1)
    bounded_end = min(len(dialogue_index.lines), end_line)
    if bounded_start > bounded_end:
        return ""
    return "\n".join(
        f"{line_number}: {dialogue_index.lines[line_number - 1]}"
        for line_number in range(bounded_start, bounded_end + 1)
    )


def _candidate_ranges_from_payload(
    payload: dict[str, Any],
    *,
    dialogue_index: Any,
    search_start: int,
    search_end: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        start_line = _int_or_default(candidate.get("start_line"), 0)
        end_line = _int_or_default(candidate.get("end_line"), 0)
        matched_line = _int_or_default(candidate.get("matched_line"), start_line)
        if start_line <= 0 or end_line <= 0 or end_line < start_line:
            continue
        bounded_start = max(1, min(start_line, len(dialogue_index.lines)))
        bounded_end = max(1, min(end_line, len(dialogue_index.lines)))
        if bounded_end < bounded_start:
            continue
        if matched_line < search_start or matched_line > search_end:
            continue
        item = {
            "start_line": bounded_start,
            "end_line": bounded_end,
            "matched_line": matched_line,
            "reason": _optional_clean(candidate.get("reason")) or "identity locator model returned this range",
            "suggested_names": _unique_strings(candidate.get("suggested_names", []))
            if isinstance(candidate.get("suggested_names"), list)
            else [],
            "text": _line_context(dialogue_index, bounded_start, bounded_end),
            "model_agent": True,
        }
        result.append(item)
        if len(result) >= max_candidates:
            break
    return result


def _resolver_review_from_payload(payload: dict[str, Any], *, speaker: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "speaker": speaker,
            "verdict": "not_enough_evidence",
            "same_person": None,
            "recommended_speaker": None,
            "evidence_lines": [],
            "reason": "identity resolver model returned an unsupported payload",
            "confidence": "low",
            "candidate_range": dict(candidate),
        }
    same_person = payload.get("same_person") if isinstance(payload.get("same_person"), bool) else None
    verdict = _optional_clean(payload.get("verdict")) or ""
    if verdict not in {"resolved", "not_same_person", "not_enough_evidence"}:
        if same_person is True:
            verdict = "resolved"
        elif same_person is False:
            verdict = "not_same_person"
        else:
            verdict = "not_enough_evidence"
    recommended = _optional_clean(payload.get("recommended_speaker"))
    if verdict == "resolved" and recommended is None:
        verdict = "not_enough_evidence"
        same_person = None
    if verdict == "resolved":
        same_person = True
    elif verdict == "not_same_person":
        same_person = False
        recommended = None
    else:
        same_person = None
        recommended = None
    return {
        "speaker": speaker,
        "verdict": verdict,
        "same_person": same_person,
        "recommended_speaker": recommended,
        "evidence_lines": _unique_positive_ints(payload.get("evidence_lines", [])),
        "reason": _optional_clean(payload.get("reason")) or "identity resolver model returned no reason",
        "confidence": _confidence_or_low(payload.get("confidence")),
        "candidate_range": dict(candidate),
    }


def _confidence_or_low(value: Any) -> str:
    cleaned = value.strip().lower() if isinstance(value, str) else ""
    return cleaned if cleaned in VALID_CONFIDENCE else "low"


def _last_identity_error(attempts: list[dict[str, Any]]) -> Optional[str]:
    for attempt in reversed(attempts):
        error = attempt.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
    return None


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


def should_coordinate_identity_lookup(speaker: str) -> bool:
    return isinstance(speaker, str) and speaker.strip() in COORDINATOR_IDENTITY_SPEAKERS


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


def _int_or_default(value: Any, default: int) -> int:
    if type(value) is int:
        return value
    return default
