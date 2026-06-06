from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .identity import (
    DEFAULT_LOOKAHEAD_LINES,
    DEFAULT_LOOKAHEAD_ROUNDS,
    CharacterLibrary,
    IdentityArbiterAgent,
    IdentityLocatorAgent,
    IdentityResolverAgent,
    IdentityValidationError,
)


DIALOGUE_PATTERN = re.compile(r"「([^」]+)」")


class ToolValidationError(ValueError):
    """Raised when a local tool receives invalid input."""


@dataclass(frozen=True)
class Dialogue:
    index: int
    line_number: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line_number": self.line_number,
            "text": self.text,
        }


class SpeakerCountMismatchError(ToolValidationError):
    def __init__(
        self,
        expected_count: int,
        received_count: int,
        active_batch: list[Dialogue],
        received_speakers: list[str],
    ):
        self.expected_count = expected_count
        self.received_count = received_count
        self.active_batch = active_batch
        self.received_speakers = received_speakers
        super().__init__(
            f"speaker count mismatch: expected {expected_count}, got {received_count}. "
            "Submit exactly one speaker per active dialogue only; do not submit labels for "
            "previous_dialogues, following_dialogues, or raw context lines."
        )

    def to_result(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "error": str(self),
            "expected_count": self.expected_count,
            "received_count": self.received_count,
            "active_dialogues": [dialogue.to_dict() for dialogue in self.active_batch],
            "received_speakers": self.received_speakers,
            "instruction": (
                f"Call submit_labels again with exactly {self.expected_count} speaker(s), "
                "in active_dialogues order. Ignore previous_dialogues and following_dialogues."
            ),
        }


@dataclass(frozen=True)
class SearchMatch:
    line_number: int
    line: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "line": self.line,
        }


class DialogueIndex:
    def __init__(self, novel_path: Path, lines: list[str], dialogues: list[Dialogue]):
        self.novel_path = novel_path
        self.lines = lines
        self.dialogues = dialogues

    @classmethod
    def from_file(cls, novel_path: Path) -> "DialogueIndex":
        resolved = novel_path.expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as file:
            text = file.read()
        return cls.from_text(text, novel_path=resolved)

    @classmethod
    def from_text(cls, text: str, novel_path: Optional[Path] = None) -> "DialogueIndex":
        lines = text.splitlines()
        dialogues: list[Dialogue] = []

        for line_number, line in enumerate(lines, start=1):
            for match in DIALOGUE_PATTERN.finditer(line):
                dialogues.append(
                    Dialogue(
                        index=len(dialogues),
                        line_number=line_number,
                        text=match.group(1),
                    )
                )

        return cls(novel_path or Path("novel.txt"), lines, dialogues)

    @property
    def total(self) -> int:
        return len(self.dialogues)

    def next_batch(
        self,
        labeled_count: int,
        batch_size: int,
        max_line_gap: Optional[int] = None,
    ) -> list[Dialogue]:
        if labeled_count < 0:
            raise ToolValidationError("labeled_count must be 0 or greater")
        if batch_size <= 0:
            raise ToolValidationError("batch_size must be greater than 0")
        if labeled_count >= self.total:
            return []

        batch: list[Dialogue] = []
        last_line_number: Optional[int] = None
        index = labeled_count

        while index < self.total and len(batch) < batch_size:
            dialogue = self.dialogues[index]

            if last_line_number is not None and max_line_gap is not None:
                if dialogue.line_number - last_line_number > max_line_gap:
                    break

            batch.append(dialogue)
            last_line_number = dialogue.line_number
            index += 1

        return batch

    def read_lines(self, start_line: int, end_line: int, line_limit: int = 300) -> dict[str, Any]:
        if line_limit <= 0:
            raise ToolValidationError("line_limit must be greater than 0")
        if start_line <= 0 or end_line <= 0:
            raise ToolValidationError("line numbers are 1-based and must be positive")
        if end_line < start_line:
            raise ToolValidationError("end_line must be greater than or equal to start_line")

        bounded_start = min(max(1, start_line), len(self.lines) + 1)
        bounded_end = min(len(self.lines), end_line)
        if bounded_start > bounded_end:
            return {
                "start_line": bounded_start,
                "end_line": bounded_end,
                "requested_end_line": end_line,
                "line_limit": line_limit,
                "truncated": False,
                "text": "",
            }

        requested_count = bounded_end - bounded_start + 1
        actual_count = min(requested_count, line_limit)
        actual_end = bounded_start + actual_count - 1
        selected = self.lines[bounded_start - 1 : actual_end]

        return {
            "start_line": bounded_start,
            "end_line": actual_end,
            "requested_end_line": end_line,
            "line_limit": line_limit,
            "truncated": requested_count > line_limit,
            "text": "\n".join(
                f"{line_number}: {line}"
                for line_number, line in enumerate(selected, start=bounded_start)
            ),
        }

    def search(self, keyword: str, limit: int = 20) -> dict[str, Any]:
        if not keyword:
            raise ToolValidationError("keyword must not be empty")
        if limit <= 0:
            raise ToolValidationError("limit must be greater than 0")

        matches: list[SearchMatch] = []
        total_matches = 0

        for line_number, line in enumerate(self.lines, start=1):
            if keyword not in line:
                continue
            total_matches += 1
            if len(matches) < limit:
                matches.append(SearchMatch(line_number=line_number, line=f"{line_number}: {line}"))

        return {
            "keyword": keyword,
            "matches": [match.to_dict() for match in matches],
            "total_matches": total_matches,
            "truncated": total_matches > len(matches),
        }


class LabelStore:
    def __init__(self, labels_path: Path):
        self.labels_path = labels_path.expanduser().resolve()

    def labels(self) -> list[str]:
        if not self.labels_path.exists():
            return []
        with self.labels_path.open("r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def count(self) -> int:
        return len(self.labels())

    def append(self, speakers: list[str]) -> int:
        cleaned = [speaker.strip() for speaker in speakers]
        if not cleaned:
            raise ToolValidationError("speakers must not be empty")
        if any(not speaker for speaker in cleaned):
            raise ToolValidationError("speaker names must not be empty")

        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8") as file:
            file.write("".join(f"{speaker}\n" for speaker in cleaned))
        return len(cleaned)


class DialoopLocalTools:
    def __init__(
        self,
        dialogue_index: DialogueIndex,
        label_store: LabelStore,
        batch_size: int = 1,
        max_line_gap: Optional[int] = None,
        read_window_limit: int = 300,
        search_limit: int = 20,
        previous_context_dialogues: int = 8,
        following_context_dialogues: int = 8,
        identity_lookahead_lines: int = DEFAULT_LOOKAHEAD_LINES,
        identity_lookahead_rounds: int = DEFAULT_LOOKAHEAD_ROUNDS,
        character_library: Optional[CharacterLibrary] = None,
    ):
        if batch_size <= 0:
            raise ToolValidationError("batch_size must be greater than 0")
        if max_line_gap is not None and max_line_gap < 0:
            raise ToolValidationError("max_line_gap must be 0 or greater")
        if previous_context_dialogues < 0:
            raise ToolValidationError("previous_context_dialogues must be 0 or greater")
        if following_context_dialogues < 0:
            raise ToolValidationError("following_context_dialogues must be 0 or greater")
        if identity_lookahead_lines <= 0:
            raise ToolValidationError("identity_lookahead_lines must be greater than 0")
        if identity_lookahead_rounds < 0:
            raise ToolValidationError("identity_lookahead_rounds must be 0 or greater")
        self.dialogue_index = dialogue_index
        self.label_store = label_store
        self.batch_size = batch_size
        self.max_line_gap = max_line_gap
        self.read_window_limit = read_window_limit
        self.search_limit = search_limit
        self.previous_context_dialogues = previous_context_dialogues
        self.following_context_dialogues = following_context_dialogues
        self.identity_lookahead_lines = identity_lookahead_lines
        self.identity_lookahead_rounds = identity_lookahead_rounds
        self.character_library = character_library or CharacterLibrary()
        self.identity_locator = IdentityLocatorAgent(dialogue_index)
        self.identity_resolver = IdentityResolverAgent(dialogue_index)
        self.identity_arbiter = IdentityArbiterAgent()
        self._active_batch: list[Dialogue] = []
        self._identity_lookup_counts: dict[str, int] = {}

    @property
    def active_batch_size(self) -> int:
        return len(self._active_batch)

    @classmethod
    def from_paths(
        cls,
        novel_path: Path,
        labels_path: Path,
        batch_size: int = 1,
        max_line_gap: Optional[int] = None,
        read_window_limit: int = 300,
        search_limit: int = 20,
        previous_context_dialogues: int = 8,
        following_context_dialogues: int = 8,
        identity_lookahead_lines: int = DEFAULT_LOOKAHEAD_LINES,
        identity_lookahead_rounds: int = DEFAULT_LOOKAHEAD_ROUNDS,
    ) -> "DialoopLocalTools":
        return cls(
            dialogue_index=DialogueIndex.from_file(novel_path),
            label_store=LabelStore(labels_path),
            batch_size=batch_size,
            max_line_gap=max_line_gap,
            read_window_limit=read_window_limit,
            search_limit=search_limit,
            previous_context_dialogues=previous_context_dialogues,
            following_context_dialogues=following_context_dialogues,
            identity_lookahead_lines=identity_lookahead_lines,
            identity_lookahead_rounds=identity_lookahead_rounds,
        )

    def get_progress(self) -> dict[str, Any]:
        labeled = self.label_store.count()
        total = self.dialogue_index.total
        return {
            "labeled": labeled,
            "total": total,
            "remaining": max(0, total - labeled),
            "output_path": str(self.label_store.labels_path),
        }

    def get_next_dialogue(self, batch_size: Optional[int] = None) -> dict[str, Any]:
        size = batch_size or self.batch_size
        progress = self.get_progress()
        batch = self.dialogue_index.next_batch(progress["labeled"], size, max_line_gap=self.max_line_gap)
        self._active_batch = batch
        labels = self.label_store.labels()

        return {
            "done": len(batch) == 0,
            "progress": progress,
            "dialogues": [dialogue.to_dict() for dialogue in batch],
            "previous_dialogues": self._previous_labeled_dialogues(
                labeled_count=progress["labeled"],
                labels=labels,
            ),
            "following_dialogues": self._following_unlabeled_dialogues(
                start_index=progress["labeled"] + len(batch),
            ),
            "known_characters": self.character_library.to_list(),
        }

    def _previous_labeled_dialogues(self, labeled_count: int, labels: list[str]) -> list[dict[str, Any]]:
        start = max(0, labeled_count - self.previous_context_dialogues)
        end = min(labeled_count, len(labels), self.dialogue_index.total)
        return [
            {
                **self.dialogue_index.dialogues[index].to_dict(),
                "speaker": labels[index],
            }
            for index in range(start, end)
        ]

    def _following_unlabeled_dialogues(self, start_index: int) -> list[dict[str, Any]]:
        end = min(self.dialogue_index.total, start_index + self.following_context_dialogues)
        return [dialogue.to_dict() for dialogue in self.dialogue_index.dialogues[start_index:end]]

    def read_novel(self, start_line: int, end_line: int) -> dict[str, Any]:
        return self.dialogue_index.read_lines(
            start_line=start_line,
            end_line=end_line,
            line_limit=self.read_window_limit,
        )

    def read_active_context(self, context_window_lines: int) -> dict[str, Any]:
        if context_window_lines <= 0:
            raise ToolValidationError("context_window_lines must be greater than 0")
        if not self._active_batch:
            raise ToolValidationError("no active batch; call get_next_dialogue first")

        first_line = min(dialogue.line_number for dialogue in self._active_batch)
        last_line = max(dialogue.line_number for dialogue in self._active_batch)
        return self.read_novel(
            start_line=max(1, first_line - context_window_lines),
            end_line=last_line + context_window_lines,
        )

    def search_novel(self, keyword: str, limit: Optional[int] = None) -> dict[str, Any]:
        return self.dialogue_index.search(keyword=keyword, limit=limit or self.search_limit)

    def locate_identity(
        self,
        speaker: str,
        dialogue_index: Optional[int] = None,
        search_after_line: Optional[int] = None,
        lookahead_lines: Optional[int] = None,
        max_candidates: int = 3,
    ) -> dict[str, Any]:
        dialogue = self._active_or_indexed_dialogue(dialogue_index)
        lookup_key = f"{dialogue.index}:{speaker.strip()}"
        lookup_count = self._identity_lookup_counts.get(lookup_key, 0)
        search_start = max(dialogue.line_number + 1, search_after_line or dialogue.line_number + 1)
        search_end = min(
            len(self.dialogue_index.lines),
            search_start + (lookahead_lines or self.identity_lookahead_lines) - 1,
        )
        if lookup_count >= self.identity_lookahead_rounds:
            return {
                "speaker": speaker.strip(),
                "current_line": dialogue.line_number,
                "search_start_line": search_start,
                "search_end_line": search_end,
                "lookahead_lines": lookahead_lines or self.identity_lookahead_lines,
                "round": lookup_count,
                "round_limit": self.identity_lookahead_rounds,
                "round_limit_reached": True,
                "candidates": [],
                "suggest_continue": False,
                "reason": "identity lookahead round limit reached",
            }
        self._identity_lookup_counts[lookup_key] = lookup_count + 1
        try:
            result = self.identity_locator.locate(
                speaker=speaker,
                current_line=dialogue.line_number,
                search_after_line=search_after_line,
                lookahead_lines=lookahead_lines or self.identity_lookahead_lines,
                max_candidates=max_candidates,
            )
        except IdentityValidationError as error:
            raise ToolValidationError(str(error)) from error
        result["round"] = lookup_count + 1
        result["round_limit"] = self.identity_lookahead_rounds
        result["round_limit_reached"] = False
        return result

    def resolve_identity(
        self,
        speaker: str,
        start_line: int,
        end_line: int,
        dialogue_index: Optional[int] = None,
    ) -> dict[str, Any]:
        dialogue = self._active_or_indexed_dialogue(dialogue_index)
        try:
            return self.identity_resolver.resolve(
                speaker=speaker,
                start_line=start_line,
                end_line=end_line,
                current_dialogue=dialogue,
            )
        except IdentityValidationError as error:
            raise ToolValidationError(str(error)) from error

    def record_character(
        self,
        display_name: str,
        aliases: Optional[list[str]] = None,
        summary: str = "",
        evidence_lines: Optional[list[int]] = None,
        last_seen_dialogue_index: Optional[int] = None,
        last_seen_line_number: Optional[int] = None,
        confidence: str = "medium",
    ) -> dict[str, Any]:
        try:
            record = self.character_library.add_or_update(
                display_name=display_name,
                aliases=aliases,
                summary=summary,
                evidence_lines=evidence_lines,
                last_seen_dialogue_index=last_seen_dialogue_index,
                last_seen_line_number=last_seen_line_number,
                confidence=confidence,
            )
        except IdentityValidationError as error:
            raise ToolValidationError(str(error)) from error
        return {"record": record.to_dict(), "known_characters": self.character_library.to_list()}

    def normalize_speaker(self, speaker: str) -> dict[str, Any]:
        try:
            return self.character_library.normalize(speaker)
        except IdentityValidationError as error:
            raise ToolValidationError(str(error)) from error

    def arbitrate_identity(
        self,
        labeler_speaker: str,
        verifier_verdict: Optional[str] = None,
        resolver_verdict: Optional[str] = None,
        resolver_speaker: Optional[str] = None,
        normalizer_speaker: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            return self.identity_arbiter.arbitrate(
                labeler_speaker=labeler_speaker,
                verifier_verdict=verifier_verdict,
                resolver_verdict=resolver_verdict,
                resolver_speaker=resolver_speaker,
                normalizer_speaker=normalizer_speaker,
            )
        except IdentityValidationError as error:
            raise ToolValidationError(str(error)) from error

    def submit_labels(self, speakers: list[str]) -> dict[str, Any]:
        result = self.validate_labels(speakers)
        result.update(self.commit_labels(result["speakers"]))
        return result

    def _active_or_indexed_dialogue(self, dialogue_index: Optional[int]) -> Dialogue:
        if dialogue_index is not None:
            if dialogue_index < 0 or dialogue_index >= self.dialogue_index.total:
                raise ToolValidationError("dialogue_index is outside the novel dialogue index")
            return self.dialogue_index.dialogues[dialogue_index]
        if not self._active_batch:
            raise ToolValidationError("no active batch; call get_next_dialogue first or pass dialogue_index")
        return self._active_batch[0]

    def validate_labels(self, speakers: list[str]) -> dict[str, Any]:
        if not self._active_batch:
            raise ToolValidationError("no active batch; call get_next_dialogue first")
        expected_count = len(self._active_batch)
        received_count = len(speakers)
        warning = None
        ignored_speakers: list[str] = []

        if received_count < expected_count:
            raise SpeakerCountMismatchError(
                expected_count=expected_count,
                received_count=received_count,
                active_batch=list(self._active_batch),
                received_speakers=speakers,
            )
        if received_count > expected_count:
            ignored_speakers = speakers[expected_count:]
            speakers = speakers[:expected_count]
            warning = (
                f"received {received_count} speakers for {expected_count} active dialogue(s); "
                f"wrote the first {expected_count} and ignored {len(ignored_speakers)} extra speaker(s). "
                "Only the active batch is accepted; following context labels are ignored."
            )

        cleaned_speakers = [speaker.strip() for speaker in speakers]
        if not cleaned_speakers:
            raise ToolValidationError("speakers must not be empty")
        if any(not speaker for speaker in cleaned_speakers):
            raise ToolValidationError("speaker names must not be empty")

        result = {
            "accepted": True,
            "speakers": cleaned_speakers,
        }
        if warning is not None:
            result.update(
                {
                    "warning": warning,
                    "expected_count": expected_count,
                    "received_count": received_count,
                    "ignored_speakers": ignored_speakers,
                }
            )
        return result

    def commit_labels(self, speakers: list[str]) -> dict[str, Any]:
        if not self._active_batch:
            raise ToolValidationError("no active batch; call get_next_dialogue first")
        if len(speakers) != len(self._active_batch):
            raise ToolValidationError(
                f"speaker count mismatch at commit: expected {len(self._active_batch)}, got {len(speakers)}"
            )
        written = self.label_store.append(speakers)
        self._active_batch = []
        return {
            "written": written,
            "progress": self.get_progress(),
        }
