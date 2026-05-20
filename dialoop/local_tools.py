from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
    ):
        if batch_size <= 0:
            raise ToolValidationError("batch_size must be greater than 0")
        if max_line_gap is not None and max_line_gap < 0:
            raise ToolValidationError("max_line_gap must be 0 or greater")
        if previous_context_dialogues < 0:
            raise ToolValidationError("previous_context_dialogues must be 0 or greater")
        if following_context_dialogues < 0:
            raise ToolValidationError("following_context_dialogues must be 0 or greater")
        self.dialogue_index = dialogue_index
        self.label_store = label_store
        self.batch_size = batch_size
        self.max_line_gap = max_line_gap
        self.read_window_limit = read_window_limit
        self.search_limit = search_limit
        self.previous_context_dialogues = previous_context_dialogues
        self.following_context_dialogues = following_context_dialogues
        self._active_batch: list[Dialogue] = []

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

    def search_novel(self, keyword: str, limit: Optional[int] = None) -> dict[str, Any]:
        return self.dialogue_index.search(keyword=keyword, limit=limit or self.search_limit)

    def submit_labels(self, speakers: list[str]) -> dict[str, Any]:
        if not self._active_batch:
            raise ToolValidationError("no active batch; call get_next_dialogue first")
        if len(speakers) != len(self._active_batch):
            raise ToolValidationError(
                f"speaker count mismatch: expected {len(self._active_batch)}, got {len(speakers)}"
            )

        written = self.label_store.append(speakers)
        self._active_batch = []
        return {
            "accepted": True,
            "written": written,
            "progress": self.get_progress(),
        }
