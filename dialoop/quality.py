from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .local_tools import DialogueIndex


ANSWER_DIALOGUE_PATTERN = re.compile(r"【([^】]+)】「([^」]+)」")
DEFAULT_SCAN_PATHS = (Path("dialoop"), Path("pyproject.toml"))
SKIPPED_DIRS = {".git", ".hg", ".mypy_cache", ".pytest_cache", "__pycache__"}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class QualityError(ValueError):
    """Raised when quality evaluation input is invalid."""


@dataclass(frozen=True)
class ExpectedDialogue:
    index: int
    line_number: int
    text: str
    speaker: str

    @property
    def accepted_speakers(self) -> list[str]:
        return [speaker.strip() for speaker in self.speaker.split("|") if speaker.strip()]


@dataclass(frozen=True)
class LabelMismatch:
    index: int
    line_number: int
    text: str
    expected: str
    actual: str


@dataclass(frozen=True)
class EvaluationReport:
    expected_count: int
    label_count: int
    compared_count: int
    correct_count: int
    mismatches: list[LabelMismatch] = field(default_factory=list)
    missing_count: int = 0
    extra_labels: list[str] = field(default_factory=list)
    novel_dialogue_count: Optional[int] = None

    @property
    def incorrect_count(self) -> int:
        return len(self.mismatches)

    @property
    def accuracy(self) -> float:
        if self.compared_count == 0:
            return 0.0
        return self.correct_count / self.compared_count


@dataclass(frozen=True)
class TermMatch:
    path: Path
    line_number: int
    term: str
    line: str


def extract_expected_dialogues(answer_path: Path) -> list[ExpectedDialogue]:
    expected: list[ExpectedDialogue] = []
    for resolved in resolve_answer_files(answer_path):
        with resolved.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                for match in ANSWER_DIALOGUE_PATTERN.finditer(line):
                    expected.append(
                        ExpectedDialogue(
                            index=len(expected),
                            line_number=line_number,
                            text=match.group(2).strip(),
                            speaker=match.group(1).strip(),
                        )
                    )
    if not expected:
        raise QualityError(f"no annotated dialogues found in {answer_path.expanduser().resolve()}")
    return expected


def resolve_answer_files(answer_path: Path) -> list[Path]:
    resolved = answer_path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if resolved.is_dir():
        files = sorted(path for path in resolved.glob("*.txt") if path.is_file())
        if files:
            return files
        raise QualityError(f"answer directory contains no .txt files: {resolved}")
    raise QualityError(f"answer path does not exist: {resolved}")


def load_label_lines(labels_path: Path) -> list[str]:
    resolved = labels_path.expanduser().resolve()
    if not resolved.exists():
        raise QualityError(f"labels file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def evaluate_labels(
    *,
    answer_path: Path,
    labels_path: Path,
    novel_path: Optional[Path] = None,
) -> EvaluationReport:
    expected = extract_expected_dialogues(answer_path)
    labels = load_label_lines(labels_path)
    compared_count = min(len(expected), len(labels))
    correct_count = 0
    mismatches: list[LabelMismatch] = []

    for index in range(compared_count):
        actual = labels[index]
        expected_dialogue = expected[index]
        if speaker_matches(actual, expected_dialogue):
            correct_count += 1
            continue
        mismatches.append(
            LabelMismatch(
                index=index,
                line_number=expected_dialogue.line_number,
                text=expected_dialogue.text,
                expected=expected_dialogue.speaker,
                actual=actual,
            )
        )

    novel_dialogue_count = None
    if novel_path is not None:
        novel_dialogue_count = DialogueIndex.from_file(novel_path).total

    return EvaluationReport(
        expected_count=len(expected),
        label_count=len(labels),
        compared_count=compared_count,
        correct_count=correct_count,
        mismatches=mismatches,
        missing_count=max(0, len(expected) - len(labels)),
        extra_labels=labels[len(expected) :],
        novel_dialogue_count=novel_dialogue_count,
    )


def speaker_matches(actual: str, expected: ExpectedDialogue) -> bool:
    return actual.strip() in expected.accepted_speakers


def render_evaluation_report(report: EvaluationReport, max_errors: Optional[int] = 50) -> str:
    lines = [
        "Dialoop evaluation",
        f"  expected: {report.expected_count}",
        f"  labels: {report.label_count}",
        f"  compared: {report.compared_count}",
        f"  correct: {report.correct_count}",
        f"  incorrect: {report.incorrect_count}",
        f"  accuracy: {report.accuracy * 100:.2f}%",
        f"  missing_labels: {report.missing_count}",
        f"  extra_labels: {len(report.extra_labels)}",
    ]
    if report.novel_dialogue_count is not None:
        lines.append(f"  novel_dialogues: {report.novel_dialogue_count}")
    if report.mismatches:
        lines.append("")
        lines.append("Mismatches:")
        shown = report.mismatches if max_errors is None else report.mismatches[:max_errors]
        for mismatch in shown:
            lines.append(
                f"  - index={mismatch.index} line={mismatch.line_number} "
                f"expected={mismatch.expected} actual={mismatch.actual} text={mismatch.text}"
            )
        hidden = len(report.mismatches) - len(shown)
        if hidden > 0:
            lines.append(f"  ... {hidden} more mismatch(es) omitted")
    return "\n".join(lines)


def render_error_labels(report: EvaluationReport) -> str:
    return "\n".join(
        f'{mismatch.line_number}行"「{mismatch.text}」"，'
        f"标注【{mismatch.actual}】，答案【{mismatch.expected}】"
        for mismatch in report.mismatches
    )


def load_terms(*, terms: Iterable[str] = (), terms_file: Optional[Path] = None) -> list[str]:
    loaded = [term.strip() for term in terms if term.strip()]
    if terms_file is not None:
        resolved = terms_file.expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    loaded.append(stripped)
    unique: list[str] = []
    seen = set()
    for term in loaded:
        if term in seen:
            continue
        seen.add(term)
        unique.append(term)
    if not unique:
        raise QualityError("at least one term must be provided")
    return unique


def scan_terms(paths: Iterable[Path], terms: Iterable[str]) -> list[TermMatch]:
    terms_list = list(terms)
    matches: list[TermMatch] = []
    for path in iter_scan_files(paths):
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    for term in terms_list:
                        if term in line:
                            matches.append(
                                TermMatch(
                                    path=path,
                                    line_number=line_number,
                                    term=term,
                                    line=line.rstrip("\n"),
                                )
                            )
        except UnicodeDecodeError:
            continue
    return matches


def iter_scan_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            if is_text_candidate(resolved):
                yield resolved
            continue
        if not resolved.is_dir():
            raise QualityError(f"scan path does not exist: {resolved}")
        for child in resolved.rglob("*"):
            if child.is_dir():
                continue
            if any(part in SKIPPED_DIRS for part in child.parts):
                continue
            if is_text_candidate(child):
                yield child


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def render_term_scan_report(matches: list[TermMatch], terms: list[str], paths: list[Path]) -> str:
    lines = [
        "Dialoop proprietary term scan",
        f"  paths: {', '.join(str(path) for path in paths)}",
        f"  terms: {', '.join(terms)}",
        f"  matches: {len(matches)}",
    ]
    if matches:
        lines.append("")
        lines.append("Matches:")
        for match in matches:
            lines.append(
                f"  - {match.path}:{match.line_number} term={match.term} line={match.line.strip()}"
            )
    return "\n".join(lines)
