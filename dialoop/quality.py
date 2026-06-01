from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

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


@dataclass(frozen=True)
class AnnotationJsonError:
    jsonl_line: int
    error: str
    raw: str


@dataclass(frozen=True)
class AnnotationMissingField:
    jsonl_line: int
    field: str
    index: Optional[int] = None
    line_number: Optional[int] = None


@dataclass(frozen=True)
class AnnotationProblem:
    kind: str
    jsonl_line: int
    detail: str
    index: Optional[int] = None
    line_number: Optional[int] = None
    speaker: Optional[str] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class AnnotationSummary:
    path: Path
    total_lines: int
    valid_records: int
    json_errors: list[AnnotationJsonError] = field(default_factory=list)
    missing_fields: list[AnnotationMissingField] = field(default_factory=list)
    risk_level_counts: Counter[str] = field(default_factory=Counter)
    needs_verifier_counts: Counter[str] = field(default_factory=Counter)
    verifier_verdict_counts: Counter[str] = field(default_factory=Counter)
    verifier_enabled_counts: Counter[str] = field(default_factory=Counter)
    confidence_counts: Counter[str] = field(default_factory=Counter)
    problems: list[AnnotationProblem] = field(default_factory=list)

    @property
    def has_structural_errors(self) -> bool:
        return bool(self.json_errors or self.missing_fields)


ANNOTATION_REQUIRED_FIELDS = (
    "index",
    "line_number",
    "text",
    "speaker",
    "evidence_lines",
    "reason",
    "rejected_candidates",
    "confidence",
    "tool_summary",
    "risk",
    "verifier",
)


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


def summarize_annotations(annotations_path: Path) -> AnnotationSummary:
    resolved = annotations_path.expanduser().resolve()
    if not resolved.exists():
        raise QualityError(f"annotations file does not exist: {resolved}")
    if not resolved.is_file():
        raise QualityError(f"annotations path is not a file: {resolved}")

    total_lines = 0
    valid_records = 0
    json_errors: list[AnnotationJsonError] = []
    missing_fields: list[AnnotationMissingField] = []
    risk_level_counts: Counter[str] = Counter()
    needs_verifier_counts: Counter[str] = Counter()
    verifier_verdict_counts: Counter[str] = Counter()
    verifier_enabled_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    problems: list[AnnotationProblem] = []

    with resolved.open("r", encoding="utf-8") as file:
        for jsonl_line, raw_line in enumerate(file, start=1):
            raw = raw_line.rstrip("\n")
            if not raw.strip():
                continue
            total_lines += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                json_errors.append(
                    AnnotationJsonError(jsonl_line=jsonl_line, error=str(error), raw=raw)
                )
                problems.append(
                    AnnotationProblem(
                        kind="json_error",
                        jsonl_line=jsonl_line,
                        detail=str(error),
                    )
                )
                continue
            if not isinstance(row, dict):
                error = "annotation row must be a JSON object"
                json_errors.append(AnnotationJsonError(jsonl_line=jsonl_line, error=error, raw=raw))
                problems.append(AnnotationProblem(kind="json_error", jsonl_line=jsonl_line, detail=error))
                continue

            valid_records += 1
            index = _optional_int(row.get("index"))
            line_number = _optional_int(row.get("line_number"))
            missing = [field for field in ANNOTATION_REQUIRED_FIELDS if field not in row]
            for field in missing:
                missing_fields.append(
                    AnnotationMissingField(
                        jsonl_line=jsonl_line,
                        field=field,
                        index=index,
                        line_number=line_number,
                    )
                )
                problems.append(
                    AnnotationProblem(
                        kind="missing_field",
                        jsonl_line=jsonl_line,
                        detail=f"missing field: {field}",
                        index=index,
                        line_number=line_number,
                        speaker=_optional_str(row.get("speaker")),
                        text=_optional_str(row.get("text")),
                    )
                )

            risk = row.get("risk") if isinstance(row.get("risk"), dict) else {}
            verifier = row.get("verifier") if isinstance(row.get("verifier"), dict) else {}
            risk_level = _counter_key(risk.get("level"))
            needs_verifier = _counter_key(risk.get("needs_verifier"))
            verifier_verdict = _counter_key(verifier.get("verdict"))
            verifier_enabled = _counter_key(verifier.get("enabled"))

            risk_level_counts[risk_level] += 1
            needs_verifier_counts[needs_verifier] += 1
            verifier_verdict_counts[verifier_verdict] += 1
            verifier_enabled_counts[verifier_enabled] += 1
            confidence_counts[_counter_key(row.get("confidence"))] += 1

            if verifier_verdict in {"fail", "uncertain", "error"}:
                problems.append(
                    AnnotationProblem(
                        kind=f"verifier_{verifier_verdict}",
                        jsonl_line=jsonl_line,
                        detail=_optional_str(verifier.get("reason")) or f"verifier verdict: {verifier_verdict}",
                        index=index,
                        line_number=line_number,
                        speaker=_optional_str(row.get("speaker")),
                        text=_optional_str(row.get("text")),
                    )
                )
            if needs_verifier == "true" and verifier_verdict == "none":
                problems.append(
                    AnnotationProblem(
                        kind="missing_verifier_review",
                        jsonl_line=jsonl_line,
                        detail="risk.needs_verifier is true but verifier review is missing",
                        index=index,
                        line_number=line_number,
                        speaker=_optional_str(row.get("speaker")),
                        text=_optional_str(row.get("text")),
                    )
                )

    return AnnotationSummary(
        path=resolved,
        total_lines=total_lines,
        valid_records=valid_records,
        json_errors=json_errors,
        missing_fields=missing_fields,
        risk_level_counts=risk_level_counts,
        needs_verifier_counts=needs_verifier_counts,
        verifier_verdict_counts=verifier_verdict_counts,
        verifier_enabled_counts=verifier_enabled_counts,
        confidence_counts=confidence_counts,
        problems=problems,
    )


def render_annotation_summary(summary: AnnotationSummary, show_problems: int = 0) -> str:
    lines = [
        "Dialoop annotations summary",
        f"  path: {summary.path}",
        f"  total_lines: {summary.total_lines}",
        f"  valid_records: {summary.valid_records}",
        f"  json_errors: {len(summary.json_errors)}",
        f"  missing_fields: {len(summary.missing_fields)}",
        f"  risk.level: {_render_counter(summary.risk_level_counts, ['high', 'medium', 'low', 'none'])}",
        (
            "  risk.needs_verifier: "
            f"{_render_counter(summary.needs_verifier_counts, ['true', 'false', 'none'])}"
        ),
        (
            "  verifier.verdict: "
            f"{_render_counter(summary.verifier_verdict_counts, ['fail', 'error', 'uncertain', 'pass', 'none'])}"
        ),
        (
            "  verifier.enabled: "
            f"{_render_counter(summary.verifier_enabled_counts, ['true', 'false', 'none'])}"
        ),
        f"  confidence: {_render_counter(summary.confidence_counts, ['low', 'medium', 'high', 'none'])}",
        f"  problems: {len(summary.problems)}",
    ]
    if show_problems > 0 and summary.problems:
        lines.append("")
        lines.append("Problems:")
        for problem in summary.problems[:show_problems]:
            location = []
            if problem.index is not None:
                location.append(f"index={problem.index}")
            if problem.line_number is not None:
                location.append(f"line={problem.line_number}")
            if problem.speaker:
                location.append(f"speaker={problem.speaker}")
            prefix = f"  - jsonl_line={problem.jsonl_line} kind={problem.kind}"
            if location:
                prefix = f"{prefix} {' '.join(location)}"
            text = f" text={problem.text}" if problem.text else ""
            lines.append(f"{prefix} detail={problem.detail}{text}")
        hidden = len(summary.problems) - min(show_problems, len(summary.problems))
        if hidden > 0:
            lines.append(f"  ... {hidden} more problem(s) omitted")
    return "\n".join(lines)


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


def _optional_int(value: object) -> Optional[int]:
    if type(value) is int:
        return value
    return None


def _optional_str(value: object) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _counter_key(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "none"


def _render_counter(counter: Counter[str], preferred_order: list[str]) -> str:
    keys = [key for key in preferred_order if counter.get(key, 0)]
    keys.extend(sorted(key for key in counter if key not in preferred_order))
    if not keys:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in keys)
