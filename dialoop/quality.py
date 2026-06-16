from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .identity import should_coordinate_identity_lookup
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


@dataclass(frozen=True)
class CoordinatorTraceAudit:
    path: Path
    total_lines: int
    valid_records: int
    json_errors: list[AnnotationJsonError] = field(default_factory=list)
    records_with_trace: int = 0
    risk_level_counts: Counter[str] = field(default_factory=Counter)
    needs_verifier_counts: Counter[str] = field(default_factory=Counter)
    verifier_action_counts: Counter[str] = field(default_factory=Counter)
    verifier_action_by_risk: Counter[str] = field(default_factory=Counter)
    identity_locator_action_counts: Counter[str] = field(default_factory=Counter)
    identity_resolver_action_counts: Counter[str] = field(default_factory=Counter)
    tool_call_counts: Counter[str] = field(default_factory=Counter)
    tool_observed_counts: Counter[str] = field(default_factory=Counter)
    problems: list[AnnotationProblem] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.json_errors and not self.problems


@dataclass(frozen=True)
class AnnotationAttribution:
    jsonl_line: int
    risk_level: str
    needs_verifier: str
    verifier_verdict: str
    verifier_reason: Optional[str]
    confidence: str
    risk_signal_codes: list[str]
    identity_triggered: str = "none"
    identity_verdict: str = "none"
    identity_recommended_speaker: Optional[str] = None
    identity_evidence_lines: list[int] = field(default_factory=list)
    blocked_identity_review_count: int = 0


@dataclass(frozen=True)
class MismatchAttribution:
    mismatch: LabelMismatch
    annotation: Optional[AnnotationAttribution]
    diagnostic_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MismatchAttributionReport:
    evaluation: EvaluationReport
    annotations_path: Path
    annotation_records: int
    items: list[MismatchAttribution] = field(default_factory=list)
    missing_annotation_indices: list[int] = field(default_factory=list)
    risk_level_counts: Counter[str] = field(default_factory=Counter)
    verifier_verdict_counts: Counter[str] = field(default_factory=Counter)
    confidence_counts: Counter[str] = field(default_factory=Counter)
    risk_signal_counts: Counter[str] = field(default_factory=Counter)
    diagnostic_hint_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)

    @property
    def matched_annotations(self) -> int:
        return len(self.items) - len(self.missing_annotation_indices)


@dataclass(frozen=True)
class VerifierFalsePassReport:
    attribution: MismatchAttributionReport
    items: list[MismatchAttribution] = field(default_factory=list)

    @property
    def total_false_passes(self) -> int:
        return len(self.items)

    @property
    def high_risk_false_passes(self) -> int:
        return sum(
            1
            for item in self.items
            if item.annotation is not None and item.annotation.risk_level == "high"
        )


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
COORDINATOR_TRACE_TOOL_AGENTS = {
    "locate_identity": "identity_locator",
    "resolve_identity": "identity_resolver",
    "normalize_speaker": "normalizer",
    "arbitrate_identity": "arbiter",
}

DIAGNOSTIC_SECOND_PERSON_MARKERS = ("\u4f60", "\u60a8", "\u6c5d")
DIAGNOSTIC_PUNCTUATION_CHARS = set(
    " \t\r\n.,!?;:'\"()[]{}<>~"
    + "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u201c\u201d\u2018\u2019"
    + "\uff08\uff09\u3010\u3011\u300a\u300b\u2026\u2014"
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


def audit_coordinator_trace(
    annotations_path: Path,
    verifier_mode: str = "risk",
) -> CoordinatorTraceAudit:
    if verifier_mode not in {"off", "risk", "all"}:
        raise QualityError(f"unsupported verifier_mode: {verifier_mode}")

    resolved = annotations_path.expanduser().resolve()
    if not resolved.exists():
        raise QualityError(f"annotations file does not exist: {resolved}")
    if not resolved.is_file():
        raise QualityError(f"annotations path is not a file: {resolved}")

    total_lines = 0
    valid_records = 0
    records_with_trace = 0
    json_errors: list[AnnotationJsonError] = []
    problems: list[AnnotationProblem] = []
    risk_level_counts: Counter[str] = Counter()
    needs_verifier_counts: Counter[str] = Counter()
    verifier_action_counts: Counter[str] = Counter()
    verifier_action_by_risk: Counter[str] = Counter()
    identity_locator_action_counts: Counter[str] = Counter()
    identity_resolver_action_counts: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()
    tool_observed_counts: Counter[str] = Counter()

    with resolved.open("r", encoding="utf-8") as file:
        for jsonl_line, raw_line in enumerate(file, start=1):
            raw = raw_line.rstrip("\n")
            if not raw.strip():
                continue
            total_lines += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                json_errors.append(AnnotationJsonError(jsonl_line=jsonl_line, error=str(error), raw=raw))
                problems.append(AnnotationProblem(kind="json_error", jsonl_line=jsonl_line, detail=str(error)))
                continue
            if not isinstance(row, dict):
                detail = "annotation row must be a JSON object"
                json_errors.append(AnnotationJsonError(jsonl_line=jsonl_line, error=detail, raw=raw))
                problems.append(AnnotationProblem(kind="json_error", jsonl_line=jsonl_line, detail=detail))
                continue

            valid_records += 1
            index = _optional_int(row.get("index"))
            line_number = _optional_int(row.get("line_number"))
            speaker = _optional_str(row.get("speaker"))
            text = _optional_str(row.get("text"))
            risk = row.get("risk") if isinstance(row.get("risk"), dict) else {}
            risk_level = _counter_key(risk.get("level"))
            needs_verifier = _counter_key(risk.get("needs_verifier"))
            risk_level_counts[risk_level] += 1
            needs_verifier_counts[needs_verifier] += 1
            tool_summary = row.get("tool_summary") if isinstance(row.get("tool_summary"), dict) else {}
            called_trace_tools = []
            for tool_name, agent_name in COORDINATOR_TRACE_TOOL_AGENTS.items():
                calls = tool_summary.get(tool_name)
                if not isinstance(calls, list) or not calls:
                    continue
                tool_call_counts[tool_name] += 1
                called_trace_tools.append((tool_name, agent_name))

            trace = row.get("coordinator_trace")
            if not isinstance(trace, list) or not trace:
                problems.append(
                    _trace_problem(
                        kind="missing_coordinator_trace",
                        jsonl_line=jsonl_line,
                        detail="coordinator_trace is missing or empty",
                        index=index,
                        line_number=line_number,
                        speaker=speaker,
                        text=text,
                    )
                )
                continue
            records_with_trace += 1

            events = [event for event in trace if isinstance(event, dict)]
            if len(events) != len(trace):
                problems.append(
                    _trace_problem(
                        kind="invalid_coordinator_trace_event",
                        jsonl_line=jsonl_line,
                        detail="coordinator_trace contains a non-object event",
                        index=index,
                        line_number=line_number,
                        speaker=speaker,
                        text=text,
                    )
                )

            action_pairs = {
                (_optional_str(event.get("agent")) or "none", _optional_str(event.get("action")) or "none")
                for event in events
            }
            if ("labeler", "accepted") not in action_pairs:
                problems.append(
                    _trace_problem(
                        kind="missing_labeler_trace",
                        jsonl_line=jsonl_line,
                        detail="coordinator_trace does not include labeler accepted",
                        index=index,
                        line_number=line_number,
                        speaker=speaker,
                        text=text,
                    )
                )

            verifier_actions = []
            for event in events:
                agent = _optional_str(event.get("agent")) or "none"
                action = _optional_str(event.get("action")) or "none"
                if agent == "verifier":
                    verifier_actions.append(action)
                    verifier_action_counts[action] += 1
                    verifier_action_by_risk[f"{risk_level}:{action}"] += 1
                elif agent == "identity_locator":
                    identity_locator_action_counts[action] += 1
                elif agent == "identity_resolver":
                    identity_resolver_action_counts[action] += 1
            problems.extend(
                _coordinator_verifier_problems(
                    verifier_mode=verifier_mode,
                    needs_verifier=needs_verifier,
                    verifier_actions=verifier_actions,
                    jsonl_line=jsonl_line,
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )

            for tool_name, agent_name in called_trace_tools:
                if (agent_name, "observed") in action_pairs:
                    tool_observed_counts[tool_name] += 1
                    continue
                problems.append(
                    _trace_problem(
                        kind="missing_tool_observed_trace",
                        jsonl_line=jsonl_line,
                        detail=f"{tool_name} was called but {agent_name} observed trace is missing",
                        index=index,
                        line_number=line_number,
                        speaker=speaker,
                        text=text,
                    )
                )

    return CoordinatorTraceAudit(
        path=resolved,
        total_lines=total_lines,
        valid_records=valid_records,
        json_errors=json_errors,
        records_with_trace=records_with_trace,
        risk_level_counts=risk_level_counts,
        needs_verifier_counts=needs_verifier_counts,
        verifier_action_counts=verifier_action_counts,
        verifier_action_by_risk=verifier_action_by_risk,
        identity_locator_action_counts=identity_locator_action_counts,
        identity_resolver_action_counts=identity_resolver_action_counts,
        tool_call_counts=tool_call_counts,
        tool_observed_counts=tool_observed_counts,
        problems=problems,
    )


def render_coordinator_trace_audit(audit: CoordinatorTraceAudit, show_problems: int = 0) -> str:
    lines = [
        "Dialoop coordinator trace audit",
        f"  path: {audit.path}",
        f"  total_lines: {audit.total_lines}",
        f"  valid_records: {audit.valid_records}",
        f"  json_errors: {len(audit.json_errors)}",
        f"  records_with_trace: {audit.records_with_trace}",
        f"  risk.level: {_render_counter(audit.risk_level_counts, ['high', 'medium', 'low', 'none'])}",
        (
            "  risk.needs_verifier: "
            f"{_render_counter(audit.needs_verifier_counts, ['true', 'false', 'none'])}"
        ),
        (
            "  verifier.trace_actions: "
            f"{_render_counter(audit.verifier_action_counts, ['called', 'accepted', 'rejected', 'uncertain', 'skipped'])}"
        ),
        (
            "  identity_locator.trace_actions: "
            f"{_render_counter(audit.identity_locator_action_counts, ['called', 'located', 'not_enough_evidence', 'observed'])}"
        ),
        (
            "  identity_resolver.trace_actions: "
            f"{_render_counter(audit.identity_resolver_action_counts, ['called', 'resolved', 'not_same_person', 'not_enough_evidence', 'observed'])}"
        ),
        f"  verifier.by_risk: {_render_counter(audit.verifier_action_by_risk, _verifier_by_risk_order())}",
        f"  tool_calls: {_render_counter(audit.tool_call_counts, list(COORDINATOR_TRACE_TOOL_AGENTS))}",
        (
            "  tool_observed: "
            f"{_render_counter(audit.tool_observed_counts, list(COORDINATOR_TRACE_TOOL_AGENTS))}"
        ),
        f"  problems: {len(audit.problems)}",
    ]
    if show_problems > 0 and audit.problems:
        lines.append("")
        lines.append("Problems:")
        for problem in audit.problems[:show_problems]:
            lines.append(_render_problem(problem))
        hidden = len(audit.problems) - min(show_problems, len(audit.problems))
        if hidden > 0:
            lines.append(f"  ... {hidden} more problem(s) omitted")
    return "\n".join(lines)


def _coordinator_verifier_problems(
    *,
    verifier_mode: str,
    needs_verifier: str,
    verifier_actions: list[str],
    jsonl_line: int,
    index: Optional[int],
    line_number: Optional[int],
    speaker: Optional[str],
    text: Optional[str],
) -> list[AnnotationProblem]:
    terminal_actions = {"accepted", "rejected", "uncertain"}
    problems: list[AnnotationProblem] = []
    if verifier_mode == "off":
        if "skipped" not in verifier_actions:
            problems.append(
                _trace_problem(
                    kind="missing_verifier_skipped_trace",
                    jsonl_line=jsonl_line,
                    detail="verifier_mode=off but verifier skipped trace is missing",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
        return problems
    if verifier_mode == "all":
        if "called" not in verifier_actions:
            problems.append(
                _trace_problem(
                    kind="missing_verifier_called_trace",
                    jsonl_line=jsonl_line,
                    detail="verifier_mode=all but verifier called trace is missing",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
        if not any(action in terminal_actions for action in verifier_actions):
            problems.append(
                _trace_problem(
                    kind="missing_verifier_terminal_trace",
                    jsonl_line=jsonl_line,
                    detail="verifier was expected to finish with accepted/rejected/uncertain",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
        return problems
    if needs_verifier == "true":
        if "called" not in verifier_actions:
            problems.append(
                _trace_problem(
                    kind="missing_verifier_called_trace",
                    jsonl_line=jsonl_line,
                    detail="risk.needs_verifier=true but verifier called trace is missing",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
        if not any(action in terminal_actions for action in verifier_actions):
            problems.append(
                _trace_problem(
                    kind="missing_verifier_terminal_trace",
                    jsonl_line=jsonl_line,
                    detail="risk.needs_verifier=true but verifier accepted/rejected/uncertain trace is missing",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
    elif needs_verifier == "false":
        if "skipped" not in verifier_actions:
            problems.append(
                _trace_problem(
                    kind="missing_verifier_skipped_trace",
                    jsonl_line=jsonl_line,
                    detail="risk.needs_verifier=false but verifier skipped trace is missing",
                    index=index,
                    line_number=line_number,
                    speaker=speaker,
                    text=text,
                )
            )
    return problems


def _trace_problem(
    *,
    kind: str,
    jsonl_line: int,
    detail: str,
    index: Optional[int],
    line_number: Optional[int],
    speaker: Optional[str],
    text: Optional[str],
) -> AnnotationProblem:
    return AnnotationProblem(
        kind=kind,
        jsonl_line=jsonl_line,
        detail=detail,
        index=index,
        line_number=line_number,
        speaker=speaker,
        text=text,
    )


def _render_problem(problem: AnnotationProblem) -> str:
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
    return f"{prefix} detail={problem.detail}{text}"


def _verifier_by_risk_order() -> list[str]:
    return [
        f"{risk}:{action}"
        for risk in ("high", "medium", "low", "none")
        for action in ("called", "accepted", "rejected", "uncertain", "skipped")
    ]


def attribute_mismatches(
    *,
    answer_path: Path,
    labels_path: Path,
    annotations_path: Path,
    novel_path: Optional[Path] = None,
) -> MismatchAttributionReport:
    evaluation = evaluate_labels(answer_path=answer_path, labels_path=labels_path, novel_path=novel_path)
    labels = load_label_lines(labels_path)
    annotations = load_annotation_attributions(annotations_path)
    line_dialogue_counts: Counter[int] = Counter()
    if novel_path is not None:
        line_dialogue_counts = Counter(
            dialogue.line_number for dialogue in DialogueIndex.from_file(novel_path).dialogues
        )

    items: list[MismatchAttribution] = []
    missing_annotation_indices: list[int] = []
    risk_level_counts: Counter[str] = Counter()
    verifier_verdict_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    risk_signal_counts: Counter[str] = Counter()
    diagnostic_hint_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for mismatch in evaluation.mismatches:
        annotation = annotations.get(mismatch.index)
        if annotation is None:
            missing_annotation_indices.append(mismatch.index)
            risk_level = "missing"
            verifier_verdict = "missing"
            confidence = "missing"
            risk_signal_codes: list[str] = []
        else:
            risk_level = annotation.risk_level
            verifier_verdict = annotation.verifier_verdict
            confidence = annotation.confidence
            risk_signal_codes = annotation.risk_signal_codes

        hints = mismatch_diagnostic_hints(
            mismatch=mismatch,
            annotation=annotation,
            labels=labels,
            line_dialogue_counts=line_dialogue_counts,
        )
        item = MismatchAttribution(mismatch=mismatch, annotation=annotation, diagnostic_hints=hints)
        items.append(item)

        risk_level_counts[risk_level] += 1
        verifier_verdict_counts[verifier_verdict] += 1
        confidence_counts[confidence] += 1
        diagnostic_hint_counts.update(hints or ["none"])
        if risk_signal_codes:
            risk_signal_counts.update(risk_signal_codes)
        else:
            risk_signal_counts["none"] += 1
        category_counts.update(_mismatch_categories(annotation, mismatch))

    return MismatchAttributionReport(
        evaluation=evaluation,
        annotations_path=annotations_path.expanduser().resolve(),
        annotation_records=len(annotations),
        items=items,
        missing_annotation_indices=missing_annotation_indices,
        risk_level_counts=risk_level_counts,
        verifier_verdict_counts=verifier_verdict_counts,
        confidence_counts=confidence_counts,
        risk_signal_counts=risk_signal_counts,
        diagnostic_hint_counts=diagnostic_hint_counts,
        category_counts=category_counts,
    )


def load_annotation_attributions(annotations_path: Path) -> dict[int, AnnotationAttribution]:
    resolved = annotations_path.expanduser().resolve()
    if not resolved.exists():
        raise QualityError(f"annotations file does not exist: {resolved}")
    if not resolved.is_file():
        raise QualityError(f"annotations path is not a file: {resolved}")

    annotations: dict[int, AnnotationAttribution] = {}
    with resolved.open("r", encoding="utf-8") as file:
        for jsonl_line, raw_line in enumerate(file, start=1):
            raw = raw_line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise QualityError(f"annotation jsonl line {jsonl_line} is invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise QualityError(f"annotation jsonl line {jsonl_line} must be a JSON object")
            index = _optional_int(row.get("index"))
            if index is None:
                raise QualityError(f"annotation jsonl line {jsonl_line} has no integer index")
            if index in annotations:
                raise QualityError(f"duplicate annotation index: {index}")

            risk = row.get("risk") if isinstance(row.get("risk"), dict) else {}
            verifier = row.get("verifier") if isinstance(row.get("verifier"), dict) else {}
            identity = row.get("identity") if isinstance(row.get("identity"), dict) else {}
            recovery = row.get("recovery") if isinstance(row.get("recovery"), dict) else {}
            blocked_identity_reviews = _blocked_identity_reviews(recovery)
            annotations[index] = AnnotationAttribution(
                jsonl_line=jsonl_line,
                risk_level=_counter_key(risk.get("level")),
                needs_verifier=_counter_key(risk.get("needs_verifier")),
                verifier_verdict=_counter_key(verifier.get("verdict")),
                verifier_reason=_optional_str(verifier.get("reason")),
                confidence=_counter_key(row.get("confidence")),
                risk_signal_codes=_risk_signal_codes(risk),
                identity_triggered=_counter_key(identity.get("triggered")),
                identity_verdict=_counter_key(identity.get("verdict")),
                identity_recommended_speaker=_optional_str(identity.get("recommended_speaker")),
                identity_evidence_lines=_line_numbers(identity.get("evidence_lines")),
                blocked_identity_review_count=len(blocked_identity_reviews),
            )
    return annotations


def mismatch_diagnostic_hints(
    *,
    mismatch: LabelMismatch,
    annotation: Optional[AnnotationAttribution],
    labels: list[str],
    line_dialogue_counts: Counter[int],
) -> list[str]:
    hints: list[str] = []
    existing_signals = set(annotation.risk_signal_codes if annotation else [])
    semantic_length = _diagnostic_semantic_length(mismatch.text)

    if semantic_length <= 4 and not (
        {"short_dialogue", "very_short_dialogue"} & existing_signals
    ):
        hints.append("short_dialogue_hint")
    if ("?" in mismatch.text or "\uff1f" in mismatch.text) and semantic_length <= 8:
        if "short_question" not in existing_signals:
            hints.append("short_question_hint")
    if any(marker in mismatch.text for marker in DIAGNOSTIC_SECOND_PERSON_MARKERS):
        if "second_person_address" not in existing_signals:
            hints.append("second_person_hint")
    if line_dialogue_counts.get(mismatch.line_number, 0) > 1:
        hints.append("same_line_multiple_dialogues")
    if _adjacent_label_matches_expected(mismatch, labels):
        hints.append("adjacent_turn_order")
    if annotation is not None and annotation.confidence == "high":
        hints.append("high_confidence_wrong")
    if annotation is None or annotation.verifier_verdict == "none":
        hints.append("no_verifier_review")

    return hints


def render_mismatch_attribution_report(
    report: MismatchAttributionReport,
    max_errors: Optional[int] = 50,
) -> str:
    evaluation = report.evaluation
    lines = [
        "Dialoop mismatch attribution",
        f"  expected: {evaluation.expected_count}",
        f"  labels: {evaluation.label_count}",
        f"  compared: {evaluation.compared_count}",
        f"  correct: {evaluation.correct_count}",
        f"  incorrect: {evaluation.incorrect_count}",
        f"  accuracy: {evaluation.accuracy * 100:.2f}%",
        f"  annotations: {report.annotations_path}",
        f"  annotation_records: {report.annotation_records}",
        f"  matched_mismatches: {report.matched_annotations}",
        f"  missing_annotations: {len(report.missing_annotation_indices)}",
    ]
    if evaluation.novel_dialogue_count is not None:
        lines.append(f"  novel_dialogues: {evaluation.novel_dialogue_count}")
    lines.extend(
        [
            "",
            "Mismatch groups:",
            f"  risk.level: {_render_counter(report.risk_level_counts, ['high', 'medium', 'low', 'none', 'missing'])}",
            (
                "  verifier.verdict: "
                f"{_render_counter(report.verifier_verdict_counts, ['fail', 'error', 'uncertain', 'pass', 'none', 'missing'])}"
            ),
            f"  confidence: {_render_counter(report.confidence_counts, ['low', 'medium', 'high', 'none', 'missing'])}",
            f"  risk.signals: {_render_counter(report.risk_signal_counts, ['none'])}",
            f"  diagnostic_hints: {_render_counter(report.diagnostic_hint_counts, ['none'])}",
            (
                "  categories: "
                f"{_render_counter(report.category_counts, ['identity_related', 'medium_or_lower_risk_no_verifier', 'high_risk_verifier_pass', 'verifier_uncertain', 'high_confidence', 'missing_annotation'])}"
            ),
        ]
    )
    if report.items:
        lines.append("")
        lines.append("Mismatches:")
        shown = report.items if max_errors is None else report.items[:max_errors]
        for item in shown:
            mismatch = item.mismatch
            annotation = item.annotation
            risk = annotation.risk_level if annotation is not None else "missing"
            verifier = annotation.verifier_verdict if annotation is not None else "missing"
            confidence = annotation.confidence if annotation is not None else "missing"
            signals = ",".join(annotation.risk_signal_codes) if annotation and annotation.risk_signal_codes else "none"
            hints = ",".join(item.diagnostic_hints) if item.diagnostic_hints else "none"
            verifier_reason = annotation.verifier_reason if annotation and annotation.verifier_reason else "none"
            lines.append(
                f"  - index={mismatch.index} line={mismatch.line_number} "
                f"expected={mismatch.expected} actual={mismatch.actual} "
                f"risk={risk} verifier={verifier} confidence={confidence} "
                f"signals={signals} hints={hints} verifier_reason={verifier_reason} "
                f"text={mismatch.text}"
            )
        hidden = len(report.items) - len(shown)
        if hidden > 0:
            lines.append(f"  ... {hidden} more mismatch(es) omitted")
    return "\n".join(lines)


def report_verifier_false_passes(
    *,
    answer_path: Path,
    labels_path: Path,
    annotations_path: Path,
    novel_path: Optional[Path] = None,
) -> VerifierFalsePassReport:
    attribution = attribute_mismatches(
        answer_path=answer_path,
        labels_path=labels_path,
        annotations_path=annotations_path,
        novel_path=novel_path,
    )
    items = [
        item
        for item in attribution.items
        if item.annotation is not None and item.annotation.verifier_verdict == "pass"
    ]
    return VerifierFalsePassReport(attribution=attribution, items=items)


def render_verifier_false_pass_report(
    report: VerifierFalsePassReport,
    max_errors: Optional[int] = 50,
) -> str:
    attribution = report.attribution
    evaluation = attribution.evaluation
    lines = [
        "Dialoop verifier false-pass report",
        f"  expected: {evaluation.expected_count}",
        f"  labels: {evaluation.label_count}",
        f"  incorrect: {evaluation.incorrect_count}",
        f"  annotations: {attribution.annotations_path}",
        f"  annotation_records: {attribution.annotation_records}",
        f"  verifier_false_passes: {report.total_false_passes}",
        f"  high_risk_verifier_pass: {report.high_risk_false_passes}",
    ]
    if evaluation.novel_dialogue_count is not None:
        lines.append(f"  novel_dialogues: {evaluation.novel_dialogue_count}")
    lines.extend(
        [
            "",
            "False-pass samples:",
        ]
    )
    if not report.items:
        lines.append("  none")
        return "\n".join(lines)

    shown = report.items if max_errors is None else report.items[:max_errors]
    for item in shown:
        mismatch = item.mismatch
        annotation = item.annotation
        risk = annotation.risk_level if annotation is not None else "missing"
        signals = ",".join(annotation.risk_signal_codes) if annotation and annotation.risk_signal_codes else "none"
        hints = ",".join(item.diagnostic_hints) if item.diagnostic_hints else "none"
        verifier_reason = annotation.verifier_reason if annotation and annotation.verifier_reason else "none"
        lines.append(
            f"  - index={mismatch.index} line={mismatch.line_number} "
            f"expected={mismatch.expected} actual={mismatch.actual} "
            f"risk={risk} signals={signals} hints={hints} "
            f"verifier_reason={verifier_reason} text={mismatch.text}"
        )
    hidden = len(report.items) - len(shown)
    if hidden > 0:
        lines.append(f"  ... {hidden} more false-pass sample(s) omitted")
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


def _risk_signal_codes(risk: dict[str, Any]) -> list[str]:
    signals = risk.get("signals")
    if not isinstance(signals, list):
        return []
    codes: list[str] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        code = _optional_str(signal.get("code"))
        if code:
            codes.append(code)
    return codes


def _blocked_identity_reviews(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    blocked_reviews = recovery.get("blocked_reviews")
    if not isinstance(blocked_reviews, list):
        return []
    return [
        review["identity"]
        for review in blocked_reviews
        if isinstance(review, dict) and isinstance(review.get("identity"), dict)
    ]


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


def _mismatch_categories(annotation: Optional[AnnotationAttribution], mismatch: LabelMismatch) -> list[str]:
    if annotation is None:
        return ["missing_annotation"]
    categories: list[str] = []
    if _identity_related_mismatch(annotation, mismatch):
        categories.append("identity_related")
    if annotation.risk_level in {"low", "medium"} and annotation.verifier_verdict == "none":
        categories.append("medium_or_lower_risk_no_verifier")
    if annotation.risk_level == "high" and annotation.verifier_verdict == "pass":
        categories.append("high_risk_verifier_pass")
    if annotation.verifier_verdict == "uncertain":
        categories.append("verifier_uncertain")
    if annotation.confidence == "high":
        categories.append("high_confidence")
    return categories or ["other"]


def _identity_related_mismatch(annotation: AnnotationAttribution, mismatch: LabelMismatch) -> bool:
    if should_coordinate_identity_lookup(mismatch.actual):
        return True
    if annotation.identity_triggered == "true":
        return True
    if annotation.identity_verdict in {"resolved", "not_same_person", "not_enough_evidence"}:
        return True
    return annotation.blocked_identity_review_count > 0


def _diagnostic_semantic_length(text: str) -> int:
    return sum(1 for char in text if char not in DIAGNOSTIC_PUNCTUATION_CHARS)


def _adjacent_label_matches_expected(mismatch: LabelMismatch, labels: list[str]) -> bool:
    expected_speakers = [speaker.strip() for speaker in mismatch.expected.split("|") if speaker.strip()]
    if mismatch.index > 0 and labels[mismatch.index - 1] in expected_speakers:
        return True
    next_index = mismatch.index + 1
    return next_index < len(labels) and labels[next_index] in expected_speakers


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
