from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dialoop.quality import (
    QualityError,
    audit_coordinator_trace,
    attribute_mismatches,
    evaluate_labels,
    extract_expected_dialogues,
    load_terms,
    render_annotation_summary,
    render_coordinator_trace_audit,
    render_error_labels,
    render_mismatch_attribution_report,
    scan_terms,
    summarize_annotations,
)
from dialoop.quality_cli import main


ANSWER_TEXT = "\n".join(
    [
        "叙述。",
        "【甲】「你好。」【乙】「你好呀。」",
        "【甲】「再见。」",
    ]
)


class QualityTest(unittest.TestCase):
    def test_extract_expected_dialogues_from_answer_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer_path = Path(directory) / "answer.txt"
            answer_path.write_text(ANSWER_TEXT, encoding="utf-8")

            expected = extract_expected_dialogues(answer_path)

        self.assertEqual([item.speaker for item in expected], ["甲", "乙", "甲"])
        self.assertEqual([item.line_number for item in expected], [2, 2, 3])
        self.assertEqual(expected[1].text, "你好呀。")

    def test_extract_expected_dialogues_accepts_answer_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answers = Path(directory) / "answers"
            answers.mkdir()
            (answers / "b.txt").write_text("【乙】「第二句。」\n", encoding="utf-8")
            (answers / "a.txt").write_text("【甲】「第一句。」\n", encoding="utf-8")

            expected = extract_expected_dialogues(answers)

        self.assertEqual([item.speaker for item in expected], ["甲", "乙"])

    def test_evaluate_labels_reports_accuracy_and_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer_path = Path(directory) / "answer.txt"
            labels_path = Path(directory) / "labels.txt"
            answer_path.write_text(ANSWER_TEXT, encoding="utf-8")
            labels_path.write_text("甲\n甲\n", encoding="utf-8")

            report = evaluate_labels(answer_path=answer_path, labels_path=labels_path)

        self.assertEqual(report.expected_count, 3)
        self.assertEqual(report.label_count, 2)
        self.assertEqual(report.compared_count, 2)
        self.assertEqual(report.correct_count, 1)
        self.assertEqual(report.incorrect_count, 1)
        self.assertEqual(report.missing_count, 1)
        self.assertAlmostEqual(report.accuracy, 0.5)
        self.assertEqual(report.mismatches[0].line_number, 2)
        self.assertEqual(render_error_labels(report), '2行"「你好呀。」"，标注【甲】，答案【乙】')

    def test_evaluate_labels_reports_extra_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer_path = Path(directory) / "answer.txt"
            labels_path = Path(directory) / "labels.txt"
            answer_path.write_text("【甲】「你好。」\n", encoding="utf-8")
            labels_path.write_text("甲\n乙\n", encoding="utf-8")

            report = evaluate_labels(answer_path=answer_path, labels_path=labels_path)

        self.assertEqual(report.extra_labels, ["乙"])
        self.assertEqual(report.missing_count, 0)

    def test_evaluate_labels_accepts_pipe_separated_answer_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer_path = Path(directory) / "answer.txt"
            labels_path = Path(directory) / "labels.txt"
            answer_path.write_text("【甲|乙】「你好。」\n", encoding="utf-8")
            labels_path.write_text("乙\n", encoding="utf-8")

            report = evaluate_labels(answer_path=answer_path, labels_path=labels_path)

        self.assertEqual(report.correct_count, 1)
        self.assertEqual(report.incorrect_count, 0)

    def test_load_terms_deduplicates_terms_and_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            terms_path = Path(directory) / "terms.txt"
            terms_path.write_text("# comment\n甲\n乙\n甲\n", encoding="utf-8")

            terms = load_terms(terms=["乙", "丙"], terms_file=terms_path)

        self.assertEqual(terms, ["乙", "丙", "甲"])

    def test_load_terms_rejects_empty_input(self) -> None:
        with self.assertRaises(QualityError):
            load_terms()

    def test_scan_terms_finds_matches_in_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            source = root / "pkg" / "module.py"
            source.write_text("PROMPT = 'specific phrase'\n", encoding="utf-8")

            matches = scan_terms([root / "pkg"], ["specific phrase"])

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].line_number, 1)
        self.assertEqual(matches[0].term, "specific phrase")

    def test_quality_cli_evaluate_writes_error_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer_path = Path(directory) / "answer.txt"
            labels_path = Path(directory) / "labels.txt"
            error_path = Path(directory) / "errors.txt"
            answer_path.write_text(ANSWER_TEXT, encoding="utf-8")
            labels_path.write_text("甲\n甲\n甲\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate",
                        "--answers",
                        str(answer_path),
                        "--labels",
                        str(labels_path),
                        "--error-output",
                        str(error_path),
                    ]
                )
            error_text = error_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("accuracy: 66.67%", stdout.getvalue())
        self.assertIn("标注【甲】，答案【乙】", error_text)

    def test_quality_cli_scan_terms_returns_nonzero_on_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "module.py"
            source.write_text("bad_term = True\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["scan-terms", "--path", str(source), "--term", "bad_term"])

        self.assertEqual(exit_code, 1)
        self.assertIn("matches: 1", stdout.getvalue())

    def test_quality_cli_scan_terms_returns_zero_without_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "module.py"
            source.write_text("safe = True\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["scan-terms", "--path", str(source), "--term", "bad_term"])

        self.assertEqual(exit_code, 0)
        self.assertIn("matches: 0", stdout.getvalue())

    def test_summarize_annotations_counts_risk_verifier_and_structural_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.jsonl"
            rows = [
                annotation_row(index=0, risk_level="high", needs_verifier=True, verifier_verdict="pass"),
                annotation_row(index=1, risk_level="low", needs_verifier=False, verifier_verdict=None),
                annotation_row(index=2, risk_level="high", needs_verifier=True, verifier_verdict="error"),
            ]
            broken = dict(rows[0])
            broken.pop("risk")
            annotations_path.write_text(
                "\n".join(
                    [
                        json.dumps(rows[0], ensure_ascii=False),
                        json.dumps(rows[1], ensure_ascii=False),
                        json.dumps(rows[2], ensure_ascii=False),
                        json.dumps(broken, ensure_ascii=False),
                        "{bad json",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_annotations(annotations_path)
            rendered = render_annotation_summary(summary, show_problems=3)

        self.assertEqual(summary.total_lines, 5)
        self.assertEqual(summary.valid_records, 4)
        self.assertEqual(len(summary.json_errors), 1)
        self.assertEqual(len(summary.missing_fields), 1)
        self.assertEqual(summary.risk_level_counts["high"], 2)
        self.assertEqual(summary.risk_level_counts["none"], 1)
        self.assertEqual(summary.verifier_verdict_counts["error"], 1)
        self.assertTrue(summary.has_structural_errors)
        self.assertIn("verifier.verdict: error=1, pass=2, none=1", rendered)
        self.assertIn("kind=verifier_error", rendered)

    def test_quality_cli_annotations_summary_reports_problems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.jsonl"
            annotations_path.write_text(
                json.dumps(
                    annotation_row(index=0, risk_level="high", needs_verifier=True, verifier_verdict="uncertain"),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "annotations-summary",
                        "--annotations",
                        str(annotations_path),
                        "--show-problems",
                        "1",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Dialoop annotations summary", stdout.getvalue())
        self.assertIn("verifier.verdict: uncertain=1", stdout.getvalue())
        self.assertIn("kind=verifier_uncertain", stdout.getvalue())

    def test_audit_coordinator_trace_accepts_expected_risk_and_tool_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.jsonl"
            high = annotation_row(index=0, risk_level="high", needs_verifier=True, verifier_verdict="pass")
            high["coordinator_trace"] = [
                trace_event("labeler", "accepted"),
                trace_event("verifier", "called"),
                trace_event("verifier", "accepted"),
            ]
            low = annotation_row(index=1, risk_level="low", needs_verifier=False, verifier_verdict=None)
            low["coordinator_trace"] = [
                trace_event("labeler", "accepted"),
                trace_event("verifier", "skipped"),
            ]
            tool_row = annotation_row(index=2, risk_level="medium", needs_verifier=False, verifier_verdict=None)
            tool_row["tool_summary"] = {
                "resolve_identity": [{"verdict": "resolved", "recommended_speaker": "Stable Name"}],
                "normalize_speaker": [{"suggested_display_name": "Stable Name"}],
            }
            tool_row["coordinator_trace"] = [
                trace_event("labeler", "accepted"),
                trace_event("identity_resolver", "observed"),
                trace_event("normalizer", "observed"),
                trace_event("verifier", "skipped"),
            ]
            annotations_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in [high, low, tool_row]),
                encoding="utf-8",
            )

            audit = audit_coordinator_trace(annotations_path)
            rendered = render_coordinator_trace_audit(audit)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.records_with_trace, 3)
        self.assertEqual(audit.verifier_action_counts["called"], 1)
        self.assertEqual(audit.verifier_action_counts["skipped"], 2)
        self.assertEqual(audit.tool_call_counts["resolve_identity"], 1)
        self.assertEqual(audit.tool_observed_counts["resolve_identity"], 1)
        self.assertIn("problems: 0", rendered)

    def test_audit_coordinator_trace_reports_missing_verifier_and_tool_observed_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.jsonl"
            missing_trace = annotation_row(
                index=0,
                risk_level="low",
                needs_verifier=False,
                verifier_verdict=None,
            )
            missing_verifier = annotation_row(
                index=1,
                risk_level="high",
                needs_verifier=True,
                verifier_verdict="pass",
            )
            missing_verifier["coordinator_trace"] = [trace_event("labeler", "accepted")]
            missing_observed = annotation_row(
                index=2,
                risk_level="low",
                needs_verifier=False,
                verifier_verdict=None,
            )
            missing_observed["tool_summary"] = {
                "locate_identity": [{"candidate_count": 1}],
            }
            missing_observed["coordinator_trace"] = [
                trace_event("labeler", "accepted"),
                trace_event("verifier", "skipped"),
            ]
            annotations_path.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in [missing_trace, missing_verifier, missing_observed]
                ),
                encoding="utf-8",
            )

            audit = audit_coordinator_trace(annotations_path)
            rendered = render_coordinator_trace_audit(audit, show_problems=5)
            problem_kinds = {problem.kind for problem in audit.problems}

        self.assertFalse(audit.passed)
        self.assertIn("missing_coordinator_trace", problem_kinds)
        self.assertIn("missing_verifier_called_trace", problem_kinds)
        self.assertIn("missing_verifier_terminal_trace", problem_kinds)
        self.assertIn("missing_tool_observed_trace", problem_kinds)
        self.assertIn("Problems:", rendered)

    def test_quality_cli_coordinator_trace_reports_audit_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations_path = Path(directory) / "annotations.jsonl"
            row = annotation_row(index=0, risk_level="low", needs_verifier=False, verifier_verdict=None)
            row["coordinator_trace"] = [
                trace_event("labeler", "accepted"),
                trace_event("verifier", "skipped"),
            ]
            annotations_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["coordinator-trace", "--annotations", str(annotations_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("Dialoop coordinator trace audit", stdout.getvalue())
        self.assertIn("records_with_trace: 1", stdout.getvalue())
        self.assertIn("problems: 0", stdout.getvalue())

    def test_attribute_mismatches_groups_annotation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.txt"
            labels_path = root / "labels.txt"
            annotations_path = root / "annotations.jsonl"
            novel_path = root / "novel.txt"
            answers_path.write_text(
                "\u3010A\u3011\u300cwho?\u300d\u3010B\u3011\u300chi\u300d\n"
                "\u3010C\u3011\u300cthanks\u300d\n",
                encoding="utf-8",
            )
            labels_path.write_text("B\nA\nC\n", encoding="utf-8")
            novel_path.write_text("\u300cwho?\u300d\u300chi\u300d\n\u300cthanks\u300d\n", encoding="utf-8")
            rows = [
                {
                    "index": 0,
                    "confidence": "high",
                    "risk": {
                        "level": "high",
                        "needs_verifier": True,
                        "signals": [{"code": "short_question", "level": "medium"}],
                    },
                    "verifier": {"enabled": True, "verdict": "pass", "reason": "looks supported"},
                },
                {
                    "index": 1,
                    "confidence": "high",
                    "risk": {"level": "medium", "needs_verifier": False, "signals": []},
                    "verifier": None,
                },
                {
                    "index": 2,
                    "confidence": "high",
                    "risk": {"level": "low", "needs_verifier": False, "signals": []},
                    "verifier": None,
                },
            ]
            annotations_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )

            report = attribute_mismatches(
                answer_path=answers_path,
                labels_path=labels_path,
                annotations_path=annotations_path,
                novel_path=novel_path,
            )
            rendered = render_mismatch_attribution_report(report, max_errors=1)

        self.assertEqual(report.evaluation.incorrect_count, 2)
        self.assertEqual(report.risk_level_counts["high"], 1)
        self.assertEqual(report.risk_level_counts["medium"], 1)
        self.assertEqual(report.verifier_verdict_counts["pass"], 1)
        self.assertEqual(report.verifier_verdict_counts["none"], 1)
        self.assertEqual(report.category_counts["high_risk_verifier_pass"], 1)
        self.assertEqual(report.category_counts["medium_or_lower_risk_no_verifier"], 1)
        self.assertEqual(report.category_counts["high_confidence"], 2)
        self.assertIn("risk.level: high=1, medium=1", rendered)
        self.assertIn("verifier.verdict: pass=1, none=1", rendered)
        self.assertIn("risk.signals: none=1, short_question=1", rendered)
        self.assertIn("diagnostic_hints:", rendered)
        self.assertIn("same_line_multiple_dialogues", rendered)
        self.assertIn("index=0 line=1 expected=A actual=B risk=high verifier=pass", rendered)
        self.assertIn("... 1 more mismatch(es) omitted", rendered)

    def test_quality_cli_mismatch_attribution_reports_grouped_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.txt"
            labels_path = root / "labels.txt"
            annotations_path = root / "annotations.jsonl"
            answers_path.write_text(
                "\u3010A\u3011\u300cfirst\u300d\n\u3010B\u3011\u300csecond\u300d\n",
                encoding="utf-8",
            )
            labels_path.write_text("B\nB\n", encoding="utf-8")
            annotations_path.write_text(
                json.dumps(
                    {
                        "index": 0,
                        "confidence": "high",
                        "risk": {"level": "high", "needs_verifier": True, "signals": []},
                        "verifier": {"enabled": True, "verdict": "pass", "reason": "pass reason"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "mismatch-attribution",
                        "--answers",
                        str(answers_path),
                        "--labels",
                        str(labels_path),
                        "--annotations",
                        str(annotations_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Dialoop mismatch attribution", stdout.getvalue())
        self.assertIn("incorrect: 1", stdout.getvalue())
        self.assertIn("verifier.verdict: pass=1", stdout.getvalue())
        self.assertIn("verifier_reason=pass reason", stdout.getvalue())

def annotation_row(
    *,
    index: int,
    risk_level: str,
    needs_verifier: bool,
    verifier_verdict: str | None,
) -> dict:
    verifier = None
    if verifier_verdict is not None:
        verifier = {
            "enabled": True,
            "verdict": verifier_verdict,
            "reason": f"{verifier_verdict} reason",
            "counter_evidence_lines": [],
            "risk_signal_codes": ["sample_signal"],
        }
    return {
        "index": index,
        "line_number": index + 10,
        "text": f"dialogue {index}",
        "speaker": "甲",
        "evidence_lines": [index + 10],
        "reason": "sample reason",
        "rejected_candidates": [],
        "confidence": "high",
        "tool_summary": {},
        "recovery": None,
        "risk": {
            "level": risk_level,
            "needs_verifier": needs_verifier,
            "signals": [],
        },
        "verifier": verifier,
    }


def trace_event(agent: str, action: str) -> dict:
    return {
        "step": 1,
        "agent": agent,
        "action": action,
        "reason": f"{agent} {action}",
    }


if __name__ == "__main__":
    unittest.main()
