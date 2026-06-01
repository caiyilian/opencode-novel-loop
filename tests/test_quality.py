from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dialoop.quality import (
    QualityError,
    evaluate_labels,
    extract_expected_dialogues,
    load_terms,
    render_annotation_summary,
    render_error_labels,
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


if __name__ == "__main__":
    unittest.main()
