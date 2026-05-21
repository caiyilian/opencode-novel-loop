from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dialoop.quality import (
    QualityError,
    evaluate_labels,
    extract_expected_dialogues,
    load_terms,
    render_error_labels,
    scan_terms,
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


if __name__ == "__main__":
    unittest.main()
