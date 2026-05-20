from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dialoop.cli import main
from dialoop.model_client import ChatResult, ToolCall


class FakeOpenAICompatibleClient:
    def __init__(self, _config):
        self.config = _config
        self.responses = [
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-read",
                        name="read_novel",
                        arguments={"start_line": 1, "end_line": 1},
                    )
                ],
            ),
            ChatResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-submit",
                        name="submit_labels",
                        arguments={"speakers": ["Lawrence"]},
                    )
                ],
            ),
        ]

    def chat(self, **_kwargs):
        if not self.responses:
            raise AssertionError("model was called more times than expected")
        return self.responses.pop(0)


class FakeTwoBatchClient:
    def __init__(self, _config):
        self.responses = [
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="read-1", name="read_novel", arguments={"start_line": 1, "end_line": 1})],
            ),
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="submit-1", name="submit_labels", arguments={"speakers": ["Lawrence"]})],
            ),
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="read-2", name="read_novel", arguments={"start_line": 2, "end_line": 2})],
            ),
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="submit-2", name="submit_labels", arguments={"speakers": ["Holo"]})],
            ),
        ]

    def chat(self, **_kwargs):
        if not self.responses:
            raise AssertionError("model was called more times than expected")
        return self.responses.pop(0)


class FakeInterruptedClient:
    def __init__(self, _config):
        self.responses = [
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="read-1", name="read_novel", arguments={"start_line": 1, "end_line": 1})],
            ),
            ChatResult(
                content="",
                tool_calls=[ToolCall(id="submit-1", name="submit_labels", arguments={"speakers": ["Lawrence"]})],
            ),
        ]

    def chat(self, **_kwargs):
        if self.responses:
            return self.responses.pop(0)
        raise KeyboardInterrupt()


class CliTest(unittest.TestCase):
    def test_dry_run_reports_python_and_model_without_opencode_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            novel_path.write_text("hello\n", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(novel_path),
                        "--dry-run",
                        "--protocol",
                        "json",
                        "--model-timeout",
                        "1",
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Dialoop dry run", output)
        self.assertIn("Environment:", output)
        self.assertIn("python: found", output)
        self.assertIn("context_window_lines: 80", output)
        self.assertIn("Model backend:", output)
        self.assertIn("protocol: json", output)
        self.assertNotIn("OpenCode:", output)
        self.assertNotIn("opencode:", output)
        self.assertNotIn("Install OpenCode", output)

    def test_run_without_dry_run_processes_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            output_path = Path(temp_dir) / "labels.txt"
            novel_path.write_text("Lawrence said: \u300cHello.\u300d\n", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("dialoop.cli.OpenAICompatibleClient", FakeOpenAICompatibleClient):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main([str(novel_path), "--output", str(output_path)])
            labels_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("submitted: true", stdout.getvalue())
        self.assertIn("iteration: 1/100", stdout.getvalue())
        self.assertIn("index: 0, line: 1", stdout.getvalue())
        self.assertEqual(labels_text, "Lawrence\n")

    def test_show_prompt_prints_prompt_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            output_path = Path(temp_dir) / "labels.txt"
            novel_path.write_text("Lawrence said: \u300cHello.\u300d\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch("dialoop.cli.OpenAICompatibleClient", FakeOpenAICompatibleClient):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            str(novel_path),
                            "--output",
                            str(output_path),
                            "--show-prompt",
                            "--context-window-lines",
                            "40",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertIn("Dialoop prompt:", stdout.getvalue())
        self.assertIn("--- system ---", stdout.getvalue())
        self.assertIn("--- user ---", stdout.getvalue())
        self.assertIn("read_novel(start_line=1, end_line=41)", stdout.getvalue())
        self.assertIn("Dialoop batch result:", stdout.getvalue())

    def test_max_iterations_one_processes_only_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            output_path = Path(temp_dir) / "labels.txt"
            novel_path.write_text("Lawrence said: \u300cHello.\u300d\nHolo said: \u300cHi.\u300d\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch("dialoop.cli.OpenAICompatibleClient", FakeTwoBatchClient):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            str(novel_path),
                            "--output",
                            str(output_path),
                            "--max-iterations",
                            "1",
                        ]
                    )
            labels_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(labels_text, "Lawrence\n")
        self.assertIn("Dialoop stopped after reaching --max-iterations=1.", stdout.getvalue())

    def test_max_iterations_runs_multiple_batches_until_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            output_path = Path(temp_dir) / "labels.txt"
            novel_path.write_text("Lawrence said: \u300cHello.\u300d\nHolo said: \u300cHi.\u300d\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch("dialoop.cli.OpenAICompatibleClient", FakeTwoBatchClient):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            str(novel_path),
                            "--output",
                            str(output_path),
                            "--max-iterations",
                            "4",
                        ]
                    )
            labels_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(labels_text, "Lawrence\nHolo\n")
        self.assertIn("iteration: 1/4", stdout.getvalue())
        self.assertIn("iteration: 2/4", stdout.getvalue())
        self.assertIn("Dialoop run complete.", stdout.getvalue())

    def test_keyboard_interrupt_preserves_completed_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            output_path = Path(temp_dir) / "labels.txt"
            novel_path.write_text("Lawrence said: \u300cHello.\u300d\nHolo said: \u300cHi.\u300d\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch("dialoop.cli.OpenAICompatibleClient", FakeInterruptedClient):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            str(novel_path),
                            "--output",
                            str(output_path),
                            "--max-iterations",
                            "4",
                        ]
                    )
            labels_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 130)
        self.assertEqual(labels_text, "Lawrence\n")
        self.assertIn("Dialoop interrupted", stdout.getvalue())
        self.assertIn("labeled: 1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
