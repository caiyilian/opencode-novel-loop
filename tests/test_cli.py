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

    def chat(self, **_kwargs):
        return ChatResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-submit",
                    name="submit_labels",
                    arguments={"speakers": ["Lawrence"]},
                )
            ],
        )


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
        self.assertEqual(labels_text, "Lawrence\n")


if __name__ == "__main__":
    unittest.main()
