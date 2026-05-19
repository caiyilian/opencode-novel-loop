from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dialoop.cli import main


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

    def test_run_without_dry_run_waits_for_independent_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "novel.txt"
            novel_path.write_text("hello\n", encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main([str(novel_path)])

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("independent Python agent loop is not implemented yet", stderr.getvalue())
        self.assertNotIn("OpenCode", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
