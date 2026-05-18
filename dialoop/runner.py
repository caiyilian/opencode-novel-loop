from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

from .environment import CommandStatus, EnvironmentStatus


DEFAULT_MODEL = "alibaba/qwen-plus"
COMPLETION_PROMISE = re.compile(r"<promise>\s*DONE\s*</promise>", re.IGNORECASE)


class ConfigError(ValueError):
    """Raised when user-provided CLI configuration is invalid."""


class RunnerError(RuntimeError):
    """Raised when the OpenCode loop cannot continue safely."""


@dataclass(frozen=True)
class DialoopConfig:
    novel_path: Path
    output_path: Path
    workdir: Path
    batch_size: int
    threshold: int
    max_iterations: int
    model: str
    opencode_template_path: Path
    get_dialogue_path: Path
    write_label_path: Path


@dataclass(frozen=True)
class RunResult:
    iterations: int
    completed: bool


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def opencode_template_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / "opencode.json"


def _resolve_existing_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ConfigError(f"novel file does not exist: {resolved}")
    if not resolved.is_file():
        raise ConfigError(f"novel path is not a file: {resolved}")
    return resolved


def _resolve_output_path(path: Path, workdir: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (workdir / expanded).resolve()


def build_config(
    novel_path: Path,
    output_path: Path,
    batch_size: int,
    threshold: int,
    max_iterations: int,
) -> DialoopConfig:
    resolved_novel = _resolve_existing_file(novel_path)
    workdir = resolved_novel.parent
    root = project_root()

    return DialoopConfig(
        novel_path=resolved_novel,
        output_path=_resolve_output_path(output_path, workdir),
        workdir=workdir,
        batch_size=batch_size,
        threshold=threshold,
        max_iterations=max_iterations,
        model=DEFAULT_MODEL,
        opencode_template_path=opencode_template_path(),
        get_dialogue_path=(root / "get_dialogue.py").resolve(),
        write_label_path=(root / "write_label.py").resolve(),
    )


def _format_command_status(status: CommandStatus) -> list[str]:
    state = "found" if status.available else "missing"
    lines = [f"{status.name}: {state}"]
    if status.executable:
        lines.append(f"  executable: {status.executable}")
    if status.version:
        lines.append(f"  version: {status.version}")
    if status.install_hint:
        lines.append(f"  install: {status.install_hint}")
    return lines


def render_status_report(config: DialoopConfig, environment: EnvironmentStatus, title: str) -> str:
    lines = [
        title,
        "",
        "Paths:",
        f"  novel: {config.novel_path}",
        f"  output: {config.output_path}",
        f"  workdir: {config.workdir}",
        f"  get_dialogue.py: {config.get_dialogue_path}",
        f"  write_label.py: {config.write_label_path}",
        "",
        "OpenCode:",
        f"  model: {config.model}",
        f"  config template: {config.opencode_template_path}",
        "",
        "Dialogue options:",
        f"  batch_size: {config.batch_size}",
        f"  threshold: {config.threshold}",
        f"  max_iterations: {config.max_iterations}",
        "",
        "Environment:",
    ]
    lines.extend(f"  {line}" for line in _format_command_status(environment.python))
    lines.extend(f"  {line}" for line in _format_command_status(environment.opencode))
    return "\n".join(lines)


def quote_for_prompt(path: Path) -> str:
    return f'"{path}"'


def get_dialogue_command(config: DialoopConfig) -> str:
    return " ".join(
        [
            "python",
            quote_for_prompt(config.get_dialogue_path),
            "--novel",
            quote_for_prompt(config.novel_path),
            "--labels",
            quote_for_prompt(config.output_path),
            "--batch-size",
            str(config.batch_size),
            "--threshold",
            str(config.threshold),
        ]
    )


def write_label_command(config: DialoopConfig) -> str:
    return " ".join(
        [
            "python",
            quote_for_prompt(config.write_label_path),
            "--labels",
            quote_for_prompt(config.output_path),
            "--name",
            "<角色名>",
        ]
    )


def initial_prompt(config: DialoopConfig) -> str:
    return "\n".join(
        [
            "You are labeling the speaker for Japanese-style novel dialogue.",
            "",
            "Workflow:",
            f"1. Run: {get_dialogue_command(config)}",
            "2. If it says `已经标注完毕`, output `<promise>DONE</promise>` and stop.",
            "3. Otherwise inspect the requested line(s) and nearby context in the novel file.",
            f"4. Append exactly one speaker name per dialogue, in order, by running: {write_label_command(config)}",
            "   For a batch, repeat `--name` once for each dialogue in the same order.",
            "5. Stop after one batch. Do not claim completion unless every dialogue is labeled.",
            "",
            "Rules:",
            "- Use the write_label.py command to write labels; do not edit the labels file directly.",
            "- Preserve the novel text unchanged.",
            "- If the speaker is ambiguous, choose the most likely character from context.",
            "- Only output `<promise>DONE</promise>` after get_dialogue.py reports there is nothing left to label.",
        ]
    )


def continuation_prompt(config: DialoopConfig, iteration: int) -> str:
    return "\n".join(
        [
            f"[DIALOOP CONTINUATION {iteration}/{config.max_iterations}]",
            "Continue the same labeling task from the current labeled.txt progress.",
            "",
            f"Run: {get_dialogue_command(config)}",
            "If it says `已经标注完毕`, output `<promise>DONE</promise>` and stop.",
            f"Otherwise label the returned batch with: {write_label_command(config)}",
            "For a batch, repeat `--name` once for each dialogue in the same order.",
            "Do not output `<promise>DONE</promise>` until all dialogues are labeled.",
        ]
    )


def opencode_command(config: DialoopConfig, environment: EnvironmentStatus, prompt: str, continue_session: bool) -> list[str]:
    if not environment.opencode.executable:
        raise RunnerError("OpenCode is not available. Run with --dry-run to see installation hints.")

    command = [
        environment.opencode.executable,
        "run",
        "--dangerously-skip-permissions",
        "--model",
        config.model,
        "--dir",
        str(config.workdir),
    ]
    if continue_session:
        command.append("--continue")
    command.append(prompt)
    return command


def run_opencode_once(
    config: DialoopConfig,
    environment: EnvironmentStatus,
    prompt: str,
    continue_session: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(config.opencode_template_path)

    command = opencode_command(config, environment, prompt, continue_session)
    return subprocess.run(
        command,
        cwd=str(config.workdir),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _write_process_output(completed: subprocess.CompletedProcess[str], output: TextIO) -> None:
    if completed.stdout:
        output.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            output.write("\n")
    if completed.stderr:
        output.write(completed.stderr)
        if not completed.stderr.endswith("\n"):
            output.write("\n")


def run_loop(
    config: DialoopConfig,
    environment: EnvironmentStatus,
    output: Optional[TextIO] = None,
) -> RunResult:
    stream = output if output is not None else sys.stdout

    if not environment.opencode.available or not environment.opencode.executable:
        hint = environment.opencode.install_hint or "Install OpenCode and make sure `opencode` is on PATH."
        raise RunnerError(f"OpenCode is missing. {hint}")

    for iteration in range(1, config.max_iterations + 1):
        prompt = initial_prompt(config) if iteration == 1 else continuation_prompt(config, iteration)
        continue_session = iteration > 1

        print(f"[dialoop] OpenCode iteration {iteration}/{config.max_iterations}", file=stream)
        completed = run_opencode_once(config, environment, prompt, continue_session)
        _write_process_output(completed, stream)

        transcript = "\n".join([completed.stdout or "", completed.stderr or ""])
        if completed.returncode != 0:
            raise RunnerError(f"OpenCode exited with code {completed.returncode}.")
        if COMPLETION_PROMISE.search(transcript):
            return RunResult(iterations=iteration, completed=True)

    raise RunnerError(f"Reached max iterations ({config.max_iterations}) without `<promise>DONE</promise>`.")
