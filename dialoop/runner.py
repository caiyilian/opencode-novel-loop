from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .environment import CommandStatus, EnvironmentStatus


DEFAULT_MODEL = "alibaba/qwen-plus"


class ConfigError(ValueError):
    """Raised when user-provided CLI configuration is invalid."""


@dataclass(frozen=True)
class DialoopConfig:
    novel_path: Path
    output_path: Path
    workdir: Path
    batch_size: int
    threshold: int
    model: str
    opencode_template_path: Path
    get_dialogue_path: Path
    write_label_path: Path


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


def render_dry_run_report(config: DialoopConfig, environment: EnvironmentStatus) -> str:
    lines = [
        "Dialoop dry run",
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
        "",
        "Environment:",
    ]
    lines.extend(f"  {line}" for line in _format_command_status(environment.python))
    lines.extend(f"  {line}" for line in _format_command_status(environment.opencode))
    return "\n".join(lines)
