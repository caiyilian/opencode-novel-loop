from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CommandStatus:
    name: str
    available: bool
    executable: Optional[str] = None
    version: Optional[str] = None
    install_hint: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentStatus:
    python: CommandStatus
    opencode: CommandStatus


def _command_version(command: str, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            [command, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = (completed.stdout or completed.stderr).strip()
    if not output:
        return None
    return output.splitlines()[0]


def _opencode_install_hint() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "Install OpenCode with `npm install -g opencode-ai`, `scoop install opencode`, or `choco install opencode`."
    if system == "darwin":
        return "Install OpenCode with `brew install anomalyco/tap/opencode` or `curl -fsSL https://opencode.ai/install | bash`."
    return "Install OpenCode with `curl -fsSL https://opencode.ai/install | bash` or `npm install -g opencode-ai`."


def check_python() -> CommandStatus:
    executable = Path(sys.executable).resolve()
    return CommandStatus(
        name="python",
        available=True,
        executable=str(executable),
        version=platform.python_version(),
    )


def check_opencode() -> CommandStatus:
    executable = shutil.which("opencode")
    if not executable:
        return CommandStatus(
            name="opencode",
            available=False,
            install_hint=_opencode_install_hint(),
        )

    return CommandStatus(
        name="opencode",
        available=True,
        executable=executable,
        version=_command_version(executable, "--version"),
    )


def check_environment() -> EnvironmentStatus:
    return EnvironmentStatus(
        python=check_python(),
        opencode=check_opencode(),
    )
