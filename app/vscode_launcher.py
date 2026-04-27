"""Utilities for opening the current project in VS Code."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import VSCODE_COMMAND

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def launch_vscode() -> str:
    project_path = str(PROJECT_ROOT)
    commands: list[list[str]] = []

    if VSCODE_COMMAND:
        commands.append([*shlex.split(VSCODE_COMMAND), "-r", project_path])
    elif shutil.which("code"):
        commands.append(["code", "-r", project_path])
    elif sys.platform == "darwin":
        commands.append(["open", "-a", "Visual Studio Code", project_path])
    else:
        commands.append(["code", "-r", project_path])

    last_error = None
    for cmd in commands:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return (
                f"Opened VS Code for {PROJECT_ROOT.name}. "
                "If the Akane bridge extension is installed, it should connect automatically."
            )
        except Exception as exc:  # pragma: no cover - platform specific
            last_error = exc

    raise RuntimeError(f"Could not open VS Code: {last_error}")
