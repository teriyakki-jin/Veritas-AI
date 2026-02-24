#!/usr/bin/env python3
"""Portable API launcher for Windows + WSL.

Usage:
  python scripts/run_api.py

Env:
  HOST=0.0.0.0
  PORT=8000
  RELOAD=0
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REQUIRED = ("torch", "fastapi", "uvicorn")


def _has_required_modules() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_uvicorn(python_exe: Path, project_root: Path) -> int:
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    cmd = [
        str(python_exe),
        "-m",
        "uvicorn",
        "src.api_server:app",
        "--host",
        host,
        "--port",
        port,
    ]
    if _bool_env("RELOAD", default=False):
        cmd.append("--reload")

    print(f"[run_api] exec: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(project_root))


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    current_python = Path(sys.executable)

    if _has_required_modules():
        print(f"[run_api] using current interpreter: {current_python}")
        return _run_uvicorn(current_python, project_root)

    windows_venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if os.name != "nt" and windows_venv_python.exists():
        print("[run_api] current interpreter is missing dependencies.")
        print(f"[run_api] fallback to Windows venv: {windows_venv_python}")
        return _run_uvicorn(windows_venv_python, project_root)

    print("[run_api] Required modules are missing in the current interpreter.")
    print("[run_api] Install dependencies with: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
