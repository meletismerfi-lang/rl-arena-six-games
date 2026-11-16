"""Windows launcher: double-click to open the RL Arena.

Deliberately NOT a PyInstaller bundle of the whole app — bundling torch + SB3 + pygame makes a
1-2 GB exe that takes ~30 s to start. This exe is a ~1 MB wrapper that finds the project's
virtualenv and starts `arena.py` with pythonw (no console window).

Search order for the project directory:
  1. the folder this exe lives in (so you can drop the exe anywhere, next to the project)
  2. C:\\Users\\melet\\AI\\rl-games
  3. RL_GAMES_DIR environment variable
"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter.messagebox as mb
from pathlib import Path

FALLBACKS = [Path(r"C:\Users\melet\AI\rl-games")]


def find_root() -> Path | None:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(__file__).resolve().parent)
    if os.environ.get("RL_GAMES_DIR"):
        candidates.append(Path(os.environ["RL_GAMES_DIR"]))
    candidates.extend(FALLBACKS)
    for c in candidates:
        if (c / "arena.py").exists():
            return c
    return None


def main() -> int:
    root = find_root()
    if root is None:
        mb.showerror("RL Arena", "Could not find arena.py.\n\n"
                                 "Put this exe in the rl-games folder, or set RL_GAMES_DIR.")
        return 1

    py = root / ".venv" / "Scripts" / "pythonw.exe"
    if not py.exists():
        py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        mb.showerror("RL Arena", f"No virtualenv python found under:\n{root}\\.venv\\Scripts\n\n"
                                 "Create it with:  uv venv --python 3.12 .venv")
        return 1

    # already running? don't stack a second arena on top of the first
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process python,pythonw -ErrorAction SilentlyContinue | "
             "Where-Object { $_.MainWindowTitle -like '*RL Arena*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        if out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0:
            return 0
    except Exception:
        pass

    subprocess.Popen([str(py), str(root / "arena.py")], cwd=str(root),
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                     close_fds=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
