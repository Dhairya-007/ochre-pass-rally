"""Where the game keeps the player's files.

Saves and settings live in the platform's per-user data folder, not next to
the code, so they survive the game being moved, updated or reinstalled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "OchrePassRally"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "assets"


def user_data_dir() -> Path:
    override = os.environ.get("OCHRE_RALLY_DATA")
    if override:
        path = Path(override)
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP
    elif sys.platform.startswith("win"):
        path = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = user_data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
