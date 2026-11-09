"""Player settings, saved as JSON in the user data folder."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import user_data_dir

QUALITY = ("low", "medium", "high", "ultra")
DIFFICULTY = ("easy", "normal", "hard")
DIFFICULTY_SKILL = {"easy": -0.14, "normal": 0.0, "hard": 0.08}


@dataclass
class Settings:
    name: str = "Driver"
    quality: str = "high"
    fullscreen: bool = False
    width: int = 1600
    height: int = 900
    volume: float = 0.8
    difficulty: str = "normal"
    camera: str = "chase"
    last_host: str = ""
    port: int = 47800
    show_fps: bool = False
    vsync: bool = True

    @classmethod
    def path(cls) -> Path:
        return user_data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in known})
        if settings.quality not in QUALITY:
            settings.quality = "high"
        if settings.difficulty not in DIFFICULTY:
            settings.difficulty = "normal"
        settings.volume = max(0.0, min(1.0, float(settings.volume)))
        return settings

    def save(self) -> None:
        path = self.path()
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-", suffix=".json")
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(self), handle, indent=2)
        os.replace(tmp, path)
