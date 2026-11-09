"""Lap timing, progress tracking and drift scoring.

Progress is measured by which centreline sample the car is nearest to. Because
`Track` spaces its samples evenly by arc length, that index divided by the
sample count is a fair 0..1 position around the loop, and a lap is simply the
moment that value wraps from near 1 back to near 0.

Pure Python, so the lap and scoring rules are covered by the unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .config import GAME, GameConfig
from .track import Track


class LapEvent(Enum):
    NONE = auto()
    LAP_COMPLETED = auto()
    STAGE_FINISHED = auto()
    WENT_BACKWARDS = auto()


def format_time(seconds: Optional[float]) -> str:
    """m:ss.hh, or an em dash when there is no time yet."""
    if seconds is None:
        return "—"
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:05.2f}"


# --------------------------------------------------------------------------- #
# Lap timing
# --------------------------------------------------------------------------- #
@dataclass
class LapTimer:
    track: Track
    config: GameConfig = field(default_factory=lambda: GAME)

    lap: int = 0                       # laps completed
    laps: List[float] = field(default_factory=list)
    best: Optional[float] = None
    last: Optional[float] = None
    lap_start: float = 0.0

    index: int = 0
    progress: float = 0.0
    _previous_progress: float = 0.0
    _wrong_way_for: float = 0.0

    def reset(self, start_index: int, behind_line: bool = False) -> None:
        """Start timing from `start_index`.

        With `behind_line`, the car starts on a grid slot short of the
        start/finish line: its first crossing begins lap 1 instead of
        completing it, and the run-up counts towards lap 1's time.
        """
        self.lap = -1 if behind_line else 0
        self.laps = []
        self.last = None
        self.lap_start = 0.0
        self.index = start_index
        self.progress = start_index / self.track.count
        self._previous_progress = self.progress
        self._wrong_way_for = 0.0

    def resync(self, index: int) -> None:
        """Move progress to `index` without it counting as a line crossing.

        Used after a respawn: the car teleports, and without this the jump in
        progress would read as a lap (or an un-lap).
        """
        self.index = index
        self.progress = index / self.track.count
        self._previous_progress = self.progress
        self._wrong_way_for = 0.0

    # ------------------------------------------------------------------ #
    def update(self, x: float, z: float, elapsed: float) -> LapEvent:
        """Advance progress. `elapsed` is total stage time in seconds."""
        self.index = self.track.nearest_index(x, z, hint=self.index)
        self._previous_progress = self.progress
        self.progress = self.index / self.track.count

        delta = self.progress - self._previous_progress

        if delta < -0.5:
            # Wrapped forwards past the start/finish line.
            return self._complete_lap(elapsed)
        if delta > 0.5:
            # Reversed back over the line.
            self.lap = max(-1 if not self.laps else 0, self.lap - 1)
            if self.laps:
                self.laps.pop()
                self.last = self.laps[-1] if self.laps else None
            return LapEvent.WENT_BACKWARDS

        return LapEvent.NONE

    def _complete_lap(self, elapsed: float) -> LapEvent:
        if self.lap < 0:
            # Rolled over the line from the grid: lap 1 is under way, and its
            # clock has been running since the start.
            self.lap = 0
            return LapEvent.NONE
        lap_time = elapsed - self.lap_start
        self.lap += 1
        self.laps.append(lap_time)
        self.last = lap_time
        if self.best is None or lap_time < self.best:
            self.best = lap_time
        self.lap_start = elapsed

        if self.lap >= self.config.laps:
            return LapEvent.STAGE_FINISHED
        return LapEvent.LAP_COMPLETED

    # ------------------------------------------------------------------ #
    def current_lap_time(self, elapsed: float) -> float:
        return max(0.0, elapsed - self.lap_start)

    @property
    def display_lap(self) -> int:
        """Lap number to show, 1-based and capped at the stage length."""
        return max(1, min(self.config.laps, self.lap + 1))

    def race_distance(self) -> float:
        """Laps completed plus progress into the current one, for standings.

        On the grid, behind the line, this is slightly negative.
        """
        if self.lap < 0:
            return self.progress - 1.0
        return self.lap + self.progress

    @property
    def finished(self) -> bool:
        return self.lap >= self.config.laps

    def check_wrong_way(self, yaw: float, v_long: float, dt: float) -> bool:
        """True once the car has been facing backwards long enough to warn."""
        tangent = self.track.tangents[self.index % self.track.count]
        facing = math.sin(yaw) * float(tangent[0]) + math.cos(yaw) * float(tangent[2])

        if facing < self.config.wrong_way_dot and abs(v_long) > 5.0:
            self._wrong_way_for += dt
        else:
            self._wrong_way_for = max(0.0, self._wrong_way_for - dt * 2.0)

        return self._wrong_way_for > self.config.wrong_way_delay

    def pace_note(self) -> str:
        return self.track.pace_note(
            (self.index + self.config.pace_note_lookahead) % self.track.count
        )


# --------------------------------------------------------------------------- #
# Drift scoring
# --------------------------------------------------------------------------- #
@dataclass
class DriftScorer:
    """Builds a chain while the car slides and banks it when the car straightens.

    Points accrue faster the harder and faster you slide. Hold a slide and the
    multiplier climbs a step every `drift_multiplier_step` seconds. Straighten
    up and the chain drains over `drift_grace` seconds before paying out, which
    lets you link two corners into one score.
    """

    config: GameConfig = field(default_factory=lambda: GAME)

    banked: float = 0.0
    chain: float = 0.0
    multiplier: int = 1
    _sliding_for: float = 0.0

    def reset(self) -> None:
        self.banked = 0.0
        self.chain = 0.0
        self.multiplier = 1
        self._sliding_for = 0.0

    @property
    def pending(self) -> float:
        """What the current chain is worth if it banks right now."""
        return self.chain * self.multiplier

    @property
    def total(self) -> float:
        return self.banked + self.pending

    @property
    def active(self) -> bool:
        return self.chain > 0.0

    def update(self, slip_angle: float, speed: float, airborne: bool,
               dt: float) -> None:
        cfg = self.config
        sliding = (
            slip_angle > cfg.drift_min_slip
            and speed > cfg.drift_min_speed
            and not airborne
        )

        if sliding:
            self._sliding_for += dt
            self.chain += slip_angle * speed * cfg.drift_points_rate * dt
            steps = int(self._sliding_for / cfg.drift_multiplier_step)
            self.multiplier = min(cfg.drift_multiplier_max, 1 + steps)
        elif self.chain > 0.0:
            self._sliding_for -= dt * cfg.drift_grace
            if self._sliding_for <= 0.0:
                self.bank()

    def bank(self) -> float:
        """Pay the chain into the running total and clear it."""
        paid = self.pending
        self.banked += paid
        self.chain = 0.0
        self.multiplier = 1
        self._sliding_for = 0.0
        return paid

    def crash_out(self) -> None:
        """Lose the chain without paying - used on a respawn."""
        self.chain = 0.0
        self.multiplier = 1
        self._sliding_for = 0.0
