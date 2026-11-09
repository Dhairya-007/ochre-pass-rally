"""The stage centreline.

A closed Catmull-Rom spline through the control points, resampled so every
sample sits the same arc-length apart. Equal spacing matters: lap progress,
the road ribbon and the marker posts all index into these arrays directly.

Coordinates follow Ursina: +X right, +Y up, +Z forward, and a heading of `yaw`
points along (sin yaw, 0, cos yaw).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .config import TRACK, TrackConfig


def _catmull_rom_closed(points: np.ndarray, dense: int) -> np.ndarray:
    """Sample a closed uniform Catmull-Rom spline at `dense` points per segment."""
    n = len(points)
    out = np.empty((n * dense, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, dense, endpoint=False)
    t2, t3 = t * t, t * t * t

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        # Standard uniform Catmull-Rom basis.
        a = 2.0 * p1
        b = -p0 + p2
        c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        seg = 0.5 * (
            a[None, :]
            + b[None, :] * t[:, None]
            + c[None, :] * t2[:, None]
            + d[None, :] * t3[:, None]
        )
        out[i * dense:(i + 1) * dense] = seg
    return out


@dataclass
class Track:
    """Equally spaced centreline samples plus everything derived from them."""

    points: np.ndarray        # (N, 3) centreline positions
    tangents: np.ndarray      # (N, 3) unit forward vectors
    rights: np.ndarray        # (N, 3) unit right vectors (horizontal)
    cumulative: np.ndarray    # (N + 1,) arc length at each sample, last == length
    length: float             # total loop length in metres
    config: TrackConfig

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, config: TrackConfig = TRACK) -> "Track":
        control = np.asarray(config.control_points, dtype=np.float64)
        dense = _catmull_rom_closed(control, dense=320)

        # Arc length along the dense polyline, wrapping back to the start.
        closed = np.vstack([dense, dense[0]])
        seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(arc[-1])

        # Resample at N equal arc-length stations.
        n = config.samples
        targets = np.linspace(0.0, total, n, endpoint=False)
        points = np.empty((n, 3), dtype=np.float64)
        for axis in range(3):
            points[:, axis] = np.interp(targets, arc, closed[:, axis])

        # Tangents by central difference on the (now uniform) samples.
        forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        forward[:, 1] = forward[:, 1]  # keep the vertical component for slopes
        norms = np.linalg.norm(forward, axis=1, keepdims=True)
        tangents = forward / np.maximum(norms, 1e-9)

        # Horizontal right vector: (tz, 0, -tx), renormalised.
        rights = np.stack(
            [tangents[:, 2], np.zeros(n), -tangents[:, 0]], axis=1
        )
        rights /= np.maximum(np.linalg.norm(rights, axis=1, keepdims=True), 1e-9)

        step = total / n
        cumulative = np.arange(n + 1, dtype=np.float64) * step

        return cls(
            points=points,
            tangents=tangents,
            rights=rights,
            cumulative=cumulative,
            length=total,
            config=config,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def count(self) -> int:
        return len(self.points)

    def edge_positions(self, index: int, offset: float) -> Tuple[np.ndarray, np.ndarray]:
        """Left and right points `offset` metres either side of the centreline."""
        centre = self.points[index % self.count]
        right = self.rights[index % self.count]
        return centre - right * offset, centre + right * offset

    def nearest_index(self, x: float, z: float, hint: int = 0, window: int = 40) -> int:
        """Index of the closest centreline sample.

        Searches a window around `hint` first, which is what the car needs every
        frame, and falls back to a full search if the car has clearly jumped
        (a respawn, or driving far off the stage).
        """
        n = self.count
        idx = np.arange(hint - window, hint + window + 1) % n
        dx = self.points[idx, 0] - x
        dz = self.points[idx, 2] - z
        d2 = dx * dx + dz * dz
        local_best = int(np.argmin(d2))
        if d2[local_best] <= 9000.0:
            return int(idx[local_best])

        dx = self.points[:, 0] - x
        dz = self.points[:, 2] - z
        return int(np.argmin(dx * dx + dz * dz))

    def curvature_at(self, index: int, span: int = 6) -> Tuple[float, int]:
        """(turn angle in radians, direction) where direction is +1 right, -1 left."""
        n = self.count
        a = self.tangents[(index - span) % n]
        b = self.tangents[(index + span) % n]
        cross_y = a[2] * b[0] - a[0] * b[2]
        dot = float(np.clip(a[0] * b[0] + a[2] * b[2], -1.0, 1.0))
        return math.acos(dot), (1 if cross_y > 0 else -1)

    def pace_note(self, index: int) -> str:
        """A rally-style call for the corner at `index`.

        Grades run 1 (hairpin) to 6 (barely a kink), which is the convention
        most co-drivers use.
        """
        turn, direction = self.curvature_at(index)
        if turn < 0.12:
            return "Flat out"

        grade = int(round(6 - min(5.0, turn * 7.5)))
        grade = max(1, min(6, grade))
        side = "right" if direction > 0 else "left"

        n = self.count
        rise = float(self.points[(index + 20) % n][1] - self.points[index % n][1])
        if rise > 3.0:
            tail = " over crest"
        elif rise < -3.0:
            tail = " into dip"
        elif grade <= 2:
            tail = " tightens"
        else:
            tail = " long"
        return f"{grade} {side}{tail}"

    def heading_at(self, index: int) -> float:
        """Yaw in radians that points along the track at `index`."""
        t = self.tangents[index % self.count]
        return math.atan2(float(t[0]), float(t[2]))

    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, max_x, min_z, max_z) of the centreline, for the minimap."""
        return (
            float(self.points[:, 0].min()),
            float(self.points[:, 0].max()),
            float(self.points[:, 2].min()),
            float(self.points[:, 2].max()),
        )

    def sharpest_corners(self, count: int, min_separation: int = 24) -> Sequence[int]:
        """Indices of the tightest corners, spread out around the loop."""
        step = 8
        candidates = [(self.curvature_at(i)[0], i) for i in range(0, self.count, step)]
        candidates.sort(reverse=True)

        chosen: list[int] = []
        for _, idx in candidates:
            if len(chosen) >= count:
                break
            if all(
                min(abs(idx - c), self.count - abs(idx - c)) >= min_separation
                for c in chosen
            ):
                chosen.append(idx)
        return chosen
