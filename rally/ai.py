"""AI rally drivers.

Two pieces, computed at very different rates:

  `RacingLine` - built once per stage and car. A lateral offset for every
  centreline sample (outside on the way in, inside at the apex, outside on the
  way out), and a target speed for every sample. The speed profile is the
  standard construction: the cornering limit v = sqrt(a_lat * r) at each point,
  then a backward pass so the car can brake down to each corner in time, then a
  forward pass so it can only gain speed as fast as the engine allows.

  `Driver` - runs every frame. Pure-pursuit steering towards a point on the
  line ahead (the look-ahead grows with speed), counter-steer into slides,
  throttle and brake to track the profile, and a little personality: skill
  scales how close to the limit it drives, and lower-skill drivers wander off
  the ideal line and misjudge braking now and then.

Pure Python and numpy; the tests run these drivers round every stage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import VehicleConfig
from .mathutils import clamp, wrap_angle
from .track import Track
from .vehicle import Control, Vehicle

GRAVITY = 9.81
CORNER_C = 2.80          # measured; see RacingLine.build
CORNER_MARGIN = 0.93     # stay a little inside the measured limit


def _smooth_loop(values: np.ndarray, width: int) -> np.ndarray:
    """Circular moving average, used to turn per-sample decisions into a line."""
    kernel = np.ones(2 * width + 1) / (2 * width + 1)
    padded = np.concatenate([values[-width:], values, values[:width]])
    return np.convolve(padded, kernel, mode="valid")


def _signed_curvature(points_xz: np.ndarray, span: int) -> np.ndarray:
    """Signed curvature (1/m) of a closed polyline via three-point circles.

    Positive curvature turns right, matching `Track.curvature_at`.
    """
    a = np.roll(points_xz, span, axis=0)
    b = points_xz
    c = np.roll(points_xz, -span, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    # Track.curvature_at calls a right turn positive; that is negative cross
    # in (x, z), so flip the sign to agree.
    return -2.0 * cross / np.maximum(ab * bc * ca, 1e-9)


@dataclass
class RacingLine:
    track: Track
    offsets: np.ndarray        # lateral offset from the centreline, +right, metres
    points: np.ndarray         # (N, 2) world x, z of the line
    curvature: np.ndarray      # signed 1/m along the line
    speed: np.ndarray          # target speed, m/s

    @classmethod
    def build(cls, track: Track, config: VehicleConfig, surface_grip: float,
              edge_margin: float = 1.9) -> "RacingLine":
        n = track.count
        half = track.config.road_half_width
        centre = track.points[:, [0, 2]]
        rights = track.rights[:, [0, 2]]

        # Use the inside of each corner: offset against the turn direction,
        # in proportion to how hard it turns, then smooth widely so the car
        # sets up wide before the apex and runs out wide after it.
        k_centre = _signed_curvature(centre, span=6)
        want = np.sign(k_centre) * np.clip(np.abs(k_centre) * 55.0, 0.0, 1.0)
        offsets = _smooth_loop(want, width=10) * (half - edge_margin)
        offsets = _smooth_loop(offsets, width=6)
        points = centre + rights * offsets[:, None]

        curvature = _signed_curvature(points, span=5)
        curvature = _smooth_loop(curvature, width=3)
        spacing = track.length / n

        # Cornering limit. Lateral grip in the physics is a decay rate, not a
        # friction coefficient, so the available lateral acceleration grows
        # with speed. Measured on skidpans of 25, 50 and 90 m (see
        # tests/test_ai.py): a = C * sqrt(v), with C = 2.80 at full grip and
        # falling roughly as grip^0.7. Setting a = v^2 / r gives the corner
        # speed v = (C * r)^(2/3).
        traction = surface_grip + (1.0 - surface_grip) * config.loose_traction
        c = CORNER_C * traction ** 0.7 * (config.lateral_grip / 7.4)
        radius = 1.0 / np.maximum(np.abs(curvature), 1e-4)
        corner = (c * radius) ** (2.0 / 3.0) * CORNER_MARGIN
        speed = np.minimum(corner, config.top_speed * 0.97)

        brake = config.brake_force * surface_grip * 0.80

        def accel(v: float) -> float:
            taper = 1.0 - min(0.78, v / config.top_speed)
            return max(0.4, config.engine_force * taper * 0.92
                       - config.drag * v * v - config.rolling)

        for _ in range(2):          # twice, so the closure point settles
            for i in range(n - 1, -1, -1):          # braking
                nxt = speed[(i + 1) % n]
                speed[i] = min(speed[i], math.sqrt(nxt * nxt + 2.0 * brake * spacing))
            for i in range(n):                       # acceleration
                prev = speed[i - 1]
                speed[i] = min(speed[i], math.sqrt(prev * prev + 2.0 * accel(prev) * spacing))

        return cls(track, offsets, points, curvature, speed)

    def target(self, index: int) -> Tuple[float, float]:
        p = self.points[index % self.track.count]
        return float(p[0]), float(p[1])


@dataclass
class Personality:
    skill: float = 0.8          # 0..1: how close to the limit
    aggression: float = 0.5     # 0..1: how readily it barges past
    consistency: float = 0.8    # 0..1: how rarely it makes mistakes


@dataclass
class Driver:
    line: RacingLine
    personality: Personality = field(default_factory=Personality)
    seed: int = 0

    index: int = 0
    wants_reset: bool = False
    _stuck_for: float = 0.0
    _lapse_for: float = 0.0
    _lapse_gain: float = 1.0
    _wander: float = 0.0
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, index: int) -> None:
        self.index = index
        self.wants_reset = False
        self._stuck_for = 0.0
        self._lapse_for = 0.0

    # ------------------------------------------------------------------ #
    def control(self, car: Vehicle, dt: float,
                others: Iterable[Tuple[float, float]] = ()) -> Control:
        track = self.line.track
        n = track.count
        spacing = track.length / n
        p = self.personality
        self.index = track.nearest_index(car.x, car.z, hint=self.index)
        speed = car.speed

        self._update_mistakes(dt)

        # --- where to aim ----------------------------------------------
        lateral = self._lateral_error(car)
        off_road = abs(lateral) > track.config.road_half_width + 1.5
        lookahead_m = clamp(7.0 + speed * 0.42, 9.0, 30.0)
        if off_road:
            lookahead_m = 10.0
        ahead = (self.index + int(lookahead_m / spacing)) % n

        if off_road:
            target = track.points[ahead]
            tx, tz = float(target[0]), float(target[2])
        else:
            tx, tz = self.line.target(ahead)
            # Lower skill drifts around the ideal line.
            wander = self._wander * (1.0 - p.skill) * 2.2
            r = track.rights[ahead]
            tx += float(r[0]) * wander
            tz += float(r[2]) * wander
            tx, tz = self._avoid_traffic(car, tx, tz, others, ahead)

        # --- steering: pure pursuit plus counter-steer -------------------
        desired = math.atan2(tx - car.x, tz - car.z)
        alpha = wrap_angle(desired - car.yaw)
        distance = max(4.0, math.hypot(tx - car.x, tz - car.z))
        steer_angle = math.atan(2.0 * car.config.wheelbase * math.sin(alpha) / distance)
        authority = car.config.max_steer * (1.0 - min(0.56, speed / car.config.top_speed
                                                     * car.config.steer_speed_falloff))
        steer = steer_angle / max(authority, 1e-3)
        travel = math.atan2(car.v_lat, max(1.0, abs(car.v_long)))
        steer += travel * 0.9                                 # catch the slide
        steer = clamp(steer, -1.0, 1.0)

        # --- speed --------------------------------------------------------
        brake_look = (self.index + int((4.0 + speed * 0.25) / spacing)) % n
        target_speed = float(self.line.speed[brake_look])
        target_speed *= 0.80 + 0.20 * p.skill
        target_speed *= self._lapse_gain
        if off_road:
            target_speed = min(target_speed, 14.0)
        if abs(alpha) > 1.2:                                  # pointing the wrong way
            target_speed = min(target_speed, 6.0)

        error = target_speed - speed
        throttle = brake = 0.0
        if error > 0.0:
            throttle = clamp(0.55 + error * 0.25, 0.0, 1.0)
            if abs(travel) > 0.45:                            # don't spin it up mid-slide
                throttle *= 0.6
        elif error < -1.2:
            brake = clamp(-error * 0.22, 0.0, 1.0)

        self._check_stuck(car, dt, throttle)
        return Control(throttle=throttle, brake=brake, steer=steer, handbrake=False)

    # ------------------------------------------------------------------ #
    def _lateral_error(self, car: Vehicle) -> float:
        track = self.line.track
        c, r = track.points[self.index], track.rights[self.index]
        return (car.x - float(c[0])) * float(r[0]) + (car.z - float(c[2])) * float(r[2])

    def _avoid_traffic(self, car: Vehicle, tx: float, tz: float,
                       others: Iterable[Tuple[float, float]], ahead: int
                       ) -> Tuple[float, float]:
        """Steer round a car directly ahead instead of driving into its boot."""
        fx, fz = car.forward
        rx, rz = car.right
        nudge = 0.0
        for ox, oz in others:
            dx, dz = ox - car.x, oz - car.z
            along = dx * fx + dz * fz
            across = dx * rx + dz * rz
            if 0.0 < along < 14.0 and abs(across) < 2.6:
                side = -1.0 if across > 0 else 1.0
                nudge += side * (2.8 - abs(across)) * (1.0 - along / 14.0)
        if nudge == 0.0:
            return tx, tz
        half = self.line.track.config.road_half_width - 1.2
        r = self.line.track.rights[ahead]
        current = self.line.offsets[ahead]
        nudge = clamp(current + nudge * (1.0 + self.personality.aggression), -half, half) - current
        return tx + float(r[0]) * nudge, tz + float(r[2]) * nudge

    def _update_mistakes(self, dt: float) -> None:
        p = self.personality
        # A slowly varying wander off the ideal line.
        self._wander += (self._rng.uniform(-1.0, 1.0) - self._wander * 0.3) * dt
        self._wander = clamp(self._wander, -1.0, 1.0)

        if self._lapse_for > 0.0:
            self._lapse_for -= dt
            if self._lapse_for <= 0.0:
                self._lapse_gain = 1.0
            return
        # Roughly once a minute at consistency 0, almost never at 1.
        chance = (1.0 - p.consistency) * dt / 60.0 * 3.0
        if self._rng.random() < chance:
            self._lapse_for = self._rng.uniform(1.0, 2.5)
            self._lapse_gain = self._rng.choice((1.10, 1.14, 0.82))

    def _check_stuck(self, car: Vehicle, dt: float, throttle: float) -> None:
        if car.speed < 1.5 and throttle > 0.3:
            self._stuck_for += dt
        else:
            self._stuck_for = max(0.0, self._stuck_for - dt * 2.0)
        if self._stuck_for > 3.0:
            self.wants_reset = True
            self._stuck_for = 0.0


# --------------------------------------------------------------------------- #
# Rival drivers for the career and quick races
# --------------------------------------------------------------------------- #
RIVAL_NAMES: Tuple[str, ...] = (
    "Aino Lehtinen", "Marco Bellandi", "Sofia Duarte", "Jules Marchetti",
    "Henrik Strand", "Priya Raman", "Tomás Écija", "Lena Vogt",
    "Kenji Arakawa", "Owen Pryce", "Nadia Kowalczyk", "Rafael Ortega",
)
