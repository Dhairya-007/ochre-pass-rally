"""Rally car handling.

The model keeps longitudinal and lateral velocity separately in the car's own
frame. That split is the whole trick:

  * `v_long` is how fast the car is travelling along its nose
  * `v_lat`  is how fast it is sliding sideways

Steering rotates the car. Lateral grip bleeds `v_lat` away. When grip is high
the car goes where it points; when grip falls, the nose rotates while the mass
keeps going the old way, which is exactly what a slide is. Crucially, grip
*falls off as slip angle rises*, so once the back steps out it keeps coming
round - the car rotates instead of snapping straight. That single non-linearity
is the difference between a rally car and a go-kart.

Pure Python. The only thing it needs from the world is a `Surface`, so the
tests drive it over a flat stub with no engine running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .config import VEHICLE, VehicleConfig
from .mathutils import clamp, sign


class Surface(Protocol):
    """What the physics needs to know about the ground."""

    def height_at(self, x: float, z: float) -> float: ...
    def grip_at(self, x: float, z: float) -> float: ...
    def rolling_at(self, x: float, z: float) -> float: ...
    def slope_ahead(self, x: float, z: float, dx: float, dz: float,
                    reach: float = 4.2) -> float: ...
    def world_limit(self) -> float: ...


@dataclass
class Control:
    """One frame of driver input. Throttle and brake are 0..1, steer is -1..1."""

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    handbrake: bool = False

    def released(self) -> "Control":
        return Control()


@dataclass
class Vehicle:
    """Position, attitude and velocity of the car."""

    config: VehicleConfig = field(default_factory=lambda: VEHICLE)

    # Pose
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    # Velocity, in the car's frame
    v_long: float = 0.0
    v_lat: float = 0.0
    v_y: float = 0.0

    # Driveline and steering
    steer_angle: float = 0.0
    rpm: float = 0.0
    gear: int = 1

    # Cosmetic attitude
    roll: float = 0.0
    pitch: float = 0.0
    wheel_spin: float = 0.0

    airborne: bool = False
    landing_impact: float = 0.0

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def speed(self) -> float:
        """Forward speed in m/s, unsigned."""
        return abs(self.v_long)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def slip_angle(self) -> float:
        """Angle between where the car points and where it is going, in radians."""
        return abs(math.atan2(self.v_lat, max(1.0, self.speed)))

    @property
    def forward(self) -> tuple[float, float]:
        return math.sin(self.yaw), math.cos(self.yaw)

    @property
    def right(self) -> tuple[float, float]:
        return math.cos(self.yaw), -math.sin(self.yaw)

    def world_velocity(self) -> tuple[float, float]:
        """Horizontal velocity in world axes (x, z)."""
        fx, fz = self.forward
        rx, rz = self.right
        return (fx * self.v_long + rx * self.v_lat,
                fz * self.v_long + rz * self.v_lat)

    def set_world_velocity(self, vx: float, vz: float) -> None:
        """Inverse of `world_velocity`: re-express (vx, vz) in the car's frame."""
        fx, fz = self.forward
        rx, rz = self.right
        self.v_long = vx * fx + vz * fz
        self.v_lat = vx * rx + vz * rz

    def is_sliding(self, min_slip: float, min_speed: float) -> bool:
        return (
            self.slip_angle > min_slip
            and self.speed > min_speed
            and not self.airborne
        )

    # ------------------------------------------------------------------ #
    # Placement
    # ------------------------------------------------------------------ #
    def place(self, x: float, z: float, yaw: float, surface: Surface,
              keep_speed: float = 0.0) -> None:
        """Drop the car at a spot, pointing along `yaw`, with the slate wiped."""
        self.x, self.z, self.yaw = x, z, yaw
        self.y = surface.height_at(x, z)
        self.v_long = keep_speed
        self.v_lat = 0.0
        self.v_y = 0.0
        self.steer_angle = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.airborne = False
        self.landing_impact = 0.0

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #
    def step(self, control: Control, surface: Surface, dt: float) -> None:
        cfg = self.config
        self.landing_impact = 0.0

        grip = surface.grip_at(self.x, self.z)
        grip += (1.0 - grip) * cfg.loose_traction
        rolling = surface.rolling_at(self.x, self.z)
        rolling = 1.0 + (rolling - 1.0) * cfg.rough_terrain

        self._apply_longitudinal(control, cfg, grip, rolling, dt)
        self._apply_steering(control, cfg, dt)
        yaw_rate = self._apply_yaw(cfg, dt)
        self._apply_lateral(control, cfg, grip, yaw_rate, dt)
        self._integrate_position(surface, dt)
        self._integrate_vertical(surface, cfg, dt)
        self._update_attitude(control, cfg, yaw_rate, dt)
        self._update_driveline(control, cfg, dt)

    # -- longitudinal ---------------------------------------------------- #
    def _apply_longitudinal(self, control: Control, cfg: VehicleConfig,
                            grip: float, rolling: float, dt: float) -> None:
        if control.throttle > 0.0:
            # Power tapers off as the car approaches its terminal speed, and a
            # loose surface costs some of it.
            taper = 1.0 - min(0.78, abs(self.v_long) / cfg.top_speed)
            traction = 0.72 + grip * 0.28
            self.v_long += cfg.engine_force * control.throttle * taper * traction * dt

        if control.brake > 0.0:
            if self.v_long > 0.4:
                self.v_long -= cfg.brake_force * control.brake * grip * dt
            else:
                # Holding brake at a standstill selects reverse.
                self.v_long -= cfg.reverse_speed * 0.55 * control.brake * dt

        if control.handbrake:
            self.v_long -= sign(self.v_long) * 9.0 * dt

        # Drag (quadratic) and rolling resistance (linear). Both scale with the
        # surface, so long grass hauls the car down hard while the gravel line
        # lets it run.
        self.v_long -= self.v_long * cfg.drag * rolling * abs(self.v_long) * dt
        self.v_long -= sign(self.v_long) * min(abs(self.v_long), cfg.rolling * rolling * dt)

        self.v_long = clamp(self.v_long, -cfg.reverse_speed, cfg.top_speed)

    # -- steering -------------------------------------------------------- #
    def _apply_steering(self, control: Control, cfg: VehicleConfig, dt: float) -> None:
        authority = 1.0 - min(0.56, self.speed / cfg.top_speed * cfg.steer_speed_falloff)
        target = control.steer * cfg.max_steer * authority
        self.steer_angle += (target - self.steer_angle) * min(1.0, dt * cfg.steer_rate)

    def _apply_yaw(self, cfg: VehicleConfig, dt: float) -> float:
        """Rotate the car from the front axle. Returns the yaw rate used."""
        yaw_rate = (self.v_long / cfg.wheelbase) * math.tan(self.steer_angle)
        if self.airborne:
            yaw_rate *= 0.22          # almost no steering authority in the air
        self.yaw += yaw_rate * dt
        return yaw_rate

    # -- lateral: where the drift lives ---------------------------------- #
    def _apply_lateral(self, control: Control, cfg: VehicleConfig, grip: float,
                       yaw_rate: float, dt: float) -> None:
        lateral = cfg.lateral_grip * grip

        if control.handbrake:
            lateral *= cfg.handbrake_grip
        if control.throttle > 0.5 and self.speed > 12.0:
            lateral *= cfg.power_on_grip        # power-on oversteer
        if self.airborne:
            lateral *= cfg.airborne_grip

        # Grip falls away as the slide develops, so a slide sustains itself.
        lateral *= 1.0 / (1.0 + self.slip_angle * cfg.slip_falloff)

        self.v_lat -= self.v_lat * min(0.96, lateral * dt)

        # Handbrake plus steering kicks the tail out.
        if control.handbrake and self.speed > 6.0:
            self.v_lat += -self.steer_angle * self.speed * cfg.handbrake_kick * dt

        # Cornering throws weight sideways, which starts the rotation.
        self.v_lat += -yaw_rate * self.v_long * cfg.weight_transfer * dt

    # -- position -------------------------------------------------------- #
    def _integrate_position(self, surface: Surface, dt: float) -> None:
        fx, fz = self.forward
        rx, rz = self.right
        nx = self.x + (fx * self.v_long + rx * self.v_lat) * dt
        nz = self.z + (fz * self.v_long + rz * self.v_lat) * dt

        limit = surface.world_limit()
        if abs(nx) > limit or abs(nz) > limit:
            # Soft wall: kill most of the momentum rather than teleporting.
            self.v_long *= 0.25
            self.v_lat *= 0.25
        else:
            self.x, self.z = nx, nz

    # -- vertical -------------------------------------------------------- #
    def _integrate_vertical(self, surface: Surface, cfg: VehicleConfig,
                            dt: float) -> None:
        ground = surface.height_at(self.x, self.z)

        if self.airborne:
            self.v_y -= cfg.gravity * dt
            self.y += self.v_y * dt
            if self.y <= ground:
                self.landing_impact = abs(min(0.0, self.v_y))
                self.y = ground
                self.v_y = max(0.0, -self.v_y * cfg.landing_bounce)
                self.airborne = False
                if self.landing_impact > 6.0:
                    self.v_lat *= 0.6      # the landing scrubs off some slide
            return

        # On the ground: follow the surface, and let a crest launch the car.
        self.y = ground
        fx, fz = self.forward
        slope = surface.slope_ahead(self.x, self.z, fx, fz)
        self.v_y = slope * self.v_long * cfg.crest_launch
        if self.v_y > 1.2:
            self.airborne = True

    # -- cosmetic attitude ----------------------------------------------- #
    def _update_attitude(self, control: Control, cfg: VehicleConfig,
                         yaw_rate: float, dt: float) -> None:
        """Body roll and pitch. Cosmetic, but it is what sells the weight.

        Sign convention, which the renderer relies on:
          roll  > 0  the body leans to its right
          pitch > 0  the nose points up
        """
        # Lateral acceleration in a corner is yaw rate times forward speed,
        # positive to the right. The body leans away from it.
        lateral_accel = yaw_rate * self.v_long
        target_roll = clamp(-lateral_accel * 0.012, -cfg.max_roll, cfg.max_roll)
        self.roll += (target_roll - self.roll) * min(1.0, dt * cfg.roll_rate)

        # Accelerating squats the rear and lifts the nose; braking dives.
        target_pitch = (control.throttle - control.brake) * 0.045
        self.pitch += (target_pitch - self.pitch) * min(1.0, dt * cfg.pitch_rate)

        self.wheel_spin += (self.v_long / cfg.wheel_radius) * dt

    # -- driveline ------------------------------------------------------- #
    def _update_driveline(self, control: Control, cfg: VehicleConfig,
                          dt: float) -> None:
        speed = self.speed
        gear = 1
        for threshold in cfg.gear_up_speeds:
            if speed > threshold:
                gear += 1
        self.gear = gear

        ratio = cfg.gear_ratios[gear - 1]
        target = (speed * ratio) / 48.0 + (0.14 if control.throttle > 0 else 0.0)
        if speed < 0.6:
            target = 0.1 + control.throttle * 0.5

        target = min(1.08, target)
        self.rpm += (target - self.rpm) * min(1.0, dt * cfg.rev_smoothing)

    # ------------------------------------------------------------------ #
    def wheel_positions(self) -> list[tuple[float, float, bool]]:
        """Local (x, z, is_front) offsets of the four wheels."""
        cfg = self.config
        return [
            (-cfg.track_width, cfg.axle_offset, True),
            (cfg.track_width, cfg.axle_offset, True),
            (-cfg.track_width, -cfg.axle_offset, False),
            (cfg.track_width, -cfg.axle_offset, False),
        ]

    def rear_axle_world(self) -> tuple[float, float, float, float]:
        """World (x, z) of the left and right rear contact patches."""
        cfg = self.config
        fx, fz = self.forward
        rx, rz = self.right
        bx = self.x - fx * cfg.axle_offset
        bz = self.z - fz * cfg.axle_offset
        return (
            bx - rx * cfg.track_width,
            bz - rz * cfg.track_width,
            bx + rx * cfg.track_width,
            bz + rz * cfg.track_width,
        )
