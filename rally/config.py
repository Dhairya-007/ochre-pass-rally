"""Tunable constants for the whole game.

Everything a designer would want to change lives here. Nothing in this module
imports the engine, so the tests and the headless tools can read it freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

RGB = Tuple[float, float, float]


def _hex(code: int) -> RGB:
    """0xRRGGBB -> (r, g, b) floats in 0..1, which is what Ursina's color.rgb wants."""
    return (
        ((code >> 16) & 0xFF) / 255.0,
        ((code >> 8) & 0xFF) / 255.0,
        (code & 0xFF) / 255.0,
    )


# --------------------------------------------------------------------------- #
# Palette - a golden-hour gravel stage: ochre dust, pine, rally red.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Palette:
    ink: RGB = _hex(0x14110F)
    bone: RGB = _hex(0xF2EADF)
    dust: RGB = _hex(0xD9A441)
    rust: RGB = _hex(0xC2452D)
    moss: RGB = _hex(0x3B4A32)
    pine_dark: RGB = _hex(0x2F3D28)
    bark: RGB = _hex(0x4A3827)
    grass: RGB = _hex(0x6F7A4A)
    dry_grass: RGB = _hex(0x9C8D5C)
    rock: RGB = _hex(0x8A7C6C)
    gravel: RGB = _hex(0x9D7F58)
    gravel_dark: RGB = _hex(0x6E573B)
    verge: RGB = _hex(0xA98B5E)
    hay: RGB = _hex(0xCBAE63)
    sky_top: RGB = _hex(0x4A74A8)
    sky_mid: RGB = _hex(0xD9B68C)
    sky_low: RGB = _hex(0xE8C89A)
    fog: RGB = _hex(0xD9B68C)


PALETTE = Palette()

# Direction *towards* the sun: low, golden, and off to the left so the stage
# reads with long raking light. Static geometry has this baked into its vertex
# colours at build time, and the car's dynamic light is aimed to match.
SUN_DIRECTION: RGB = (-0.55, 0.72, 0.42)


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VehicleConfig:
    """Arcade rally handling. Units are metres, seconds, radians."""

    # Longitudinal.
    # These four settle the speed curve. Power tapers towards `top_speed` while
    # drag grows with the square of speed, so terminal velocity is where they
    # meet - about 40 m/s (145 km/h) on gravel and 30 m/s (108 km/h) off it.
    # `top_speed` is only the hard clamp; the drag balance is what you feel.
    engine_force: float = 12.0      # m/s^2 at zero speed, tapers with speed
    brake_force: float = 16.0       # m/s^2 (~1.6g, so 22 m from 100 km/h)
    reverse_speed: float = 12.0     # m/s cap in reverse
    top_speed: float = 48.0         # m/s hard ceiling
    drag: float = 0.0010            # quadratic, multiplied by surface rolling
    rolling: float = 1.2            # linear, multiplied by surface rolling

    # Steering
    max_steer: float = 0.60         # rad at the front wheels
    steer_rate: float = 9.5         # how fast the wheels reach the target angle
    steer_speed_falloff: float = 0.78   # steering authority lost at top speed
    wheelbase: float = 2.65

    # Grip - this is what makes it a rally car rather than a go-kart
    lateral_grip: float = 7.4       # base rear-end grip on the racing line
    handbrake_grip: float = 0.17    # multiplier while the handbrake is up
    power_on_grip: float = 0.80     # multiplier at full throttle above 12 m/s
    airborne_grip: float = 0.25
    slip_falloff: float = 1.5       # how fast grip dies once the car slides
    handbrake_kick: float = 1.5     # tail-out impulse from handbrake + steering
    weight_transfer: float = 0.55   # cornering throws weight sideways

    # Vertical
    gravity: float = 26.0
    crest_launch: float = 0.92      # how hard crests throw the car
    landing_bounce: float = 0.12

    # Body attitude (cosmetic, but it sells the weight)
    max_roll: float = 0.17
    roll_rate: float = 6.0
    pitch_rate: float = 5.0

    # Drivetrain, for the tacho and the engine note
    gear_ratios: Tuple[float, ...] = (3.30, 2.15, 1.55, 1.18, 0.94)
    gear_up_speeds: Tuple[float, ...] = (8.0, 16.0, 25.0, 33.0)
    rev_smoothing: float = 7.0

    # Character traits that separate one car from another.
    mass: float = 1230.0            # kg; only matters when cars collide
    loose_traction: float = 0.0     # 0..1, how much loose-surface grip loss a
                                    # 4x4 claws back: grip + (1 - grip) * this
    rough_terrain: float = 1.0      # multiplies the off-stage rolling penalty

    wheel_radius: float = 0.46
    track_width: float = 0.95       # half-distance between left and right wheels
    axle_offset: float = 1.41       # half-distance between front and rear axles


VEHICLE = VehicleConfig()


# --------------------------------------------------------------------------- #
# Track and world
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackConfig:
    samples: int = 600              # centreline samples, equally spaced by arc length
    road_half_width: float = 7.2
    # Control points: (x, y, z). The loop closes automatically, so these must
    # run monotonically around the circuit - a point that sits nearer the
    # middle than its neighbours puts a hairpin in the spline where the loop
    # closes. The start/finish point below is deliberately on a gentle bend
    # between the last corner and the first, not at the origin.
    control_points: Tuple[Tuple[float, float, float], ...] = (
        (190.0, 0.0, -8.0),       # start / finish
        (130.0, 1.0, -58.0),
        (228.0, 4.0, -176.0),
        (186.0, 9.0, -304.0),
        (44.0, 12.0, -352.0),
        (-96.0, 9.0, -292.0),
        (-146.0, 4.0, -162.0),
        (-268.0, 5.0, -66.0),
        (-292.0, 10.0, 84.0),
        (-168.0, 13.0, 168.0),
        (-24.0, 10.0, 200.0),
        (124.0, 5.0, 156.0),
        (206.0, 2.0, 62.0),
    )


TRACK = TrackConfig()


@dataclass(frozen=True)
class WorldConfig:
    size: float = 1200.0            # terrain is size x size, centred on the origin
    resolution: int = 144           # grid cells per side
    hill_height: float = 34.0
    hill_scale: float = 0.0016
    detail_height: float = 5.5
    detail_scale: float = 0.0075
    flatten_start: float = 4.0      # metres past the road edge where blending begins
    flatten_end: float = 42.0       # metres past the road edge where terrain is free
    boundary_margin: float = 30.0   # invisible wall inset from the terrain edge

    # Surface grip by distance from the centreline
    grip_line: float = 1.00         # packed gravel
    grip_edge: float = 0.84         # loose stuff on the verge
    grip_offroad: float = 0.56      # scrub and grass
    edge_band: float = 2.5
    offroad_rolling: float = 2.9    # rolling-resistance multiplier off the stage

    tree_count: int = 520
    rock_count: int = 220
    post_spacing: int = 9           # centreline samples between marker posts
    bale_corners: int = 3


WORLD = WorldConfig()


# --------------------------------------------------------------------------- #
# Game rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GameConfig:
    laps: int = 3
    countdown_step: float = 0.8     # seconds per countdown number
    grid_sample: int = 3            # centreline index the car starts on
    drift_min_slip: float = 0.17    # rad of slip angle before it counts
    drift_min_speed: float = 9.0    # m/s
    drift_points_rate: float = 11.0
    drift_multiplier_step: float = 1.6   # seconds of sliding per multiplier step
    drift_multiplier_max: int = 6
    drift_grace: float = 2.2        # how fast the chain drains once you straighten
    wrong_way_dot: float = -0.35
    wrong_way_delay: float = 0.6
    pace_note_lookahead: int = 30   # centreline samples ahead


GAME = GameConfig()


# --------------------------------------------------------------------------- #
# Liveries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Livery:
    name: str
    body: RGB
    accent: RGB


LIVERIES: Tuple[Livery, ...] = (
    Livery("Ochre", PALETTE.dust, PALETTE.ink),
    Livery("Rally Red", PALETTE.rust, PALETTE.bone),
    Livery("Pine", PALETTE.moss, PALETTE.dust),
    Livery("Bone", _hex(0xE8DDC9), PALETTE.rust),
    Livery("Slate", _hex(0x47566B), PALETTE.dust),
)


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraConfig:
    chase_distance: float = 8.4
    chase_distance_per_speed: float = 0.09
    chase_height: float = 3.5
    chase_height_per_speed: float = 0.022
    drift_lean: float = 0.035       # how far the camera swings into a slide
    drift_lean_max: float = 0.55
    position_lerp: float = 4.6
    look_lerp: float = 6.2
    look_ahead: float = 6.5
    fov_base: float = 64.0
    fov_per_speed: float = 0.28
    fov_max_gain: float = 14.0
    ground_clearance: float = 1.9
    bonnet_forward: float = 0.9
    bonnet_height: float = 1.42


CAMERA = CameraConfig()


@dataclass(frozen=True)
class DustConfig:
    """Dust is the hottest loop in the game, so the pool is sized deliberately.

    `max_particles` x `lifetime` x `spawn_per_second` sets how many quads are
    alive at once. Around a hundred during a hard drift looks dense already,
    because big billboards overlap heavily; three hundred cost frame time and
    looked identical.
    """

    max_particles: int = 260
    spawn_per_second: float = 40.0
    lifetime: float = 1.1
    rise: float = 2.1
    drag: float = 0.975

    # Size is in metres, and these are billboards a few metres behind a camera
    # that sits eight metres back. A puff that grows past about three metres
    # fills the screen and hides the car, so growth stays modest and alpha
    # stays low - the plume reads from dozens of overlapping quads, not from
    # any single opaque one.
    growth: float = 1.5             # metres per second
    start_size: float = 0.40
    size_jitter: float = 0.45
    alpha: float = 0.22


DUST = DustConfig()
