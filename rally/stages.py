"""The rally stages: where they are, what they are made of, and how they drive.

Each stage is pure data plus a seed. The track itself is generated, not
hand-placed: control points sit on a closed curve whose radius is a sum of a
few low-frequency sines around the loop,

    r(theta) = R * (1 + a1 sin(k1 theta + p1) + a2 sin(k2 theta + p2) + ...)

which can only ever produce a smooth, non-self-intersecting loop. The original
hand-placed track had a control point inside the circuit and so a 139 degree
kink at the start line; this construction makes that class of bug impossible.

No engine imports - the career, the AI and the tests all read this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TRACK, WORLD, TrackConfig, WorldConfig

RGB = Tuple[float, float, float]


@dataclass(frozen=True)
class Vegetation:
    """Which models dress the stage, how densely, and how they are tinted."""

    trees: Tuple[str, ...]
    tree_count: int
    rocks: Tuple[str, ...]
    rock_count: int
    clutter: Tuple[str, ...] = ()          # bushes, grass, stumps, logs
    clutter_count: int = 0
    # Kenney material name -> sRGB tint. Replaces the stylised source colours.
    tints: Tuple[Tuple[str, RGB], ...] = ()
    tree_scale: Tuple[float, float] = (8.0, 13.0)
    rock_scale: Tuple[float, float] = (2.5, 6.5)
    clutter_scale: Tuple[float, float] = (2.0, 4.0)


@dataclass(frozen=True)
class Atmosphere:
    sky: str                                # asset key under assets/skies
    sun_direction: Tuple[float, float, float]   # towards the sun, Y up
    sun_color: RGB
    sun_strength: float
    ambient: RGB
    fog_color: RGB                          # linear
    fog_density: float
    exposure: float
    env_strength: float = 0.35              # image-based lighting scale
    sky_rotation: float = 0.0               # degrees, to line the sun up


@dataclass(frozen=True)
class Surface:
    road_texture: str                       # asset key, e.g. "gravel_forest"
    ground_texture: str
    road_tile: float = 6.0                  # metres per texture repeat, along
    ground_tile: float = 5.0
    grip_line: float = 1.00
    grip_edge: float = 0.84
    grip_offroad: float = 0.56
    offroad_rolling: float = 2.9
    dust: RGB = (0.78, 0.68, 0.52)          # particle tint, sRGB
    dust_amount: float = 1.0
    ground_tint: RGB = (1.0, 1.0, 1.0)      # multiplies the ground texture
    road_tint: RGB = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Layout:
    """Parameters for the generated loop.

    On a loop of radius R, a wave with k cycles and amplitude a tightens the
    sharpest corner by roughly a * (k^2 - 1), so the tightest radius is about
    R / (1 + sum of those). That is the knob for how twisty a stage is. Every
    radius stays positive, so the loop is star-shaped and cannot cross itself.
    """

    seed: int
    radius: float = 230.0
    stretch: Tuple[float, float] = (1.15, 0.85)   # x and z scale of the loop
    waves: Tuple[Tuple[int, float], ...] = ((2, 0.16), (3, 0.10), (5, 0.05))
    elevation: float = 14.0
    points: int = 44   # >= ~4 per cycle of the highest wave, or the spline smooths it away
    road_half_width: float = 7.0
    samples: int = 600


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    country: str
    blurb: str
    layout: Layout
    surface: Surface
    atmosphere: Atmosphere
    vegetation: Vegetation
    hill_height: float = 34.0

    # ------------------------------------------------------------------ #
    def track_config(self) -> TrackConfig:
        return replace(
            TRACK,
            control_points=generate_control_points(self.layout),
            road_half_width=self.layout.road_half_width,
            samples=self.layout.samples,
        )

    def world_config(self) -> WorldConfig:
        s = self.surface
        return replace(
            WORLD,
            resolution=200,                 # ~6 m cells: smooth hills under PBR light
            hill_height=self.hill_height,
            grip_line=s.grip_line,
            grip_edge=s.grip_edge,
            grip_offroad=s.grip_offroad,
            offroad_rolling=s.offroad_rolling,
            tree_count=self.vegetation.tree_count,
            rock_count=self.vegetation.rock_count,
        )


# --------------------------------------------------------------------------- #
# Track generation
# --------------------------------------------------------------------------- #
def generate_control_points(layout: Layout) -> Tuple[Tuple[float, float, float], ...]:
    """Control points for a smooth closed loop, deterministic per seed.

    Point 0 is the start/finish. The loop runs clockwise seen from above.
    """
    rng = np.random.default_rng(layout.seed)
    phases = rng.uniform(0.0, math.tau, size=len(layout.waves))
    height_phases = rng.uniform(0.0, math.tau, size=3)
    rotation = rng.uniform(0.0, math.tau)

    points = []
    for i in range(layout.points):
        theta = -i / layout.points * math.tau            # clockwise
        r = 1.0
        for (k, amplitude), phase in zip(layout.waves, phases):
            r += amplitude * math.sin(k * theta + phase)
        r *= layout.radius

        angle = theta + rotation
        x = math.cos(angle) * r * layout.stretch[0]
        z = math.sin(angle) * r * layout.stretch[1]
        y = layout.elevation * (
            0.6 * math.sin(theta + height_phases[0])
            + 0.3 * math.sin(2 * theta + height_phases[1])
            + 0.1 * math.sin(4 * theta + height_phases[2])
        )
        points.append((round(x, 2), round(y, 2), round(z, 2)))
    return tuple(points)


# --------------------------------------------------------------------------- #
# The stages
# --------------------------------------------------------------------------- #
# Kenney Nature Kit material names -> sRGB colour, per biome. Twelve names
# cover every model used (listed by loading each .glb and reading its
# materials). `_defaultMat` and `dirt` are rock bodies, `grass` is moss on rock
# tops and bush foliage, `leafsGreen` is cacti and broadleaf trees alike.
def _tints(leafs_dark, leafs_green, grass, rock, rock_alt, bark_dark, bark,
           wood=(0.50, 0.37, 0.24), red=(0.68, 0.12, 0.10)):
    return (
        ("leafsDark", leafs_dark), ("leafsGreen", leafs_green), ("grass", grass),
        ("_defaultMat", rock), ("dirt", rock_alt),
        ("woodBarkDark", bark_dark), ("woodBark", bark), ("woodInner", (0.72, 0.60, 0.45)),
        ("wood", wood), ("woodDark", (0.33, 0.25, 0.18)),
        ("colorRed", red), ("colorRedDark", (red[0] * 0.78, red[1] * 0.75, red[2] * 0.75)),
    )


PINE_TINTS = _tints(leafs_dark=(0.12, 0.21, 0.12), leafs_green=(0.20, 0.30, 0.14),
                    grass=(0.26, 0.32, 0.15), rock=(0.50, 0.49, 0.47),
                    rock_alt=(0.44, 0.42, 0.39), bark_dark=(0.27, 0.20, 0.15),
                    bark=(0.40, 0.30, 0.21))
SNOW_TINTS = _tints(leafs_dark=(0.09, 0.16, 0.11), leafs_green=(0.14, 0.22, 0.13),
                    grass=(0.84, 0.87, 0.92), rock=(0.47, 0.48, 0.50),
                    rock_alt=(0.40, 0.41, 0.43), bark_dark=(0.24, 0.19, 0.15),
                    bark=(0.33, 0.26, 0.20), red=(0.62, 0.14, 0.12))
DESERT_TINTS = _tints(leafs_dark=(0.33, 0.35, 0.18), leafs_green=(0.28, 0.40, 0.19),
                      grass=(0.56, 0.46, 0.29), rock=(0.64, 0.40, 0.28),
                      rock_alt=(0.58, 0.35, 0.24), bark_dark=(0.38, 0.28, 0.20),
                      bark=(0.46, 0.34, 0.24), wood=(0.60, 0.46, 0.32))
AUTUMN_TINTS = _tints(leafs_dark=(0.11, 0.18, 0.11), leafs_green=(0.55, 0.30, 0.10),
                      grass=(0.33, 0.30, 0.14), rock=(0.40, 0.39, 0.37),
                      rock_alt=(0.34, 0.33, 0.31), bark_dark=(0.23, 0.17, 0.13),
                      bark=(0.30, 0.22, 0.16))

PINES = ("tree_pineTallA_detailed", "tree_pineTallB_detailed",
         "tree_pineTallC_detailed", "tree_pineTallD_detailed",
         "tree_pineRoundA", "tree_pineRoundC")
ROCKS = ("rock_largeA", "rock_largeB", "rock_largeC", "rock_largeD",
         "rock_largeE", "rock_tallA", "rock_tallB")
SMALL_ROCKS = ("rock_smallA", "rock_smallB", "rock_smallC")


STAGES: Tuple[Stage, ...] = (
    Stage(
        id="pine_ridge",
        name="Pine Ridge",
        country="Finland",
        blurb="Fast, flowing gravel through dense pine. Crests hide the exit of "
              "half the corners, so trust the pace notes.",
        layout=Layout(seed=11, radius=235.0, elevation=16.0,
                      waves=((2, 0.14), (3, 0.12), (6, 0.075), (9, 0.062), (11, 0.024))),
        surface=Surface("gravel_forest", "ground_forest",
                        dust=(0.74, 0.68, 0.58)),
        atmosphere=Atmosphere(
            sky="sky_cloudy", sun_direction=(-0.62, 0.48, 0.30),
            sun_color=(1.0, 0.95, 0.86), sun_strength=2.7,
            ambient=(0.10, 0.12, 0.15), fog_color=(0.42, 0.50, 0.60),
            fog_density=0.0011, exposure=0.85),
        vegetation=Vegetation(
            trees=PINES, tree_count=900, rocks=ROCKS + SMALL_ROCKS, rock_count=160,
            clutter=("plant_bush", "plant_bushLarge", "stump_old", "log",
                     "grass_large", "grass_leafsLarge"),
            clutter_count=700, tints=PINE_TINTS),
    ),
    Stage(
        id="red_mesa",
        name="Red Mesa",
        country="Morocco",
        blurb="Wide-open desert piste on loose red laterite. High speed, little "
              "grip at the edges, and nothing to hide behind if you get it wrong.",
        layout=Layout(seed=29, radius=255.0, elevation=9.0, stretch=(1.25, 0.8),
                      waves=((2, 0.20), (3, 0.08), (5, 0.06), (8, 0.03))),
        surface=Surface("gravel_desert", "ground_desert", grip_line=0.94,
                        grip_edge=0.76, grip_offroad=0.58, offroad_rolling=2.2,
                        dust=(0.80, 0.52, 0.36), dust_amount=1.6,
                        ground_tint=(1.0, 0.86, 0.74)),
        atmosphere=Atmosphere(
            sky="sky_clear", sun_direction=(0.40, 0.62, -0.35),
            sun_color=(1.0, 0.92, 0.80), sun_strength=3.1,
            ambient=(0.13, 0.12, 0.11), fog_color=(0.62, 0.58, 0.55),
            fog_density=0.0008, exposure=0.75),
        vegetation=Vegetation(
            trees=("cactus_tall", "cactus_short", "tree_fat"), tree_count=260,
            rocks=ROCKS, rock_count=260,
            clutter=("plant_bushSmall", "rock_smallA", "rock_smallB", "grass"),
            clutter_count=420, tints=DESERT_TINTS,
            tree_scale=(3.0, 6.0), rock_scale=(3.0, 9.0)),
        hill_height=42.0,
    ),
    Stage(
        id="frostfell",
        name="Frostfell",
        country="Sweden",
        blurb="Packed snow over ice. Grip is low everywhere and almost gone off "
              "the line - lean on the snowbanks and keep the car sideways.",
        layout=Layout(seed=47, radius=220.0, elevation=12.0,
                      waves=((2, 0.12), (3, 0.13), (7, 0.065), (10, 0.048))),
        surface=Surface("gravel_snow", "ground_snow", grip_line=0.70,
                        grip_edge=0.60, grip_offroad=0.46, offroad_rolling=3.4,
                        dust=(0.93, 0.95, 0.98), dust_amount=1.4),
        atmosphere=Atmosphere(
            sky="sky_snow", sun_direction=(0.55, 0.30, 0.55),
            sun_color=(1.0, 0.93, 0.85), sun_strength=2.4,
            ambient=(0.14, 0.16, 0.20), fog_color=(0.62, 0.68, 0.78),
            fog_density=0.0016, exposure=0.70, env_strength=0.30),
        vegetation=Vegetation(
            trees=PINES, tree_count=800, rocks=ROCKS, rock_count=120,
            clutter=("stump_round", "log_large", "rock_smallC"),
            clutter_count=260, tints=SNOW_TINTS),
    ),
    Stage(
        id="dusk_valley",
        name="Dusk Valley",
        country="Wales",
        blurb="Muddy forestry tracks at sunset, low light through the trees. "
              "Rutted and slippery - smooth hands win here.",
        layout=Layout(seed=73, radius=215.0, elevation=20.0,
                      waves=((2, 0.12), (4, 0.11), (7, 0.06), (11, 0.03))),
        surface=Surface("gravel_mud", "ground_leaves", grip_line=0.86,
                        grip_edge=0.72, grip_offroad=0.50, offroad_rolling=3.2,
                        dust=(0.40, 0.32, 0.24), dust_amount=0.8,
                        road_tint=(1.0, 0.86, 0.72), ground_tint=(0.95, 0.85, 0.72)),
        atmosphere=Atmosphere(
            sky="sky_sunset", sun_direction=(0.70, 0.17, 0.30),
            sun_color=(1.0, 0.62, 0.36), sun_strength=2.8,
            ambient=(0.08, 0.07, 0.10), fog_color=(0.52, 0.38, 0.30),
            fog_density=0.0016, exposure=0.80, env_strength=0.45),
        vegetation=Vegetation(
            trees=("tree_detailed", "tree_oak", "tree_default") + PINES[:2],
            tree_count=850, rocks=ROCKS, rock_count=140,
            clutter=("plant_bush", "plant_bushLarge", "stump_old", "log",
                     "grass_leafsLarge"),
            clutter_count=650, tints=AUTUMN_TINTS),
        hill_height=40.0,
    ),
)

STAGES_BY_ID: Dict[str, Stage] = {s.id: s for s in STAGES}


def get_stage(stage_id: str) -> Stage:
    try:
        return STAGES_BY_ID[stage_id]
    except KeyError as error:
        raise KeyError(f"unknown stage {stage_id!r}; "
                       f"have {', '.join(STAGES_BY_ID)}") from error
