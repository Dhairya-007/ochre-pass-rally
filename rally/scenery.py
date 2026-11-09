"""Where every tree, rock, bush and piece of rally furniture goes.

Placement is pure data so that two consumers can share it exactly: the renderer
draws a model at each placement, and the collision system puts a circle at
each one that is solid. If they were computed separately, sooner or later a car
would hit an invisible tree or drive through a visible one.

Deterministic per stage, so every player in a multiplayer race sees the same
forest without it ever being sent over the network.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .stages import Stage
from .terrain import Terrain
from .track import Track


@dataclass(frozen=True)
class Placement:
    kind: str           # "tree" | "rock" | "clutter" | "furniture"
    model: str          # file stem under assets/models/nature
    x: float
    y: float            # ground height at (x, z)
    z: float
    scale: float
    yaw: float          # radians
    radius: float       # collision radius in metres; 0 means drive-through


# Collision radius as a fraction of model scale. Measured from the Kenney
# models' bounds: a pine trunk is ~4% of its scaled height across, a large rock
# ~40% of its scale. Bushes and grass are deliberately soft (0).
RADIUS_PER_SCALE = {
    "tree_": 0.045, "cactus_tall": 0.06, "cactus_short": 0.08,
    "rock_large": 0.40, "rock_tall": 0.30, "rock_small": 0.0,
    "stump": 0.12, "log_large": 0.10, "log": 0.0,
    "plant_": 0.0, "grass": 0.0, "fence": 0.0, "tent": 0.25, "sign": 0.03,
}


def collision_radius(model: str, scale: float) -> float:
    for prefix, factor in RADIUS_PER_SCALE.items():
        if model.startswith(prefix):
            return factor * scale
    return 0.0


class SceneryPlanner:
    def __init__(self, stage: Stage, track: Track, terrain: Terrain) -> None:
        self.stage = stage
        self.track = track
        self.terrain = terrain
        self.rng = random.Random(stage.layout.seed * 7919 + 3)
        self.half = track.config.road_half_width
        self.limit = terrain.world_limit() - 15.0

    # ------------------------------------------------------------------ #
    def plan(self) -> List[Placement]:
        veg = self.stage.vegetation
        out: List[Placement] = []
        out += self._scatter("tree", veg.trees, veg.tree_count, veg.tree_scale,
                             inner=9.0, outer=95.0, bias=1.8)
        out += self._scatter("rock", veg.rocks, veg.rock_count, veg.rock_scale,
                             inner=3.5, outer=60.0, bias=1.4)
        out += self._scatter("clutter", veg.clutter, veg.clutter_count,
                             veg.clutter_scale, inner=2.5, outer=40.0, bias=1.6)
        out += self._furniture()
        return out

    def _scatter(self, kind: str, models: Tuple[str, ...], count: int,
                 scale_range: Tuple[float, float], inner: float, outer: float,
                 bias: float) -> List[Placement]:
        """Scatter `count` models in a band either side of the road.

        `bias` > 1 crowds them towards the road edge, which is what makes a
        forest stage feel like a corridor rather than a park.
        """
        if not models or count <= 0:
            return []
        out: List[Placement] = []
        attempts = 0
        while len(out) < count and attempts < count * 12:
            attempts += 1
            i = self.rng.randrange(self.track.count)
            centre, right = self.track.points[i], self.track.rights[i]
            side = -1.0 if self.rng.random() < 0.5 else 1.0
            distance = self.half + inner + (outer - inner) * self.rng.random() ** bias
            x = float(centre[0] + right[0] * distance * side) + self.rng.uniform(-6, 6)
            z = float(centre[2] + right[2] * distance * side) + self.rng.uniform(-6, 6)
            if abs(x) > self.limit or abs(z) > self.limit:
                continue
            if self.terrain.distance_at(x, z) < self.half + inner * 0.8:
                continue
            model = self.rng.choice(models)
            scale = self.rng.uniform(*scale_range)
            out.append(Placement(kind, model, x, self.terrain.height_at(x, z), z,
                                 scale, self.rng.uniform(0, math.tau),
                                 collision_radius(model, scale)))
        return out

    def _furniture(self) -> List[Placement]:
        """Signs at the hairpins, log barriers on their outside, tents by the start."""
        out: List[Placement] = []
        for corner in self.track.sharpest_corners(5, min_separation=40):
            _, direction = self.track.curvature_at(corner)
            outward = -1.0 if direction > 0 else 1.0
            for step in range(-2, 3):
                i = (corner + step * 4) % self.track.count
                c, r, t = self.track.points[i], self.track.rights[i], self.track.tangents[i]
                x = float(c[0] + r[0] * (self.half + 3.0) * outward)
                z = float(c[2] + r[2] * (self.half + 3.0) * outward)
                out.append(Placement("furniture", "log_large", x,
                                     self.terrain.height_at(x, z), z, 3.2,
                                     math.atan2(float(t[0]), float(t[2])) + math.pi / 2,
                                     collision_radius("log_large", 3.2)))
            i = (corner - 30) % self.track.count
            c, r = self.track.points[i], self.track.rights[i]
            x = float(c[0] - r[0] * (self.half + 2.0) * outward)
            z = float(c[2] - r[2] * (self.half + 2.0) * outward)
            out.append(Placement("furniture", "sign", x, self.terrain.height_at(x, z),
                                 z, 3.0, self.track.heading_at(i) + math.pi, 0.0))

        for j, offset in enumerate((-14, -9, 12, 17)):
            i = (offset * 2) % self.track.count
            c, r = self.track.points[i], self.track.rights[i]
            side = 1.0 if j % 2 else -1.0
            x = float(c[0] + r[0] * (self.half + 11.0) * side)
            z = float(c[2] + r[2] * (self.half + 11.0) * side)
            model = "tent_detailedOpen" if j < 2 else "tent_smallOpen"
            out.append(Placement("furniture", model, x, self.terrain.height_at(x, z),
                                 z, 4.0, self.track.heading_at(i) + side * math.pi / 2,
                                 collision_radius(model, 4.0)))
        return out
