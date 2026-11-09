"""Car-versus-scenery and car-versus-car collisions.

A car is approximated by two circles along its axis - one over each axle -
which hugs a long, narrow shape far better than one big circle and costs
almost nothing. Scenery is a set of circles (tree trunks, rock bodies) in a
uniform grid, so each car only ever tests the handful of obstacles near it.

Response is a plain impulse: separate the shapes, then remove the closing
velocity along the contact normal with some restitution, and bleed off a
share of the sliding velocity as friction. Both functions return the impact
speed so the game can play a sound, shake the camera or spray debris.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .vehicle import Vehicle

CAR_CIRCLE_RADIUS = 1.0
CAR_CIRCLE_OFFSET = 1.15        # metres either side of the centre, along the car


@dataclass(frozen=True)
class Obstacle:
    x: float
    z: float
    radius: float


def car_circles(vehicle: Vehicle) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    fx, fz = vehicle.forward
    return ((vehicle.x + fx * CAR_CIRCLE_OFFSET, vehicle.z + fz * CAR_CIRCLE_OFFSET),
            (vehicle.x - fx * CAR_CIRCLE_OFFSET, vehicle.z - fz * CAR_CIRCLE_OFFSET))


class StaticColliders:
    """Solid scenery in a uniform spatial hash."""

    def __init__(self, obstacles: Iterable[Obstacle], cell: float = 8.0) -> None:
        self.cell = cell
        self.grid: Dict[Tuple[int, int], List[Obstacle]] = {}
        self.count = 0
        for obstacle in obstacles:
            if obstacle.radius <= 0.0:
                continue
            reach = obstacle.radius + CAR_CIRCLE_RADIUS
            for gx in range(self._cell(obstacle.x - reach), self._cell(obstacle.x + reach) + 1):
                for gz in range(self._cell(obstacle.z - reach), self._cell(obstacle.z + reach) + 1):
                    self.grid.setdefault((gx, gz), []).append(obstacle)
            self.count += 1

    def _cell(self, value: float) -> int:
        return int(math.floor(value / self.cell))

    def nearby(self, x: float, z: float) -> Sequence[Obstacle]:
        return self.grid.get((self._cell(x), self._cell(z)), ())

    # ------------------------------------------------------------------ #
    def resolve(self, vehicle: Vehicle, restitution: float = 0.25,
                friction: float = 0.35) -> float:
        """Push the car out of anything it is inside. Returns impact speed."""
        worst = 0.0
        for cx, cz in car_circles(vehicle):
            for obstacle in self.nearby(cx, cz):
                dx, dz = cx - obstacle.x, cz - obstacle.z
                distance = math.hypot(dx, dz)
                overlap = obstacle.radius + CAR_CIRCLE_RADIUS - distance
                if overlap <= 0.0:
                    continue
                if distance < 1e-6:
                    nx, nz = -vehicle.forward[0], -vehicle.forward[1]
                else:
                    nx, nz = dx / distance, dz / distance

                vehicle.x += nx * overlap
                vehicle.z += nz * overlap
                cx += nx * overlap
                cz += nz * overlap

                vx, vz = vehicle.world_velocity()
                closing = vx * nx + vz * nz
                if closing < 0.0:
                    # Remove the closing component, bounce a little, and drag
                    # the tangential component as the bodywork scrapes.
                    tx, tz = vx - closing * nx, vz - closing * nz
                    vx = tx * (1.0 - friction) - closing * restitution * nx
                    vz = tz * (1.0 - friction) - closing * restitution * nz
                    vehicle.set_world_velocity(vx, vz)
                    worst = max(worst, -closing)
        return worst


def resolve_cars(a: Vehicle, b: Vehicle, restitution: float = 0.35,
                 a_movable: bool = True, b_movable: bool = True) -> float:
    """Separate two cars and exchange momentum. Returns impact speed.

    Set `b_movable=False` for a remote car in multiplayer: its owner is the
    authority on where it is, so only the local car reacts.
    """
    if not (a_movable or b_movable):
        return 0.0
    worst = 0.0
    for ax, az in car_circles(a):
        for bx, bz in car_circles(b):
            dx, dz = ax - bx, az - bz
            distance = math.hypot(dx, dz)
            overlap = 2.0 * CAR_CIRCLE_RADIUS - distance
            if overlap <= 0.0:
                continue
            nx, nz = (dx / distance, dz / distance) if distance > 1e-6 else (1.0, 0.0)

            ma = a.config.mass if a_movable else math.inf
            mb = b.config.mass if b_movable else math.inf
            inv_a = 0.0 if math.isinf(ma) else 1.0 / ma
            inv_b = 0.0 if math.isinf(mb) else 1.0 / mb
            share_a = inv_a / (inv_a + inv_b)
            share_b = inv_b / (inv_a + inv_b)

            a.x += nx * overlap * share_a
            a.z += nz * overlap * share_a
            b.x -= nx * overlap * share_b
            b.z -= nz * overlap * share_b

            avx, avz = a.world_velocity()
            bvx, bvz = b.world_velocity()
            closing = (avx - bvx) * nx + (avz - bvz) * nz
            if closing >= 0.0:
                continue
            impulse = -(1.0 + restitution) * closing / (inv_a + inv_b)
            if a_movable:
                a.set_world_velocity(avx + impulse * inv_a * nx, avz + impulse * inv_a * nz)
            if b_movable:
                b.set_world_velocity(bvx - impulse * inv_b * nx, bvz - impulse * inv_b * nz)
            worst = max(worst, -closing)
    return worst
