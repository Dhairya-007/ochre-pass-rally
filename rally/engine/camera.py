"""Race cameras: chase, close chase, bonnet, and a slow orbit for menus.

The chase camera follows where the car is *going*, not where it points. In a
slide those differ by thirty degrees or more, and a camera glued to the nose
hides the slide; trailing the velocity shows it and makes countersteer
something you can judge by eye. A little spring lag and speed-dependent field
of view do the rest of the work of conveying pace.
"""

from __future__ import annotations

import math
from typing import Tuple

from panda3d.core import LPoint3f, LVector3f, NodePath

from ..terrain import Terrain
from ..vehicle import Vehicle
from .coords import to_panda

MODES = ("chase", "chase_far", "bonnet")


class RaceCamera:
    def __init__(self, base, terrain: Terrain) -> None:
        self.base = base
        self.terrain = terrain
        self.mode = "chase"
        self._pos = LPoint3f(0, 0, 0)
        self._look = LPoint3f(0, 0, 0)
        self._heading = 0.0
        self._fov = 68.0
        self.shake = 0.0

    def cycle(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def snap(self, car: Vehicle) -> None:
        self._heading = car.yaw
        self._pos, self._look = self._targets(car)

    def _targets(self, car: Vehicle) -> Tuple[LPoint3f, LPoint3f]:
        far = self.mode == "chase_far"
        distance = (9.5 if far else 6.6) + car.speed * 0.05
        height = (3.6 if far else 2.25) + car.speed * 0.012
        x = car.x - math.sin(self._heading) * distance
        z = car.z - math.cos(self._heading) * distance
        y = max(car.y + height, self.terrain.height_at(x, z) + 1.3)
        fx, fz = car.forward
        look = to_panda(car.x + fx * 5.0, car.y + (1.3 if far else 1.1), car.z + fz * 5.0)
        return to_panda(x, y, z), look

    def update(self, car: Vehicle, dt: float) -> None:
        cam = self.base.cam
        if self.mode == "bonnet":
            fx, fz = car.forward
            cam.setPos(to_panda(car.x + fx * 0.6, car.y + 1.25, car.z + fz * 0.6))
            cam.lookAt(to_panda(car.x + fx * 30, car.y + 0.9, car.z + fz * 30))
            cam.setR(cam.getR() + math.degrees(car.roll) * 0.6)
            self._set_fov(74 + min(18, car.speed * 0.32), dt)
            return

        # Swing towards the direction of travel, faster when there is speed.
        travel = math.atan2(*car.world_velocity()) if car.speed > 3 else car.yaw
        target = car.yaw + max(-0.6, min(0.6, math.atan2(math.sin(travel - car.yaw),
                                                         math.cos(travel - car.yaw)) * 0.65))
        diff = math.atan2(math.sin(target - self._heading), math.cos(target - self._heading))
        self._heading += diff * min(1.0, dt * 4.5)

        pos, look = self._targets(car)
        k = min(1.0, dt * 7.0)
        self._pos = self._pos + (pos - self._pos) * k
        self._look = self._look + (look - self._look) * min(1.0, dt * 10.0)
        cam.setPos(self._pos)
        if self.shake > 0.01:
            import random
            jitter = self.shake * 0.08
            cam.setPos(cam.getPos() + LVector3f(random.uniform(-jitter, jitter),
                                                random.uniform(-jitter, jitter),
                                                random.uniform(-jitter, jitter)))
            self.shake *= 0.86
        cam.lookAt(self._look)
        self._set_fov(64 + min(16, car.speed * 0.28), dt)

    def _set_fov(self, target: float, dt: float) -> None:
        self._fov += (target - self._fov) * min(1.0, dt * 3.0)
        self.base.camLens.setFov(self._fov)

    def orbit(self, x: float, y: float, z: float, t: float, radius: float = 12.0,
              height: float = 2.1, offset: float = 1.8) -> None:
        """Slow turntable around a car, framing it right of the menu panel.

        `offset` aims the camera that far to the car's left as seen from the
        camera, so the car sits in the open right-hand part of the screen.
        """
        cam = self.base.cam
        a = 0.6 + math.sin(t * 0.12) * 0.9            # sweep back and forth
        cx, cz = x + math.sin(a) * radius, z + math.cos(a) * radius
        # Camera-relative left of the car, in the x-z plane.
        dx, dz = x - cx, z - cz
        n = math.hypot(dx, dz) or 1.0
        lx, lz = -dz / n, dx / n
        cam.setPos(to_panda(cx, y + height, cz))
        cam.lookAt(to_panda(x + lx * offset, y + 0.75, z + lz * offset))
        self.base.camLens.setFov(42)
