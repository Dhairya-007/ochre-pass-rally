"""Core (Y-up) <-> Panda3D (Z-up) conversion.

The core uses x right, y up, z forward, with a heading `yaw` pointing along
(sin yaw, cos yaw) in the x-z plane. Panda3D uses x right, y forward, z up,
with heading H pointing along (-sin H, cos H).

So a core point (x, y, z) is Panda (x, z, y), and H = -degrees(yaw).

Swapping two axes is a reflection, which flips the handedness of every cross
product - and Ursina-style left-handed winding (what `meshbuild` produces)
becomes standard right-handed counter-clockwise winding. Meshes therefore
convert with a plain axis swap and no index reordering.
"""

from __future__ import annotations

import math
from typing import Tuple

from panda3d.core import LPoint3f, LVector3f


def to_panda(x: float, y: float, z: float) -> LPoint3f:
    return LPoint3f(x, z, y)


def direction_to_panda(x: float, y: float, z: float) -> LVector3f:
    return LVector3f(x, z, y)


def heading(yaw: float) -> float:
    return -math.degrees(yaw)


def hpr(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> Tuple[float, float, float]:
    """Panda HPR for a core yaw, nose-up pitch and lean-right roll (radians)."""
    return (-math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
