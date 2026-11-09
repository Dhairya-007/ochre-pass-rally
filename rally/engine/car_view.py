"""A car on screen.

Node layout, so suspension movement looks right:

    root   - position and heading, plus the slope of the ground under it
    ├── body     - leans and pitches on its suspension (Vehicle.roll/pitch)
    └── pivot x4 - wheel hubs: front ones steer
        └── wheel   - spins at road speed

The wheels stay planted while the body rolls over them, which is what sells
weight transfer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from panda3d.core import (BillboardEffect, Material, NodePath, TextNode)

from ..cars import CARS_BY_ID, LIVERY_COLOURS, CarSpec
from ..terrain import Terrain
from ..vehicle import Vehicle
from .assets import AssetLibrary, linear_colour
from .coords import to_panda


def _accent(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """A contrasting stripe colour for two-tone liveries."""
    lum = 0.3 * rgb[0] + 0.6 * rgb[1] + 0.1 * rgb[2]
    return (0.06, 0.06, 0.07) if lum > 0.45 else (0.92, 0.92, 0.90)


class CarView:
    def __init__(self, car_id: str, livery: int, assets: AssetLibrary, parent: NodePath,
                 terrain: Terrain, label: Optional[str] = None, font=None) -> None:
        self.spec: CarSpec = CARS_BY_ID[car_id]
        self.terrain = terrain
        self.root = parent.attachNewNode(f"car-{car_id}")
        model = assets.model(f"models/cars/{self.spec.model}.glb", roughness=0.5)
        model.reparentTo(self.root)
        model.setScale(self.spec.scale)
        self.model = model

        self._paint(model, livery)
        self.brake_lights: List[Material] = [m for m in model.findAllMaterials()
                                             if m.getName() == "lightBack"]
        for m in model.findAllMaterials():
            if m.getName() == "lightFront":
                m.setEmission((2.5, 2.4, 2.1, 1))

        self.body = model.find("**/body")
        self.pivots: List[Tuple[NodePath, NodePath, bool]] = []
        for wheel in model.findAllMatches("**/wheel_*"):
            pivot = model.attachNewNode(f"pivot-{wheel.getName()}")
            pivot.setPos(wheel.getPos(model))
            wheel.wrtReparentTo(pivot)
            self.pivots.append((pivot, wheel, "front" in wheel.getName()))
        self._wheel_radius = 0.30 * self.spec.scale

        self.label: Optional[NodePath] = None
        if label:
            text = TextNode("name")
            text.setText(label)
            text.setAlign(TextNode.ACenter)
            if font:
                text.setFont(font)
            text.setTextColor(1, 1, 1, 1)
            text.setShadow(0.04, 0.04)
            text.setShadowColor(0, 0, 0, 0.7)
            self.label = self.root.attachNewNode(text)
            self.label.setPos(0, 0, 2.4)
            self.label.setScale(0.30)
            self.label.setEffect(BillboardEffect.makePointEye())
            self.label.setLightOff(1)
            self.label.setShaderOff(1)
            self.label.setDepthWrite(False)
            self.label.setBin("fixed", 50)

    @staticmethod
    def _body_paint(model: NodePath) -> str:
        """Name of the paint material on the bodywork (not the wheel rims).

        Kenney models vary: on most the only paint is the body, but the race
        car's yellow paint is on its rims and its body is red.
        """
        from collections import Counter
        from panda3d.core import MaterialAttrib
        counts = Counter()
        for gnp in model.findAllMatches("**/+GeomNode"):
            if "wheel" in gnp.getName():
                continue
            node = gnp.node()
            for i in range(node.getNumGeoms()):
                attrib = node.getGeomState(i).getAttrib(MaterialAttrib)
                material = attrib.getMaterial() if attrib else None
                if material and material.getName().startswith("paint"):
                    counts[material.getName()] += 1
        return counts.most_common(1)[0][0] if counts else ""

    def _paint(self, model: NodePath, livery: int) -> None:
        _, primary = LIVERY_COLOURS[livery % len(LIVERY_COLOURS)]
        body = self._body_paint(model)
        paints = [m for m in model.findAllMaterials() if m.getName().startswith("paint")]
        for m in paints:
            colour = primary if m.getName() == body else _accent(primary)
            m.setBaseColor(linear_colour(colour))
            m.setRoughness(0.32)
            m.setMetallic(0.35)
        for m in model.findAllMaterials():
            name = m.getName()
            if name == "window":
                m.setBaseColor(linear_colour((0.06, 0.08, 0.10)))
                m.setRoughness(0.06)
                m.setMetallic(0.85)
            elif name == "carTire":
                m.setBaseColor(linear_colour((0.07, 0.07, 0.07)))
                m.setRoughness(0.92)
            elif name == "plastic":
                m.setBaseColor(linear_colour((0.13, 0.13, 0.14)))
                m.setRoughness(0.7)

    # ------------------------------------------------------------------ #
    def update(self, vehicle: Vehicle, braking: float = 0.0, distance: float = 0.0) -> None:
        x, y, z = vehicle.x, vehicle.y, vehicle.z
        self.root.setPos(to_panda(x, y, z))

        ground_pitch, ground_roll = 0.0, 0.0
        if not vehicle.airborne:
            fx, fz = vehicle.forward
            rx, rz = vehicle.right
            reach, half = 1.35, 0.85
            h = self.terrain.height_at
            front = h(x + fx * reach, z + fz * reach)
            back = h(x - fx * reach, z - fz * reach)
            left = h(x - rx * half, z - rz * half)
            right = h(x + rx * half, z + rz * half)
            ground_pitch = math.atan2(front - back, reach * 2)
            ground_roll = math.atan2(left - right, half * 2)
        self.root.setHpr(-math.degrees(vehicle.yaw), math.degrees(ground_pitch),
                         math.degrees(ground_roll))

        if not self.body.isEmpty():
            self.body.setHpr(0, math.degrees(vehicle.pitch) * 1.4,
                             math.degrees(vehicle.roll) * 1.3)

        steer = -math.degrees(vehicle.steer_angle)
        spin = -math.degrees(vehicle.wheel_spin)
        for pivot, wheel, front in self.pivots:
            pivot.setH(steer if front else 0.0)
            wheel.setP(spin)

        glow = 0.4 + 3.6 * braking
        for m in self.brake_lights:
            m.setEmission((glow, glow * 0.12, glow * 0.08, 1))

        if self.label is not None:
            # Hidden on cars right beside you (it would fill the screen), faded
            # out in the distance where it would just be clutter.
            near = max(0.0, min(1.0, (distance - 10.0) / 8.0))
            far = max(0.0, min(1.0, (90.0 - distance) / 30.0))
            self.label.setAlphaScale(near * far)

    def destroy(self) -> None:
        self.root.removeNode()
