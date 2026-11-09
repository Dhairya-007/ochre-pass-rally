"""Dust, snow spray and mud: one particle system per race.

All particles live in a single dynamic Geom. Each frame the simulation is a
handful of numpy operations on fixed-size arrays and the camera-facing quads
are written back with one memoryview copy - no per-particle Python objects,
no per-particle scene-graph nodes. That keeps 600 particles under a
millisecond, where the old per-entity version spent 7 ms on 300.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData, Material,
                          NodePath, OmniBoundingVolume, SamplerState, TransparencyAttrib)

from .assets import AssetLibrary, linear_colour
from .geom import _format

RGB = Tuple[float, float, float]


class DustSystem:
    def __init__(self, parent: NodePath, assets: AssetLibrary, colour: RGB,
                 capacity: int = 600, seed: int = 5) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.colour = np.array(linear_colour(colour)[:3], dtype=np.float32)

        self.pos = np.zeros((capacity, 3), np.float32)       # Panda Z-up
        self.vel = np.zeros((capacity, 3), np.float32)
        self.size = np.zeros(capacity, np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.spin = np.zeros(capacity, np.float32)
        self.cursor = 0
        self._credit = 0.0

        vdata = GeomVertexData("dust", _format(), Geom.UH_dynamic)
        vdata.uncleanSetNumRows(capacity * 4)
        self.vdata = vdata
        prim = GeomTriangles(Geom.UH_static)
        base = np.arange(capacity, dtype=np.uint32)[:, None] * 4
        quads = (base + np.array([0, 1, 2, 0, 2, 3], np.uint32)).ravel()
        prim.setIndexType(Geom.NT_uint32)
        handle = prim.modifyVertices()
        handle.uncleanSetNumRows(len(quads))
        memoryview(handle).cast("B").cast("I")[:] = quads
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode("dust")
        node.addGeom(geom)
        node.setBounds(OmniBoundingVolume())      # particles roam; never cull
        node.setFinal(True)
        self.np = parent.attachNewNode(node)

        tex = assets.texture("particles/puff12.png", srgb=True)
        tex.setWrapU(SamplerState.WM_clamp)
        tex.setWrapV(SamplerState.WM_clamp)
        self.np.setTexture(tex, 1)
        mat = Material("dust")
        mat.setBaseColor((1, 1, 1, 1)); mat.setRoughness(1.0); mat.setMetallic(0.0)
        self.np.setMaterial(mat, 1)
        self.np.setTransparency(TransparencyAttrib.M_alpha)
        self.np.setDepthWrite(False)
        self.np.setBin("fixed", 10)
        self.np.setTwoSided(True)
        self._buffer = np.zeros((capacity * 4, 16), np.float32)
        self._corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
        self._uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)

    # ------------------------------------------------------------------ #
    def emit(self, x: float, y: float, z: float, count: int, spread: float,
             lift: float = 1.6, size: float = 0.7, life: float = 1.3,
             push: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Emit at a Panda-space point. `push` adds horizontal drift (x, y)."""
        for _ in range(count):
            i = self.cursor
            self.cursor = (self.cursor + 1) % self.capacity
            self.pos[i] = (x + self.rng.uniform(-0.5, 0.5), y + self.rng.uniform(-0.5, 0.5),
                           z + self.rng.uniform(0.0, 0.25))
            self.vel[i] = (push[0] + self.rng.uniform(-spread, spread),
                           push[1] + self.rng.uniform(-spread, spread),
                           lift * self.rng.uniform(0.5, 1.2))
            self.size[i] = size * self.rng.uniform(0.7, 1.3)
            self.max_life[i] = life * self.rng.uniform(0.7, 1.2)
            self.life[i] = self.max_life[i]
            self.spin[i] = self.rng.uniform(0, math.tau)

    def emit_rate(self, x, y, z, rate: float, dt: float, **kwargs) -> None:
        """Frame-rate independent emission at `rate` particles per second."""
        self._credit += rate * dt
        count = int(self._credit)
        if count:
            self._credit -= count
            self.emit(x, y, z, min(count, 10), **kwargs)

    # ------------------------------------------------------------------ #
    def update(self, dt: float, camera: NodePath) -> None:
        alive = self.life > 0.0
        self.life[alive] -= dt
        self.vel[alive] *= np.float32(0.96 ** (dt * 60))
        self.vel[alive, 2] -= np.float32(0.4 * dt)
        self.pos[alive] += self.vel[alive] * np.float32(dt)
        self.size[alive] += np.float32(1.9 * dt)

        quat = camera.getQuat(self.np.getParent())
        right = np.array(quat.getRight(), np.float32)
        up = np.array(quat.getUp(), np.float32)
        forward = np.array(quat.getForward(), np.float32)

        t = np.clip(self.life / self.max_life, 0.0, 1.0)
        # Fade in fast, out slowly.
        alpha = np.where(t > 0.85, (1.0 - t) / 0.15, t / 0.85) * 0.38
        alpha[self.life <= 0.0] = 0.0
        half = np.where(self.life > 0.0, self.size, 0.0)

        c, s = np.cos(self.spin), np.sin(self.spin)
        b = self._buffer.reshape(self.capacity, 4, 16)
        for k, (cx, cy) in enumerate(self._corners):
            rx = cx * c - cy * s
            ry = cx * s + cy * c
            b[:, k, 0:3] = self.pos + (right[None] * rx[:, None] + up[None] * ry[:, None]) * half[:, None]
            b[:, k, 3:6] = -forward
            b[:, k, 6:9] = right
            b[:, k, 9] = 1.0
            b[:, k, 10:13] = self.colour
            b[:, k, 13] = alpha
            b[:, k, 14:16] = self._uv[k]
        memoryview(self.vdata.modifyArray(0)).cast("B").cast("f")[:] = self._buffer.ravel()

    def clear(self) -> None:
        self.life[:] = 0.0

    def destroy(self) -> None:
        self.np.removeNode()
