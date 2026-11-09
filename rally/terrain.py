"""The ground: a height field, a distance-to-road field, and the surface grip
that follows from them.

One idea does a lot of work here. We compute, for every grid point, the distance
to the nearest centreline sample. That single field then drives three things:

  * height  - the terrain is flattened into the road near the centreline, so the
              stage sits in the landscape instead of on top of it
  * grip    - packed gravel on the line, loose stuff on the verge, scrub beyond
  * drag    - rolling resistance climbs sharply once you leave the stage

Built with numpy; nothing here imports the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import WORLD, WorldConfig
from .mathutils import fbm, smoothstep
from .meshbuild import MeshArrays
from .track import Track


class Terrain:
    """Sampleable surface fields plus the terrain mesh."""

    def __init__(self, track: Track, config: WorldConfig = WORLD) -> None:
        self.track = track
        self.config = config
        self.size = config.size
        self.resolution = config.resolution
        self.cell = config.size / config.resolution

        self._build_fields()

    # ------------------------------------------------------------------ #
    # Field construction
    # ------------------------------------------------------------------ #
    def _build_fields(self) -> None:
        cfg = self.config
        res = self.resolution
        half = self.size * 0.5

        axis = np.linspace(-half, half, res + 1)
        self.grid_x, self.grid_z = np.meshgrid(axis, axis, indexing="xy")

        # --- distance to the centreline, and the road height there ---------
        px = self.track.points[:, 0]
        pz = self.track.points[:, 2]
        py = self.track.points[:, 1]

        best_d2 = np.full(self.grid_x.shape, np.inf)
        best_k = np.zeros(self.grid_x.shape, dtype=np.int32)

        # Every third sample is plenty for a field this coarse and keeps the
        # build under a second.
        stride = 3
        for k in range(0, self.track.count, stride):
            dx = self.grid_x - px[k]
            dz = self.grid_z - pz[k]
            d2 = dx * dx + dz * dz
            closer = d2 < best_d2
            best_d2 = np.where(closer, d2, best_d2)
            best_k = np.where(closer, k, best_k)

        self.distance = np.sqrt(best_d2)
        self.road_height = py[best_k]

        # --- natural terrain, then flattened towards the road --------------
        natural = (
            fbm(self.grid_x * cfg.hill_scale + 11.0, self.grid_z * cfg.hill_scale + 7.0)
            * cfg.hill_height
            - 10.0
        )
        natural += (
            fbm(self.grid_x * cfg.detail_scale, self.grid_z * cfg.detail_scale)
            * cfg.detail_height
        )

        edge = self.track.config.road_half_width
        blend = smoothstep(edge + cfg.flatten_start, edge + cfg.flatten_end, self.distance)
        self.height = self.road_height * (1.0 - blend) + natural * blend

        # Cached for colouring the mesh.
        self._surface_noise = fbm(self.grid_x * 0.02 + 3.0, self.grid_z * 0.02 + 9.0)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _bilinear(self, field: np.ndarray, x: float, z: float) -> float:
        """Bilinear lookup in grid space, clamped at the edges."""
        res = self.resolution
        half = self.size * 0.5
        fx = (x + half) / self.cell
        fz = (z + half) / self.cell
        fx = min(max(fx, 0.0), res - 1e-3)
        fz = min(max(fz, 0.0), res - 1e-3)

        x0, z0 = int(fx), int(fz)
        tx, tz = fx - x0, fz - z0

        a = field[z0, x0]
        b = field[z0, x0 + 1]
        c = field[z0 + 1, x0]
        d = field[z0 + 1, x0 + 1]
        top = a * (1.0 - tx) + b * tx
        bottom = c * (1.0 - tx) + d * tx
        return float(top * (1.0 - tz) + bottom * tz)

    def height_at(self, x: float, z: float) -> float:
        return self._bilinear(self.height, x, z)

    def distance_at(self, x: float, z: float) -> float:
        """Metres from the centreline."""
        return self._bilinear(self.distance, x, z)

    def grip_at(self, x: float, z: float) -> float:
        cfg = self.config
        edge = self.track.config.road_half_width
        d = self.distance_at(x, z)
        if d < edge - 0.5:
            return cfg.grip_line
        if d < edge + cfg.edge_band:
            return cfg.grip_edge
        return cfg.grip_offroad

    def rolling_at(self, x: float, z: float) -> float:
        """Rolling-resistance multiplier; off the stage it bites."""
        edge = self.track.config.road_half_width
        if self.distance_at(x, z) < edge + 1.5:
            return 1.0
        return self.config.offroad_rolling

    def is_offroad(self, x: float, z: float) -> bool:
        return self.distance_at(x, z) > self.track.config.road_half_width + 2.0

    def slope_ahead(self, x: float, z: float, dx: float, dz: float,
                    reach: float = 4.2) -> float:
        """Gradient over `reach` metres in the given horizontal direction."""
        here = self.height_at(x, z)
        there = self.height_at(x + dx * reach, z + dz * reach)
        return (there - here) / reach

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def build_mesh(self) -> MeshArrays:
        """Terrain geometry with vertex colours and smooth normals."""
        res = self.resolution
        h = self.height

        vertices = np.stack(
            [self.grid_x.ravel(), h.ravel(), self.grid_z.ravel()], axis=1
        )

        # Normals from central differences on the height grid - far cheaper and
        # smoother than accumulating face normals.
        dzdx = np.gradient(h, self.cell, axis=1)
        dzdz = np.gradient(h, self.cell, axis=0)
        normals = np.stack(
            [-dzdx.ravel(), np.ones(h.size), -dzdz.ravel()], axis=1
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Triangle indices, two per cell, built vectorised. Wound for
        # Ursina's left-handed front faces - see meshbuild's docstring.
        row = np.arange(res)
        col = np.arange(res)
        cc, rr = np.meshgrid(col, row, indexing="xy")
        a = (rr * (res + 1) + cc).ravel()
        b = a + 1
        c = a + (res + 1)
        d = c + 1
        triangles = np.concatenate(
            [np.stack([a, b, c], axis=1), np.stack([b, d, c], axis=1)], axis=0
        ).astype(np.int32)

        colors = self._vertex_colors()
        uvs = np.stack(
            [(self.grid_x.ravel() / 12.0), (self.grid_z.ravel() / 12.0)], axis=1
        )
        return MeshArrays(vertices, triangles, normals, colors, uvs)

    def _vertex_colors(self) -> np.ndarray:
        """Grass through dry grass, rock on the tops, dust near the verge."""
        from .config import PALETTE

        grass = np.array(PALETTE.grass)
        dry = np.array(PALETTE.dry_grass)
        rock = np.array(PALETTE.rock)
        verge = np.array(PALETTE.verge)

        n = self._surface_noise.ravel()[:, None]
        base = grass[None, :] * (1.0 - n) + dry[None, :] * n

        height_t = smoothstep(16.0, 34.0, self.height.ravel())[:, None]
        base = base * (1.0 - height_t) + rock[None, :] * height_t

        edge = self.track.config.road_half_width
        verge_t = 1.0 - smoothstep(edge, edge + 16.0, self.distance.ravel())[:, None]
        base = base * (1.0 - verge_t) + verge[None, :] * verge_t

        shade = (0.86 + n * 0.24)
        rgb = np.clip(base * shade, 0.0, 1.0)
        return np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)

    # ------------------------------------------------------------------ #
    def world_limit(self) -> float:
        """Half-extent of the drivable area."""
        return self.size * 0.5 - self.config.boundary_margin
