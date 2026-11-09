"""Builds a stage's visible world from the pure track/terrain/scenery data."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from panda3d.core import (Material, NodePath, SamplerState, Texture,
                          TransparencyAttrib, PNMImage, CullFaceAttrib)

from ..mathutils import fbm, smoothstep
from ..meshbuild import MeshArrays, MeshBuilder, box
from ..scenery import Placement
from ..stages import Stage
from ..terrain import Terrain
from ..track import Track
from .assets import AssetLibrary, linear_colour
from .coords import to_panda
from .geom import make_geom_node

CHUNK = 100.0


class StageView:
    """Every visible part of a stage, under one root node."""

    def __init__(self, stage: Stage, track: Track, terrain: Terrain,
                 placements: Sequence[Placement], assets: AssetLibrary,
                 parent: NodePath, density: float = 1.0, progress=None) -> None:
        self.stage = stage
        self.track = track
        self.terrain = terrain
        self.assets = assets
        self.root = parent.attachNewNode(f"stage-{stage.id}")
        self._progress = progress or (lambda message, fraction: None)

        self._progress("Grading terrain", 0.1)
        self._build_ground()
        self._progress("Laying gravel", 0.25)
        self._build_road()
        self._build_posts()
        self._build_start()
        self._progress("Planting the forest", 0.35)
        self._build_scenery(placements, density)
        self._progress("Ready", 1.0)

    # ------------------------------------------------------------------ #
    def _build_ground(self) -> None:
        t = self.terrain
        res = t.resolution
        s = self.stage.surface
        vertices = np.stack([t.grid_x.ravel(), t.height.ravel(), t.grid_z.ravel()], axis=1)

        dhdx = np.gradient(t.height, t.cell, axis=1)
        dhdz = np.gradient(t.height, t.cell, axis=0)
        normals = np.stack([-dhdx.ravel(), np.ones(t.height.size), -dhdz.ravel()], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
        a = (rr * (res + 1) + cc).ravel()
        b, c = a + 1, a + (res + 1)
        d = c + 1
        triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.int32)

        # Macro variation stops the tiling texture reading as a grid at range.
        macro = fbm(t.grid_x * 0.006 + 5, t.grid_z * 0.006 + 2).ravel()
        detail = fbm(t.grid_x * 0.05, t.grid_z * 0.05).ravel()
        shade = 0.78 + macro * 0.30 + (detail - 0.5) * 0.10
        tint = np.array(linear_colour(s.ground_tint)[:3])
        rgb = np.clip(shade[:, None] * tint[None, :], 0, 1)
        # Slopes go rockier and darker; the verge picks up road dust.
        slope = 1.0 - normals[:, 1]
        rgb *= (1.0 - np.clip(slope * 1.6, 0, 0.35))[:, None]
        edge = self.track.config.road_half_width
        verge = 1.0 - smoothstep(edge + 1.0, edge + 9.0, t.distance.ravel())
        dust = np.array(linear_colour(s.dust)[:3])
        rgb = rgb * (1.0 - verge[:, None] * 0.45) + dust[None, :] * verge[:, None] * 0.45
        colors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
        uvs = np.stack([t.grid_x.ravel(), t.grid_z.ravel()], axis=1) / s.ground_tile

        mesh = MeshArrays(vertices, triangles, normals, colors, uvs)
        ground = self.root.attachNewNode(make_geom_node(mesh, "ground"))
        self.assets.apply_pbr(ground, s.ground_texture)

    # ------------------------------------------------------------------ #
    def _build_road(self) -> None:
        """A ribbon with feathered edges so gravel fades into the verge."""
        track = self.track
        s = self.stage.surface
        half = track.config.road_half_width
        n = track.count
        # (fraction of half-width, alpha, rut darkening)
        columns = ((-1.40, 0.0, 0.0), (-1.08, 0.85, 0.0), (-0.62, 1.0, 0.30),
                   (-0.42, 1.0, 0.30), (0.0, 1.0, 0.0), (0.42, 1.0, 0.30),
                   (0.62, 1.0, 0.30), (1.08, 0.85, 0.0), (1.40, 0.0, 0.0))
        cols = len(columns)
        verts, uvs, colors, norms = [], [], [], []
        tint = np.array(linear_colour(s.road_tint)[:3])
        rng = np.random.default_rng(self.stage.layout.seed)
        width = half * 2 * 1.40
        for i in range(n + 1):
            k = i % n
            centre, right = track.points[k], track.rights[k]
            along = float(track.cumulative[i]) / s.road_tile
            wear = 0.92 + rng.uniform(-0.04, 0.04)
            for frac, alpha, rut in columns:
                x = float(centre[0] + right[0] * half * frac)
                z = float(centre[2] + right[2] * half * frac)
                ground = self.terrain.height_at(x, z)
                y = max(float(centre[1]), ground) + 0.06 + (0.03 if abs(frac) < 0.3 else 0.0)
                verts.append((x, y, z))
                norms.append((0.0, 1.0, 0.0))
                uvs.append(((frac + 1.40) * half / s.road_tile * 1.1, along))
                rgb = tint * wear * (1.0 - rut * 0.45)
                colors.append((*rgb, alpha))
        tris = []
        for i in range(n):
            base = i * cols
            for c in range(cols - 1):
                a, b = base + c, base + c + 1
                cc, d = a + cols, b + cols
                tris += [(a, b, cc), (b, d, cc)]
        mesh = MeshArrays(np.array(verts), np.array(tris, dtype=np.int32),
                          np.array(norms), np.array(colors), np.array(uvs))
        road = self.root.attachNewNode(make_geom_node(mesh, "road"))
        self.assets.apply_pbr(road, s.road_texture)
        road.setTransparency(TransparencyAttrib.M_alpha)
        road.setDepthOffset(2)
        road.setBin("fixed", 0)

    # ------------------------------------------------------------------ #
    def _build_posts(self) -> None:
        """Marker posts every ~40 m down both edges; tall orange poles on snow."""
        track = self.track
        half = track.config.road_half_width
        snow = self.stage.surface.road_texture == "gravel_snow"
        builder = MeshBuilder()
        step = max(1, int(40.0 / (track.length / track.count)))
        post = box(0.12, 2.4 if snow else 1.1, 0.12)
        cap = box(0.13, 0.25, 0.13)
        body_colour = (0.95, 0.42, 0.05) if snow else (0.92, 0.92, 0.90)
        for i in range(0, track.count, step):
            centre, right = track.points[i], track.rights[i]
            for side in (-1.0, 1.0):
                x = float(centre[0] + right[0] * (half + 2.2) * side)
                z = float(centre[2] + right[2] * (half + 2.2) * side)
                y = self.terrain.height_at(x, z)
                height = 2.4 if snow else 1.1
                builder.add(post, position=(x, y + height / 2, z), color=body_colour)
                if not snow:
                    builder.add(cap, position=(x, y + height - 0.12, z), color=(0.80, 0.10, 0.08))
        posts = self.root.attachNewNode(make_geom_node(builder.build(), "posts"))
        material = Material("posts")
        material.setBaseColor((1, 1, 1, 1))
        material.setRoughness(0.55)
        material.setMetallic(0.0)
        posts.setMaterial(material, 1)

    # ------------------------------------------------------------------ #
    def _build_start(self) -> None:
        track = self.track
        half = track.config.road_half_width
        centre, right, tangent = track.points[0], track.rights[0], track.tangents[0]
        yaw = math.atan2(float(tangent[0]), float(tangent[2]))
        ground = float(centre[1])

        frame = MeshBuilder()
        leg = box(0.6, 7.6, 0.6)
        for side in (-1.0, 1.0):
            x = float(centre[0] + right[0] * (half + 1.8) * side)
            z = float(centre[2] + right[2] * (half + 1.8) * side)
            frame.add(leg, position=(x, self.terrain.height_at(x, z) + 3.8, z),
                      rotation=(0.0, yaw, 0.0), color=(0.08, 0.08, 0.09))
        truss = box((half + 2.4) * 2, 0.35, 0.5)
        frame.add(truss, position=(float(centre[0]), ground + 7.9, float(centre[2])),
                  rotation=(0.0, yaw, 0.0), color=(0.08, 0.08, 0.09))
        node = self.root.attachNewNode(make_geom_node(frame.build(), "gantry"))
        mat = Material("gantry")
        mat.setBaseColor((1, 1, 1, 1)); mat.setRoughness(0.45); mat.setMetallic(0.7)
        node.setMaterial(mat, 1)

        banner_mesh = MeshBuilder()
        banner_mesh.add(box((half + 1.6) * 2, 1.9, 0.12),
                        position=(float(centre[0]), ground + 6.6, float(centre[2])),
                        rotation=(0.0, yaw, 0.0))
        banner = self.root.attachNewNode(make_geom_node(banner_mesh.build(), "banner"))
        banner.setTexture(self._banner_texture(), 1)
        bmat = Material("banner")
        bmat.setBaseColor((1, 1, 1, 1)); bmat.setRoughness(0.7); bmat.setMetallic(0.0)
        banner.setMaterial(bmat, 1)

        line = MeshBuilder()
        squares = 16
        for i in range(squares):
            for row in range(2):
                frac = (i + 0.5) / squares * 2 - 1
                along = (row - 0.5) * 0.9
                x = float(centre[0] + right[0] * half * frac + tangent[0] * along)
                z = float(centre[2] + right[2] * half * frac + tangent[2] * along)
                colour = (0.95, 0.95, 0.95) if (i + row) % 2 == 0 else (0.03, 0.03, 0.03)
                line.add(box(half * 2 / squares, 0.04, 0.9),
                         position=(x, max(ground, self.terrain.height_at(x, z)) + 0.12, z),
                         rotation=(0.0, yaw, 0.0), color=colour)
        line_np = self.root.attachNewNode(make_geom_node(line.build(), "start-line"))
        lmat = Material("line")
        lmat.setBaseColor((1, 1, 1, 1)); lmat.setRoughness(0.8); lmat.setMetallic(0.0)
        line_np.setMaterial(lmat, 1)
        line_np.setDepthOffset(3)

    def _banner_texture(self) -> Texture:
        from PIL import Image, ImageDraw, ImageFont
        w, h = 1024, 128
        img = Image.new("RGB", (w, h), (178, 28, 22))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w, 10), fill=(245, 245, 240))
        draw.rectangle((0, h - 10, w, h), fill=(245, 245, 240))
        try:
            font = ImageFont.truetype(str(self.assets.path(
                "fonts/BarlowCondensed-ExtraBoldItalic.ttf")), 82)
        except OSError:
            font = ImageFont.load_default()
        text = f"{self.stage.name.upper()}  ·  START / FINISH"
        box_ = draw.textbbox((0, 0), text, font=font)
        draw.text(((w - (box_[2] - box_[0])) / 2, (h - (box_[3] - box_[1])) / 2 - box_[1]),
                  text, font=font, fill=(250, 250, 245))
        from ..paths import cache_dir
        from panda3d.core import Filename
        path = cache_dir() / f"banner-{self.stage.id}.png"
        img.save(path)
        tex = self.assets.loader.loadTexture(Filename.fromOsSpecific(str(path)))
        tex.setFormat(Texture.F_srgb)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        return tex

    # ------------------------------------------------------------------ #
    def _build_scenery(self, placements: Sequence[Placement], density: float) -> None:
        veg = self.stage.vegetation
        tints = dict(veg.tints)
        prototypes: Dict[str, NodePath] = {}
        chunks: Dict[Tuple[int, int], List[Placement]] = defaultdict(list)

        stride = 1.0 / max(density, 1e-3)
        accumulator = 0.0
        for p in placements:
            if p.kind in ("tree", "clutter") and density < 1.0:
                accumulator += 1.0
                if accumulator < stride:
                    continue
                accumulator -= stride
            chunks[(int(p.x // CHUNK), int(p.z // CHUNK))].append(p)

        scenery = self.root.attachNewNode("scenery")
        total = len(chunks)
        for n_chunk, (key, items) in enumerate(chunks.items()):
            chunk = scenery.attachNewNode(f"chunk-{key[0]}-{key[1]}")
            for p in items:
                if p.model not in prototypes:
                    prototypes[p.model] = self.assets.model(f"models/nature/{p.model}.glb", tints)
                inst = prototypes[p.model].copyTo(chunk)
                inst.setPos(to_panda(p.x, p.y - 0.05, p.z))
                inst.setH(-math.degrees(p.yaw))
                inst.setScale(p.scale)
            chunk.flattenStrong()
            if n_chunk % 8 == 0:
                self._progress("Planting the forest", 0.35 + 0.6 * n_chunk / max(total, 1))

    def destroy(self) -> None:
        self.root.removeNode()
