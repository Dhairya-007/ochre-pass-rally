"""Procedural geometry.

Primitives are generated as plain numpy arrays and merged into a single mesh by
`MeshBuilder`. Merging matters: 500 trees as 500 entities is 500 draw calls,
but 500 trees baked into one vertex buffer is one. Per-vertex colour lets the
whole forest - trunks, needles, rocks, posts - share a single material.

No engine imports here, so geometry can be built and inspected headlessly.

Winding convention - read this before adding a primitive
---------------------------------------------------------
Ursina runs Panda3D in a *left-handed* Y-up coordinate system. The practical
consequence: a triangle is front-facing when the right-hand cross product
(b - a) x (c - a) points *away* from the viewer, i.e. opposite to the
surface's true outward normal. Get it backwards and the face is culled. On a
closed shape that is easy to miss (you see the inside of the far wall and it
looks almost right); on an open one like the ground it simply vanishes.
`tests/test_core.py` checks every primitive against this rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

RGB = Tuple[float, float, float]


@dataclass
class MeshArrays:
    """Raw geometry, ready to be handed to whichever renderer is in use."""

    vertices: np.ndarray   # (V, 3) float
    triangles: np.ndarray  # (T, 3) int indices
    normals: np.ndarray    # (V, 3) float
    colors: np.ndarray     # (V, 4) rgba in 0..1
    uvs: np.ndarray        # (V, 2) float

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
def euler_matrix(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation matrix for intrinsic Y (yaw), X (pitch), Z (roll), in radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return ry @ rx @ rz


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _finish(vertices: List[Sequence[float]], triangles: List[Sequence[int]],
            normals: Optional[List[Sequence[float]]] = None,
            uvs: Optional[List[Sequence[float]]] = None) -> MeshArrays:
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if normals is None:
        n = _face_normals(v, t)
    else:
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    u = (np.zeros((len(v), 2)) if uvs is None
         else np.asarray(uvs, dtype=np.float64).reshape(-1, 2))
    c = np.ones((len(v), 4))
    return MeshArrays(v, t, n, c, u)


def _face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Smooth vertex normals by accumulating area-weighted face normals."""
    normals = np.zeros_like(vertices)
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    face = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-9)


def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> MeshArrays:
    """Axis-aligned box centred on the origin, with flat per-face normals."""
    hw, hh, hd = width * 0.5, height * 0.5, depth * 0.5
    faces = [
        # (corner offsets, normal)
        ([(-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd)], (0, 0, 1)),
        ([(hw, -hh, -hd), (-hw, -hh, -hd), (-hw, hh, -hd), (hw, hh, -hd)], (0, 0, -1)),
        ([(hw, -hh, hd), (hw, -hh, -hd), (hw, hh, -hd), (hw, hh, hd)], (1, 0, 0)),
        ([(-hw, -hh, -hd), (-hw, -hh, hd), (-hw, hh, hd), (-hw, hh, -hd)], (-1, 0, 0)),
        ([(-hw, hh, hd), (hw, hh, hd), (hw, hh, -hd), (-hw, hh, -hd)], (0, 1, 0)),
        ([(-hw, -hh, -hd), (hw, -hh, -hd), (hw, -hh, hd), (-hw, -hh, hd)], (0, -1, 0)),
    ]
    vertices: List[Sequence[float]] = []
    normals: List[Sequence[float]] = []
    uvs: List[Sequence[float]] = []
    triangles: List[Sequence[int]] = []
    for corners, normal in faces:
        base = len(vertices)
        vertices.extend(corners)
        normals.extend([normal] * 4)
        uvs.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
        triangles.extend([(base, base + 2, base + 1), (base, base + 3, base + 2)])
    return _finish(vertices, triangles, normals, uvs)


def cylinder(radius_bottom: float = 0.5, radius_top: float = 0.5,
             height: float = 1.0, segments: int = 12,
             caps: bool = True) -> MeshArrays:
    """Y-aligned cylinder or truncated cone, centred on the origin."""
    hh = height * 0.5
    vertices: List[Sequence[float]] = []
    normals: List[Sequence[float]] = []
    uvs: List[Sequence[float]] = []
    triangles: List[Sequence[int]] = []

    slope = (radius_bottom - radius_top) / max(height, 1e-6)
    for i in range(segments + 1):
        angle = (i / segments) * math.tau
        cx, cz = math.cos(angle), math.sin(angle)
        base = len(vertices)
        vertices.append((cx * radius_bottom, -hh, cz * radius_bottom))
        vertices.append((cx * radius_top, hh, cz * radius_top))
        normal = np.array([cx, slope, cz], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        normals.extend([tuple(normal), tuple(normal)])
        uvs.extend([(i / segments, 0.0), (i / segments, 1.0)])
        if i < segments:
            triangles.extend([
                (base, base + 2, base + 1),
                (base + 1, base + 2, base + 3),
            ])

    if caps:
        for y, radius, normal, flip in ((hh, radius_top, (0, 1, 0), False),
                                        (-hh, radius_bottom, (0, -1, 0), True)):
            if radius <= 1e-6:
                continue
            centre = len(vertices)
            vertices.append((0.0, y, 0.0))
            normals.append(normal)
            uvs.append((0.5, 0.5))
            for i in range(segments + 1):
                angle = (i / segments) * math.tau
                vertices.append((math.cos(angle) * radius, y, math.sin(angle) * radius))
                normals.append(normal)
                uvs.append((0.5 + math.cos(angle) * 0.5, 0.5 + math.sin(angle) * 0.5))
            for i in range(segments):
                a, b = centre + 1 + i, centre + 2 + i
                triangles.append((centre, b, a) if flip else (centre, a, b))

    return _finish(vertices, triangles, normals, uvs)


def cone(radius: float = 1.0, height: float = 2.0, segments: int = 10) -> MeshArrays:
    """Y-aligned cone, centred on the origin."""
    return cylinder(radius_bottom=radius, radius_top=0.0, height=height,
                    segments=segments, caps=True)


def rock(seed: int = 0, facets: int = 1) -> MeshArrays:
    """A lumpy low-poly boulder of radius ~1, deterministic per seed."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=np.float64)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    faces = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int32)

    faces = faces[:, [0, 2, 1]]   # the icosahedron table is right-handed outward

    rng = np.random.default_rng(seed)
    radii = 1.0 + rng.uniform(-0.28, 0.28, size=len(base))
    vertices = base * radii[:, None]
    return MeshArrays(
        vertices=vertices,
        triangles=faces,
        normals=_face_normals(vertices, faces),
        colors=np.ones((len(vertices), 4)),
        uvs=np.zeros((len(vertices), 2)),
    )


def gradient_sky(radius: float = 900.0, segments: int = 24, rings: int = 16,
                 top: RGB = (0.29, 0.45, 0.66), mid: RGB = (0.85, 0.71, 0.55),
                 low: RGB = (0.91, 0.78, 0.60)) -> MeshArrays:
    """Inward-facing dome with a vertical colour gradient baked into vertices."""
    vertices: List[Sequence[float]] = []
    colors: List[Sequence[float]] = []
    triangles: List[Sequence[int]] = []

    top_c = np.array(top)
    mid_c = np.array(mid)
    low_c = np.array(low)

    for ring in range(rings + 1):
        v = ring / rings
        theta = v * math.pi
        y = math.cos(theta)
        r = math.sin(theta)
        # Blend low -> mid -> top as we rise.
        if y < 0.16:
            t = max(0.0, (y + 0.06) / 0.22)
            rgb = low_c * (1.0 - t) + mid_c * t
        else:
            t = min(1.0, (y - 0.12) / 0.60)
            rgb = mid_c * (1.0 - t) + top_c * t
        for seg in range(segments + 1):
            u = seg / segments
            angle = u * math.tau
            vertices.append((math.cos(angle) * r * radius, y * radius,
                             math.sin(angle) * r * radius))
            colors.append((rgb[0], rgb[1], rgb[2], 1.0))

    stride = segments + 1
    for ring in range(rings):
        for seg in range(segments):
            a = ring * stride + seg
            b = a + 1
            c = a + stride
            d = c + 1
            # Wound inward: we are inside the dome looking out.
            triangles.extend([(a, b, c), (b, d, c)])

    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int32)
    return MeshArrays(
        vertices=v,
        triangles=t,
        normals=-v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9),
        colors=np.asarray(colors, dtype=np.float64),
        uvs=np.zeros((len(v), 2)),
    )


# --------------------------------------------------------------------------- #
# Baked lighting
# --------------------------------------------------------------------------- #
def bake_directional_light(mesh: MeshArrays, direction: Sequence[float],
                           sun: RGB = (1.0, 0.89, 0.69), sun_strength: float = 0.85,
                           ambient: RGB = (0.46, 0.52, 0.60),
                           ambient_strength: float = 0.55,
                           wrap: float = 0.35) -> MeshArrays:
    """Multiply a mesh's vertex colours by a fixed sun plus ambient term.

    The sun never moves in this game, so there is no reason to pay for dynamic
    lighting: shading it into the vertex colours once at build time looks the
    same, costs nothing per frame, and - the reason it matters here - needs no
    GLSL at all. Ursina's lit shaders require GLSL 1.50, which macOS only
    offers through a 3.2 core profile that breaks other things.

    `direction` points *towards* the light. `wrap` softens the terminator so
    surfaces facing away keep some shape instead of going flat black.
    """
    light = np.asarray(direction, dtype=np.float64)
    light /= np.linalg.norm(light)

    lambert = mesh.normals @ light
    # Wrapped diffuse: remap [-1, 1] so the dark side falls off gently.
    lambert = np.clip((lambert + wrap) / (1.0 + wrap), 0.0, 1.0)

    sun_rgb = np.asarray(sun, dtype=np.float64) * sun_strength
    ambient_rgb = np.asarray(ambient, dtype=np.float64) * ambient_strength
    lighting = ambient_rgb[None, :] + sun_rgb[None, :] * lambert[:, None]

    rgb = np.clip(mesh.colors[:, :3] * lighting, 0.0, 1.0)
    colors = np.concatenate([rgb, mesh.colors[:, 3:4]], axis=1)

    return MeshArrays(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        normals=mesh.normals,
        colors=colors,
        uvs=mesh.uvs,
    )


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
class MeshBuilder:
    """Accumulates transformed primitives into one mesh."""

    def __init__(self) -> None:
        self._vertices: List[np.ndarray] = []
        self._triangles: List[np.ndarray] = []
        self._normals: List[np.ndarray] = []
        self._colors: List[np.ndarray] = []
        self._uvs: List[np.ndarray] = []
        self._offset = 0

    def add(self, mesh: MeshArrays, position: Sequence[float] = (0.0, 0.0, 0.0),
            rotation: Sequence[float] = (0.0, 0.0, 0.0),
            scale: Sequence[float] | float = 1.0,
            color: Optional[RGB] = None,
            shade: float = 1.0) -> "MeshBuilder":
        """Add one primitive.

        `rotation` is (pitch, yaw, roll) in radians. `shade` multiplies the
        colour, which is a cheap way to vary a forest without new materials.
        """
        verts = mesh.vertices.copy()

        if isinstance(scale, (int, float)):
            verts *= float(scale)
        else:
            verts *= np.asarray(scale, dtype=np.float64)[None, :]

        normals = mesh.normals.copy()
        if any(rotation):
            matrix = euler_matrix(*rotation)
            verts = verts @ matrix.T
            normals = normals @ matrix.T

        verts += np.asarray(position, dtype=np.float64)[None, :]

        if color is None:
            colors = mesh.colors.copy()
        else:
            rgb = np.clip(np.asarray(color, dtype=np.float64) * shade, 0.0, 1.0)
            colors = np.concatenate(
                [np.tile(rgb, (len(verts), 1)), np.ones((len(verts), 1))], axis=1
            )

        self._vertices.append(verts)
        self._normals.append(normals)
        self._colors.append(colors)
        self._uvs.append(mesh.uvs.copy())
        self._triangles.append(mesh.triangles + self._offset)
        self._offset += len(verts)
        return self

    @property
    def is_empty(self) -> bool:
        return self._offset == 0

    def build(self) -> MeshArrays:
        if self.is_empty:
            raise ValueError("MeshBuilder is empty - nothing to build")
        return MeshArrays(
            vertices=np.concatenate(self._vertices, axis=0),
            triangles=np.concatenate(self._triangles, axis=0),
            normals=np.concatenate(self._normals, axis=0),
            colors=np.concatenate(self._colors, axis=0),
            uvs=np.concatenate(self._uvs, axis=0),
        )
