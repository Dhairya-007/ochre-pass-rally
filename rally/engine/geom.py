"""MeshArrays (numpy, Y-up) -> Panda3D Geom (Z-up), fast.

Vertex data is packed into one interleaved numpy buffer and copied into the
GeomVertexArrayData in a single memoryview assignment, which is two orders of
magnitude faster than GeomVertexWriter for the 50k-vertex terrain.

simplepbr's normal mapping needs a tangent per vertex; `compute_tangents`
derives them from the UVs when a mesh has none.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from panda3d.core import (Geom, GeomEnums, GeomNode, GeomTriangles,
                          GeomVertexArrayFormat, GeomVertexData, GeomVertexFormat,
                          InternalName, NodePath)

from ..meshbuild import MeshArrays

_FORMAT = None


def _format() -> GeomVertexFormat:
    global _FORMAT
    if _FORMAT is None:
        array = GeomVertexArrayFormat()
        array.addColumn(InternalName.getVertex(), 3, Geom.NT_float32, Geom.C_point)
        array.addColumn(InternalName.getNormal(), 3, Geom.NT_float32, Geom.C_normal)
        array.addColumn(InternalName.getTangent(), 4, Geom.NT_float32, Geom.C_vector)
        array.addColumn(InternalName.getColor(), 4, Geom.NT_float32, Geom.C_color)
        array.addColumn(InternalName.getTexcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        _FORMAT = GeomVertexFormat.registerFormat(array)
    return _FORMAT


def compute_tangents(mesh: MeshArrays) -> np.ndarray:
    """Per-vertex tangents (xyz + handedness w) from positions and UVs."""
    v, uv, t = mesh.vertices, mesh.uvs, mesh.triangles
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    w0, w1, w2 = uv[t[:, 0]], uv[t[:, 1]], uv[t[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1, det), 0.0)
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]

    acc_t = np.zeros_like(v)
    acc_b = np.zeros_like(v)
    for i in range(3):
        np.add.at(acc_t, t[:, i], tangent)
        np.add.at(acc_b, t[:, i], bitangent)

    n = mesh.normals
    # Gram-Schmidt against the normal, then pick handedness.
    acc_t -= n * np.sum(n * acc_t, axis=1, keepdims=True)
    length = np.linalg.norm(acc_t, axis=1, keepdims=True)
    fallback = np.tile([1.0, 0.0, 0.0], (len(v), 1))
    acc_t = np.where(length > 1e-9, acc_t / np.maximum(length, 1e-9), fallback)
    w = np.where(np.sum(np.cross(n, acc_t) * acc_b, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([acc_t, w[:, None]], axis=1)


def make_geom_node(mesh: MeshArrays, name: str = "mesh",
                   tangents: Optional[np.ndarray] = None) -> GeomNode:
    """Build a GeomNode from core-space arrays, converting to Z-up."""
    if tangents is None:
        tangents = compute_tangents(mesh)
    n = mesh.vertex_count
    swap = [0, 2, 1]
    data = np.empty((n, 16), dtype=np.float32)
    data[:, 0:3] = mesh.vertices[:, swap]
    data[:, 3:6] = mesh.normals[:, swap]
    data[:, 6:9] = tangents[:, :3][:, swap]
    data[:, 9] = tangents[:, 3]
    data[:, 10:14] = mesh.colors
    data[:, 14:16] = mesh.uvs

    vdata = GeomVertexData(name, _format(), Geom.UH_static)
    vdata.uncleanSetNumRows(n)
    memoryview(vdata.modifyArray(0)).cast("B").cast("f")[:] = data.ravel()

    prim = GeomTriangles(Geom.UH_static)
    index_type = GeomEnums.NT_uint32 if n > 65535 else GeomEnums.NT_uint16
    prim.setIndexType(index_type)
    dtype = np.uint32 if n > 65535 else np.uint16
    indices = mesh.triangles.astype(dtype).ravel()
    handle = prim.modifyVertices()
    handle.uncleanSetNumRows(len(indices))
    memoryview(handle).cast("B").cast("I" if dtype is np.uint32 else "H")[:] = indices

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def mesh_nodepath(mesh: MeshArrays, parent: NodePath, name: str = "mesh") -> NodePath:
    return parent.attachNewNode(make_geom_node(mesh, name))
