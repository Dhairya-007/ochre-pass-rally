"""Loading the downloaded assets, with the corrections they need.

Two corrections matter:

  * Kenney's glTF files store flat sRGB colours in `baseColorFactor`, which
    glTF defines as linear. Rendered as-is through a physically based
    pipeline they come out pale and washed (the pines go teal). `load_model`
    converts them to linear, and can replace them with per-stage tints.

  * Colour textures are sRGB; normal and ARM maps are data and must stay
    linear. `texture()` sets the format accordingly.

HDR skies are equirectangular; simplepbr wants a cubemap. The conversion is
done in numpy and cached in the user cache folder, since it takes a second or
two per sky.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from panda3d.core import (Filename, Material, NodePath, PfmFile, SamplerState,
                          Texture, TextureStage)

from ..paths import ASSETS, cache_dir

RGB = Tuple[float, float, float]


class MissingAssets(Exception):
    pass


def srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_colour(rgb: Iterable[float], alpha: float = 1.0) -> Tuple[float, float, float, float]:
    r, g, b = (srgb_to_linear(c) for c in rgb)
    return (r, g, b, alpha)


def check_assets() -> None:
    from ..assets_manifest import manifest
    missing = [a.dest for a in manifest("high")
               if not (ASSETS / a.dest).exists() and not a.dest.startswith("textures/")]
    textures = [a.dest for a in manifest("high") if a.dest.startswith("textures/")]
    if not any((ASSETS / t).exists() for t in textures):
        missing += textures[:1]
    if missing:
        raise MissingAssets(
            f"{len(missing)} asset file(s) are missing (e.g. {missing[0]}).\n"
            "Download them once with:\n\n    python tools/fetch_assets.py\n")


class AssetLibrary:
    def __init__(self, loader) -> None:
        self.loader = loader
        self._textures: Dict[Tuple[str, bool], Texture] = {}
        self._fonts: Dict[str, object] = {}
        self._cubes: Dict[str, Texture] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def path(relative: str) -> Path:
        return ASSETS / relative

    def model(self, relative: str, tints: Optional[Mapping[str, RGB]] = None,
              roughness: float = 0.82) -> NodePath:
        """Load a glTF model with its colours corrected to linear.

        A fresh, uncached copy every call, so per-stage tints on one copy never
        leak into another.
        """
        path = Filename.fromOsSpecific(str(self.path(relative)))
        node = self.loader.loadModel(path, noCache=True)
        for material in node.findAllMaterials():
            name = material.getName()
            source = tints.get(name) if tints else None
            if source is None:
                c = material.getBaseColor()
                source = (c[0], c[1], c[2])
            material.setBaseColor(linear_colour(source))
            material.setRoughness(roughness)
            material.setMetallic(0.0)
        return node

    def texture(self, relative: str, srgb: bool, anisotropy: int = 8) -> Texture:
        key = (relative, srgb)
        if key in self._textures:
            return self._textures[key]
        tex = self.loader.loadTexture(Filename.fromOsSpecific(str(self.path(relative))))
        if srgb:
            # Keep the alpha channel: Kenney's particle PNGs are grey + alpha,
            # and forcing them to 3-channel sRGB throws the alpha away, which
            # turns soft puffs into hard squares.
            components = tex.getNumComponents()
            if components == 4:
                tex.setFormat(Texture.F_srgb_alpha)
            elif components == 3:
                tex.setFormat(Texture.F_srgb)
            elif components == 2:
                tex.setFormat(Texture.F_sluminance_alpha)
            else:
                tex.setFormat(Texture.F_sluminance)
        tex.setWrapU(SamplerState.WM_repeat)
        tex.setWrapV(SamplerState.WM_repeat)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        tex.setMagfilter(SamplerState.FT_linear)
        tex.setAnisotropicDegree(anisotropy)
        self._textures[key] = tex
        return tex

    def apply_pbr(self, node: NodePath, name: str, tint: RGB = (1.0, 1.0, 1.0),
                  roughness: float = 1.0) -> None:
        """Give `node` the albedo / normal / ARM set `textures/<name>_*.jpg`."""
        material = Material(name)
        material.setBaseColor(linear_colour(tint))
        material.setRoughness(roughness)
        material.setMetallic(1.0)        # scaled by the ARM map's blue channel
        node.setMaterial(material, 1)

        base = TextureStage(f"{name}-albedo")
        base.setMode(TextureStage.MModulate)
        base.setSort(0)
        node.setTexture(base, self.texture(f"textures/{name}_albedo.jpg", srgb=True), 1)

        arm = TextureStage(f"{name}-arm")
        arm.setMode(TextureStage.MSelector)          # glTF metal-roughness slot
        arm.setSort(1)
        node.setTexture(arm, self.texture(f"textures/{name}_arm.jpg", srgb=False), 1)

        normal = TextureStage(f"{name}-normal")
        normal.setMode(TextureStage.MNormal)
        normal.setSort(2)
        node.setTexture(normal, self.texture(f"textures/{name}_normal.jpg", srgb=False), 1)

    def font(self, name: str = "BarlowCondensed-SemiBold"):
        if name not in self._fonts:
            path = Filename.fromOsSpecific(str(self.path(f"fonts/{name}.ttf")))
            font = self.loader.loadFont(path.getFullpath())
            font.setPixelsPerUnit(110)
            font.setPageSize(1024, 1024)
            self._fonts[name] = font
        return self._fonts[name]

    def sound_path(self, relative: str) -> Filename:
        return Filename.fromOsSpecific(str(self.path(relative)))

    # ------------------------------------------------------------------ #
    # Sky
    # ------------------------------------------------------------------ #
    def sky_cubemap(self, key: str, size: int = 512, clamp: float = 40.0,
                    rotation_deg: float = 0.0, scale: float = 1.0) -> Texture:
        """Equirectangular HDR -> float cubemap, cached on disk as .npy.

        Very bright pixels (the sun disc is ~12,000x mid-grey) are clamped:
        the directional light already provides the sun, and leaving them in
        makes every glossy surface flare.
        """
        cache_key = f"{key}-{size}-{clamp:g}-{rotation_deg:g}-{scale:g}"
        if cache_key in self._cubes:
            return self._cubes[cache_key]
        cache = cache_dir() / f"cube-{cache_key}.npy"
        if cache.exists():
            faces = np.load(cache)
        else:
            faces = _equirect_to_faces(self.path(f"skies/{key}.hdr"), size, clamp,
                                       rotation_deg) * scale
            np.save(cache, faces)

        tex = Texture(f"sky-{key}")
        tex.setupCubeMap(size, Texture.T_float, Texture.F_rgb16)
        tex.setRamImage(faces.astype(np.float32).tobytes())
        tex.setMinfilter(SamplerState.FT_linear)
        tex.setMagfilter(SamplerState.FT_linear)
        self._cubes[cache_key] = tex
        return tex


def _equirect_to_faces(path: Path, size: int, clamp: float, rotation_deg: float) -> np.ndarray:
    pfm = PfmFile()
    if not pfm.read(Filename.fromOsSpecific(str(path))):
        raise MissingAssets(f"Could not read sky {path.name}")
    w, h = pfm.getXSize(), pfm.getYSize()
    img = np.array(pfm.getPoints(), dtype=np.float32).reshape(h, w, pfm.getNumChannels())[..., :3]
    img = np.minimum(img, clamp)

    s = (np.arange(size) + 0.5) / size * 2 - 1
    sc, tc = np.meshgrid(s, s)
    one = np.ones_like(sc)
    # OpenGL cube face order and orientation (+X, -X, +Y, -Y, +Z, -Z),
    # interpreted directly as Panda's Z-up world directions.
    faces = [(one, -tc, -sc), (-one, -tc, sc), (sc, one, tc),
             (sc, -one, -tc), (sc, -tc, one), (-sc, -tc, -one)]
    rot = math.radians(rotation_deg)
    out = []
    for dx, dy, dz in faces:
        d = np.stack([dx, dy, dz], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        lon = np.arctan2(d[..., 1], d[..., 0]) + rot
        lat = np.arcsin(np.clip(d[..., 2], -1, 1))
        u = ((0.5 + lon / (2 * math.pi)) * w).astype(int) % w
        v = np.clip(((0.5 - lat / math.pi) * h).astype(int), 0, h - 1)
        out.append(img[v, u][..., ::-1])              # Panda stores BGR
    return np.stack(out)
