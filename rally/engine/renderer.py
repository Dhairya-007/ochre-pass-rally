"""The physically based render setup, and the per-stage lighting.

simplepbr gives us PBR materials, normal maps, image-based lighting from an
HDR sky, a shadow-mapped sun and filmic tone mapping. Its shaders are GLSL
1.20, which matters on macOS: a default OpenGL 2.1 context there cannot
compile GLSL 1.50 (Ursina's shaders, which is why the old version rendered
flat), but compiles these fine.

The shadow map is fitted to a box around the player rather than the whole
stage, and moved in whole-texel steps so the shadow edges don't crawl as the
car drives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import simplepbr
import simplepbr.utils
from panda3d.core import (AmbientLight, DirectionalLight, Fog, LVector3f, NodePath)

from ..stages import Atmosphere
from .assets import AssetLibrary, linear_colour


@dataclass
class Quality:
    shadows: bool = True
    shadow_size: int = 2048
    shadow_extent: float = 110.0        # metres covered by the shadow map
    msaa: int = 4
    sky_size: int = 512
    scenery_density: float = 1.0        # multiplies tree/rock/clutter counts

    @classmethod
    def preset(cls, name: str) -> "Quality":
        return {
            "low": cls(shadows=False, msaa=0, sky_size=256, scenery_density=0.5),
            "medium": cls(shadow_size=1024, msaa=2, sky_size=384, scenery_density=0.75),
            "high": cls(),
            "ultra": cls(shadow_size=4096, shadow_extent=140.0, msaa=8, scenery_density=1.25),
        }.get(name, cls())


class Renderer:
    def __init__(self, base, assets: AssetLibrary, quality: Quality) -> None:
        self.base = base
        self.assets = assets
        self.quality = quality
        self.pipeline = simplepbr.init(
            render_node=base.render,
            window=base.win,
            camera_node=base.cam,
            msaa_samples=quality.msaa,
            enable_shadows=quality.shadows,
            use_normal_maps=True,
            enable_fog=True,
            exposure=0.85,
            max_lights=4,
        )
        base.camLens.setNearFar(0.25, 2200.0)

        self.scene: Optional[NodePath] = None
        self.sun_np: Optional[NodePath] = None
        self.ambient_np: Optional[NodePath] = None
        self.skybox: Optional[NodePath] = None
        self._sun_dir = LVector3f(0, 0, 1)

    # ------------------------------------------------------------------ #
    def set_stage(self, scene: NodePath, atmosphere: Atmosphere) -> None:
        """Light `scene` for a stage. Replaces any previous stage's lighting."""
        self.clear()
        self.scene = scene
        a = atmosphere

        cube = self.assets.sky_cubemap(a.sky, size=self.quality.sky_size,
                                       rotation_deg=a.sky_rotation)
        self.pipeline.env_map = self._env_map(a)
        self.skybox = simplepbr.utils.make_skybox(cube)
        self.skybox.reparentTo(scene)
        self.skybox.setColorScale(1, 1, 1, 1)
        self.pipeline.exposure = a.exposure

        sun = DirectionalLight("sun")
        r, g, b = a.sun_color
        sun.setColor((r * a.sun_strength, g * a.sun_strength, b * a.sun_strength, 1))
        if self.quality.shadows:
            sun.setShadowCaster(True, self.quality.shadow_size, self.quality.shadow_size)
            lens = sun.getLens()
            lens.setFilmSize(self.quality.shadow_extent, self.quality.shadow_extent)
            lens.setNearFar(1.0, 600.0)
        self.sun_np = scene.attachNewNode(sun)
        dx, dy, dz = a.sun_direction            # core Y-up, towards the sun
        self._sun_dir = LVector3f(dx, dz, dy).normalized()
        self.sun_np.lookAt(-self._sun_dir)
        scene.setLight(self.sun_np)

        ambient = AmbientLight("ambient")
        ambient.setColor((*a.ambient, 1))
        self.ambient_np = scene.attachNewNode(ambient)
        scene.setLight(self.ambient_np)

        fog = Fog("stage-fog")
        fog.setColor(*a.fog_color)
        fog.setExpDensity(a.fog_density)
        scene.setFog(fog)
        self.base.setBackgroundColor(*a.fog_color, 1)

    def _env_map(self, a: Atmosphere) -> "simplepbr.EnvMap":
        """Image-based lighting for the stage's sky, computed once and cached.

        simplepbr integrates the lighting on the CPU in pure Python. Fed the
        512px display cubemap that took 8.7 s per stage; lighting only needs
        the low frequencies, so it gets a 64px cube (1/64 of the work) and the
        result is written to disk in simplepbr's own .env format, after which
        loading it is instant.
        """
        from ..paths import cache_dir
        path = cache_dir() / f"env-{a.sky}-{a.sky_rotation:g}-{a.env_strength:g}.env"
        if path.exists():
            try:
                return simplepbr.EnvMap.from_file_path(path)
            except Exception:                    # noqa: BLE001 - rebuild a bad cache
                path.unlink(missing_ok=True)
        small = self.assets.sky_cubemap(a.sky, size=64, rotation_deg=a.sky_rotation,
                                        scale=a.env_strength)
        env = simplepbr.EnvMap(small, blocking_prepare=True)
        try:
            env.write(path)
        except Exception:                        # noqa: BLE001 - caching is optional
            pass
        return env

    def follow(self, x: float, y: float, z: float) -> None:
        """Centre the shadow map on a point (Panda coordinates)."""
        if self.sun_np is None:
            return
        texel = self.quality.shadow_extent / self.quality.shadow_size
        step = max(texel * 4, 0.5)
        cx, cy, cz = (round(x / step) * step, round(y / step) * step, z)
        self.sun_np.setPos(cx + self._sun_dir.x * 250.0,
                           cy + self._sun_dir.y * 250.0,
                           cz + self._sun_dir.z * 250.0)
        self.sun_np.lookAt(cx, cy, cz)

    def clear(self) -> None:
        if self.scene is not None:
            self.scene.clearLight()
            self.scene.clearFog()
        for np in (self.sun_np, self.ambient_np, self.skybox):
            if np is not None:
                np.removeNode()
        self.sun_np = self.ambient_np = self.skybox = None
        self.scene = None
