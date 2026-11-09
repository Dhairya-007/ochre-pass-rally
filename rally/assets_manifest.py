"""Every third-party file the game uses, where it comes from, and its licence.

All of it is free:

  * Kenney (kenney.nl) ........ CC0 1.0 - public domain, no credit required
  * Poly Haven (polyhaven.com) . CC0 1.0 - public domain, no credit required
  * Google Fonts ............... SIL Open Font License 1.1 - free to use and
                                 bundle, may not be sold on its own

Kenney files are fetched from a public GitHub mirror of the extracted packs,
pinned to one commit so a download today is byte-identical to one next year.
Poly Haven files come from Poly Haven's own CDN.

Nothing here imports the engine; `tools/fetch_assets.py` reads this table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

KENNEY_SHA = "45df48c4d45f8716216b1a9e22df0b69cd9f5932"
KENNEY_ROOT = f"https://raw.githubusercontent.com/ETdoFresh/kenney.nl/{KENNEY_SHA}"
POLYHAVEN_TEX = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg"
POLYHAVEN_HDRI = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr"
GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main/ofl"

CC0 = "CC0 1.0 (public domain)"
OFL = "SIL Open Font License 1.1"


@dataclass(frozen=True)
class Asset:
    dest: str          # path under assets/
    url: str
    licence: str
    author: str
    source: str        # human-facing page, for CREDITS.md


def _kenney(pack: str, inner: str, dest: str, page: str) -> Asset:
    url = f"{KENNEY_ROOT}/{pack}/{inner}".replace(" ", "%20")
    return Asset(dest, url, CC0, "Kenney", f"https://kenney.nl/assets/{page}")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
CAR_MODELS = ("hatchbackSports", "sedanSports", "suv", "race", "suvLuxury")

NATURE_MODELS = (
    "tree_pineTallA_detailed", "tree_pineTallB_detailed",
    "tree_pineTallC_detailed", "tree_pineTallD_detailed",
    "tree_pineRoundA", "tree_pineRoundC", "tree_pineSmallA", "tree_pineSmallC",
    "tree_detailed", "tree_oak", "tree_fat", "tree_default",
    "cactus_tall", "cactus_short",
    "rock_largeA", "rock_largeB", "rock_largeC", "rock_largeD", "rock_largeE",
    "rock_tallA", "rock_tallB", "rock_smallA", "rock_smallB", "rock_smallC",
    "stump_old", "stump_round", "log", "log_large",
    "plant_bush", "plant_bushLarge", "plant_bushSmall",
    "grass", "grass_large", "grass_leafsLarge",
    "fence_simple", "tent_detailedOpen", "tent_smallOpen", "sign",
)


def _models() -> List[Asset]:
    out = [
        _kenney("carkit_v1.4", f"Models/GLTF format/{name}.glb",
                f"models/cars/{name}.glb", "car-kit")
        for name in CAR_MODELS
    ]
    out += [
        _kenney("kenney_natureKit_2.1", f"Models/GLTF format/{name}.glb",
                f"models/nature/{name}.glb", "nature-kit")
        for name in NATURE_MODELS
    ]
    out.append(_kenney("carkit_v1.4", "License.txt", "licences/kenney-car-kit.txt", "car-kit"))
    out.append(_kenney("kenney_natureKit_2.1", "License.txt",
                       "licences/kenney-nature-kit.txt", "nature-kit"))
    return out


# --------------------------------------------------------------------------- #
# Particles and audio
# --------------------------------------------------------------------------- #
def _particles() -> List[Asset]:
    out = [
        _kenney("smokeparticleassets", f"PNG/White puff/whitePuff{i:02d}.png",
                f"particles/puff{i:02d}.png", "smoke-particles")
        for i in range(0, 25, 3)
    ]
    out.append(_kenney("smokeparticleassets", "license.txt",
                       "licences/kenney-smoke-particles.txt", "smoke-particles"))
    return out


IMPACTS = ("impactMetal_heavy", "impactMetal_medium", "impactWood_heavy",
           "impactWood_medium", "impactSoft_heavy", "impactPlate_medium")
UI_SOUNDS = ("click1", "click3", "rollover2", "rollover4", "switch2", "switch7")


def _audio() -> List[Asset]:
    out = []
    for name in IMPACTS:
        for i in range(3):
            out.append(_kenney("kenney_impactsounds", f"Audio/{name}_{i:03d}.ogg",
                               f"audio/impact/{name}_{i:03d}.ogg", "impact-sounds"))
    out += [_kenney("kenney_uiaudio", f"Audio/{name}.ogg", f"audio/ui/{name}.ogg",
                    "ui-audio") for name in UI_SOUNDS]
    out.append(_kenney("kenney_impactsounds", "License.txt",
                       "licences/kenney-impact-sounds.txt", "impact-sounds"))
    out.append(_kenney("kenney_uiaudio", "License.txt",
                       "licences/kenney-ui-audio.txt", "ui-audio"))
    return out


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
FONTS = ("BarlowCondensed-Medium", "BarlowCondensed-SemiBold",
         "BarlowCondensed-Bold", "BarlowCondensed-ExtraBoldItalic")


def _fonts() -> List[Asset]:
    page = "https://fonts.google.com/specimen/Barlow+Condensed"
    out = [Asset(f"fonts/{name}.ttf", f"{GOOGLE_FONTS}/barlowcondensed/{name}.ttf",
                 OFL, "Jeremy Tribby", page) for name in FONTS]
    out.append(Asset("licences/barlow-condensed-OFL.txt",
                     f"{GOOGLE_FONTS}/barlowcondensed/OFL.txt", OFL,
                     "Jeremy Tribby", page))
    return out


# --------------------------------------------------------------------------- #
# Poly Haven textures and skies
# --------------------------------------------------------------------------- #
# (poly haven id, local name). Each texture fetches colour, normal and ARM
# (ambient occlusion / roughness / metalness, which is glTF's packed layout).
TEXTURES: Tuple[Tuple[str, str], ...] = (
    ("rocky_trail", "gravel_forest"),
    ("forrest_ground_01", "ground_forest"),
    ("red_laterite_soil_stones", "gravel_desert"),
    ("coast_sand_01", "ground_desert"),
    ("snow_field_aerial", "gravel_snow"),
    ("snow_02", "ground_snow"),
    ("brown_mud_rocks_01", "gravel_mud"),
    ("forest_leaves_02", "ground_leaves"),
)
TEXTURE_MAPS = (("diff", "albedo"), ("nor_gl", "normal"), ("arm", "arm"))

# Poly Haven is not consistent about the colour map's file suffix. These were
# read from its API (api.polyhaven.com/files/<id>), not guessed.
ALBEDO_SUFFIX = {"snow_field_aerial": "col", "forest_leaves_02": "diffuse"}

SKIES: Tuple[Tuple[str, str], ...] = (
    ("kloofendal_48d_partly_cloudy_puresky", "sky_cloudy"),
    ("qwantani_afternoon_puresky", "sky_clear"),
    ("horn-koppe_snow", "sky_snow"),
    ("belfast_sunset_puresky", "sky_sunset"),
)


def _polyhaven(resolution: str) -> List[Asset]:
    out = []
    for ph_id, local in TEXTURES:
        for ph_map, local_map in TEXTURE_MAPS:
            if local_map == "albedo":
                ph_map = ALBEDO_SUFFIX.get(ph_id, ph_map)
            out.append(Asset(
                f"textures/{local}_{local_map}.jpg",
                f"{POLYHAVEN_TEX}/{resolution}/{ph_id}/{ph_id}_{ph_map}_{resolution}.jpg",
                CC0, "Poly Haven", f"https://polyhaven.com/a/{ph_id}"))
    sky_res = "2k" if resolution != "1k" else "1k"
    for ph_id, local in SKIES:
        out.append(Asset(f"skies/{local}.hdr",
                         f"{POLYHAVEN_HDRI}/{sky_res}/{ph_id}_{sky_res}.hdr",
                         CC0, "Poly Haven", f"https://polyhaven.com/a/{ph_id}"))
    return out


# --------------------------------------------------------------------------- #
def manifest(quality: str = "high") -> List[Asset]:
    """The full list. `quality` picks texture resolution: high=2k, low=1k."""
    resolution = {"high": "2k", "low": "1k"}[quality]
    return _models() + _particles() + _audio() + _fonts() + _polyhaven(resolution)


def by_dest(quality: str = "high") -> Dict[str, Asset]:
    return {a.dest: a for a in manifest(quality)}
