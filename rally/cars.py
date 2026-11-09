"""The car roster, the upgrade tree, and how both turn into a VehicleConfig.

A car is a base handling package plus a 3D model. An upgrade is a multiplier on
a few of those numbers. `build_config` folds a car and its upgrade levels into
the frozen `VehicleConfig` the physics actually runs on, so there is exactly
one place where "this car with these parts" is defined.

The names are invented; the models are Kenney's CC0 Car Kit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Tuple

from .config import VEHICLE, VehicleConfig

RGB = Tuple[float, float, float]


@dataclass(frozen=True)
class CarSpec:
    id: str
    name: str
    model: str                       # file under assets/models/cars, no suffix
    price: int
    category: str                    # shown in the garage
    blurb: str
    tweaks: Tuple[Tuple[str, float], ...]   # VehicleConfig field -> multiplier
    traits: Tuple[Tuple[str, float], ...] = ()   # VehicleConfig field -> absolute
    scale: float = 1.55              # model units -> metres
    upgrade_cost_factor: float = 1.0


CARS: Tuple[CarSpec, ...] = (
    CarSpec(
        id="kestrel", name="Kestrel R2", model="hatchbackSports", price=0,
        category="Rally2 hatch",
        blurb="Light, chuckable front-driver. Forgiving at the limit and "
              "quick to rotate on the handbrake.",
        tweaks=(("engine_force", 1.00), ("top_speed", 1.00)),
        traits=(("mass", 1080.0),),
    ),
    CarSpec(
        id="vanta", name="Vanta GT-S", model="sedanSports", price=18_000,
        category="Group N saloon",
        blurb="More power and a longer wheelbase. Stable at speed, lazier to "
              "turn in - brake in a straight line.",
        tweaks=(("engine_force", 1.14), ("top_speed", 1.08), ("lateral_grip", 0.95),
                ("max_steer", 0.94), ("handbrake_kick", 0.85)),
        traits=(("mass", 1320.0), ("wheelbase", 2.80)),
        upgrade_cost_factor=1.3,
    ),
    CarSpec(
        id="ridgeback", name="Ridgeback T1", model="suv", price=26_000,
        category="T1 raid 4x4",
        blurb="Heavy, tall and unstoppable off the line. Shrugs off snow, mud "
              "and the scenery's edge; gives away pace on dry gravel.",
        tweaks=(("engine_force", 1.06), ("top_speed", 0.94), ("lateral_grip", 0.98),
                ("slip_falloff", 0.80)),
        traits=(("mass", 1650.0), ("loose_traction", 0.35), ("rough_terrain", 0.55)),
        scale=1.6, upgrade_cost_factor=1.5,
    ),
    CarSpec(
        id="torque", name="Torque RS 4x4", model="suvLuxury", price=34_000,
        category="Rally1 crossover",
        blurb="All-wheel drive and a big turbo. Pulls out of hairpins like "
              "nothing else, and keeps its grip when the surface goes soft.",
        tweaks=(("engine_force", 1.22), ("top_speed", 1.05), ("lateral_grip", 1.04),
                ("slip_falloff", 0.88)),
        traits=(("mass", 1400.0), ("loose_traction", 0.22), ("rough_terrain", 0.75)),
        scale=1.6, upgrade_cost_factor=1.7,
    ),
    CarSpec(
        id="apex", name="Apex Proto R5", model="race", price=52_000,
        category="Prototype",
        blurb="A stage car built from a circuit racer. Savage power, sharp "
              "front end, and very little patience for mistakes.",
        tweaks=(("engine_force", 1.40), ("top_speed", 1.18), ("lateral_grip", 1.10),
                ("max_steer", 1.04), ("slip_falloff", 1.12), ("brake_force", 1.15)),
        traits=(("mass", 1150.0),),
        upgrade_cost_factor=2.2,
    ),
)

CARS_BY_ID: Dict[str, CarSpec] = {c.id: c for c in CARS}


# --------------------------------------------------------------------------- #
# Upgrades
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UpgradeKind:
    id: str
    name: str
    describe: str
    per_level: Tuple[Tuple[str, float], ...]   # field -> multiplier per level
    additive: Tuple[Tuple[str, float], ...] = ()   # field -> + per level
    base_prices: Tuple[int, ...] = (3_500, 7_500, 13_000)

    @property
    def max_level(self) -> int:
        return len(self.base_prices)


UPGRADES: Tuple[UpgradeKind, ...] = (
    UpgradeKind("engine", "Engine", "Remap, intake and exhaust: more power, "
                "higher top speed.",
                per_level=(("engine_force", 1.08), ("top_speed", 1.03))),
    UpgradeKind("tyres", "Tyres", "Softer compounds and deeper tread: more grip, "
                "and less of it lost on loose surfaces.",
                per_level=(("lateral_grip", 1.05),),
                additive=(("loose_traction", 0.05),)),
    UpgradeKind("suspension", "Suspension", "Longer travel and stiffer bars: "
                "slides stay catchable and rough ground slows you less.",
                per_level=(("slip_falloff", 0.92), ("rough_terrain", 0.92)),
                base_prices=(3_000, 6_500, 11_000)),
    UpgradeKind("brakes", "Brakes", "Bigger discs and calipers: stop later.",
                per_level=(("brake_force", 1.10),),
                base_prices=(2_000, 4_500, 8_000)),
)

UPGRADES_BY_ID: Dict[str, UpgradeKind] = {u.id: u for u in UPGRADES}


def upgrade_price(car: CarSpec, kind: UpgradeKind, next_level: int) -> int:
    """Cost of buying `next_level` (1-based) of `kind` for `car`."""
    if not 1 <= next_level <= kind.max_level:
        raise ValueError(f"{kind.id} has levels 1..{kind.max_level}")
    raw = kind.base_prices[next_level - 1] * car.upgrade_cost_factor
    return int(round(raw / 250.0)) * 250


# --------------------------------------------------------------------------- #
# Folding it all into physics
# --------------------------------------------------------------------------- #
def build_config(car: CarSpec, levels: Mapping[str, int] | None = None,
                 base: VehicleConfig = VEHICLE) -> VehicleConfig:
    """The VehicleConfig for `car` fitted with the given upgrade levels."""
    values: Dict[str, float] = {}

    def current(name: str) -> float:
        return values.get(name, getattr(base, name))

    for name, multiplier in car.tweaks:
        values[name] = current(name) * multiplier
    for name, absolute in car.traits:
        values[name] = absolute

    for kind in UPGRADES:
        level = int((levels or {}).get(kind.id, 0))
        level = max(0, min(kind.max_level, level))
        for name, multiplier in kind.per_level:
            values[name] = current(name) * multiplier ** level
        for name, step in kind.additive:
            values[name] = current(name) + step * level

    values["loose_traction"] = min(0.8, max(0.0, current("loose_traction")))
    return replace(base, **values)


# --------------------------------------------------------------------------- #
# Ratings for the garage screen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ratings:
    power: float
    top_speed: float
    grip: float
    handling: float
    braking: float
    rough: float

    @property
    def overall(self) -> float:
        return (self.power * 1.3 + self.top_speed + self.grip * 1.2
                + self.handling + self.braking * 0.6 + self.rough * 0.5) / 5.6


def _scale(value: float, low: float, high: float) -> float:
    return max(0.0, min(10.0, (value - low) / (high - low) * 10.0))


def rate(config: VehicleConfig) -> Ratings:
    """0-10 bars for the UI. Ranges cover the stock Kestrel to a maxed Apex."""
    power_to_weight = config.engine_force / (config.mass / 1200.0)
    return Ratings(
        power=_scale(power_to_weight, 8.0, 24.0),
        top_speed=_scale(config.top_speed, 40.0, 72.0),
        grip=_scale(config.lateral_grip * (1 + config.loose_traction), 6.0, 12.5),
        handling=_scale(config.max_steer / config.slip_falloff * 2.5, 0.75, 1.55),
        braking=_scale(config.brake_force, 12.0, 26.0),
        rough=_scale(1.0 / config.rough_terrain, 0.9, 2.4),
    )


# --------------------------------------------------------------------------- #
# Liveries
# --------------------------------------------------------------------------- #
LIVERY_COLOURS: Tuple[Tuple[str, RGB], ...] = (
    ("Ochre", (0.93, 0.60, 0.12)),
    ("Rally red", (0.78, 0.11, 0.10)),
    ("Arctic white", (0.92, 0.93, 0.94)),
    ("Midnight", (0.08, 0.10, 0.16)),
    ("Racing green", (0.06, 0.30, 0.17)),
    ("Electric blue", (0.07, 0.32, 0.78)),
    ("Lime", (0.55, 0.78, 0.10)),
    ("Graphite", (0.22, 0.23, 0.25)),
)
