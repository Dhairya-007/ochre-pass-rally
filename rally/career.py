"""Career mode: championships, rivals, money, the garage, and the save file.

Structure
---------
Three championship tiers. Each is a season of events, one per stage visit,
contested against a fixed roster of AI rivals who carry points from event to
event. Finish a season in the top three to unlock the next tier. A season can
be replayed as often as you like, which is how you earn money for the garage.

Every event pays prize money by finishing position and championship points on
the usual 25-18-15-12-10-8-6-4 scale. Winning the title pays a bonus.

Persistence
-----------
`Career.save()` writes JSON atomically (write to a temp file, then rename) so a
crash mid-save can never corrupt the only copy. The file carries a version
number; `Career.load()` refuses files from the future rather than guessing.

Pure Python - the career screens just call into this.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ai import Personality, RIVAL_NAMES
from .cars import CARS, CARS_BY_ID, UPGRADES_BY_ID, upgrade_price
from .race import Result

SAVE_VERSION = 1
STARTING_MONEY = 5_000
POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Static championship data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rival:
    id: str
    name: str
    car_id: str
    skill: float
    consistency: float
    aggression: float

    def personality(self) -> Personality:
        return Personality(skill=self.skill, aggression=self.aggression,
                           consistency=self.consistency)


@dataclass(frozen=True)
class EventSpec:
    stage_id: str
    laps: int


@dataclass(frozen=True)
class Championship:
    tier: int
    name: str
    blurb: str
    events: Tuple[EventSpec, ...]
    rivals: Tuple[Rival, ...]
    prizes: Tuple[int, ...]            # by finishing position, 1st first
    title_bonus: int
    unlock_position: int = 3           # finish this high or better to unlock next


def _rivals(tier: int, cars: Sequence[str], skills: Sequence[float]) -> Tuple[Rival, ...]:
    out = []
    for i, (car, skill) in enumerate(zip(cars, skills)):
        name = RIVAL_NAMES[(tier * 4 + i) % len(RIVAL_NAMES)]
        out.append(Rival(f"t{tier}r{i}", name, car, skill,
                         consistency=0.55 + 0.4 * skill, aggression=0.35 + 0.1 * i % 0.6))
    return tuple(out)


CHAMPIONSHIPS: Tuple[Championship, ...] = (
    Championship(
        tier=0, name="Rookie Cup",
        blurb="One-make club rallies. Learn the stages, bank some prize money, "
              "and finish in the top three to get noticed.",
        events=(EventSpec("pine_ridge", 2), EventSpec("red_mesa", 2),
                EventSpec("dusk_valley", 2), EventSpec("frostfell", 2)),
        rivals=_rivals(0, ("kestrel",) * 5, (0.40, 0.48, 0.55, 0.62, 0.68)),
        prizes=(6_000, 4_200, 3_000, 2_200, 1_600, 1_100),
        title_bonus=8_000,
    ),
    Championship(
        tier=1, name="National Series",
        blurb="Bigger grids, faster cars and rivals who don't lift. Three laps "
              "a stage. Upgrades start to matter here.",
        events=(EventSpec("red_mesa", 3), EventSpec("pine_ridge", 3),
                EventSpec("frostfell", 3), EventSpec("dusk_valley", 3),
                EventSpec("pine_ridge", 3)),
        rivals=_rivals(1, ("vanta", "kestrel", "ridgeback", "vanta", "torque", "kestrel", "ridgeback"),
                       (0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.85)),
        prizes=(14_000, 10_000, 7_500, 5_500, 4_000, 3_000, 2_200, 1_500),
        title_bonus=20_000,
    ),
    Championship(
        tier=2, name="World Rally Championship",
        blurb="The best drivers in the fastest machinery on every surface. "
              "Nothing short of a tuned Rally1 or prototype will do.",
        events=(EventSpec("frostfell", 3), EventSpec("red_mesa", 4),
                EventSpec("dusk_valley", 3), EventSpec("pine_ridge", 4),
                EventSpec("frostfell", 3), EventSpec("red_mesa", 3)),
        rivals=_rivals(2, ("torque", "apex", "torque", "apex", "vanta", "apex", "torque"),
                       (0.80, 0.84, 0.87, 0.90, 0.92, 0.95, 0.98)),
        prizes=(30_000, 21_000, 15_000, 11_000, 8_000, 6_000, 4_500, 3_000),
        title_bonus=60_000,
        unlock_position=1,
    ),
)


# --------------------------------------------------------------------------- #
# Mutable career state
# --------------------------------------------------------------------------- #
@dataclass
class OwnedCar:
    upgrades: Dict[str, int] = field(default_factory=dict)
    livery: int = 0


@dataclass
class EventOutcome:
    """What one finished event did to the career - shown on the results screen."""
    position: int
    points: int
    prize: int
    standings: List[Tuple[str, str, int]]      # (id, name, points), sorted
    season_complete: bool = False
    championship_position: Optional[int] = None
    title_bonus: int = 0
    unlocked_tier: Optional[int] = None


class CareerError(Exception):
    """A garage or career action that the player is not allowed to take."""


@dataclass
class Career:
    player_name: str = "Driver"
    money: int = STARTING_MONEY
    owned: Dict[str, OwnedCar] = field(default_factory=lambda: {"kestrel": OwnedCar()})
    current_car: str = "kestrel"
    tier: int = 0
    unlocked_tier: int = 0
    event_index: int = 0
    season: int = 1
    standings: Dict[str, int] = field(default_factory=dict)
    best_laps: Dict[str, float] = field(default_factory=dict)
    races: int = 0
    wins: int = 0
    podiums: int = 0
    earned: int = 0
    titles: List[str] = field(default_factory=list)

    PLAYER_ID = "player"

    # ------------------------------------------------------------------ #
    # Championship
    # ------------------------------------------------------------------ #
    @property
    def championship(self) -> Championship:
        return CHAMPIONSHIPS[self.tier]

    @property
    def next_event(self) -> EventSpec:
        return self.championship.events[self.event_index]

    @property
    def season_in_progress(self) -> bool:
        return self.event_index > 0

    def choose_tier(self, tier: int) -> None:
        if not 0 <= tier <= self.unlocked_tier:
            raise CareerError("That championship is still locked.")
        if self.season_in_progress and tier != self.tier:
            raise CareerError("Finish the current season before switching championship.")
        if tier != self.tier:
            self.tier = tier
            self.standings = {}

    def table(self) -> List[Tuple[str, str, int]]:
        """Championship standings as (id, name, points), leader first."""
        rows = [(self.PLAYER_ID, self.player_name, self.standings.get(self.PLAYER_ID, 0))]
        rows += [(r.id, r.name, self.standings.get(r.id, 0)) for r in self.championship.rivals]
        return sorted(rows, key=lambda row: (-row[2], row[0] != self.PLAYER_ID, row[1]))

    def record(self, results: Sequence[Result]) -> EventOutcome:
        """Apply a finished event to the career and advance to the next one."""
        champ = self.championship
        event = self.next_event
        player = next(r for r in results if r.id == self.PLAYER_ID)

        for row in results:
            if row.dnf:
                continue
            points = POINTS[row.position - 1] if row.position <= len(POINTS) else 0
            self.standings[row.id] = self.standings.get(row.id, 0) + points

        position = player.position
        points = 0 if player.dnf else (POINTS[position - 1] if position <= len(POINTS) else 0)
        prize = 0 if player.dnf else (champ.prizes[position - 1] if position <= len(champ.prizes) else 0)
        self.money += prize
        self.earned += prize
        self.races += 1
        if not player.dnf and position == 1:
            self.wins += 1
        if not player.dnf and position <= 3:
            self.podiums += 1
        if player.best_lap is not None:
            previous = self.best_laps.get(event.stage_id)
            if previous is None or player.best_lap < previous:
                self.best_laps[event.stage_id] = player.best_lap

        outcome = EventOutcome(position=position, points=points, prize=prize,
                               standings=self.table())

        self.event_index += 1
        if self.event_index >= len(champ.events):
            self._close_season(outcome)
        return outcome

    def _close_season(self, outcome: EventOutcome) -> None:
        champ = self.championship
        table = self.table()
        place = next(i for i, row in enumerate(table) if row[0] == self.PLAYER_ID) + 1
        outcome.season_complete = True
        outcome.championship_position = place
        if place == 1:
            outcome.title_bonus = champ.title_bonus
            self.money += champ.title_bonus
            self.earned += champ.title_bonus
            self.titles.append(f"{champ.name} - season {self.season}")
        if place <= champ.unlock_position and champ.tier + 1 < len(CHAMPIONSHIPS) \
                and self.unlocked_tier < champ.tier + 1:
            self.unlocked_tier = champ.tier + 1
            outcome.unlocked_tier = self.unlocked_tier
        self.event_index = 0
        self.season += 1
        self.standings = {}

    # ------------------------------------------------------------------ #
    # Garage
    # ------------------------------------------------------------------ #
    def buy_car(self, car_id: str) -> None:
        car = CARS_BY_ID[car_id]
        if car_id in self.owned:
            raise CareerError(f"You already own the {car.name}.")
        if self.money < car.price:
            raise CareerError(f"The {car.name} costs ${car.price:,}; you have ${self.money:,}.")
        self.money -= car.price
        self.owned[car_id] = OwnedCar()
        self.current_car = car_id

    def select_car(self, car_id: str) -> None:
        if car_id not in self.owned:
            raise CareerError("You don't own that car yet.")
        self.current_car = car_id

    def upgrade_level(self, car_id: str, kind_id: str) -> int:
        return self.owned[car_id].upgrades.get(kind_id, 0)

    def next_upgrade_price(self, car_id: str, kind_id: str) -> Optional[int]:
        kind = UPGRADES_BY_ID[kind_id]
        level = self.upgrade_level(car_id, kind_id)
        if level >= kind.max_level:
            return None
        return upgrade_price(CARS_BY_ID[car_id], kind, level + 1)

    def buy_upgrade(self, car_id: str, kind_id: str) -> int:
        if car_id not in self.owned:
            raise CareerError("You don't own that car yet.")
        price = self.next_upgrade_price(car_id, kind_id)
        if price is None:
            raise CareerError("That upgrade is already fully fitted.")
        if self.money < price:
            raise CareerError(f"That upgrade costs ${price:,}; you have ${self.money:,}.")
        self.money -= price
        upgrades = self.owned[car_id].upgrades
        upgrades[kind_id] = upgrades.get(kind_id, 0) + 1
        return upgrades[kind_id]

    def set_livery(self, car_id: str, livery: int) -> None:
        if car_id not in self.owned:
            raise CareerError("You don't own that car yet.")
        self.owned[car_id].livery = livery

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        data = asdict(self)
        data["version"] = SAVE_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Career":
        version = data.get("version", 0)
        if version > SAVE_VERSION:
            raise CareerError("This save was made by a newer version of the game.")
        data = {k: v for k, v in data.items() if k != "version"}
        owned = {cid: OwnedCar(**o) for cid, o in data.pop("owned", {}).items()
                 if cid in CARS_BY_ID}
        career = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        career.owned = owned or {"kestrel": OwnedCar()}
        if career.current_car not in career.owned:
            career.current_car = next(iter(career.owned))
        career.tier = max(0, min(career.tier, len(CHAMPIONSHIPS) - 1))
        career.unlocked_tier = max(career.tier, min(career.unlocked_tier, len(CHAMPIONSHIPS) - 1))
        career.event_index = max(0, min(career.event_index,
                                        len(CHAMPIONSHIPS[career.tier].events) - 1))
        return career

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".career-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self.to_dict(), handle, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "Career":
        return cls.from_dict(json.loads(path.read_text()))
