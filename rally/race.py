"""One race: the grid, the countdown, the clock, and who finished where.

`RaceSession` is the referee. It owns every car in the race and knows three
kinds of entrant:

  * player  - driven by local input
  * ai      - driven by an `ai.Driver` running here
  * remote  - driven by someone else's machine; this session only receives
              where the car is (see `apply_remote`)

The single-player game and the multiplayer host run the same class, which is
what keeps their rules identical: same countdown, same lap counting, same
finish window, same DNF rule.

Pure Python. No rendering, no sockets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ai import Driver, Personality, RacingLine
from .cars import CARS_BY_ID, build_config
from .collision import StaticColliders, resolve_cars
from .config import GameConfig
from .scoring import LapEvent, LapTimer
from .stages import Stage
from .terrain import Terrain
from .track import Track
from .vehicle import Control, Vehicle

FINISH_WINDOW = 45.0        # seconds after the winner before the rest are DNF
SECTORS = 20                # timing points per lap, for live gaps


class Phase(Enum):
    GRID = auto()
    COUNTDOWN = auto()
    RACING = auto()
    DONE = auto()


@dataclass
class Entrant:
    id: str
    name: str
    kind: str                       # "player" | "ai" | "remote"
    car_id: str
    livery: int
    vehicle: Vehicle
    timer: LapTimer
    driver: Optional[Driver] = None
    finished_at: Optional[float] = None
    dnf: bool = False
    grid_slot: int = 0
    resets: int = 0
    # Race clock at each timing sector crossed, keyed by (lap, sector).
    splits: Dict[Tuple[int, int], float] = field(default_factory=dict)
    last_split: Optional[Tuple[int, int]] = None
    # Latest snapshot received for a remote car (time, state dict).
    remote_state: Optional[dict] = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None or self.dnf


@dataclass(frozen=True)
class Result:
    position: int
    id: str
    name: str
    kind: str
    car_id: str
    time: Optional[float]           # total race time; None for a DNF
    best_lap: Optional[float]
    gap: Optional[float]            # to the winner
    dnf: bool


@dataclass(frozen=True)
class Event:
    kind: str                       # "go" | "lap" | "finish" | "impact" | "reset" | "done"
    entrant: Optional[str] = None
    value: float = 0.0
    detail: str = ""


class RaceSession:
    def __init__(self, stage: Stage, track: Track, terrain: Terrain,
                 laps: int, colliders: Optional[StaticColliders] = None,
                 countdown: float = 3.0) -> None:
        self.stage = stage
        self.track = track
        self.terrain = terrain
        self.laps = laps
        self.colliders = colliders
        self.countdown_length = countdown
        self.config = GameConfig(laps=laps)

        self.entrants: Dict[str, Entrant] = {}
        self.order: List[str] = []           # insertion order = grid order
        self.phase = Phase.GRID
        self.countdown = countdown
        self.clock = 0.0                     # race time, starts at GO
        self.first_finish: Optional[float] = None
        self._lines: Dict[Tuple[str, Tuple[int, ...]], RacingLine] = {}

    # ------------------------------------------------------------------ #
    # Grid
    # ------------------------------------------------------------------ #
    def grid_slot(self, slot: int) -> Tuple[float, float, float]:
        """(x, z, yaw) for a grid slot: two columns, staggered, behind the line."""
        n = self.track.count
        spacing = self.track.length / n
        back = 7.0 + slot * 4.6                  # metres behind the line
        index = (n - int(round(back / spacing))) % n
        centre, right = self.track.points[index], self.track.rights[index]
        side = -2.4 if slot % 2 == 0 else 2.4
        return (float(centre[0] + right[0] * side),
                float(centre[2] + right[2] * side),
                self.track.heading_at(index))

    def add(self, entrant_id: str, name: str, kind: str, car_id: str,
            livery: int = 0, upgrades: Optional[dict] = None,
            personality: Optional[Personality] = None, seed: int = 0) -> Entrant:
        if self.phase is not Phase.GRID:
            raise RuntimeError("entrants can only join on the grid")
        if entrant_id in self.entrants:
            raise ValueError(f"duplicate entrant {entrant_id!r}")
        car = CARS_BY_ID[car_id]
        config = build_config(car, upgrades or {})
        vehicle = Vehicle(config=config)
        timer = LapTimer(self.track, self.config)
        slot = len(self.order)
        entrant = Entrant(entrant_id, name, kind, car_id, livery, vehicle, timer,
                          grid_slot=slot)
        if kind == "ai":
            line = self._line_for(car_id, upgrades or {}, config)
            entrant.driver = Driver(line, personality or Personality(), seed=seed)
        self.entrants[entrant_id] = entrant
        self.order.append(entrant_id)
        self._place_on_grid(entrant)
        return entrant

    def remove(self, entrant_id: str) -> None:
        """A remote player left. Their car is retired, not deleted."""
        entrant = self.entrants.get(entrant_id)
        if entrant and not entrant.done:
            entrant.dnf = True

    def _line_for(self, car_id: str, upgrades: dict, config) -> RacingLine:
        key = (car_id, tuple(sorted(upgrades.items())))
        if key not in self._lines:
            self._lines[key] = RacingLine.build(self.track, config,
                                                self.stage.surface.grip_line)
        return self._lines[key]

    def _place_on_grid(self, entrant: Entrant) -> None:
        x, z, yaw = self.grid_slot(entrant.grid_slot)
        entrant.vehicle.place(x, z, yaw, self.terrain)
        index = self.track.nearest_index(x, z, hint=self.track.count - 5)
        entrant.timer.reset(index, behind_line=True)
        if entrant.driver:
            entrant.driver.reset(index)

    # ------------------------------------------------------------------ #
    # Flow
    # ------------------------------------------------------------------ #
    def start_countdown(self) -> None:
        self.phase = Phase.COUNTDOWN
        self.countdown = self.countdown_length

    @property
    def countdown_label(self) -> str:
        if self.phase is Phase.COUNTDOWN:
            return str(max(1, math.ceil(self.countdown)))
        if self.phase is Phase.RACING and self.clock < 0.8:
            return "GO"
        return ""

    def update(self, dt: float, controls: Optional[Dict[str, Control]] = None) -> List[Event]:
        controls = controls or {}
        events: List[Event] = []

        if self.phase is Phase.COUNTDOWN:
            self.countdown -= dt
            if self.countdown <= 0.0:
                self.phase = Phase.RACING
                self.clock = 0.0
                events.append(Event("go"))
            # Cars sit on the grid, but physics runs so they settle on the ground.
            for entrant in self._local():
                entrant.vehicle.step(Control(), self.terrain, dt)
            return events

        if self.phase is not Phase.RACING:
            return events

        self.clock += dt
        positions = [(e.vehicle.x, e.vehicle.z) for e in self.entrants.values()]

        for entrant in self._local():
            if entrant.done:
                control = Control(brake=0.6)          # coast to a halt past the flag
            elif entrant.kind == "ai":
                others = [p for p, e in zip(positions, self.entrants.values()) if e is not entrant]
                control = entrant.driver.control(entrant.vehicle, dt, others)
            else:
                control = controls.get(entrant.id, Control())
            entrant.vehicle.step(control, self.terrain, dt)

            if self.colliders is not None:
                impact = self.colliders.resolve(entrant.vehicle)
                if impact > 2.0:
                    events.append(Event("impact", entrant.id, impact, "scenery"))

            if entrant.driver and entrant.driver.wants_reset:
                self.recover(entrant.id)
                events.append(Event("reset", entrant.id))

        events.extend(self._collide_cars())

        for entrant in self.entrants.values():
            if entrant.done:
                continue
            event = entrant.timer.update(entrant.vehicle.x, entrant.vehicle.z, self.clock)
            if entrant.timer.lap >= 0:
                key = (entrant.timer.lap, int(entrant.timer.progress * SECTORS))
                if key != entrant.last_split and key not in entrant.splits:
                    entrant.splits[key] = self.clock
                    entrant.last_split = key
            if event in (LapEvent.LAP_COMPLETED, LapEvent.STAGE_FINISHED):
                events.append(Event("lap", entrant.id, entrant.timer.last or 0.0))
            if event is LapEvent.STAGE_FINISHED:
                entrant.finished_at = self.clock
                if self.first_finish is None:
                    self.first_finish = self.clock
                events.append(Event("finish", entrant.id, self.clock,
                                    str(self.position_of(entrant.id))))

        if self.first_finish is not None and self.clock - self.first_finish > FINISH_WINDOW:
            for entrant in self.entrants.values():
                if not entrant.done:
                    entrant.dnf = True

        if all(e.done for e in self.entrants.values()):
            self.phase = Phase.DONE
            events.append(Event("done"))
        return events

    def _local(self) -> Iterable[Entrant]:
        return (e for e in self.entrants.values() if e.kind != "remote")

    def _collide_cars(self) -> List[Event]:
        events = []
        entrants = list(self.entrants.values())
        for i, a in enumerate(entrants):
            for b in entrants[i + 1:]:
                if a.kind == "remote" and b.kind == "remote":
                    continue
                dx = a.vehicle.x - b.vehicle.x
                dz = a.vehicle.z - b.vehicle.z
                if dx * dx + dz * dz > 25.0:
                    continue
                impact = resolve_cars(a.vehicle, b.vehicle,
                                      a_movable=a.kind != "remote",
                                      b_movable=b.kind != "remote")
                if impact > 1.5:
                    events.append(Event("impact", a.id, impact, b.id))
        return events

    # ------------------------------------------------------------------ #
    def recover(self, entrant_id: str) -> None:
        """Put a car back on the centreline facing the right way."""
        entrant = self.entrants[entrant_id]
        v = entrant.vehicle
        index = self.track.nearest_index(v.x, v.z, hint=entrant.timer.index)
        point = self.track.points[index]
        v.place(float(point[0]), float(point[2]), self.track.heading_at(index),
                self.terrain, keep_speed=min(max(v.v_long, 0.0), 8.0))
        entrant.timer.resync(index)
        if entrant.driver:
            entrant.driver.reset(index)
        entrant.resets += 1

    def apply_remote(self, entrant_id: str, state: dict) -> None:
        """Accept a position report for a remote car."""
        entrant = self.entrants.get(entrant_id)
        if entrant is None or entrant.kind != "remote":
            return
        v = entrant.vehicle
        v.x, v.y, v.z = state["x"], state["y"], state["z"]
        v.yaw = state["yaw"]
        v.v_long, v.v_lat = state["vl"], state["vt"]
        v.steer_angle = state.get("st", 0.0)
        v.rpm = state.get("rpm", 0.0)
        entrant.remote_state = state

    # ------------------------------------------------------------------ #
    # Standings
    # ------------------------------------------------------------------ #
    def standings(self) -> List[Entrant]:
        """Running order: finishers by time, then by distance covered, DNFs last."""
        def key(e: Entrant):
            if e.finished_at is not None:
                return (0, e.finished_at)
            if e.dnf:
                return (2, -e.timer.race_distance())
            return (1, -e.timer.race_distance())
        return sorted(self.entrants.values(), key=key)

    def gaps(self) -> Dict[str, str]:
        """Live gap to the leader, like a timing screen: "+3.2" or "+1 LAP".

        Each car's last timing sector is compared with the moment the leader
        crossed that same sector, which is an honest time gap rather than a
        distance guess.
        """
        order = self.standings()
        if not order:
            return {}
        leader = order[0]
        out = {leader.id: "LEADER" if leader.finished_at is None else "WINNER"}
        for e in order[1:]:
            if e.dnf:
                out[e.id] = "DNF"
            elif e.finished_at is not None and leader.finished_at is not None:
                out[e.id] = f"+{e.finished_at - leader.finished_at:.1f}"
            elif e.last_split is None:
                out[e.id] = "-"
            else:
                laps_down = leader.timer.lap - e.timer.lap
                if laps_down >= 1 and leader.timer.race_distance() - e.timer.race_distance() > 1.0:
                    out[e.id] = f"+{int(leader.timer.race_distance() - e.timer.race_distance())} LAP"
                elif e.last_split in leader.splits:
                    # Clamped: just after an overtake the order can lead the timing.
                    gap = max(0.0, e.splits[e.last_split] - leader.splits[e.last_split])
                    out[e.id] = f"+{gap:.1f}"
                else:
                    out[e.id] = "-"
        return out

    def position_of(self, entrant_id: str) -> int:
        for i, entrant in enumerate(self.standings()):
            if entrant.id == entrant_id:
                return i + 1
        raise KeyError(entrant_id)

    def results(self) -> List[Result]:
        rows = []
        winner = None
        for i, e in enumerate(self.standings()):
            if i == 0 and e.finished_at is not None:
                winner = e.finished_at
            gap = (e.finished_at - winner) if (e.finished_at is not None and winner is not None) else None
            rows.append(Result(i + 1, e.id, e.name, e.kind, e.car_id, e.finished_at,
                               e.timer.best if e.timer.laps else None, gap, e.dnf))
        return rows
