"""Multiplayer sessions: the host's lobby and race control, and the client side.

Who decides what
----------------
Each player's own machine simulates their own car - that is what keeps
driving responsive however far away the host is. The host is the referee for
everything else: who is in the lobby, which stage and how many laps, the grid,
when the countdown starts, and the finishing order. The host also runs any AI
bots and relays every car's state to everyone at 20 Hz.

Clients render other cars `INTERP_DELAY` seconds in the past and interpolate
between the two snapshots that straddle that moment, so remote cars move
smoothly even though updates arrive in steps.

Threading: sockets have reader threads (see `transport`), but everything in
this module runs on the game's main thread via `poll()`, so the game never has
to lock anything.
"""

from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .protocol import (DEFAULT_PORT, MAX_PLAYERS, PROTOCOL_VERSION, clean_car,
                       clean_int, clean_name, clean_state, clean_upgrades)
from .transport import Connection, Listener, connect, lan_addresses
from ..stages import STAGES_BY_ID

INTERP_DELAY = 0.10
HOST_ID = "p0"


@dataclass
class LobbyPlayer:
    id: str
    name: str
    car: str
    livery: int = 0
    upgrades: Dict[str, int] = field(default_factory=dict)
    ready: bool = False
    is_host: bool = False
    is_bot: bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "car": self.car, "livery": self.livery,
                "upgrades": self.upgrades, "ready": self.ready, "host": self.is_host,
                "bot": self.is_bot}

    @classmethod
    def from_dict(cls, data: dict) -> "LobbyPlayer":
        return cls(id=str(data.get("id", "")), name=clean_name(data.get("name")),
                   car=clean_car(data.get("car")), livery=clean_int(data.get("livery"), 0, 63),
                   upgrades=clean_upgrades(data.get("upgrades")),
                   ready=bool(data.get("ready")), is_host=bool(data.get("host")),
                   is_bot=bool(data.get("bot")))


@dataclass
class NetEvent:
    kind: str           # joined | left | lobby | start | states | standings | results | closed | error
    data: Any = None


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #
class RemoteBuffer:
    """Recent snapshots of one remote car, sampled slightly in the past."""

    def __init__(self, capacity: int = 32) -> None:
        self.snapshots: List[Tuple[float, Dict[str, float]]] = []
        self.capacity = capacity

    def push(self, stamp: float, state: Dict[str, float]) -> None:
        if self.snapshots and stamp <= self.snapshots[-1][0]:
            return
        self.snapshots.append((stamp, state))
        if len(self.snapshots) > self.capacity:
            del self.snapshots[:-self.capacity]

    def sample(self, when: float) -> Optional[Dict[str, float]]:
        if not self.snapshots:
            return None
        if when <= self.snapshots[0][0]:
            return dict(self.snapshots[0][1])
        for (t0, a), (t1, b) in zip(self.snapshots, self.snapshots[1:]):
            if t0 <= when <= t1:
                f = (when - t0) / max(t1 - t0, 1e-6)
                out = {k: a[k] + (b[k] - a[k]) * f for k in a if k != "yaw"}
                dyaw = math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"]))
                out["yaw"] = a["yaw"] + dyaw * f
                return out
        # Past the newest snapshot: extrapolate briefly along the velocity,
        # then hold, so a dropped packet doesn't freeze the car mid-corner.
        t_last, last = self.snapshots[-1]
        ahead = min(when - t_last, 0.25)
        out = dict(last)
        fx, fz = math.sin(last["yaw"]), math.cos(last["yaw"])
        rx, rz = math.cos(last["yaw"]), -math.sin(last["yaw"])
        out["x"] += (fx * last["vl"] + rx * last["vt"]) * ahead
        out["z"] += (fz * last["vl"] + rz * last["vt"]) * ahead
        return out


# --------------------------------------------------------------------------- #
# Host
# --------------------------------------------------------------------------- #
class HostSession:
    def __init__(self, name: str, car: str, livery: int = 0,
                 upgrades: Optional[Dict[str, int]] = None, port: int = DEFAULT_PORT) -> None:
        self.players: Dict[str, LobbyPlayer] = {
            HOST_ID: LobbyPlayer(HOST_ID, clean_name(name), clean_car(car), livery,
                                 upgrades or {}, ready=True, is_host=True)}
        self.connections: Dict[str, Connection] = {}
        self.stage_id = "pine_ridge"
        self.laps = 3
        self.bots = 3
        self.phase = "lobby"                    # lobby | racing | results
        self.latest: Dict[str, Dict[str, float]] = {}
        self.buffers: Dict[str, RemoteBuffer] = {}
        self._pending: "queue.Queue[Tuple[Connection, Tuple[str, int]]]" = queue.Queue()
        self._unidentified: List[Connection] = []
        self._next_id = 1
        self.listener = Listener(port, lambda conn, addr: self._pending.put((conn, addr)))
        self.port = self.listener.port
        self.addresses = lan_addresses()
        self._sync_bots()

    # -- lobby settings ------------------------------------------------- #
    def configure(self, stage_id: Optional[str] = None, laps: Optional[int] = None,
                  bots: Optional[int] = None) -> None:
        if self.phase != "lobby":
            return
        if stage_id is not None and stage_id in STAGES_BY_ID:
            self.stage_id = stage_id
        if laps is not None:
            self.laps = max(1, min(10, laps))
        if bots is not None:
            self.bots = max(0, min(MAX_PLAYERS - 1, bots))
        self._sync_bots()
        self._broadcast_lobby()

    def _sync_bots(self) -> None:
        humans = [p for p in self.players.values() if not p.is_bot]
        room = max(0, MAX_PLAYERS - len(humans))
        wanted = min(self.bots, room)
        bots = [p for p in self.players.values() if p.is_bot]
        for bot in bots[wanted:]:
            del self.players[bot.id]
        from ..ai import RIVAL_NAMES
        cars = ("kestrel", "vanta", "ridgeback", "torque", "apex")
        for i in range(len(bots), wanted):
            bot_id = f"b{i}"
            self.players[bot_id] = LobbyPlayer(bot_id, RIVAL_NAMES[i % len(RIVAL_NAMES)],
                                               cars[i % len(cars)], livery=i + 1,
                                               ready=True, is_bot=True)

    def lobby_dict(self) -> dict:
        return {"t": "lobby", "players": [p.as_dict() for p in self.players.values()],
                "stage": self.stage_id, "laps": self.laps, "bots": self.bots,
                "phase": self.phase}

    def _broadcast_lobby(self) -> None:
        self.broadcast(self.lobby_dict())

    @property
    def can_start(self) -> bool:
        return self.phase == "lobby" and all(
            p.ready for p in self.players.values() if not p.is_host)

    # -- networking ------------------------------------------------------ #
    def broadcast(self, message: dict) -> None:
        for conn in list(self.connections.values()):
            conn.send(message)

    def poll(self) -> List[NetEvent]:
        events: List[NetEvent] = []
        while True:
            try:
                conn, _ = self._pending.get_nowait()
            except queue.Empty:
                break
            self._unidentified.append(conn)

        for conn in list(self._unidentified):
            for message in conn.poll():
                if message is None:
                    self._unidentified.remove(conn)
                    break
                if message.get("t") == "hello":
                    self._unidentified.remove(conn)
                    joined = self._welcome(conn, message)
                    if joined:
                        events.append(NetEvent("joined", joined))
                    break

        for player_id, conn in list(self.connections.items()):
            for message in conn.poll():
                if message is None:
                    events.append(self._drop(player_id))
                    break
                events.extend(self._handle(player_id, message))
        return events

    def _welcome(self, conn: Connection, hello: dict) -> Optional[LobbyPlayer]:
        if clean_int(hello.get("version"), 0, 10**6) != PROTOCOL_VERSION:
            conn.send({"t": "reject", "reason": "Different game version."})
            conn.close("version mismatch")
            return None
        if self.phase != "lobby":
            conn.send({"t": "reject", "reason": "A race is already running - try again shortly."})
            conn.close("race in progress")
            return None
        humans = [p for p in self.players.values() if not p.is_bot]
        if len(humans) >= MAX_PLAYERS:
            conn.send({"t": "reject", "reason": "The lobby is full."})
            conn.close("full")
            return None
        player_id = f"p{self._next_id}"
        self._next_id += 1
        player = LobbyPlayer(player_id, clean_name(hello.get("name")), clean_car(hello.get("car")),
                             clean_int(hello.get("livery"), 0, 63),
                             clean_upgrades(hello.get("upgrades")))
        self.players[player_id] = player
        self.connections[player_id] = conn
        self._sync_bots()
        conn.send({"t": "welcome", "id": player_id, "host": HOST_ID})
        self._broadcast_lobby()
        return player

    def _drop(self, player_id: str) -> NetEvent:
        conn = self.connections.pop(player_id, None)
        player = self.players.pop(player_id, None)
        self.latest.pop(player_id, None)
        self.buffers.pop(player_id, None)
        self._sync_bots()
        self._broadcast_lobby()
        return NetEvent("left", {"id": player_id, "name": player.name if player else "?",
                                 "reason": conn.close_reason if conn else ""})

    def _handle(self, player_id: str, message: dict) -> List[NetEvent]:
        kind = message.get("t")
        if kind == "ready" and self.phase == "lobby":
            self.players[player_id].ready = bool(message.get("ready"))
            self._broadcast_lobby()
            return [NetEvent("lobby", self.lobby_dict())]
        if kind == "state" and self.phase == "racing":
            state = clean_state(message.get("s"))
            if state is not None:
                self.latest[player_id] = state
                self.buffers.setdefault(player_id, RemoteBuffer()).push(time.monotonic(), state)
            return []
        if kind == "ping":
            self.connections[player_id].send({"t": "pong", "n": message.get("n")})
            return []
        if kind == "bye":
            return [self._drop(player_id)]
        return []

    # -- race control ---------------------------------------------------- #
    def start_race(self, countdown: float = 3.0) -> dict:
        """Lock the grid and tell everyone to go. Returns the start message."""
        self.phase = "racing"
        self.latest.clear()
        self.buffers.clear()
        grid = []
        ordered = ([p for p in self.players.values() if p.is_host]
                   + [p for p in self.players.values() if not p.is_host and not p.is_bot]
                   + [p for p in self.players.values() if p.is_bot])
        for slot, p in enumerate(ordered):
            grid.append({**p.as_dict(), "slot": slot})
        message = {"t": "start", "stage": self.stage_id, "laps": self.laps,
                   "countdown": countdown, "grid": grid}
        self.broadcast(message)
        return message

    def remote_states(self) -> Dict[str, Dict[str, float]]:
        """Remote players' cars, interpolated for smooth display on the host."""
        when = time.monotonic() - INTERP_DELAY
        out = {}
        for player_id, buffer in self.buffers.items():
            state = buffer.sample(when)
            if state is not None:
                out[player_id] = state
        return out

    def send_states(self, states: Dict[str, Dict[str, float]], clock: float) -> None:
        """Relay every car's latest state. `states` holds the host's own cars."""
        merged = dict(self.latest)
        merged.update(states)
        self.broadcast({"t": "states", "clock": round(clock, 3), "cars": merged})

    def send_standings(self, rows: List[dict], clock: float) -> None:
        self.broadcast({"t": "standings", "clock": round(clock, 3), "rows": rows})

    def send_results(self, rows: List[dict]) -> None:
        self.phase = "results"
        self.broadcast({"t": "results", "rows": rows})

    def back_to_lobby(self) -> None:
        self.phase = "lobby"
        for p in self.players.values():
            if not p.is_host and not p.is_bot:
                p.ready = False
        self._broadcast_lobby()

    def close(self) -> None:
        self.broadcast({"t": "closed", "reason": "The host closed the lobby."})
        for conn in list(self.connections.values()) + self._unidentified:
            conn.close("host closed")
        self.connections.clear()
        self.listener.close()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class JoinError(Exception):
    pass


class ClientSession:
    """The joining side. Construction never blocks.

    The TCP connect and the hello/welcome handshake happen in the background;
    watch `status` ("connecting", "joined", "failed") and `error`, calling
    `poll()` every frame as usual.
    """

    def __init__(self, address: str, port: int, name: str, car: str, livery: int = 0,
                 upgrades: Optional[Dict[str, int]] = None, timeout: float = 5.0) -> None:
        import threading
        self.address, self.port = address, port
        self.conn: Optional[Connection] = None
        self.id: Optional[str] = None
        self.status = "connecting"
        self.error = ""
        self.lobby: Optional[dict] = None
        self.buffers: Dict[str, RemoteBuffer] = {}
        self._hello = {"t": "hello", "version": PROTOCOL_VERSION, "name": name,
                       "car": car, "livery": livery, "upgrades": upgrades or {}}
        self._deadline = time.monotonic() + timeout + 3.0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._connect, args=(timeout,), daemon=True,
                                        name="net-connect")
        self._thread.start()

    def _connect(self, timeout: float) -> None:
        try:
            conn = connect(self.address, self.port, timeout=timeout)
            conn.send(self._hello)
            self.conn = conn
        except OSError as error:
            self.error = (f"Couldn't reach {self.address}:{self.port} - "
                          f"{error.strerror or error}. Check the address, that the host "
                          "is in their lobby, and that you're on the same network.")
        self._ready.set()

    def wait_joined(self, timeout: float = 5.0, pump=None) -> None:
        """Block until joined or failed. For tests and command-line tools only."""
        deadline = time.monotonic() + timeout
        while self.status == "connecting" and time.monotonic() < deadline:
            if pump:
                pump()
            self.poll()
            time.sleep(0.01)
        if self.status != "joined":
            raise JoinError(self.error or "The host didn't answer in time.")

    def set_ready(self, ready: bool) -> None:
        if self.conn:
            self.conn.send({"t": "ready", "ready": ready})

    def send_state(self, state: Dict[str, float]) -> None:
        if self.conn and self.status == "joined":
            self.conn.send({"t": "state", "s": state})

    def _fail(self, reason: str) -> List[NetEvent]:
        self.status = "failed"
        self.error = reason
        if self.conn:
            self.conn.close(reason)
        return [NetEvent("error", reason)]

    def poll(self) -> List[NetEvent]:
        events: List[NetEvent] = []
        if self.status == "failed":
            return events
        if self.conn is None:
            if self._ready.is_set():
                return self._fail(self.error or "Couldn't connect.")
            if time.monotonic() > self._deadline:
                return self._fail("Timed out connecting to the host.")
            return events

        now = time.monotonic()
        for message in self.conn.poll():
            if message is None:
                if self.status == "connecting":
                    return self._fail(self.conn.close_reason or "The host closed the connection.")
                events.append(NetEvent("closed", self.conn.close_reason or "Connection lost."))
                self.status = "failed"
                continue
            kind = message.get("t")
            if self.status == "connecting":
                if kind == "reject":
                    return self._fail(str(message.get("reason", "The host turned you away.")))
                if kind == "welcome":
                    self.id = str(message.get("id"))
                    self.status = "joined"
                    events.append(NetEvent("joined", self.id))
                    continue
            if kind == "lobby":
                self.lobby = message
                events.append(NetEvent("lobby", message))
            elif kind == "start":
                self.buffers.clear()
                events.append(NetEvent("start", message))
            elif kind == "states":
                cars = message.get("cars")
                if isinstance(cars, dict):
                    for car_id, raw in cars.items():
                        if car_id == self.id:
                            continue
                        state = clean_state(raw)
                        if state is not None:
                            self.buffers.setdefault(car_id, RemoteBuffer()).push(now, state)
            elif kind == "standings":
                events.append(NetEvent("standings", message))
            elif kind == "results":
                events.append(NetEvent("results", message))
            elif kind == "closed":
                events.append(NetEvent("closed", str(message.get("reason", "Host left."))))
        return events

    def remote_states(self) -> Dict[str, Dict[str, float]]:
        when = time.monotonic() - INTERP_DELAY
        out = {}
        for car_id, buffer in self.buffers.items():
            state = buffer.sample(when)
            if state is not None:
                out[car_id] = state
        return out

    def close(self) -> None:
        if self.conn:
            self.conn.send({"t": "bye"})
            self.conn.close("left")
