"""Multiplayer: host or join, then the lobby."""

from __future__ import annotations

from typing import Optional

from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode

from ...cars import CARS, CARS_BY_ID, LIVERY_COLOURS
from ...net.protocol import DEFAULT_PORT
from ...net.session import ClientSession, HostSession
from ...stages import STAGES, STAGES_BY_ID
from .kit import (ACCENT, BAD, GOOD, GREY, GREY_DARK, INK, INK_SOFT, WHITE, Button, Field,
                  Fonts, Screen, Selector, panel, stripe, text, title_block)
from .menus import MainMenu, ResultsScreen, anchor, left_shade


def _player_car(app):
    """Career car if there is a career, otherwise the last quick-race pick."""
    if app.career:
        owned = app.career.owned[app.career.current_car]
        return app.career.current_car, owned.livery, dict(owned.upgrades)
    return app.mp_car, app.mp_livery, {}


class MultiplayerMenu(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Online · LAN", "Multiplayer", 0.12, 0.8)
        text(col, "One player hosts; everyone else joins with the host's IP address.\n"
                  "On the same Wi-Fi this just works. Across the internet the host must\n"
                  f"forward TCP port {DEFAULT_PORT} on their router.",
             (0.12, 0.56), 0.032, GREY, Fonts.regular)
        self.name = self.add(Field(col, "Your name", app.settings.name, (0.12, 0.37), 1.1, 20))
        if not app.career:
            self.add(Selector(col, "Car", [c.name for c in CARS],
                              [c.id for c in CARS].index(app.mp_car), (0.12, 0.275),
                              lambda i: setattr(app, "mp_car", CARS[i].id), 1.1, app.audio.ui))
            self.add(Selector(col, "Livery", [n for n, _ in LIVERY_COLOURS], app.mp_livery,
                              (0.12, 0.18), lambda i: setattr(app, "mp_livery", i), 1.1,
                              app.audio.ui))
            y = 0.06
        else:
            car = CARS_BY_ID[app.career.current_car]
            text(col, f"RACING YOUR CAREER CAR: {car.name.upper()} (WITH UPGRADES)",
                 (0.12, 0.27), 0.03, ACCENT, Fonts.bold)
            y = 0.17
        self.add(Button(col, "Host a lobby", (0.12, y), self.host, width=1.1, height=0.1,
                        scale=0.055, sound=app.audio.ui))
        self.address = self.add(Field(col, "Host IP", app.settings.last_host or "192.168.",
                                      (0.12, y - 0.13), 1.1, 40, on_submit=self.join))
        self.add(Button(col, "Join", (0.12, y - 0.225), self.join, width=1.1, height=0.1,
                        scale=0.055, sound=app.audio.ui))
        self.add(Button(col, "Back", (0.12, y - 0.345), self.back, width=1.1, sound=app.audio.ui))
        self.status = text(col, "", (0.12, y - 0.45), 0.036, GREY, Fonts.bold)
        self.client: Optional[ClientSession] = None

    def _name(self) -> str:
        name = self.name.value or "Driver"
        self.app.settings.name = name
        return name

    def host(self) -> None:
        car, livery, upgrades = _player_car(self.app)
        try:
            session = HostSession(self._name(), car, livery, upgrades, port=self.app.settings.port)
        except OSError as error:
            self.status.setText(f"COULDN'T OPEN PORT {self.app.settings.port}: {error.strerror or error}".upper())
            self.status["fg"] = BAD
            return
        self.app.settings.save()
        self.app.goto(Lobby, host=session)

    def join(self) -> None:
        raw = self.address.value.strip()
        if not raw or raw.endswith("."):
            self.status.setText("TYPE THE HOST'S IP ADDRESS FIRST")
            self.status["fg"] = BAD
            return
        address, _, port = raw.partition(":")
        port_number = int(port) if port.isdigit() else self.app.settings.port
        self.app.settings.last_host = raw
        self.app.settings.save()
        car, livery, upgrades = _player_car(self.app)
        self.client = ClientSession(address, port_number, self._name(), car, livery, upgrades)
        self.status.setText(f"CONNECTING TO {address}:{port_number} ...")
        self.status["fg"] = WHITE

    def update(self, dt: float) -> None:
        if not self.client:
            return
        for ev in self.client.poll():
            if ev.kind == "error":
                self.status.setText(ev.data.upper()[:110])
                self.status["fg"] = BAD
                self.client = None
                return
        if self.client and self.client.status == "joined":
            client, self.client = self.client, None
            self.app.goto(Lobby, client=client)

    def back(self) -> None:
        if self.client:
            self.client.close()
        self.app.goto(MainMenu)


class Lobby(Screen):
    def __init__(self, app, host: Optional[HostSession] = None,
                 client: Optional[ClientSession] = None) -> None:
        super().__init__(app)
        self.host = host
        self.client = client
        self.ready = False
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Multiplayer", "Lobby", 0.12, 0.8)
        if host:
            text(col, "FRIENDS JOIN AT", (0.12, 0.58), 0.03, GREY, Fonts.bold)
            text(col, "   ".join(f"{a}" for a in host.addresses) + f"   (port {host.port})",
                 (0.12, 0.51), 0.052, ACCENT, Fonts.display)
            self.stage = self.add(Selector(col, "Stage", [s.name for s in STAGES],
                                           [s.id for s in STAGES].index(host.stage_id),
                                           (0.12, 0.38),
                                           lambda i: host.configure(stage_id=STAGES[i].id),
                                           1.1, app.audio.ui))
            self.add(Selector(col, "Laps", [str(n) for n in range(1, 11)], host.laps - 1,
                              (0.12, 0.285), lambda i: host.configure(laps=i + 1), 1.1,
                              app.audio.ui))
            self.add(Selector(col, "AI bots", [str(n) for n in range(0, 8)], host.bots,
                              (0.12, 0.19), lambda i: host.configure(bots=i), 1.1, app.audio.ui))
            self.start = self.add(Button(col, "Start race", (0.12, 0.07), self.start_race,
                                         width=1.1, height=0.1, scale=0.055, sound=app.audio.ui))
        else:
            text(col, f"CONNECTED TO {client.address}", (0.12, 0.58), 0.03, GREY, Fonts.bold)
            self.info = text(col, "", (0.12, 0.47), 0.05, ACCENT, Fonts.display)
            self.start = self.add(Button(col, "Ready", (0.12, 0.33), self.toggle_ready,
                                         width=1.1, height=0.1, scale=0.055, sound=app.audio.ui))
        self.add(Button(col, "Leave", (0.12, -0.05 if host else 0.21), self.back, width=1.1,
                        sound=app.audio.ui))
        self.status = text(col, "", (0.12, -0.16 if host else 0.1), 0.034, GREY, Fonts.bold)

        side = anchor(self, "right")
        panel(side, (-0.95, -0.08, -0.38, 0.5), INK)
        stripe(side, (-0.95, -0.08, 0.492, 0.5))
        text(side, "DRIVERS", (-0.9, 0.42), 0.032, ACCENT, Fonts.bold)
        self.rows = []
        for i in range(8):
            y = 0.33 - i * 0.085
            name = text(side, "", (-0.9, y), 0.04, WHITE, Fonts.bold)
            car = text(side, "", (-0.9, y - 0.035), 0.028, GREY, Fonts.semibold)
            state = text(side, "", (-0.12, y), 0.034, GOOD, Fonts.bold, TextNode.ARight)
            self.rows.append((name, car, state))
        self._refresh_rows()

    # ------------------------------------------------------------------ #
    def _lobby(self) -> Optional[dict]:
        return self.host.lobby_dict() if self.host else (self.client.lobby if self.client else None)

    def _refresh_rows(self) -> None:
        lobby = self._lobby()
        players = lobby["players"] if lobby else []
        for i, (name, car, state) in enumerate(self.rows):
            if i < len(players):
                p = players[i]
                tag = " (HOST)" if p["host"] else (" (BOT)" if p["bot"] else "")
                me = self.client and p["id"] == self.client.id
                name.setText(p["name"].upper() + tag + ("  ← YOU" if me else ""))
                car.setText(CARS_BY_ID[p["car"]].name.upper())
                ready = p["ready"] or p["host"] or p["bot"]
                state.setText("READY" if ready else "NOT READY")
                state["fg"] = GOOD if ready else GREY
            else:
                name.setText(""); car.setText(""); state.setText("")
        if self.host:
            self.start.set_enabled(self.host.can_start)
            self.status.setText("" if self.host.can_start else "WAITING FOR EVERYONE TO BE READY")
        elif lobby:
            stage = STAGES_BY_ID.get(lobby["stage"])
            self.info.setText(f"{stage.name.upper() if stage else '?'}  ·  {lobby['laps']} LAPS")

    def update(self, dt: float) -> None:
        if self.host:
            self.host.poll()
            self._refresh_rows()
            return
        for ev in self.client.poll():
            if ev.kind in ("closed", "error"):
                self.app.toast(str(ev.data), BAD)
                self.client.close()
                self.app.goto(MainMenu)
                return
            if ev.kind == "start":
                self._begin(ev.data)
                return
        self._refresh_rows()

    def toggle_ready(self) -> None:
        self.ready = not self.ready
        self.client.set_ready(self.ready)
        self.start.set_label("Not ready" if self.ready else "Ready")

    def start_race(self) -> None:
        if self.host.can_start:
            self._begin(self.host.start_race())

    def _begin(self, start: dict) -> None:
        from ..race_controller import EntrantSpec, RaceSetup
        me = "p0" if self.host else self.client.id
        entrants = []
        for g in start["grid"]:
            if g["id"] == me:
                kind = "player"
            elif self.host and g["bot"]:
                kind = "ai"
            else:
                kind = "remote"
            entrants.append(EntrantSpec(g["id"], g["name"], kind, g["car"], g["livery"],
                                        g.get("upgrades", {}), skill=0.72, consistency=0.8,
                                        seed=int(g["slot"])))
        setup = RaceSetup(start["stage"], int(start["laps"]), entrants, me,
                          "host" if self.host else "client")
        net = self.host or self.client

        def finished(results):
            def back_to_lobby():
                if self.host:
                    self.host.back_to_lobby()
                    self.app.goto(Lobby, host=self.host)
                elif self.client and self.client.status == "joined":
                    self.app.goto(Lobby, client=self.client)
                else:
                    self.app.goto(MainMenu)
            self.app.goto(ResultsScreen, results=results, setup=setup, on_continue=back_to_lobby)
        self.app.start_race(setup, finished, net=net)

    def back(self) -> None:
        if self.host:
            self.host.close()
        if self.client:
            self.client.close()
        self.app.goto(MainMenu)
