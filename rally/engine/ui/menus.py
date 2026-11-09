"""Main menu, quick race, settings, loading, pause and results."""

from __future__ import annotations

import random
from typing import List, Optional

from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode

from ...ai import RIVAL_NAMES
from ...cars import CARS, CARS_BY_ID, LIVERY_COLOURS
from ...race import Result
from ...scoring import format_time
from ...settings import DIFFICULTY, DIFFICULTY_SKILL, QUALITY
from ...stages import STAGES
from .kit import (ACCENT, BAD, GOLD, GOOD, GREY, GREY_DARK, INK, INK_SOFT, WHITE, Button,
                  Field, Fonts, Screen, Selector, money, panel, stripe, text, title_block)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "TH"
    return {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")


def left_shade(screen: Screen, width: float = 1.05) -> None:
    """A dark band down the left so menu text reads over the 3D scene."""
    shade = DirectFrame(parent=screen.app.a2dLeftCenter, frameColor=(0.02, 0.025, 0.03, 0.72),
                        frameSize=(0, width, -1.2, 1.2))
    shade.reparentTo(screen.root)
    shade.setPos(screen.app.a2dLeftCenter, 0, 0, 0)
    return shade


def anchor(screen: Screen, where: str) -> DirectFrame:
    parent = {"left": screen.app.a2dLeftCenter, "right": screen.app.a2dRightCenter,
              "top": screen.app.a2dTopCenter, "bottom": screen.app.a2dBottomCenter,
              "topleft": screen.app.a2dTopLeft, "topright": screen.app.a2dTopRight,
              "bottomleft": screen.app.a2dBottomLeft, "bottomright": screen.app.a2dBottomRight,
              "centre": screen.app.aspect2d}[where]
    frame = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0))
    frame.wrtReparentTo(screen.root)
    return frame


# --------------------------------------------------------------------------- #
class MainMenu(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self)
        col = anchor(self, "left")
        x = 0.12
        text(col, "FREE · OPEN SOURCE · PYTHON", (x, 0.66), 0.032, ACCENT, Fonts.bold)
        text(col, "OCHRE PASS", (x, 0.5), 0.15, WHITE, Fonts.display, shadow=True)
        text(col, "RALLY", (x, 0.37), 0.15, ACCENT, Fonts.display, shadow=True)

        career = app.career
        career_sub = (f"{career.championship.name} · event {career.event_index + 1}"
                      if career else "Start a new career")
        items = [
            ("Career", app.open_career, career_sub),
            ("Quick race", lambda: app.goto(QuickRace), "Any stage, any car"),
            ("Multiplayer", app.open_multiplayer, "Host or join by IP"),
            ("Settings", lambda: app.goto(SettingsScreen), ""),
            ("Quit", app.quit, ""),
        ]
        for i, (label, command, sub) in enumerate(items):
            self.add(Button(col, label, (x, 0.18 - i * 0.105), command, width=0.82,
                            height=0.09, sub=sub, scale=0.05, sound=app.audio.ui))

        foot = anchor(self, "bottomleft")
        text(foot, "ARROWS / MOUSE TO NAVIGATE   ·   ENTER TO SELECT   ·   ESC TO GO BACK",
             (0.12, 0.06), 0.026, GREY_DARK, Fonts.bold)
        info = anchor(self, "bottomright")
        if career:
            text(info, f"{career.player_name.upper()}   ·   {money(career.money)}",
                 (-0.08, 0.06), 0.036, WHITE, Fonts.bold, TextNode.ARight)

    def back(self) -> None:
        pass


# --------------------------------------------------------------------------- #
class QuickRace(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Single player", "Quick race", 0.12, 0.72)
        s = app.settings
        self.stage = 0
        self.car = 0
        self.livery = 0
        self.laps = 2
        self.opponents = 5
        self.difficulty = DIFFICULTY.index(s.difficulty)
        y = 0.42
        rows = [
            ("Stage", [f"{st.name} · {st.country}" for st in STAGES], self.stage, "stage"),
            ("Car", [c.name for c in CARS], self.car, "car"),
            ("Livery", [name for name, _ in LIVERY_COLOURS], self.livery, "livery"),
            ("Laps", [str(n) for n in range(1, 6)], self.laps - 1, "laps"),
            ("Opponents", [str(n) for n in range(0, 8)], self.opponents, "opponents"),
            ("Difficulty", list(DIFFICULTY), self.difficulty, "difficulty"),
        ]
        for i, (label, options, index, attr) in enumerate(rows):
            self.add(Selector(col, label, options, index, (0.12, y - i * 0.095),
                              self._setter(attr), width=1.05, sound=app.audio.ui))
        self.blurb = text(col, "", (0.12, y - len(rows) * 0.095 - 0.03), 0.034, GREY,
                          Fonts.regular, wrap=30)
        self.add(Button(col, "Start race", (0.12, -0.42), self.start, width=1.05,
                        sound=app.audio.ui))
        self.add(Button(col, "Back", (0.12, -0.53), self.back, width=1.05, sound=app.audio.ui))
        self._refresh()

    def _setter(self, attr):
        def set_value(index: int) -> None:
            value = index + 1 if attr == "laps" else index
            setattr(self, attr, value)
            self._refresh()
        return set_value

    def _refresh(self) -> None:
        self.blurb.setText(STAGES[self.stage].blurb + "\n\n" + CARS[self.car].blurb)
        self.app.menu_scene.show_car(CARS[self.car].id, self.livery, STAGES[self.stage].id)

    def start(self) -> None:
        from ..race_controller import EntrantSpec, RaceSetup
        stage = STAGES[self.stage]
        car = CARS[self.car]
        offset = DIFFICULTY_SKILL[DIFFICULTY[self.difficulty]]
        entrants = [EntrantSpec("player", self.app.settings.name, "player", car.id, self.livery)]
        rng = random.Random()
        names = rng.sample(RIVAL_NAMES, self.opponents)
        for i in range(self.opponents):
            rival_car = rng.choice([c.id for c in CARS])
            entrants.append(EntrantSpec(f"ai{i}", names[i], "ai", rival_car,
                                        livery=rng.randrange(len(LIVERY_COLOURS)),
                                        skill=max(0.2, min(1.0, 0.62 + 0.05 * i + offset)),
                                        consistency=0.7, aggression=0.4 + 0.05 * i, seed=i))
        rng.shuffle(entrants)
        setup = RaceSetup(stage.id, self.laps, entrants, "player", "quick")
        self.app.start_race(setup, lambda results: self.app.goto(ResultsScreen, results=results,
                                                                 setup=setup))

    def back(self) -> None:
        self.app.audio.ui("back")
        self.app.goto(MainMenu)


# --------------------------------------------------------------------------- #
class SettingsScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Options", "Settings", 0.12, 0.72)
        s = app.settings
        self.name = self.add(Field(col, "Driver name", s.name, (0.12, 0.42), width=1.05,
                                   max_length=20))
        self.add(Selector(col, "Graphics", [q.title() for q in QUALITY], QUALITY.index(s.quality),
                          (0.12, 0.325), lambda i: setattr(s, "quality", QUALITY[i]), 1.05,
                          app.audio.ui))
        self.add(Selector(col, "Display", ["Window", "Fullscreen"], int(s.fullscreen),
                          (0.12, 0.23), lambda i: setattr(s, "fullscreen", bool(i)), 1.05,
                          app.audio.ui))
        vols = [f"{v}%" for v in range(0, 101, 10)]
        self.add(Selector(col, "Volume", vols, int(round(s.volume * 10)), (0.12, 0.135),
                          self._volume, 1.05, app.audio.ui))
        self.add(Selector(col, "AI difficulty", list(DIFFICULTY), DIFFICULTY.index(s.difficulty),
                          (0.12, 0.04), lambda i: setattr(s, "difficulty", DIFFICULTY[i]), 1.05,
                          app.audio.ui))
        cams = ["chase", "chase_far", "bonnet"]
        self.add(Selector(col, "Camera", ["Chase", "Far chase", "Bonnet"],
                          cams.index(s.camera) if s.camera in cams else 0, (0.12, -0.055),
                          lambda i: setattr(s, "camera", cams[i]), 1.05, app.audio.ui))
        self.add(Selector(col, "FPS counter", ["Off", "On"], int(s.show_fps), (0.12, -0.15),
                          self._fps, 1.05, app.audio.ui))
        text(col, "Graphics and display changes apply the next time the game starts.",
             (0.12, -0.24), 0.03, GREY, Fonts.regular)
        self.add(Button(col, "Save", (0.12, -0.36), self.save, width=1.05, sound=app.audio.ui))
        self.add(Button(col, "Back", (0.12, -0.47), self.back, width=1.05, sound=app.audio.ui))

    def _volume(self, i: int) -> None:
        self.app.settings.volume = i / 10.0
        self.app.audio.set_volume(i / 10.0)

    def _fps(self, i: int) -> None:
        self.app.settings.show_fps = bool(i)
        self.app.setFrameRateMeter(bool(i))

    def save(self) -> None:
        name = self.name.value or "Driver"
        self.app.settings.name = name
        if self.app.career:
            self.app.career.player_name = name
            self.app.save_career()
        self.app.settings.save()
        self.app.toast("Settings saved")
        self.app.goto(MainMenu)

    def back(self) -> None:
        self.app.audio.ui("back")
        self.app.goto(MainMenu)


# --------------------------------------------------------------------------- #
class LoadingScreen(Screen):
    def __init__(self, app, title: str, subtitle: str = "") -> None:
        super().__init__(app)
        DirectFrame(parent=self.root, frameColor=(0.02, 0.025, 0.03, 1),
                    frameSize=(-3, 3, -2, 2))
        col = anchor(self, "bottomleft")
        text(col, subtitle.upper(), (0.12, 0.42), 0.036, ACCENT, Fonts.bold)
        text(col, title.upper(), (0.12, 0.27), 0.14, WHITE, Fonts.display)
        self.message = text(col, "", (0.12, 0.17), 0.034, GREY, Fonts.bold)
        panel(col, (0.12, 1.42, 0.115, 0.125), (1, 1, 1, 0.12))
        self.bar = panel(col, (0.12, 0.121, 0.115, 0.125), ACCENT)

    def set_progress(self, message: str, fraction: float) -> None:
        if message:
            self.message.setText(message.upper())
        self.bar["frameSize"] = (0.12, 0.12 + 1.30 * max(0.0, min(1.0, fraction)), 0.115, 0.125)


# --------------------------------------------------------------------------- #
class PauseScreen(Screen):
    def __init__(self, app, allow_restart: bool) -> None:
        super().__init__(app)
        DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0.55), frameSize=(-3, 3, -2, 2))
        col = anchor(self, "left")
        title_block(col, "Paused", "Pause", 0.12, 0.45)
        self.add(Button(col, "Resume", (0.12, 0.18), app.resume, width=0.8, sound=app.audio.ui))
        if allow_restart:
            self.add(Button(col, "Restart", (0.12, 0.075), app.restart_race, width=0.8,
                            sound=app.audio.ui))
        self.add(Button(col, "Change camera", (0.12, -0.03), app.cycle_camera, width=0.8,
                        sound=app.audio.ui))
        self.add(Button(col, "Quit to menu", (0.12, -0.135), app.abandon_race, width=0.8,
                        sound=app.audio.ui))
        text(col, "R RECOVER   ·   C CAMERA   ·   SPACE HANDBRAKE", (0.12, -0.25), 0.03,
             GREY, Fonts.bold)

    def back(self) -> None:
        self.app.resume()


# --------------------------------------------------------------------------- #
class ResultsScreen(Screen):
    def __init__(self, app, results: List[Result], setup, outcome=None,
                 on_continue=None) -> None:
        super().__init__(app)
        self.on_continue = on_continue
        DirectFrame(parent=self.root, frameColor=(0.02, 0.025, 0.03, 0.9), frameSize=(-3, 3, -2, 2))
        col = anchor(self, "left")
        from ...stages import get_stage
        stage = get_stage(setup.stage_id)
        me = next((r for r in results if r.id == setup.player_id), None)
        headline = "DNF" if (me and me.dnf) else (f"P{me.position}" if me else "")
        title_block(col, f"{stage.name} · {setup.laps} laps · results", headline, 0.12, 0.8)

        # Table
        y0 = 0.5
        heads = (("POS", 0.12, TextNode.ALeft), ("DRIVER", 0.25, TextNode.ALeft),
                 ("CAR", 0.85, TextNode.ALeft), ("TIME", 1.45, TextNode.ARight),
                 ("GAP", 1.7, TextNode.ARight), ("BEST LAP", 1.98, TextNode.ARight))
        for label, x, align in heads:
            text(col, label, (x, y0), 0.03, GREY, Fonts.bold, align)
        for i, r in enumerate(results[:10]):
            y = y0 - 0.075 - i * 0.062
            mine = r.id == setup.player_id
            panel(col, (0.1, 2.0, y - 0.022, y + 0.04), (0.55, 0.28, 0.04, 0.8) if mine else INK_SOFT)
            colour = GOLD if r.position == 1 and not r.dnf else WHITE
            text(col, str(r.position), (0.13, y), 0.04, colour, Fonts.display)
            text(col, r.name.upper(), (0.25, y), 0.038, WHITE, Fonts.bold)
            text(col, CARS_BY_ID[r.car_id].name.upper(), (0.85, y), 0.034, GREY, Fonts.semibold)
            text(col, "DNF" if r.dnf else format_time(r.time), (1.45, y), 0.038,
                 BAD if r.dnf else WHITE, Fonts.bold, TextNode.ARight)
            gap = "" if r.dnf or r.gap is None else ("—" if r.gap == 0 else f"+{r.gap:.2f}")
            text(col, gap, (1.7, y), 0.034, GREY, Fonts.bold, TextNode.ARight)
            text(col, format_time(r.best_lap), (1.98, y), 0.034, GREY, Fonts.bold, TextNode.ARight)

        if outcome is not None:
            side = anchor(self, "right")
            panel(side, (-0.95, -0.1, -0.45, 0.62), INK)
            stripe(side, (-0.95, -0.1, 0.612, 0.62))
            text(side, "CAREER", (-0.9, 0.53), 0.034, ACCENT, Fonts.bold)
            text(side, f"+{outcome.points} PTS", (-0.9, 0.42), 0.075, WHITE, Fonts.display)
            text(side, f"+{money(outcome.prize)}", (-0.9, 0.33), 0.06, GOOD, Fonts.display)
            y = 0.22
            text(side, "CHAMPIONSHIP", (-0.9, y), 0.03, GREY, Fonts.bold)
            for i, (rid, name, pts) in enumerate(outcome.standings[:6]):
                yy = y - 0.06 - i * 0.05
                mine = rid == "player"
                text(side, f"{i + 1}. {name.upper()}", (-0.9, yy), 0.034,
                     ACCENT if mine else WHITE, Fonts.bold)
                text(side, str(pts), (-0.15, yy), 0.034, WHITE, Fonts.bold, TextNode.ARight)
            if outcome.season_complete:
                place = outcome.championship_position
                msg = f"SEASON OVER · {place}{_ordinal(place)} IN THE CHAMPIONSHIP"
                text(side, msg, (-0.9, -0.2), 0.04, GOLD, Fonts.display)
                if outcome.title_bonus:
                    text(side, f"CHAMPION! BONUS {money(outcome.title_bonus)}", (-0.9, -0.27),
                         0.034, GOOD, Fonts.bold)
                if outcome.unlocked_tier is not None:
                    from ...career import CHAMPIONSHIPS
                    text(side, f"UNLOCKED: {CHAMPIONSHIPS[outcome.unlocked_tier].name.upper()}",
                         (-0.9, -0.34), 0.034, ACCENT, Fonts.bold)

        self.add(Button(col, "Continue", (0.12, -0.78), self.back, width=0.7, sound=app.audio.ui))

    def back(self) -> None:
        if self.on_continue:
            self.on_continue()
        else:
            self.app.goto(MainMenu)
