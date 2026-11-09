"""Career hub, garage and championship select."""

from __future__ import annotations

from typing import Optional

from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode

from ...cars import CARS, CARS_BY_ID, LIVERY_COLOURS, UPGRADES, build_config, rate
from ...career import CHAMPIONSHIPS, Career, CareerError
from ...settings import DIFFICULTY_SKILL
from ...stages import get_stage
from .kit import (ACCENT, BAD, GOLD, GOOD, GREY, GREY_DARK, INK, INK_SOFT, WHITE, Button,
                  Field, Fonts, Screen, Selector, StatBar, money, panel, stripe, text,
                  title_block)
from .menus import MainMenu, ResultsScreen, anchor, left_shade


class NewCareer(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Career", "New career", 0.12, 0.6)
        text(col, "Three championships stand between you and the world title. You start\n"
                  "in the Rookie Cup with a Kestrel R2 and $5,000. Finish a season in\n"
                  "the top three to move up; prize money buys cars and upgrades.",
             (0.12, 0.3), 0.036, GREY, Fonts.regular)
        self.name = self.add(Field(col, "Driver name", app.settings.name, (0.12, 0.05),
                                   width=1.05, max_length=20, on_submit=self.start))
        self.add(Button(col, "Start career", (0.12, -0.06), self.start, width=1.05,
                        sound=app.audio.ui))
        self.add(Button(col, "Back", (0.12, -0.17), self.back, width=1.05, sound=app.audio.ui))

    def start(self) -> None:
        name = self.name.value or "Driver"
        self.app.settings.name = name
        self.app.settings.save()
        self.app.career = Career(player_name=name)
        self.app.save_career()
        self.app.goto(CareerHub)

    def back(self) -> None:
        self.app.goto(MainMenu)


class CareerHub(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        c: Career = app.career
        champ = c.championship
        event = c.next_event
        stage = get_stage(event.stage_id)
        app.menu_scene.show_car(c.current_car, c.owned[c.current_car].livery, stage.id)

        left_shade(self, 1.2)
        col = anchor(self, "left")
        title_block(col, f"Season {c.season} · tier {champ.tier + 1} of {len(CHAMPIONSHIPS)}",
                    champ.name, 0.12, 0.8)
        text(col, champ.blurb, (0.12, 0.58), 0.032, GREY, Fonts.regular, wrap=34)

        # Next event card
        panel(col, (0.12, 1.12, 0.06, 0.44), INK)
        stripe(col, (0.12, 0.128, 0.06, 0.44))
        text(col, f"EVENT {c.event_index + 1} OF {len(champ.events)}", (0.17, 0.39), 0.03,
             ACCENT, Fonts.bold)
        text(col, stage.name.upper(), (0.17, 0.29), 0.085, WHITE, Fonts.display)
        text(col, f"{stage.country.upper()}  ·  {event.laps} LAPS  ·  {len(champ.rivals) + 1} STARTERS",
             (0.17, 0.225), 0.032, GREY, Fonts.bold)
        best = c.best_laps.get(stage.id)
        from ...scoring import format_time
        text(col, f"WIN {money(champ.prizes[0])}   ·   YOUR BEST LAP {format_time(best)}",
             (0.17, 0.1), 0.032, WHITE, Fonts.bold)

        car = CARS_BY_ID[c.current_car]
        self.add(Button(col, "Race", (0.12, -0.03), self.race, width=1.0, height=0.1,
                        sub=f"in your {car.name}", scale=0.058, sound=app.audio.ui))
        self.add(Button(col, "Garage", (0.12, -0.15), lambda: app.goto(Garage), width=1.0,
                        sub=money(c.money), sound=app.audio.ui))
        if c.unlocked_tier > 0 and not c.season_in_progress:
            self.add(Button(col, "Change championship", (0.12, -0.255),
                            lambda: app.goto(ChampionshipSelect), width=1.0, sound=app.audio.ui))
        self.add(Button(col, "Main menu", (0.12, -0.36 if c.unlocked_tier > 0 and not c.season_in_progress else -0.255),
                        self.back, width=1.0, sound=app.audio.ui))
        text(col, f"{c.races} RACES · {c.wins} WINS · {c.podiums} PODIUMS · {len(c.titles)} TITLES",
             (0.12, -0.55), 0.03, GREY_DARK, Fonts.bold)

        # Standings
        side = anchor(self, "right")
        panel(side, (-0.9, -0.08, -0.2 - 0.055 * len(c.table()), 0.42), INK)
        stripe(side, (-0.9, -0.08, 0.412, 0.42))
        text(side, "CHAMPIONSHIP STANDINGS", (-0.86, 0.34), 0.032, ACCENT, Fonts.bold)
        for i, (rid, name, pts) in enumerate(c.table()):
            y = 0.26 - i * 0.055
            mine = rid == Career.PLAYER_ID
            if mine:
                panel(side, (-0.88, -0.1, y - 0.02, y + 0.038), (0.55, 0.28, 0.04, 0.8))
            text(side, f"{i + 1}", (-0.85, y), 0.036, WHITE if mine else GREY, Fonts.display)
            text(side, name.upper(), (-0.78, y), 0.036, WHITE, Fonts.bold)
            rival = next((r for r in champ.rivals if r.id == rid), None)
            if rival:
                text(side, CARS_BY_ID[rival.car_id].name.upper(), (-0.45, y), 0.028, GREY_DARK,
                     Fonts.semibold)
            text(side, str(pts), (-0.12, y), 0.036, WHITE, Fonts.bold, TextNode.ARight)

    def race(self) -> None:
        from ..race_controller import EntrantSpec, RaceSetup
        c: Career = self.app.career
        champ = c.championship
        event = c.next_event
        owned = c.owned[c.current_car]
        offset = DIFFICULTY_SKILL[self.app.settings.difficulty]
        entrants = []
        for i, r in enumerate(champ.rivals):
            entrants.append(EntrantSpec(r.id, r.name, "ai", r.car_id, livery=(i + 1) % len(LIVERY_COLOURS),
                                        skill=max(0.15, min(1.0, r.skill + offset)),
                                        consistency=r.consistency, aggression=r.aggression,
                                        seed=c.season * 100 + c.event_index * 10 + i))
        # Start position follows the championship: leader starts at the back.
        table = [row[0] for row in c.table()]
        player = EntrantSpec(Career.PLAYER_ID, c.player_name, "player", c.current_car,
                             owned.livery, dict(owned.upgrades))
        everyone = entrants + [player]
        if c.event_index > 0:
            everyone.sort(key=lambda e: -table.index(e.id) if e.id in table else 0)
        else:
            everyone = entrants[:len(entrants) // 2] + [player] + entrants[len(entrants) // 2:]
        setup = RaceSetup(event.stage_id, event.laps, everyone, Career.PLAYER_ID, "career")

        def finished(results):
            outcome = c.record(results)
            self.app.save_career()
            self.app.goto(ResultsScreen, results=results, setup=setup, outcome=outcome,
                          on_continue=lambda: self.app.goto(CareerHub))
        self.app.start_race(setup, finished)

    def back(self) -> None:
        self.app.goto(MainMenu)


class ChampionshipSelect(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left_shade(self, 1.3)
        col = anchor(self, "left")
        title_block(col, "Career", "Championships", 0.12, 0.8)
        c = app.career
        for i, champ in enumerate(CHAMPIONSHIPS):
            locked = i > c.unlocked_tier
            sub = "LOCKED" if locked else ("CURRENT" if i == c.tier else f"WIN {money(champ.prizes[0])}")
            b = self.add(Button(col, champ.name, (0.12, 0.45 - i * 0.12),
                                (lambda t=i: self.pick(t)), width=1.1, height=0.1, sub=sub,
                                scale=0.055, sound=app.audio.ui))
            b.set_enabled(not locked)
        self.add(Button(col, "Back", (0.12, 0.45 - len(CHAMPIONSHIPS) * 0.12), self.back,
                        width=1.1, sound=app.audio.ui))

    def pick(self, tier: int) -> None:
        try:
            self.app.career.choose_tier(tier)
            self.app.save_career()
        except CareerError as error:
            self.app.toast(str(error), BAD)
            return
        self.app.goto(CareerHub)

    def back(self) -> None:
        self.app.goto(CareerHub)


class Garage(Screen):
    def __init__(self, app, index: Optional[int] = None) -> None:
        super().__init__(app)
        c: Career = app.career
        self.index = index if index is not None else [x.id for x in CARS].index(c.current_car)
        left_shade(self, 0.95)
        col = anchor(self, "left")
        title_block(col, "Career", "Garage", 0.12, 0.8)
        self.money = text(col, money(c.money), (0.12, 0.6), 0.06, GOOD, Fonts.display)

        self.car_selector = self.add(Selector(col, "Car", [x.name for x in CARS], self.index,
                                              (0.12, 0.47), self._pick, width=0.78,
                                              sound=app.audio.ui))
        self.status = text(col, "", (0.12, 0.38), 0.034, GREY, Fonts.bold)
        self.blurb = text(col, "", (0.12, 0.31), 0.03, GREY, Fonts.regular, wrap=24)
        self.action = self.add(Button(col, "Buy", (0.12, 0.05), self._buy_or_select, width=0.78,
                                      sound=app.audio.ui))
        self.livery = self.add(Selector(col, "Livery", [n for n, _ in LIVERY_COLOURS], 0,
                                        (0.12, -0.05), self._livery, width=0.78, sound=app.audio.ui))
        self.upgrade_buttons = []
        for i, kind in enumerate(UPGRADES):
            b = self.add(Button(col, kind.name, (0.12, -0.17 - i * 0.092),
                                (lambda k=kind.id: self._upgrade(k)), width=0.78,
                                sound=app.audio.ui))
            self.upgrade_buttons.append((kind, b))
        self.add(Button(col, "Back", (0.12, -0.17 - len(UPGRADES) * 0.092 - 0.02), self.back,
                        width=0.78, sound=app.audio.ui))

        # Bottom-right, clear of the car in the showroom shot.
        side = anchor(self, "bottomright")
        panel(side, (-0.95, -0.08, 0.08, 0.66), INK)
        stripe(side, (-0.95, -0.08, 0.652, 0.66))
        text(side, "PERFORMANCE", (-0.9, 0.58), 0.03, ACCENT, Fonts.bold)
        self.bars = {}
        for i, (key, label) in enumerate((("power", "Power"), ("top_speed", "Top speed"),
                                          ("grip", "Grip"), ("handling", "Handling"),
                                          ("braking", "Braking"), ("rough", "Rough roads"))):
            self.bars[key] = StatBar(side, label, 0, (-0.9, 0.5 - i * 0.07), width=0.55)
        self._refresh()

    # ------------------------------------------------------------------ #
    @property
    def car(self):
        return CARS[self.index]

    def _pick(self, index: int) -> None:
        self.index = index
        self._refresh()

    def _refresh(self) -> None:
        c: Career = self.app.career
        car = self.car
        owned = c.owned.get(car.id)
        self.money.setText(money(c.money))
        self.blurb.setText(car.blurb)
        if owned:
            current = c.current_car == car.id
            self.status.setText(("IN USE" if current else "OWNED") + f"  ·  {car.category.upper()}")
            self.action.set_label("In use" if current else "Drive this car")
            self.action.set_enabled(not current)
            self.livery.set_options([n for n, _ in LIVERY_COLOURS], owned.livery)
            self.livery.enabled = True
        else:
            self.status.setText(f"{money(car.price)}  ·  {car.category.upper()}")
            self.action.set_label(f"Buy for {money(car.price)}")
            self.action.set_enabled(c.money >= car.price)
            self.livery.enabled = False
        for kind, button in self.upgrade_buttons:
            if owned:
                level = owned.upgrades.get(kind.id, 0)
                price = c.next_upgrade_price(car.id, kind.id)
                pips = "■" * level + "□" * (kind.max_level - level)
                button.set_label(f"{kind.name}  {level}/{kind.max_level}")
                button.set_sub("MAXED" if price is None else money(price))
                button.set_enabled(price is not None and c.money >= price)
            else:
                button.set_label(kind.name)
                button.set_sub("BUY THE CAR FIRST")
                button.set_enabled(False)
        levels = owned.upgrades if owned else {}
        ratings = rate(build_config(car, levels))
        for key, bar in self.bars.items():
            bar.set(getattr(ratings, key))
        self.app.menu_scene.show_car(car.id, owned.livery if owned else 0, None)

    def _buy_or_select(self) -> None:
        c: Career = self.app.career
        try:
            if self.car.id in c.owned:
                c.select_car(self.car.id)
            else:
                c.buy_car(self.car.id)
                self.app.audio.ui("buy")
                self.app.toast(f"Bought the {self.car.name}", GOOD)
        except CareerError as error:
            self.app.toast(str(error), BAD)
        self.app.save_career()
        self._refresh()

    def _livery(self, index: int) -> None:
        c: Career = self.app.career
        if self.car.id in c.owned:
            c.set_livery(self.car.id, index)
            self.app.save_career()
            self._refresh()

    def _upgrade(self, kind_id: str) -> None:
        c: Career = self.app.career
        try:
            level = c.buy_upgrade(self.car.id, kind_id)
            self.app.audio.ui("buy")
            self.app.toast(f"{kind_id.title()} upgraded to level {level}", GOOD)
        except CareerError as error:
            self.app.toast(str(error), BAD)
        self.app.save_career()
        self._refresh()

    def back(self) -> None:
        self.app.goto(CareerHub)
