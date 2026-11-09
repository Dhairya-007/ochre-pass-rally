"""The in-race HUD."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from direct.gui.DirectGui import DirectFrame
from panda3d.core import Filename, SamplerState, TextNode, TransparencyAttrib

from ...paths import cache_dir
from ...scoring import format_time
from ...track import Track
from .kit import (ACCENT, BAD, GOLD, GOOD, GREY, GREY_DARK, INK, INK_SOFT, WHITE,
                  Fonts, panel, stripe, text)

MAP_SIZE = 0.40
REV_SEGMENTS = 16


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "TH"
    return {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")


class HUD:
    def __init__(self, app, track: Track, stage_name: str, laps: int, entrants: int) -> None:
        self.app = app
        self.track = track
        self.laps = laps
        self.entrants = entrants

        tl = app.a2dTopLeft
        tr = app.a2dTopRight
        br = app.a2dBottomRight
        self.root = DirectFrame(parent=app.aspect2d, frameColor=(0, 0, 0, 0))
        self._nodes = [self.root]

        # --- position + lap ----------------------------------------------
        block = DirectFrame(parent=tl, frameColor=(0, 0, 0, 0), pos=(0.06, 0, -0.07))
        self._nodes.append(block)
        panel(block, (0, 0.62, -0.30, 0.02), INK)
        stripe(block, (0, 0.62, 0.012, 0.02))
        text(block, "POSITION", (0.03, -0.045), 0.032, GREY, Fonts.bold)
        self.pos_value = text(block, "1", (0.03, -0.175), 0.15, WHITE, Fonts.display)
        self.pos_suffix = text(block, "ST", (0.135, -0.105), 0.04, WHITE, Fonts.bold)
        self.pos_of = text(block, f"/ {entrants}", (0.135, -0.165), 0.045, GREY, Fonts.bold)
        text(block, "LAP", (0.36, -0.045), 0.032, GREY, Fonts.bold)
        self.lap_value = text(block, f"1/{laps}", (0.36, -0.175), 0.11, WHITE, Fonts.display)
        self.time_value = text(block, "0:00.00", (0.03, -0.255), 0.05, WHITE, Fonts.bold)
        self.best_value = text(block, "BEST  —", (0.59, -0.255), 0.034, GREY, Fonts.bold,
                               TextNode.ARight)

        # --- standings tower ---------------------------------------------
        self.tower = DirectFrame(parent=tl, frameColor=(0, 0, 0, 0), pos=(0.06, 0, -0.42))
        self._nodes.append(self.tower)
        self.rows: List[Tuple] = []
        for i in range(8):
            y = -i * 0.056
            bg = panel(self.tower, (0, 0.62, y - 0.048, y), INK_SOFT)
            num = text(self.tower, str(i + 1), (0.035, y - 0.035), 0.036, GREY, Fonts.bold,
                       TextNode.ACenter)
            name = text(self.tower, "", (0.075, y - 0.035), 0.036, WHITE, Fonts.semibold)
            gap = text(self.tower, "", (0.6, y - 0.035), 0.034, GREY, Fonts.bold, TextNode.ARight)
            self.rows.append((bg, num, name, gap))

        # --- pace note ---------------------------------------------------
        top = DirectFrame(parent=app.a2dTopCenter, frameColor=(0, 0, 0, 0), pos=(0, 0, -0.07))
        self._nodes.append(top)
        self.note_panel = panel(top, (-0.30, 0.30, -0.13, 0.0), INK)
        stripe(top, (-0.30, 0.30, -0.006, 0.0))
        self.note_arrow = text(top, "", (-0.24, -0.1), 0.09, ACCENT, Fonts.display, TextNode.ACenter)
        self.note_main = text(top, "", (0.02, -0.078), 0.065, WHITE, Fonts.display, TextNode.ACenter)
        self.note_sub = text(top, "", (0.02, -0.115), 0.03, GREY, Fonts.bold, TextNode.ACenter)

        # --- minimap -----------------------------------------------------
        self.map = DirectFrame(parent=tr, frameColor=(1, 1, 1, 1),
                               frameSize=(-MAP_SIZE, 0, -MAP_SIZE, 0),
                               pos=(-0.05, 0, -0.05), image=self._map_texture(),
                               image_scale=(MAP_SIZE / 2, 1, MAP_SIZE / 2),
                               image_pos=(-MAP_SIZE / 2, 0, -MAP_SIZE / 2))
        self.map.setTransparency(TransparencyAttrib.M_alpha)
        self._nodes.append(self.map)
        self.dots: Dict[str, DirectFrame] = {}

        # --- speedo ------------------------------------------------------
        speedo = DirectFrame(parent=br, frameColor=(0, 0, 0, 0), pos=(-0.06, 0, 0.06))
        self._nodes.append(speedo)
        panel(speedo, (-0.52, 0, 0, 0.27), INK)
        stripe(speedo, (-0.52, 0, 0.262, 0.27))
        self.rev = []
        seg_w = 0.46 / REV_SEGMENTS
        for i in range(REV_SEGMENTS):
            x0 = -0.49 + i * seg_w
            self.rev.append(panel(speedo, (x0, x0 + seg_w * 0.72, 0.215, 0.24), (1, 1, 1, 0.1)))
        self.speed_value = text(speedo, "0", (-0.2, 0.05), 0.14, WHITE, Fonts.display,
                                TextNode.ARight)
        text(speedo, "KM/H", (-0.19, 0.055), 0.028, GREY, Fonts.bold)
        panel(speedo, (-0.105, -0.025, 0.035, 0.18), INK_SOFT)
        text(speedo, "GEAR", (-0.065, 0.15), 0.022, GREY, Fonts.bold, TextNode.ACenter)
        self.gear_value = text(speedo, "1", (-0.065, 0.06), 0.08, ACCENT, Fonts.display,
                               TextNode.ACenter)

        # --- centre messages -------------------------------------------
        self.big = text(app.aspect2d, "", (0, 0.12), 0.26, WHITE, Fonts.display, TextNode.ACenter,
                        shadow=True)
        self.banner = text(app.aspect2d, "", (0, -0.02), 0.07, ACCENT, Fonts.display,
                           TextNode.ACenter, shadow=True)
        self.hint = text(app.a2dBottomCenter, "", (0, 0.1), 0.04, WHITE, Fonts.bold,
                         TextNode.ACenter, shadow=True)
        self._nodes += [self.big, self.banner, self.hint]
        self._banner_for = 0.0
        self._last_note = ""

    # ------------------------------------------------------------------ #
    def _map_texture(self):
        """Draw the stage map once with PIL and cache it."""
        from PIL import Image, ImageDraw
        size = 512
        pts = self.track.points[:, [0, 2]]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = float((hi - lo).max()) * 1.12
        centre = (lo + hi) / 2
        self._map_xform = (centre, span)

        def to_px(p):
            u = (p[0] - centre[0]) / span + 0.5
            v = 0.5 - (p[1] - centre[1]) / span
            return (u * size, v * size)

        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, size - 1, size - 1), radius=26, fill=(9, 10, 13, 205))
        line = [to_px(p) for p in pts] + [to_px(pts[0])]
        d.line(line, fill=(0, 0, 0, 255), width=22, joint="curve")
        d.line(line, fill=(232, 232, 228, 255), width=12, joint="curve")
        sx, sy = to_px(pts[0])
        d.rectangle((sx - 9, sy - 9, sx + 9, sy + 9), fill=(255, 133, 20, 255))
        path = cache_dir() / f"map-{abs(hash(tuple(pts[::40].ravel().round(1))))}.png"
        img.save(path)
        tex = self.app.loader.loadTexture(Filename.fromOsSpecific(str(path)))
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        return tex

    def _map_pos(self, x: float, z: float) -> Tuple[float, float]:
        centre, span = self._map_xform
        u = (x - centre[0]) / span + 0.5
        v = (z - centre[1]) / span + 0.5
        return (-MAP_SIZE + u * MAP_SIZE, -MAP_SIZE + v * MAP_SIZE)

    # ------------------------------------------------------------------ #
    def update(self, dt: float, *, speed_kmh: float, gear: int, rpm: float, reverse: bool,
               position: int, lap: int, lap_time: float, best: Optional[float],
               standings: Sequence[Tuple[str, str, str, bool]],
               cars: Sequence[Tuple[str, float, float, bool]], pace_note: str,
               countdown: str, wrong_way: bool, hint: str = "") -> None:
        self.pos_value.setText(str(position))
        self.pos_suffix.setText(_ordinal(position))
        self.lap_value.setText(f"{min(max(lap, 1), self.laps)}/{self.laps}")
        self.time_value.setText(format_time(lap_time))
        self.best_value.setText(f"BEST  {format_time(best)}")

        for i, row in enumerate(self.rows):
            bg, num, name, gap = row
            if i < len(standings):
                entrant_id, who, gap_text, me = standings[i]
                bg.show(); num.show(); name.show(); gap.show()
                name.setText(who.upper())
                gap.setText(gap_text)
                bg["frameColor"] = (0.55, 0.28, 0.04, 0.85) if me else INK_SOFT
                name["fg"] = WHITE
                num["fg"] = WHITE if me else GREY
            else:
                bg.hide(); num.hide(); name.hide(); gap.hide()

        seen = set()
        for car_id, x, z, me in cars:
            seen.add(car_id)
            dot = self.dots.get(car_id)
            if dot is None:
                r = 0.011 if me else 0.008
                dot = DirectFrame(parent=self.map, frameColor=ACCENT if me else (0.9, 0.9, 0.9, 1),
                                  frameSize=(-r, r, -r, r))
                self.dots[car_id] = dot
            mx, my = self._map_pos(x, z)
            dot.setPos(mx, 0, my)
            if me:
                dot.reparentTo(self.map)        # draw the player on top
        for car_id in list(self.dots):
            if car_id not in seen:
                self.dots.pop(car_id).destroy()

        self.speed_value.setText(str(int(speed_kmh)))
        self.gear_value.setText("R" if reverse else str(gear))
        lit = int(min(1.0, rpm) * REV_SEGMENTS + 0.5)
        for i, seg in enumerate(self.rev):
            if i < lit:
                seg["frameColor"] = BAD if i >= REV_SEGMENTS - 3 else (GOLD if i >= REV_SEGMENTS - 6 else WHITE)
            else:
                seg["frameColor"] = (1, 1, 1, 0.1)

        if pace_note != self._last_note:
            self._last_note = pace_note
            if pace_note == "Flat out":
                self.note_arrow.setText("")
                self.note_main.setText("FLAT OUT")
                self.note_sub.setText("")
            else:
                parts = pace_note.split(" ", 2)
                self.note_arrow.setText("<" if len(parts) > 1 and parts[1] == "left" else ">")
                self.note_main.setText(" ".join(parts[:2]).upper())
                self.note_sub.setText(parts[2].upper() if len(parts) > 2 else "")

        if countdown:
            self.big.setText(countdown)
            self.big["fg"] = GOOD if countdown == "GO" else WHITE
        elif wrong_way:
            self.big.setText("")
            self.banner.setText("WRONG WAY")
            self.banner["fg"] = BAD
            self._banner_for = 0.2
        else:
            self.big.setText("")

        if self._banner_for > 0:
            self._banner_for -= dt
            if self._banner_for <= 0:
                self.banner.setText("")
        self.hint.setText(hint)

    def flash(self, message: str, seconds: float = 2.2, colour=ACCENT) -> None:
        self.banner.setText(message)
        self.banner["fg"] = colour
        self._banner_for = seconds

    def destroy(self) -> None:
        for dot in self.dots.values():
            dot.destroy()
        for node in self._nodes:
            node.destroy()
