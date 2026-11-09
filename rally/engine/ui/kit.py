"""The UI kit: theme, widgets, and keyboard focus.

Look: motorsport broadcast graphics. Near-black translucent panels, a single
orange accent used for focus and emphasis, condensed type in capitals.

Every screen can be driven entirely from the keyboard (or a gamepad's d-pad):
up/down/left/right move focus or change a selector, Enter activates, Escape
goes back. The mouse works too - hovering a widget focuses it.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from direct.gui.DirectGui import (DGG, DirectButton, DirectEntry, DirectFrame,
                                  OnscreenText)
from panda3d.core import NodePath, TextNode, TransparencyAttrib

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
INK = (0.035, 0.040, 0.050, 0.86)
INK_SOFT = (0.06, 0.07, 0.085, 0.78)
INK_RAISED = (0.12, 0.13, 0.15, 0.92)
ACCENT = (1.00, 0.52, 0.08, 1.0)
ACCENT_DIM = (1.00, 0.52, 0.08, 0.35)
WHITE = (0.97, 0.97, 0.95, 1.0)
GREY = (0.64, 0.67, 0.71, 1.0)
GREY_DARK = (0.38, 0.40, 0.44, 1.0)
GOOD = (0.36, 0.86, 0.46, 1.0)
BAD = (0.98, 0.33, 0.28, 1.0)
GOLD = (1.0, 0.80, 0.26, 1.0)


class Fonts:
    regular = None
    bold = None
    display = None

    @classmethod
    def init(cls, assets) -> None:
        cls.regular = assets.font("BarlowCondensed-Medium")
        cls.semibold = assets.font("BarlowCondensed-SemiBold")
        cls.bold = assets.font("BarlowCondensed-Bold")
        cls.display = assets.font("BarlowCondensed-ExtraBoldItalic")


def text(parent, value: str, pos: Tuple[float, float], scale: float = 0.05,
         colour=WHITE, font=None, align=TextNode.ALeft, wrap: Optional[float] = None,
         shadow: bool = False) -> OnscreenText:
    extra = {"shadow": (0, 0, 0, 0.55), "shadowOffset": (0.05, 0.05)} if shadow else {}
    return OnscreenText(text=value, parent=parent, pos=pos, scale=scale, fg=colour,
                        font=font or Fonts.semibold, align=align, mayChange=True,
                        wordwrap=wrap, **extra)


def panel(parent, frame: Tuple[float, float, float, float], colour=INK,
          pos=(0, 0)) -> DirectFrame:
    f = DirectFrame(parent=parent, frameColor=colour, frameSize=frame, pos=(pos[0], 0, pos[1]))
    f.setTransparency(TransparencyAttrib.M_alpha)
    return f


def stripe(parent, frame: Tuple[float, float, float, float], colour=ACCENT) -> DirectFrame:
    return panel(parent, frame, colour)


# --------------------------------------------------------------------------- #
# Focusable widgets
# --------------------------------------------------------------------------- #
class Focusable:
    enabled = True
    def focus(self, on: bool) -> None: ...
    def activate(self) -> None: ...
    def nudge(self, step: int) -> bool:
        """Left/right. Return True if handled."""
        return False


class Button(Focusable):
    def __init__(self, parent, label: str, pos: Tuple[float, float], command: Callable[[], None],
                 width: float = 0.62, height: float = 0.085, sub: str = "",
                 scale: float = 0.048, align: str = "left", sound=None) -> None:
        self.command = command
        self.sound = sound
        self.width, self.height = width, height
        self.root = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0), pos=(pos[0], 0, pos[1]))
        self.bg = panel(self.root, (0, width, -height / 2, height / 2), INK_SOFT)
        self.bar = stripe(self.root, (0, 0.008, -height / 2, height / 2), GREY_DARK)
        x = 0.035 if align == "left" else width / 2
        a = TextNode.ALeft if align == "left" else TextNode.ACenter
        self.label = text(self.root, label.upper(), (x, -scale * 0.36), scale, WHITE, Fonts.bold, a)
        self.sub = text(self.root, sub, (width - 0.03, -scale * 0.33), scale * 0.72, GREY,
                        Fonts.semibold, TextNode.ARight) if sub else None
        self.hit = DirectButton(parent=self.root, frameColor=(0, 0, 0, 0),
                                frameSize=(0, width, -height / 2, height / 2),
                                relief=DGG.FLAT, command=self._click, rolloverSound=None,
                                clickSound=None, pressEffect=False)
        self.hit.bind(DGG.ENTER, lambda _e: self.on_hover and self.on_hover(self))
        self.on_hover: Optional[Callable[["Button"], None]] = None
        self._focused = False

    def _click(self) -> None:
        if self.enabled:
            if self.sound:
                self.sound("click")
            self.command()

    def focus(self, on: bool) -> None:
        self._focused = on
        self.bg["frameColor"] = INK_RAISED if on else INK_SOFT
        self.bar["frameColor"] = ACCENT if on else (GREY_DARK if self.enabled else (0.2, 0.2, 0.2, 1))
        self.label["fg"] = (ACCENT if on else WHITE) if self.enabled else GREY_DARK

    def activate(self) -> None:
        self._click()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.focus(self._focused)

    def set_label(self, label: str) -> None:
        self.label.setText(label.upper())

    def set_sub(self, value: str) -> None:
        if self.sub:
            self.sub.setText(value)

    def destroy(self) -> None:
        self.root.destroy()


class Selector(Focusable):
    """LABEL            < value >   - left/right to change."""

    def __init__(self, parent, label: str, options: Sequence[str], index: int,
                 pos: Tuple[float, float], on_change: Callable[[int], None],
                 width: float = 0.9, sound=None) -> None:
        self.options = list(options)
        self.index = index
        self.on_change = on_change
        self.sound = sound
        h = 0.08
        self.root = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0), pos=(pos[0], 0, pos[1]))
        self.bg = panel(self.root, (0, width, -h / 2, h / 2), INK_SOFT)
        self.bar = stripe(self.root, (0, 0.008, -h / 2, h / 2), GREY_DARK)
        self.label = text(self.root, label.upper(), (0.035, -0.016), 0.042, GREY, Fonts.bold)
        self.value = text(self.root, "", (width - 0.2, -0.017), 0.046, WHITE, Fonts.bold,
                          TextNode.ACenter)
        self.left = DirectButton(parent=self.root, text="<", text_font=Fonts.bold,
                                 text_scale=0.05, text_fg=GREY, frameColor=(0, 0, 0, 0),
                                 pos=(width - 0.36, 0, -0.017), command=lambda: self.nudge(-1),
                                 relief=DGG.FLAT, pressEffect=False)
        self.right = DirectButton(parent=self.root, text=">", text_font=Fonts.bold,
                                  text_scale=0.05, text_fg=GREY, frameColor=(0, 0, 0, 0),
                                  pos=(width - 0.04, 0, -0.017), command=lambda: self.nudge(1),
                                  relief=DGG.FLAT, pressEffect=False)
        self.hit = DirectButton(parent=self.root, frameColor=(0, 0, 0, 0),
                                frameSize=(0, width - 0.4, -h / 2, h / 2), relief=DGG.FLAT,
                                pressEffect=False, command=lambda: self.nudge(1))
        self.hit.bind(DGG.ENTER, lambda _e: self.on_hover and self.on_hover(self))
        self.on_hover = None
        self._refresh()

    def _refresh(self) -> None:
        self.value.setText(self.options[self.index].upper() if self.options else "")

    def set_options(self, options: Sequence[str], index: int = 0) -> None:
        self.options = list(options)
        self.index = max(0, min(index, len(self.options) - 1))
        self._refresh()

    def nudge(self, step: int) -> bool:
        if not self.enabled or not self.options:
            return True
        self.index = (self.index + step) % len(self.options)
        self._refresh()
        if self.sound:
            self.sound("toggle")
        self.on_change(self.index)
        return True

    def activate(self) -> None:
        self.nudge(1)

    def focus(self, on: bool) -> None:
        self.bg["frameColor"] = INK_RAISED if on else INK_SOFT
        self.bar["frameColor"] = ACCENT if on else GREY_DARK
        self.label["fg"] = ACCENT if on else GREY
        for b in (self.left, self.right):
            b["text_fg"] = WHITE if on else GREY

    def destroy(self) -> None:
        self.root.destroy()


class Field(Focusable):
    """A text box (player name, IP address)."""

    def __init__(self, parent, label: str, initial: str, pos: Tuple[float, float],
                 width: float = 0.9, max_length: int = 24, on_submit=None) -> None:
        h = 0.08
        self.root = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0), pos=(pos[0], 0, pos[1]))
        self.bg = panel(self.root, (0, width, -h / 2, h / 2), INK_SOFT)
        self.bar = stripe(self.root, (0, 0.008, -h / 2, h / 2), GREY_DARK)
        self.label = text(self.root, label.upper(), (0.035, -0.016), 0.042, GREY, Fonts.bold)
        self.entry = DirectEntry(parent=self.root, initialText=initial, scale=0.046,
                                 pos=(width * 0.42, 0, -0.017), width=(width * 0.55) / 0.046,
                                 entryFont=Fonts.bold, frameColor=(0, 0, 0, 0),
                                 text_fg=WHITE, numLines=1, focus=0,
                                 command=lambda _v: on_submit and on_submit(),
                                 focusInCommand=lambda: self.focus(True),
                                 focusOutCommand=lambda: self.focus(False))
        self.entry["width"] = (width * 0.55) / 0.046
        self.max_length = max_length
        self.on_hover = None

    @property
    def value(self) -> str:
        return self.entry.get()[: self.max_length].strip()

    def focus(self, on: bool) -> None:
        self.bg["frameColor"] = INK_RAISED if on else INK_SOFT
        self.bar["frameColor"] = ACCENT if on else GREY_DARK
        self.label["fg"] = ACCENT if on else GREY
        self.entry["focus"] = 1 if on else 0

    def activate(self) -> None:
        self.entry["focus"] = 1

    def destroy(self) -> None:
        self.root.destroy()


class StatBar:
    def __init__(self, parent, label: str, value: float, pos: Tuple[float, float],
                 width: float = 0.5, preview: Optional[float] = None) -> None:
        self.root = DirectFrame(parent=parent, frameColor=(0, 0, 0, 0), pos=(pos[0], 0, pos[1]))
        self.label = text(self.root, label.upper(), (0, -0.012), 0.036, GREY, Fonts.bold)
        self.width = width
        x0 = 0.24
        panel(self.root, (x0, x0 + width, -0.012, 0.012), (1, 1, 1, 0.08))
        self.preview = panel(self.root, (x0, x0 + 0.001, -0.012, 0.012), GOOD)
        self.fill = panel(self.root, (x0, x0 + 0.001, -0.012, 0.012), WHITE)
        self.x0 = x0
        self.set(value, preview)

    def set(self, value: float, preview: Optional[float] = None) -> None:
        v = max(0.0, min(10.0, value)) / 10.0 * self.width
        self.fill["frameSize"] = (self.x0, self.x0 + max(0.002, v), -0.012, 0.012)
        p = max(0.0, min(10.0, preview if preview is not None else value)) / 10.0 * self.width
        self.preview["frameSize"] = (self.x0, self.x0 + max(0.002, p), -0.012, 0.012)

    def destroy(self) -> None:
        self.root.destroy()


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
class Screen:
    """Base for every menu. Owns its widgets and the focus ring."""

    def __init__(self, app) -> None:
        self.app = app
        self.root = DirectFrame(parent=app.aspect2d, frameColor=(0, 0, 0, 0))
        self.focusables: List[Focusable] = []
        self.focus_index = 0
        self.alive = True

    # Widgets register here to join keyboard navigation.
    def add(self, widget: Focusable) -> Focusable:
        widget.on_hover = self._hovered
        self.focusables.append(widget)
        if len(self.focusables) == 1:
            widget.focus(True)
        else:
            widget.focus(False)
        return widget

    def _hovered(self, widget: Focusable) -> None:
        if widget in self.focusables:
            self.set_focus(self.focusables.index(widget), sound=False)

    def set_focus(self, index: int, sound: bool = True) -> None:
        if not self.focusables:
            return
        index %= len(self.focusables)
        if index != self.focus_index and sound:
            self.app.audio.ui("hover")
        if 0 <= self.focus_index < len(self.focusables):
            self.focusables[self.focus_index].focus(False)
        self.focus_index = index
        self.focusables[index].focus(True)

    def key(self, name: str) -> None:
        current = self.focusables[self.focus_index] if self.focusables else None
        typing = isinstance(current, Field) and current.entry["focus"]
        if typing and name not in ("arrow_up", "arrow_down", "enter", "escape", "tab"):
            return                      # letters belong to the text box
        if typing and name == "enter":
            current.entry["focus"] = 0
            self.set_focus(self.focus_index + 1)
            return
        if name == "tab":
            self.set_focus(self.focus_index + 1)
            return
        if name in ("arrow_up", "w", "gamepad-dpad_up"):
            self.set_focus(self.focus_index - 1)
        elif name in ("arrow_down", "s", "gamepad-dpad_down"):
            self.set_focus(self.focus_index + 1)
        elif name in ("arrow_left", "a", "gamepad-dpad_left"):
            if self.focusables:
                self.focusables[self.focus_index].nudge(-1)
        elif name in ("arrow_right", "d", "gamepad-dpad_right"):
            if self.focusables:
                self.focusables[self.focus_index].nudge(1)
        elif name in ("enter", "gamepad-face_a"):
            if self.focusables:
                self.focusables[self.focus_index].activate()
        elif name in ("escape", "gamepad-face_b", "gamepad-back"):
            self.back()

    def back(self) -> None: ...
    def update(self, dt: float) -> None: ...

    def destroy(self) -> None:
        self.alive = False
        self.root.destroy()


def title_block(parent, eyebrow: str, title: str, x: float, y: float) -> None:
    text(parent, eyebrow.upper(), (x, y), 0.038, ACCENT, Fonts.bold)
    text(parent, title.upper(), (x, y - 0.11), 0.105, WHITE, Fonts.display, shadow=True)


def money(amount: int) -> str:
    return f"${amount:,}"
