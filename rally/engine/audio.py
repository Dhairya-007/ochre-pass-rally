"""Sound: engines, tyres, surfaces, wind, impacts and the UI.

Every car gets its own pair of engine loops (on-throttle and overrun),
cross-faded by throttle and pitched by rpm. Opponents' engines are placed in
3D, so you hear a rival closing on you before you see them. The player also
gets tyre scrub, a surface bed (gravel or snow), and wind that rises with speed.

Audio is optional: if no device opens, every method is a quiet no-op.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..paths import cache_dir
from ..sound import ensure_loops
from .assets import AssetLibrary


class _Channel:
    def __init__(self, sound) -> None:
        self.sound = sound
        sound.setLoop(True)
        sound.setVolume(0.0)
        sound.play()

    def set(self, volume: float, rate: Optional[float] = None) -> None:
        self.sound.setVolume(max(0.0, volume))
        if rate is not None:
            self.sound.setPlayRate(max(0.2, rate))


class AudioSystem:
    def __init__(self, base, assets: AssetLibrary, volume: float = 0.8) -> None:
        self.base = base
        self.assets = assets
        self.volume = volume
        self.enabled = bool(base.sfxManagerList) and base.sfxManagerList[0].isValid()
        self._loops = ensure_loops(cache_dir() / "audio") if self.enabled else {}
        self._engines: Dict[str, List[_Channel]] = {}
        self._player: Optional[Dict[str, _Channel]] = None
        self._audio3d = None
        self._impacts: Dict[str, list] = {}
        self._ui: Dict[str, object] = {}
        if not self.enabled:
            return
        from direct.showbase.Audio3DManager import Audio3DManager
        self._audio3d = Audio3DManager(base.sfxManagerList[0], base.cam)
        self._audio3d.setDistanceFactor(1.0)
        self._audio3d.setDropOffFactor(0.35)
        for name in ("impactMetal_heavy", "impactMetal_medium", "impactWood_heavy",
                     "impactSoft_heavy", "impactPlate_medium"):
            self._impacts[name] = [self.base.loader.loadSfx(assets.sound_path(
                f"audio/impact/{name}_{i:03d}.ogg")) for i in range(3)]
        for key, name in (("click", "click1"), ("back", "click3"), ("hover", "rollover2"),
                          ("buy", "switch7"), ("toggle", "switch2")):
            self._ui[key] = self.base.loader.loadSfx(assets.sound_path(f"audio/ui/{name}.ogg"))

    def _load(self, name: str):
        from panda3d.core import Filename
        return self.base.loader.loadSfx(Filename.fromOsSpecific(str(self._loops[name])))

    # ------------------------------------------------------------------ #
    def add_car(self, car_id: str, node, is_player: bool) -> None:
        if not self.enabled:
            return
        if is_player:
            self._player = {name: _Channel(self._load(name))
                            for name in ("engine_on", "engine_off", "skid", "gravel", "snow", "wind")}
            return
        on = self._audio3d.loadSfx(self._loops["engine_on"])
        off = self._audio3d.loadSfx(self._loops["engine_off"])
        for s in (on, off):
            self._audio3d.attachSoundToObject(s, node)
        self._engines[car_id] = [_Channel(on), _Channel(off)]

    def update_player(self, rpm: float, throttle: float, slip: float, speed: float,
                      surface: str, offroad: bool, airborne: bool) -> None:
        if not self._player:
            return
        v = self.volume
        rate = 0.55 + rpm * 1.9
        load = min(1.0, throttle * 1.2)
        self._player["engine_on"].set(v * (0.10 + 0.42 * load) * (0.5 + 0.5 * rpm), rate)
        self._player["engine_off"].set(v * 0.30 * (1.0 - load) * (0.4 + 0.6 * rpm), rate)
        scrub = 0.0 if airborne else max(0.0, min(1.0, (slip - 0.12) * 3.0)) * min(1.0, speed / 12)
        self._player["skid"].set(v * 0.32 * scrub, 0.9 + scrub * 0.2)
        bed = 0.0 if airborne else min(1.0, speed / 25.0) * (1.6 if offroad else 1.0)
        snow = surface == "snow"
        self._player["gravel"].set(v * 0.18 * bed * (0.0 if snow else 1.0), 0.8 + speed / 60)
        self._player["snow"].set(v * 0.22 * bed * (1.0 if snow else 0.0), 0.9 + speed / 80)
        self._player["wind"].set(v * min(0.35, (speed / 45.0) ** 2 * 0.35), 0.7 + speed / 70)

    def update_car(self, car_id: str, rpm: float, throttle: float) -> None:
        channels = self._engines.get(car_id)
        if not channels:
            return
        rate = 0.55 + rpm * 1.9
        channels[0].set(self.volume * (0.15 + 0.5 * throttle), rate)
        channels[1].set(self.volume * 0.3 * (1 - throttle), rate)

    def tick(self) -> None:
        if self._audio3d:
            self._audio3d.update()

    def impact(self, strength: float, kind: str = "car") -> None:
        if not self.enabled:
            return
        if kind == "scenery":
            name = "impactWood_heavy" if strength > 8 else "impactSoft_heavy"
        else:
            name = "impactMetal_heavy" if strength > 8 else "impactMetal_medium"
        sound = random.choice(self._impacts[name])
        sound.setVolume(self.volume * min(1.0, 0.25 + strength / 18.0))
        sound.setPlayRate(random.uniform(0.85, 1.1))
        sound.play()

    def ui(self, key: str) -> None:
        sound = self._ui.get(key)
        if sound is not None:
            sound.setVolume(self.volume * 0.6)
            sound.play()

    def mute_race(self) -> None:
        """Silence engines and tyres, e.g. while paused. The next update restores them."""
        for channels in self._engines.values():
            for c in channels:
                c.set(0.0)
        if self._player:
            for c in self._player.values():
                c.set(0.0)

    def stop_race(self) -> None:
        for channels in self._engines.values():
            for c in channels:
                c.sound.stop()
        self._engines.clear()
        if self._player:
            for c in self._player.values():
                c.sound.stop()
        self._player = None

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
