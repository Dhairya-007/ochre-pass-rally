"""The application: window, screens, menus' 3D showroom, input, and the loop."""

from __future__ import annotations

import sys
import traceback
from typing import Callable, Optional

from direct.gui.DirectGui import DirectFrame
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (InputDevice, KeyboardButton, TextNode, WindowProperties,
                          loadPrcFileData)

from ..career import Career, CareerError
from ..paths import user_data_dir
from ..settings import Settings
from ..stages import get_stage
from ..terrain import Terrain
from ..track import Track
from ..scenery import SceneryPlanner
from ..vehicle import Control, Vehicle
from .assets import AssetLibrary, check_assets
from .audio import AudioSystem
from .camera import RaceCamera
from .car_view import CarView
from .renderer import Quality, Renderer
from .world import StageView


class MenuScene:
    """A real stage behind the menus, with the selected car parked on it."""

    def __init__(self, app) -> None:
        self.app = app
        self.stage_id: Optional[str] = None
        self.scene = None
        self.view = None
        self.car = None
        self.car_key = None
        self.camera = None
        self.spot = (0.0, 0.0, 0.0, 0.0)
        self.t = 0.0

    def show_car(self, car_id: str, livery: int, stage_id: Optional[str]) -> None:
        stage_id = stage_id or self.stage_id or "pine_ridge"
        if stage_id != self.stage_id:
            self._build(stage_id)
        if (car_id, livery) != self.car_key:
            if self.car:
                self.car.destroy()
            self.car = CarView(car_id, livery, self.app.assets, self.scene, self.terrain)
            vehicle = Vehicle()
            x, y, z, yaw = self.spot
            vehicle.place(x, z, yaw, self.terrain)
            self.car.update(vehicle)
            self.car_key = (car_id, livery)

    def _build(self, stage_id: str) -> None:
        self.destroy()
        stage = get_stage(stage_id)
        track = Track.build(stage.track_config())
        self.terrain = Terrain(track, stage.world_config())
        plan = SceneryPlanner(stage, track, self.terrain).plan()
        self.scene = self.app.render.attachNewNode("menu-scene")
        self.view = StageView(stage, track, self.terrain, plan, self.app.assets, self.scene,
                              density=self.app.quality.scenery_density * 0.8)
        self.app.renderer.set_stage(self.scene, stage.atmosphere)
        i = int(track.count * 0.04)
        p, r = track.points[i], track.rights[i]
        x, z = float(p[0] + r[0] * 1.5), float(p[2] + r[2] * 1.5)
        self.spot = (x, self.terrain.height_at(x, z), z, track.heading_at(i) + 0.6)
        self.camera = RaceCamera(self.app, self.terrain)
        self.stage_id = stage_id
        self.car_key = None
        from .coords import to_panda
        self.app.renderer.follow(*to_panda(x, self.spot[1], z))

    def update(self, dt: float) -> None:
        if self.camera:
            self.t += dt
            x, y, z, _ = self.spot
            self.camera.orbit(x, y, z, self.t)

    def destroy(self) -> None:
        if self.car:
            self.car.destroy()
            self.car = None
        if self.view:
            self.app.renderer.clear()
            self.view.destroy()
            self.scene.removeNode()
        self.view = None
        self.scene = None
        self.stage_id = None
        self.car_key = None


class RallyApp(ShowBase):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        quality = Quality.preset(settings.quality)
        loadPrcFileData("", f"""
            window-title Ochre Pass Rally
            win-size {settings.width} {settings.height}
            fullscreen {"#t" if settings.fullscreen else "#f"}
            framebuffer-multisample 0
            textures-power-2 none
            sync-video {"#t" if settings.vsync else "#f"}
            show-frame-rate-meter {"#t" if settings.show_fps else "#f"}
        """)
        super().__init__()
        self.disableMouse()
        check_assets()

        self.quality = quality
        self.assets = AssetLibrary(self.loader)
        from .ui.kit import Fonts
        Fonts.init(self.assets)
        self.renderer = Renderer(self, self.assets, quality)
        self.audio = AudioSystem(self, self.assets, settings.volume)
        self.career: Optional[Career] = self._load_career()
        self.mp_car, self.mp_livery = "kestrel", 0

        self.screen = None
        self.overlay = None                     # pause screen over a race
        self.race = None
        self.paused = False
        self._last_setup = None
        self._toast = None
        self._toast_for = 0.0
        self._steer = 0.0
        self._throttle = 0.0
        self._brake = 0.0
        self.gamepad: Optional[InputDevice] = None
        self._setup_input()

        self.menu_scene = MenuScene(self)
        loading = self._loading("Ochre Pass Rally", "Warming up")
        loading.set_progress("Preparing the showroom", 0.3)
        self._draw()
        self.menu_scene.show_car(*self._showroom_car(), "pine_ridge")
        loading.destroy()

        from .ui.menus import MainMenu
        self.goto(MainMenu)
        self.taskMgr.add(self._tick, "rally-tick")

    # ------------------------------------------------------------------ #
    # Career persistence
    # ------------------------------------------------------------------ #
    @property
    def career_path(self):
        return user_data_dir() / "career.json"

    def _load_career(self) -> Optional[Career]:
        if not self.career_path.exists():
            return None
        try:
            return Career.load(self.career_path)
        except (CareerError, ValueError, OSError, TypeError) as error:
            backup = self.career_path.with_suffix(".unreadable.json")
            self.career_path.replace(backup)
            print(f"Career save could not be read ({error}); moved it to {backup.name}.")
            return None

    def save_career(self) -> None:
        if self.career:
            self.career.save(self.career_path)

    def _showroom_car(self):
        if self.career:
            return self.career.current_car, self.career.owned[self.career.current_car].livery
        return "kestrel", 0

    # ------------------------------------------------------------------ #
    # Screens
    # ------------------------------------------------------------------ #
    def goto(self, screen_cls, **kwargs) -> None:
        if self.screen is not None:
            self.screen.destroy()
            self.screen = None
        if self.race is None and self.menu_scene.scene is None:
            self.menu_scene.show_car(*self._showroom_car(), None)
        self.screen = screen_cls(self, **kwargs)

    def open_career(self) -> None:
        from .ui.career_screens import CareerHub, NewCareer
        self.goto(CareerHub if self.career else NewCareer)

    def open_multiplayer(self) -> None:
        from .ui.multiplayer_screens import MultiplayerMenu
        self.goto(MultiplayerMenu)

    def toast(self, message: str, colour=None) -> None:
        from .ui.kit import ACCENT, Fonts, text
        if self._toast is not None:
            self._toast.destroy()
        self._toast = text(self.a2dBottomCenter, message.upper(), (0, 0.2), 0.045,
                           colour or ACCENT, Fonts.bold, TextNode.ACenter, shadow=True)
        self._toast_for = 2.6

    def _loading(self, title: str, subtitle: str):
        from .ui.menus import LoadingScreen
        return LoadingScreen(self, title, subtitle)

    def _draw(self) -> None:
        """Render a frame right now - used to animate loading screens."""
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.renderFrame()

    # ------------------------------------------------------------------ #
    # Racing
    # ------------------------------------------------------------------ #
    def start_race(self, setup, on_finish: Callable, net=None) -> None:
        from .race_controller import RaceController
        from ..stages import get_stage
        self._last_setup = (setup, on_finish, net)
        if self.screen is not None:
            self.screen.destroy()
            self.screen = None
        self.menu_scene.destroy()
        stage = get_stage(setup.stage_id)
        loading = self._loading(stage.name, f"{stage.country} · {setup.laps} laps")

        def progress(message: str, fraction: float) -> None:
            loading.set_progress(message, fraction)
            self._draw()

        def finished(results) -> None:
            self.end_race()
            on_finish(results)

        try:
            self.race = RaceController(self, setup, finished, net=net, progress=progress)
        except Exception:
            loading.destroy()
            traceback.print_exc()
            self.race = None
            from .ui.menus import MainMenu
            self.toast("Couldn't start the race - see the console")
            self.goto(MainMenu)
            return
        loading.destroy()
        self.paused = False

    def end_race(self) -> None:
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        if self.race:
            self.race.destroy()
            self.race = None
        self.paused = False

    def pause(self) -> None:
        if self.race and not self.paused:
            from .ui.menus import PauseScreen
            # Multiplayer races can't freeze for everyone; the menu overlays a live race.
            self.paused = self.race.setup.mode in ("career", "quick")
            self.overlay = PauseScreen(self, allow_restart=self.race.setup.mode in ("career", "quick"))
            if self.paused:
                self.audio.mute_race()

    def resume(self) -> None:
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.paused = False

    def restart_race(self) -> None:
        if self._last_setup:
            setup, on_finish, net = self._last_setup
            self.end_race()
            self.start_race(setup, on_finish, net)

    def abandon_race(self) -> None:
        mode = self.race.setup.mode if self.race else ""
        net = self.race.net if self.race else None
        self.end_race()
        if net is not None:
            net.close()
        from .ui.menus import MainMenu
        if mode == "career":
            from .ui.career_screens import CareerHub
            self.goto(CareerHub)
        else:
            self.goto(MainMenu)

    def cycle_camera(self) -> None:
        if self.race:
            mode = self.race.cycle_camera()
            self.toast(f"Camera: {mode.replace('_', ' ')}")

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def _setup_input(self) -> None:
        for key in ("arrow_up", "arrow_down", "arrow_left", "arrow_right", "enter", "escape",
                    "tab", "w", "a", "s", "d", "r", "c"):
            self.accept(key, self._key, [key])
            if key.startswith("arrow"):
                self.accept(key + "-repeat", self._key, [key])
        for button in ("dpad_up", "dpad_down", "dpad_left", "dpad_right", "face_a", "face_b",
                       "start", "back", "face_y", "rshoulder"):
            self.accept(f"gamepad-{button}", self._key, [f"gamepad-{button}"])
        self.accept("connect-device", self._device_connected)
        self.accept("disconnect-device", self._device_disconnected)
        for device in self.devices.getDevices(InputDevice.DeviceClass.gamepad):
            self._device_connected(device)

    def _device_connected(self, device) -> None:
        if device.device_class == InputDevice.DeviceClass.gamepad and self.gamepad is None:
            self.gamepad = device
            self.attachInputDevice(device, prefix="gamepad")

    def _device_disconnected(self, device) -> None:
        if device is self.gamepad:
            self.detachInputDevice(device)
            self.gamepad = None

    def _key(self, name: str) -> None:
        if self.overlay is not None:
            self.overlay.key(name)
            return
        if self.race is not None:
            if name in ("escape", "gamepad-start"):
                self.pause()
            elif name in ("r", "gamepad-face_y"):
                self.race.recover()
            elif name in ("c", "gamepad-rshoulder"):
                self.cycle_camera()
            return
        if self.screen is not None:
            self.screen.key(name)

    def _down(self, *keys) -> bool:
        watcher = self.mouseWatcherNode
        for key in keys:
            if watcher.isButtonDown(KeyboardButton.asciiKey(key) if len(key) == 1 else
                                    getattr(KeyboardButton, key)()):
                return True
        return False

    def read_controls(self, dt: float) -> Control:
        """Keyboard with smoothing, or a gamepad's analogue axes if one is in use."""
        throttle = 1.0 if self._down("w", "up") else 0.0
        brake = 1.0 if self._down("s", "down") else 0.0
        steer = (1.0 if self._down("d", "right") else 0.0) - (1.0 if self._down("a", "left") else 0.0)
        handbrake = self._down("space")

        pad = self.gamepad
        if pad is not None:
            def axis(a):
                ax = pad.findAxis(a)
                return ax.value if ax else 0.0
            lx = axis(InputDevice.Axis.left_x)
            rt = axis(InputDevice.Axis.right_trigger)
            lt = axis(InputDevice.Axis.left_trigger)
            if abs(lx) > 0.12:
                steer = (abs(lx) - 0.12) / 0.88 * (1 if lx > 0 else -1)
            throttle = max(throttle, rt)
            brake = max(brake, lt)
            face_a = pad.findButton("face_a")
            handbrake = handbrake or (face_a.pressed if face_a else False)
            self._steer = steer
            return Control(throttle=throttle, brake=brake, steer=steer, handbrake=handbrake)

        # Keyboard: ramp towards the target, centre faster than it turns in,
        # so taps give small corrections and holds give full lock.
        rate = 5.5 if abs(steer) > abs(self._steer) and steer * self._steer >= 0 else 9.0
        self._steer += max(-rate * dt, min(rate * dt, steer - self._steer))
        self._throttle += max(-8 * dt, min(8 * dt, throttle - self._throttle))
        self._brake += max(-10 * dt, min(10 * dt, brake - self._brake))
        return Control(throttle=self._throttle, brake=self._brake, steer=self._steer,
                       handbrake=handbrake)

    # ------------------------------------------------------------------ #
    def _tick(self, task):
        dt = min(globalClock.getDt(), 0.1)
        try:
            if self.race is not None:
                if not self.paused:
                    self.race.update(dt)
                if self.overlay is not None:
                    self.overlay.update(dt)
            else:
                self.menu_scene.update(dt)
                if self.screen is not None:
                    self.screen.update(dt)
        except Exception:
            traceback.print_exc()
            self.toast("Something went wrong - see the console")
            if self.race is not None:
                self.abandon_race()
        if self._toast_for > 0:
            self._toast_for -= dt
            if self._toast_for <= 0 and self._toast is not None:
                self._toast.destroy()
                self._toast = None
        return task.cont

    def quit(self) -> None:
        self.settings.save()
        self.save_career()
        self.userExit()
