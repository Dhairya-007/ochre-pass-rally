"""Runs one race on screen, in any mode: career, quick race, host or client."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from ..ai import Personality
from ..collision import Obstacle, StaticColliders
from ..net.protocol import STATE_HZ, state_of
from ..race import Phase, RaceSession, Result
from ..scenery import SceneryPlanner
from ..stages import get_stage
from ..terrain import Terrain
from ..track import Track
from ..vehicle import Control
from .camera import RaceCamera
from .car_view import CarView
from .coords import to_panda
from .particles import DustSystem
from .ui.hud import HUD
from .world import StageView


@dataclass
class EntrantSpec:
    id: str
    name: str
    kind: str                         # player | ai | remote
    car_id: str
    livery: int = 0
    upgrades: Dict[str, int] = field(default_factory=dict)
    skill: float = 0.75
    consistency: float = 0.8
    aggression: float = 0.5
    seed: int = 0


@dataclass
class RaceSetup:
    stage_id: str
    laps: int
    entrants: List[EntrantSpec]
    player_id: str
    mode: str                         # career | quick | host | client


def result_to_dict(r: Result) -> dict:
    return asdict(r)


def result_from_dict(d: dict) -> Result:
    keys = Result.__dataclass_fields__
    return Result(**{k: d.get(k) for k in keys})


class RaceController:
    def __init__(self, app, setup: RaceSetup, on_finish: Callable[[List[Result]], None],
                 net=None, progress: Optional[Callable[[str, float], None]] = None) -> None:
        self.app = app
        self.setup = setup
        self.on_finish = on_finish
        self.net = net
        self.progress = progress or (lambda m, f: None)
        self.finished = False
        self._finish_delay = 0.0
        self._net_clock = 0.0
        self._standings_clock = 0.0
        self._remote_standings: Optional[List[dict]] = None
        self._final_results: Optional[List[Result]] = None
        self._throttle = 0.0
        self._reverse_hint = 0.0

        self.stage = get_stage(setup.stage_id)
        self.progress("Surveying the stage", 0.02)
        self.track = Track.build(self.stage.track_config())
        self.terrain = Terrain(self.track, self.stage.world_config())
        plan = SceneryPlanner(self.stage, self.track, self.terrain).plan()
        colliders = StaticColliders(Obstacle(p.x, p.z, p.radius) for p in plan)

        self.race = RaceSession(self.stage, self.track, self.terrain, setup.laps, colliders)
        for e in setup.entrants:
            self.race.add(e.id, e.name, e.kind, e.car_id, e.livery, e.upgrades,
                          Personality(e.skill, e.aggression, e.consistency), seed=e.seed)

        self.scene = app.render.attachNewNode("race-scene")
        self.view = StageView(self.stage, self.track, self.terrain, plan, app.assets,
                              self.scene, density=app.quality.scenery_density,
                              progress=lambda m, f: self.progress(m, 0.05 + f * 0.75))
        self.progress("Lighting the stage (first visit takes a moment)", 0.85)
        app.renderer.set_stage(self.scene, self.stage.atmosphere)

        self.progress("Unloading the cars", 0.95)
        self.cars: Dict[str, CarView] = {}
        for e in setup.entrants:
            label = None if e.id == setup.player_id else e.name
            view = CarView(e.car_id, e.livery, app.assets, self.scene, self.terrain,
                           label=label, font=app.assets.font("BarlowCondensed-Bold"))
            self.cars[e.id] = view
            app.audio.add_car(e.id, view.root, is_player=e.id == setup.player_id)

        self.dust = DustSystem(self.scene, app.assets, self.stage.surface.dust)
        self.camera = RaceCamera(app, self.terrain)
        self.camera.mode = app.settings.camera if app.settings.camera in ("chase", "chase_far", "bonnet") else "chase"
        self.player = self.race.entrants[setup.player_id]
        self.camera.snap(self.player.vehicle)
        self.hud = HUD(app, self.track, self.stage.name, setup.laps, len(setup.entrants))
        self.race.start_countdown()
        self.progress("", 1.0)

    # ------------------------------------------------------------------ #
    def update(self, dt: float) -> None:
        dt = min(dt, 1 / 30)
        control = self.app.read_controls(dt) if not self.player.done else Control(brake=0.5)
        self._throttle = control.throttle

        if self.setup.mode == "client" and self.net is not None:
            self._client_net(dt)

        events = self.race.update(dt, {self.player.id: control})
        for event in events:
            self._on_event(event)

        if self.setup.mode == "host" and self.net is not None:
            self._host_net(dt)
        elif self.setup.mode == "client" and self.net is not None:
            self._net_clock += dt
            if self._net_clock >= 1.0 / STATE_HZ:
                self._net_clock = 0.0
                self.net.send_state(state_of(self.player.vehicle, control.brake))

        self._update_views(dt, control)
        self._update_hud(dt)

        done = self.race.phase is Phase.DONE or (self.setup.mode == "client" and self._final_results)
        if self.player.done or done:
            self._finish_delay += dt
            if done and self._finish_delay > 3.0 and not self.finished:
                self.finished = True
                results = self._final_results or self.race.results()
                self.on_finish(results)

    # ------------------------------------------------------------------ #
    def _on_event(self, event) -> None:
        if event.kind == "go":
            self.hud.flash("GO!", 0.8)
        elif event.kind == "impact":
            if event.entrant == self.player.id or event.detail == self.player.id:
                self.app.audio.impact(event.value, "scenery" if event.detail == "scenery" else "car")
                self.camera.shake = min(1.5, event.value / 10.0)
        elif event.kind == "lap" and event.entrant == self.player.id:
            timer = self.player.timer
            if not timer.finished:
                best = " - FASTEST" if timer.last == timer.best else ""
                from ..scoring import format_time
                self.hud.flash(f"LAP {timer.lap}  {format_time(timer.last)}{best}", 2.4)
        elif event.kind == "finish" and event.entrant == self.player.id:
            pos = self.race.position_of(self.player.id)
            self.hud.flash(f"FINISHED  P{pos}", 6.0)

    def _host_net(self, dt: float) -> None:
        net = self.net
        for ev in net.poll():
            if ev.kind == "left":
                self.race.remove(ev.data["id"])
                self.hud.flash(f"{ev.data['name'].upper()} LEFT", 2.0)
        for player_id, state in net.remote_states().items():
            self.race.apply_remote(player_id, state)
        self._net_clock += dt
        if self._net_clock >= 1.0 / STATE_HZ:
            self._net_clock = 0.0
            local = {e.id: state_of(e.vehicle) for e in self.race.entrants.values()
                     if e.kind != "remote"}
            net.send_states(local, self.race.clock)
        self._standings_clock += dt
        if self._standings_clock >= 0.25:
            self._standings_clock = 0.0
            gaps = self.race.gaps()
            net.send_standings([{"id": e.id, "gap": gaps.get(e.id, ""),
                                 "lap": e.timer.lap} for e in self.race.standings()],
                               self.race.clock)
        if self.race.phase is Phase.DONE and net.phase == "racing":
            net.send_results([result_to_dict(r) for r in self.race.results()])

    def _client_net(self, dt: float) -> None:
        for ev in self.net.poll():
            if ev.kind == "standings":
                self._remote_standings = ev.data.get("rows")
            elif ev.kind == "results":
                self._final_results = [result_from_dict(r) for r in ev.data.get("rows", [])]
            elif ev.kind == "closed":
                self.hud.flash("HOST DISCONNECTED", 8.0)
                if self._final_results is None:
                    self._final_results = self.race.results()
        for car_id, state in self.net.remote_states().items():
            self.race.apply_remote(car_id, state)

    # ------------------------------------------------------------------ #
    def _update_views(self, dt: float, control: Control) -> None:
        cam_pos = self.app.cam.getPos(self.scene)
        snow = self.stage.surface.road_texture == "gravel_snow"
        for e in self.race.entrants.values():
            v = e.vehicle
            braking = control.brake if e is self.player else (
                (e.remote_state or {}).get("br", 0.0) if e.kind == "remote" else 0.0)
            view = self.cars[e.id]
            distance = (view.root.getPos(self.scene) - cam_pos).length()
            view.update(v, braking, distance)
            if e is not self.player:
                self.app.audio.update_car(e.id, v.rpm, 0.8 if v.speed > 2 else 0.2)

            if v.speed > 3.0 and not v.airborne and distance < 160:
                offroad = self.terrain.is_offroad(v.x, v.z)
                amount = self.stage.surface.dust_amount * (
                    0.5 * min(1.0, v.speed / 20) + v.slip_angle * 5.0 + (1.8 if offroad else 0.0))
                if amount > 0.25:
                    lx, lz, rx, rz = v.rear_axle_world()
                    fx, fz = v.forward
                    push = (-fx * v.speed * 0.15, -fz * v.speed * 0.15)
                    for wx, wz in ((lx, lz), (rx, rz)):
                        p = to_panda(wx, v.y + 0.1, wz)
                        self.dust.emit_rate(p.x, p.y, p.z, amount * 12.0, dt,
                                            spread=1.0 + v.slip_angle * 4, size=0.55,
                                            lift=1.2 + (0.8 if snow else 0.0),
                                            push=(push[0], push[1]))
        self.dust.update(dt, self.app.cam)

        pv = self.player.vehicle
        self.camera.update(pv, dt)
        p = to_panda(pv.x, pv.y, pv.z)
        self.app.renderer.follow(p.x, p.y, p.z)
        self.app.audio.update_player(pv.rpm, control.throttle, pv.slip_angle, pv.speed,
                                     "snow" if snow else "gravel",
                                     self.terrain.is_offroad(pv.x, pv.z), pv.airborne)
        self.app.audio.tick()

    def _update_hud(self, dt: float) -> None:
        pv = self.player.vehicle
        timer = self.player.timer
        if self._remote_standings:
            names = {e.id: e.name for e in self.race.entrants.values()}
            rows = [(r["id"], names.get(r["id"], "?"), r.get("gap", ""), r["id"] == self.player.id)
                    for r in self._remote_standings]
            position = next((i + 1 for i, r in enumerate(rows) if r[3]), 1)
        else:
            gaps = self.race.gaps()
            order = self.race.standings()
            rows = [(e.id, e.name, gaps.get(e.id, ""), e is self.player) for e in order]
            position = next(i + 1 for i, e in enumerate(order) if e is self.player)

        cars = [(e.id, e.vehicle.x, e.vehicle.z, e is self.player)
                for e in self.race.entrants.values()]
        wrong = self.race.phase is Phase.RACING and timer.check_wrong_way(pv.yaw, pv.v_long, dt)
        stuck = self.race.phase is Phase.RACING and pv.speed < 1.0 and self._throttle > 0.5
        self._reverse_hint = self._reverse_hint + dt if stuck else 0.0
        hint = "PRESS R TO RECOVER TO THE STAGE" if (self._reverse_hint > 2.0 or
                                                   self.terrain.distance_at(pv.x, pv.z) > 40) else ""
        if self.player.done:
            hint = "WAITING FOR THE OTHERS TO FINISH" if self.race.phase is not Phase.DONE else ""
        self.hud.update(dt, speed_kmh=pv.speed_kmh, gear=pv.gear, rpm=pv.rpm,
                        reverse=pv.v_long < -0.8, position=position,
                        lap=timer.display_lap, lap_time=timer.current_lap_time(self.race.clock)
                        if self.race.phase is Phase.RACING else 0.0,
                        best=timer.best if timer.laps else None, standings=rows, cars=cars,
                        pace_note=timer.pace_note(), countdown=self.race.countdown_label,
                        wrong_way=wrong, hint=hint)

    # ------------------------------------------------------------------ #
    def recover(self) -> None:
        if self.race.phase is Phase.RACING and not self.player.done:
            self.race.recover(self.player.id)
            self.camera.snap(self.player.vehicle)

    def cycle_camera(self) -> str:
        mode = self.camera.cycle()
        self.app.settings.camera = mode
        return mode

    def destroy(self) -> None:
        self.app.audio.stop_race()
        self.hud.destroy()
        self.dust.destroy()
        for view in self.cars.values():
            view.destroy()
        self.app.renderer.clear()
        self.view.destroy()
        self.scene.removeNode()
