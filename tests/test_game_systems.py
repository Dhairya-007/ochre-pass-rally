"""Tests for stages, cars, career, AI, collisions, the race referee and
multiplayer. All headless - no window, no GPU.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import socket
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from rally.ai import Driver, Personality, RacingLine
from rally.cars import (CARS, CARS_BY_ID, UPGRADES, UPGRADES_BY_ID, build_config,
                        rate, upgrade_price)
from rally.career import (CHAMPIONSHIPS, POINTS, STARTING_MONEY, Career, CareerError)
from rally.collision import (CAR_CIRCLE_RADIUS, Obstacle, StaticColliders,
                             resolve_cars)
from rally.config import GameConfig, VEHICLE
from rally.net.protocol import (FrameDecoder, ProtocolError, clean_name, clean_state,
                                encode, state_of)
from rally.net.session import ClientSession, HostSession, RemoteBuffer
from rally.race import Phase, RaceSession, Result
from rally.scenery import SceneryPlanner
from rally.scoring import LapEvent, LapTimer
from rally.stages import STAGES, get_stage
from rally.terrain import Terrain
from rally.track import Track
from rally.vehicle import Control, Vehicle

DT = 1 / 60


class Flat:
    def __init__(self, grip=1.0, rolling=1.0):
        self.grip, self.rolling = grip, rolling

    def height_at(self, x, z): return 0.0
    def grip_at(self, x, z): return self.grip
    def rolling_at(self, x, z): return self.rolling
    def slope_ahead(self, *args, **kwargs): return 0.0
    def world_limit(self): return 1e9


_BUILT = {}


def built(stage_id):
    """Track and terrain per stage, built once for the whole test run."""
    if stage_id not in _BUILT:
        stage = get_stage(stage_id)
        track = Track.build(stage.track_config())
        _BUILT[stage_id] = (stage, track, Terrain(track, stage.world_config()))
    return _BUILT[stage_id]


# --------------------------------------------------------------------------- #
class StageTests(unittest.TestCase):
    def test_every_stage_generates_a_sane_loop(self):
        for stage in STAGES:
            with self.subTest(stage=stage.id):
                _, track, terrain = built(stage.id)
                self.assertGreater(track.length, 1200)
                self.assertLess(track.length, 2500)
                limit = terrain.world_limit()
                self.assertLess(np.abs(track.points[:, [0, 2]]).max(), limit - 50)

    def test_no_stage_crosses_itself(self):
        for stage in STAGES:
            with self.subTest(stage=stage.id):
                _, track, _ = built(stage.id)
                p = track.points[:, [0, 2]]
                d = np.linalg.norm(p[:, None] - p[None], axis=-1)
                i = np.arange(track.count)
                sep = np.abs(i[:, None] - i[None])
                sep = np.minimum(sep, track.count - sep)
                # Any two parts of the road more than 60 samples apart stay at
                # least several road widths away from each other.
                self.assertGreater(d[sep > 60].min(), 6 * track.config.road_half_width)

    def test_corners_are_tight_enough_to_be_rally_but_drivable(self):
        for stage in STAGES:
            with self.subTest(stage=stage.id):
                _, track, _ = built(stage.id)
                span = 12 * track.length / track.count
                radii = span / np.maximum([track.curvature_at(i)[0]
                                           for i in range(track.count)], 1e-6)
                self.assertGreater(radii.min(), 18.0)
                self.assertLess(radii.min(), 60.0)

    def test_generation_is_deterministic(self):
        stage = STAGES[0]
        self.assertEqual(stage.track_config().control_points,
                         stage.track_config().control_points)

    def test_unknown_stage_names_the_options(self):
        with self.assertRaises(KeyError) as ctx:
            get_stage("moon")
        self.assertIn("pine_ridge", str(ctx.exception))


class SceneryTests(unittest.TestCase):
    def test_planning_is_deterministic_and_off_the_road(self):
        stage, track, terrain = built("pine_ridge")
        a = SceneryPlanner(stage, track, terrain).plan()
        b = SceneryPlanner(stage, track, terrain).plan()
        self.assertEqual(a, b)
        half = track.config.road_half_width
        trees = [p for p in a if p.kind == "tree"]
        self.assertGreater(len(trees), stage.vegetation.tree_count * 0.9)
        for p in trees:
            self.assertGreater(terrain.distance_at(p.x, p.z), half + 5.0)

    def test_soft_props_have_no_collider(self):
        stage, track, terrain = built("pine_ridge")
        for p in SceneryPlanner(stage, track, terrain).plan():
            if p.model.startswith(("plant_", "grass")):
                self.assertEqual(p.radius, 0.0)
            if p.model.startswith("tree_"):
                self.assertGreater(p.radius, 0.2)


# --------------------------------------------------------------------------- #
class CarTests(unittest.TestCase):
    def test_every_car_builds_a_valid_config(self):
        for car in CARS:
            cfg = build_config(car)
            self.assertGreater(cfg.engine_force, 0)
            self.assertGreater(cfg.mass, 500)

    def test_upgrades_improve_what_they_claim(self):
        car = CARS_BY_ID["kestrel"]
        stock = build_config(car)
        tuned = build_config(car, {"engine": 3, "tyres": 3, "suspension": 3, "brakes": 3})
        self.assertGreater(tuned.engine_force, stock.engine_force)
        self.assertGreater(tuned.top_speed, stock.top_speed)
        self.assertGreater(tuned.lateral_grip, stock.lateral_grip)
        self.assertLess(tuned.slip_falloff, stock.slip_falloff)
        self.assertGreater(tuned.brake_force, stock.brake_force)

    def test_levels_are_clamped(self):
        car = CARS_BY_ID["kestrel"]
        self.assertEqual(build_config(car, {"engine": 99}), build_config(car, {"engine": 3}))
        self.assertEqual(build_config(car, {"engine": -5}), build_config(car))

    def test_prices_rise_with_level_and_car_tier(self):
        engine = UPGRADES_BY_ID["engine"]
        cheap, dear = CARS_BY_ID["kestrel"], CARS_BY_ID["apex"]
        self.assertLess(upgrade_price(cheap, engine, 1), upgrade_price(cheap, engine, 3))
        self.assertLess(upgrade_price(cheap, engine, 1), upgrade_price(dear, engine, 1))
        with self.assertRaises(ValueError):
            upgrade_price(cheap, engine, 4)

    def test_ratings_are_on_the_bar_scale_and_order_sensibly(self):
        for car in CARS:
            r = rate(build_config(car))
            for value in (r.power, r.top_speed, r.grip, r.handling, r.braking, r.rough):
                self.assertTrue(0.0 <= value <= 10.0)
        kestrel = rate(build_config(CARS_BY_ID["kestrel"]))
        apex = rate(build_config(CARS_BY_ID["apex"]))
        ridgeback = rate(build_config(CARS_BY_ID["ridgeback"]))
        self.assertGreater(apex.overall, kestrel.overall)
        self.assertGreater(ridgeback.rough, apex.rough)

    def test_four_by_four_holds_on_better_on_snow(self):
        """The loose-surface trait should show up in the physics, not just the bar."""
        def slip_in_corner(car_id):
            car = Vehicle(config=build_config(CARS_BY_ID[car_id]))
            surface = Flat(grip=0.6)
            car.place(0, 0, 0, surface, keep_speed=22.0)
            for _ in range(90):
                car.step(Control(throttle=0.7, steer=0.8), surface, DT)
            return car.slip_angle
        self.assertLess(slip_in_corner("ridgeback"), slip_in_corner("kestrel"))


# --------------------------------------------------------------------------- #
class CollisionTests(unittest.TestCase):
    def test_a_car_cannot_drive_through_a_tree(self):
        trees = StaticColliders([Obstacle(0.0, 20.0, 0.5)])
        car = Vehicle()
        car.place(0.0, 0.0, 0.0, Flat(), keep_speed=20.0)
        worst = 0.0
        for _ in range(180):
            car.step(Control(throttle=1.0), Flat(), DT)
            worst = max(worst, trees.resolve(car))
        self.assertLess(car.z, 20.0)
        self.assertGreater(worst, 5.0)

    def test_impact_bleeds_speed(self):
        trees = StaticColliders([Obstacle(0.0, 3.0, 0.5)])
        car = Vehicle()
        car.place(0.0, 0.0, 0.0, Flat(), keep_speed=25.0)
        car.step(Control(), Flat(), DT * 4)
        trees.resolve(car)
        self.assertLess(abs(car.v_long), 10.0)

    def test_zero_radius_obstacles_are_ignored(self):
        self.assertEqual(StaticColliders([Obstacle(0, 0, 0.0)]).count, 0)

    def test_car_to_car_conserves_momentum(self):
        a, b = Vehicle(), Vehicle()
        a.place(0.0, 0.0, 0.0, Flat(), keep_speed=20.0)
        b.place(0.0, 3.0, 0.0, Flat(), keep_speed=0.0)
        before = a.config.mass * a.world_velocity()[1] + b.config.mass * b.world_velocity()[1]
        impact = resolve_cars(a, b)
        after = a.config.mass * a.world_velocity()[1] + b.config.mass * b.world_velocity()[1]
        self.assertGreater(impact, 10.0)
        self.assertAlmostEqual(before, after, delta=1.0)
        self.assertGreater(b.v_long, 0.0)                 # it got shunted forwards

    def test_remote_car_is_not_moved_by_a_local_collision(self):
        a, b = Vehicle(), Vehicle()
        a.place(0.0, 0.0, 0.0, Flat(), keep_speed=20.0)
        b.place(0.0, 3.0, 0.0, Flat())
        resolve_cars(a, b, b_movable=False)
        self.assertEqual((b.x, b.z, b.v_long), (0.0, 3.0, 0.0))

    def test_world_velocity_round_trips(self):
        car = Vehicle()
        car.yaw, car.v_long, car.v_lat = 0.7, 13.0, -4.0
        vx, vz = car.world_velocity()
        car.set_world_velocity(vx, vz)
        self.assertAlmostEqual(car.v_long, 13.0, places=9)
        self.assertAlmostEqual(car.v_lat, -4.0, places=9)


# --------------------------------------------------------------------------- #
class AITests(unittest.TestCase):
    def _lap(self, stage_id, car_id="kestrel", skill=0.8):
        stage, track, terrain = built(stage_id)
        cfg = build_config(CARS_BY_ID[car_id])
        line = RacingLine.build(track, cfg, stage.surface.grip_line)
        car = Vehicle(config=cfg)
        p = track.points[3]
        car.place(float(p[0]), float(p[2]), track.heading_at(3), terrain)
        driver = Driver(line, Personality(skill=skill, consistency=1.0), seed=1)
        driver.reset(3)
        timer = LapTimer(track, GameConfig(laps=1))
        timer.reset(3)
        t, offroad, frames = 0.0, 0, 0
        while t < 200.0:
            car.step(driver.control(car, DT), terrain, DT)
            t += DT
            frames += 1
            offroad += terrain.is_offroad(car.x, car.z)
            if timer.update(car.x, car.z, t) is LapEvent.STAGE_FINISHED:
                return t, offroad / frames
        self.fail(f"AI did not finish a lap of {stage_id}")

    def test_ai_laps_every_stage_on_the_road(self):
        for stage in STAGES:
            with self.subTest(stage=stage.id):
                lap_time, offroad = self._lap(stage.id)
                self.assertLess(offroad, 0.05)
                self.assertLess(lap_time, 100.0)

    def test_skill_makes_the_ai_faster(self):
        slow, _ = self._lap("pine_ridge", skill=0.4)
        fast, _ = self._lap("pine_ridge", skill=1.0)
        self.assertLess(fast, slow * 0.97)

    def test_racing_line_stays_on_the_road(self):
        stage, track, _ = built("dusk_valley")
        line = RacingLine.build(track, build_config(CARS_BY_ID["apex"]), 0.86)
        self.assertLess(np.abs(line.offsets).max(), track.config.road_half_width - 1.0)
        self.assertTrue(np.isfinite(line.speed).all())
        self.assertGreater(line.speed.min(), 5.0)

    def test_racing_line_uses_the_inside_of_corners(self):
        stage, track, _ = built("pine_ridge")
        line = RacingLine.build(track, build_config(CARS_BY_ID["kestrel"]), 1.0)
        corner = track.sharpest_corners(1)[0]
        _, direction = track.curvature_at(corner)
        # Right-hand corner (+1): the apex is to the right (+offset).
        self.assertGreater(line.offsets[corner] * direction, 1.0)


# --------------------------------------------------------------------------- #
class LapTimerGridTests(unittest.TestCase):
    def test_first_crossing_from_the_grid_starts_lap_one(self):
        _, track, _ = built("pine_ridge")
        timer = LapTimer(track, GameConfig(laps=2))
        timer.reset(track.count - 5, behind_line=True)
        self.assertEqual(timer.display_lap, 1)
        self.assertLess(timer.race_distance(), 0.0)
        p = track.points[2]
        self.assertIs(timer.update(float(p[0]), float(p[2]), 1.0), LapEvent.NONE)
        self.assertEqual(timer.lap, 0)
        self.assertEqual(timer.laps, [])


class RaceSessionTests(unittest.TestCase):
    def _race(self, n_ai=4, laps=1):
        stage, track, terrain = built("red_mesa")
        race = RaceSession(stage, track, terrain, laps=laps)
        for i in range(n_ai):
            race.add(f"ai{i}", f"Bot {i}", "ai", "kestrel",
                     personality=Personality(skill=0.5 + 0.1 * i, consistency=1.0), seed=i)
        return race

    def test_grid_slots_are_distinct_and_behind_the_line(self):
        race = self._race(6)
        spots = [(e.vehicle.x, e.vehicle.z) for e in race.entrants.values()]
        for i, a in enumerate(spots):
            for b in spots[i + 1:]:
                self.assertGreater(math.dist(a, b), 3.0)
        for e in race.entrants.values():
            self.assertLess(e.timer.race_distance(), 0.0)

    def test_cars_hold_still_during_the_countdown(self):
        race = self._race(2)
        before = [(e.vehicle.x, e.vehicle.z) for e in race.entrants.values()]
        race.start_countdown()
        for _ in range(75):              # 1.25 s in: clear of the 1 s boundary
            race.update(DT)
        after = [(e.vehicle.x, e.vehicle.z) for e in race.entrants.values()]
        for a, b in zip(before, after):
            self.assertLess(math.dist(a, b), 0.05)
        self.assertEqual(race.countdown_label, "2")

    def test_a_full_race_produces_ordered_results(self):
        race = self._race(4, laps=1)
        race.start_countdown()
        kinds = set()
        while race.phase is not Phase.DONE and race.clock < 300:
            kinds.update(e.kind for e in race.update(DT))
        self.assertIs(race.phase, Phase.DONE)
        self.assertIn("go", kinds)
        self.assertIn("finish", kinds)
        results = race.results()
        self.assertEqual([r.position for r in results], [1, 2, 3, 4])
        times = [r.time for r in results if r.time is not None]
        self.assertEqual(times, sorted(times))
        self.assertEqual(results[0].gap, 0.0)

    def test_player_left_on_the_grid_is_classified_dnf(self):
        stage, track, terrain = built("red_mesa")
        race = RaceSession(stage, track, terrain, laps=1)
        race.add("ai0", "Bot", "ai", "apex", personality=Personality(skill=1.0, consistency=1.0))
        race.add("player", "You", "player", "kestrel")
        race.start_countdown()
        while race.phase is not Phase.DONE and race.clock < 400:
            race.update(DT, {"player": Control()})      # never touches the throttle
        results = {r.id: r for r in race.results()}
        self.assertFalse(results["ai0"].dnf)
        self.assertTrue(results["player"].dnf)
        self.assertEqual(results["player"].position, 2)

    def test_remote_cars_follow_their_reports(self):
        stage, track, terrain = built("red_mesa")
        race = RaceSession(stage, track, terrain, laps=1)
        race.add("p1", "Remote", "remote", "kestrel")
        race.start_countdown()
        race.update(4.0)
        race.apply_remote("p1", {"x": 12.0, "y": 1.0, "z": -4.0, "yaw": 0.5, "vl": 9.0, "vt": 0.0})
        race.update(DT)
        v = race.entrants["p1"].vehicle
        self.assertEqual((v.x, v.z, v.yaw), (12.0, -4.0, 0.5))

    def test_cannot_join_after_the_start(self):
        race = self._race(1)
        race.start_countdown()
        with self.assertRaises(RuntimeError):
            race.add("late", "Late", "ai", "kestrel")


# --------------------------------------------------------------------------- #
def _results(order, dnf=()):
    return [Result(i + 1, rid, rid, "ai", "kestrel", None if rid in dnf else 60.0 + i,
                   55.0, None, rid in dnf) for i, rid in enumerate(order)]


class CareerTests(unittest.TestCase):
    def setUp(self):
        self.career = Career(player_name="Tester")
        self.rivals = [r.id for r in self.career.championship.rivals]

    def test_new_career_starts_in_a_kestrel_with_some_money(self):
        self.assertEqual(self.career.money, STARTING_MONEY)
        self.assertEqual(self.career.current_car, "kestrel")
        self.assertEqual(self.career.tier, 0)

    def test_a_win_pays_points_and_prize(self):
        outcome = self.career.record(_results(["player"] + self.rivals))
        self.assertEqual(outcome.position, 1)
        self.assertEqual(outcome.points, POINTS[0])
        self.assertEqual(outcome.prize, CHAMPIONSHIPS[0].prizes[0])
        self.assertEqual(self.career.money, STARTING_MONEY + outcome.prize)
        self.assertEqual(self.career.wins, 1)
        self.assertEqual(self.career.event_index, 1)

    def test_dnf_pays_nothing(self):
        outcome = self.career.record(_results(self.rivals + ["player"], dnf={"player"}))
        self.assertEqual((outcome.points, outcome.prize), (0, 0))

    def test_winning_the_season_unlocks_the_next_tier(self):
        outcome = None
        for _ in CHAMPIONSHIPS[0].events:
            outcome = self.career.record(_results(["player"] + self.rivals))
        self.assertTrue(outcome.season_complete)
        self.assertEqual(outcome.championship_position, 1)
        self.assertEqual(outcome.unlocked_tier, 1)
        self.assertEqual(self.career.unlocked_tier, 1)
        self.assertEqual(self.career.event_index, 0)
        self.assertEqual(self.career.season, 2)
        self.assertGreater(outcome.title_bonus, 0)
        self.career.choose_tier(1)
        self.assertEqual(self.career.championship.name, "National Series")

    def test_finishing_last_all_season_does_not_unlock(self):
        for _ in CHAMPIONSHIPS[0].events:
            outcome = self.career.record(_results(self.rivals + ["player"]))
        self.assertEqual(self.career.unlocked_tier, 0)
        with self.assertRaises(CareerError):
            self.career.choose_tier(1)

    def test_cannot_switch_championship_mid_season(self):
        self.career.unlocked_tier = 1
        self.career.record(_results(["player"] + self.rivals))
        with self.assertRaises(CareerError):
            self.career.choose_tier(1)

    def test_garage_buying_and_upgrading(self):
        with self.assertRaises(CareerError):
            self.career.buy_car("apex")
        self.career.money = 100_000
        self.career.buy_car("vanta")
        self.assertEqual(self.career.current_car, "vanta")
        price = self.career.next_upgrade_price("vanta", "engine")
        self.career.buy_upgrade("vanta", "engine")
        self.assertEqual(self.career.upgrade_level("vanta", "engine"), 1)
        self.assertEqual(self.career.money, 100_000 - CARS_BY_ID["vanta"].price - price)
        for _ in range(2):
            self.career.buy_upgrade("vanta", "engine")
        self.assertIsNone(self.career.next_upgrade_price("vanta", "engine"))
        with self.assertRaises(CareerError):
            self.career.buy_upgrade("vanta", "engine")
        with self.assertRaises(CareerError):
            self.career.buy_car("vanta")

    def test_save_and_load_round_trip(self):
        self.career.money = 50_000
        self.career.buy_car("ridgeback")
        self.career.buy_upgrade("ridgeback", "tyres")
        self.career.record(_results(["player"] + self.rivals))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "career.json"
            self.career.save(path)
            loaded = Career.load(path)
            self.assertEqual(loaded.to_dict(), self.career.to_dict())
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["career.json"])

    def test_load_rejects_saves_from_the_future(self):
        with self.assertRaises(CareerError):
            Career.from_dict({"version": 999})

    def test_load_repairs_bad_references(self):
        career = Career.from_dict({"version": 1, "current_car": "ghost",
                                   "owned": {"ghost": {}}, "tier": 9, "event_index": 77})
        self.assertEqual(career.current_car, "kestrel")
        self.assertEqual(career.tier, len(CHAMPIONSHIPS) - 1)
        self.assertLess(career.event_index, len(career.championship.events))

    def test_standings_table_lists_everyone(self):
        table = self.career.table()
        self.assertEqual(len(table), 1 + len(self.rivals))


# --------------------------------------------------------------------------- #
class ProtocolTests(unittest.TestCase):
    def test_frames_survive_arbitrary_chunking(self):
        messages = [{"t": "a", "n": i, "pad": "x" * i} for i in range(40)]
        stream = b"".join(encode(m) for m in messages)
        decoder, out = FrameDecoder(), []
        for i in range(0, len(stream), 7):
            out += decoder.feed(stream[i:i + 7])
        self.assertEqual(out, messages)

    def test_oversized_and_garbage_frames_are_rejected(self):
        with self.assertRaises(ProtocolError):
            FrameDecoder().feed(b"\xff\xff\xff\xff")
        with self.assertRaises(ProtocolError):
            FrameDecoder().feed(len(b"nope").to_bytes(4, "big") + b"nope")
        with self.assertRaises(ProtocolError):
            FrameDecoder().feed(len(b"[1]").to_bytes(4, "big") + b"[1]")

    def test_hostile_state_is_coerced(self):
        state = clean_state({"x": float("nan"), "z": 1e12, "yaw": "spin", "vl": -1e9})
        self.assertTrue(all(math.isfinite(v) for v in state.values()))
        self.assertEqual(state["x"], 0.0)
        self.assertEqual(state["z"], 2000.0)
        self.assertEqual(state["vl"], -100.0)
        self.assertIsNone(clean_state("not a dict"))

    def test_names_are_printable_and_short(self):
        self.assertEqual(clean_name("Ann\x00\x07ie"), "Annie")
        self.assertEqual(len(clean_name("x" * 100)), 20)
        self.assertEqual(clean_name(42), "Driver")

    def test_state_of_round_trips_through_cleaning(self):
        car = Vehicle()
        car.x, car.z, car.yaw, car.v_long = 10.5, -3.25, 1.2, 17.0
        cleaned = clean_state(json.loads(json.dumps(state_of(car))))
        self.assertAlmostEqual(cleaned["x"], 10.5)
        self.assertAlmostEqual(cleaned["vl"], 17.0)


class RemoteBufferTests(unittest.TestCase):
    def _state(self, x, yaw=0.0):
        return {"x": x, "y": 0.0, "z": 0.0, "yaw": yaw, "vl": 10.0, "vt": 0.0}

    def test_interpolates_between_snapshots(self):
        buf = RemoteBuffer()
        buf.push(0.0, self._state(0.0))
        buf.push(1.0, self._state(10.0))
        self.assertAlmostEqual(buf.sample(0.25)["x"], 2.5)

    def test_yaw_takes_the_short_way_round(self):
        buf = RemoteBuffer()
        buf.push(0.0, self._state(0.0, yaw=math.pi - 0.1))
        buf.push(1.0, self._state(0.0, yaw=-math.pi + 0.1))
        yaw = buf.sample(0.5)["yaw"]
        self.assertAlmostEqual(abs(math.atan2(math.sin(yaw), math.cos(yaw))), math.pi, places=5)

    def test_extrapolation_is_capped(self):
        buf = RemoteBuffer()
        buf.push(0.0, self._state(0.0))
        # Yaw 0 points along +z; 0.25 s of extrapolation at 10 m/s, then hold.
        self.assertAlmostEqual(buf.sample(10.0)["z"], 2.5)

    def test_out_of_order_snapshots_are_ignored(self):
        buf = RemoteBuffer()
        buf.push(1.0, self._state(10.0))
        buf.push(0.5, self._state(99.0))
        self.assertEqual(len(buf.snapshots), 1)


class LoopbackMultiplayerTests(unittest.TestCase):
    def _pump(self, host, clients, seconds=0.4):
        end = time.monotonic() + seconds
        events = []
        while time.monotonic() < end:
            events += host.poll()
            for c in clients:
                events += c.poll()
            time.sleep(0.01)
        return events

    def setUp(self):
        self.host = HostSession("Host", "apex", port=0)
        self.addCleanup(self.host.close)

    def test_join_ready_start_and_relay(self):
        a = ClientSession("127.0.0.1", self.host.port, "Alice", "torque")
        b = ClientSession("127.0.0.1", self.host.port, "Bob", "vanta")
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        self._pump(self.host, [a, b])
        self.assertEqual((a.status, b.status), ("joined", "joined"))
        self.assertFalse(self.host.can_start)
        a.set_ready(True)
        b.set_ready(True)
        self._pump(self.host, [a, b])
        self.assertTrue(self.host.can_start)

        start = self.host.start_race()
        self.assertEqual(start["grid"][0]["id"], "p0")
        events = self._pump(self.host, [a, b])
        self.assertEqual(sum(e.kind == "start" for e in events), 2)

        a.send_state({"x": 5.0, "y": 0, "z": 0, "yaw": 0, "vl": 0, "vt": 0})
        self._pump(self.host, [a, b], 0.2)
        self.assertIn(a.id, self.host.latest)
        self.host.send_states({}, clock=1.0)
        self._pump(self.host, [a, b], 0.3)
        self.assertIn(a.id, b.remote_states())

    def test_joining_mid_race_is_refused_politely(self):
        self.host.start_race()
        late = ClientSession("127.0.0.1", self.host.port, "Late", "kestrel")
        self._pump(self.host, [late])
        self.assertEqual(late.status, "failed")
        self.assertIn("already running", late.error)

    def test_bots_fill_the_grid_and_respect_the_cap(self):
        self.host.configure(bots=3)
        self.assertEqual(sum(p.is_bot for p in self.host.players.values()), 3)
        self.host.configure(bots=50)
        self.assertLessEqual(len(self.host.players), 8)

    def test_malformed_peer_does_not_take_the_host_down(self):
        raw = socket.create_connection(("127.0.0.1", self.host.port))
        raw.sendall(b"\xff\xff\xff\xffjunk")
        time.sleep(0.2)
        self.host.poll()
        raw.close()
        good = ClientSession("127.0.0.1", self.host.port, "Good", "kestrel")
        self.addCleanup(good.close)
        self._pump(self.host, [good])
        self.assertEqual(good.status, "joined")

    def test_unreachable_host_fails_with_a_readable_reason(self):
        c = ClientSession("127.0.0.1", 1, "Nobody", "kestrel", timeout=1.0)
        for _ in range(200):
            c.poll()
            if c.status != "connecting":
                break
            time.sleep(0.01)
        self.assertEqual(c.status, "failed")
        self.assertIn("Couldn't reach", c.error)


if __name__ == "__main__":
    unittest.main()
