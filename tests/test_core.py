"""Unit tests for the simulation core.

These run headlessly - nothing here imports Ursina, so there is no window and
no GPU involved. Run them with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rally.config import GAME, TRACK, VEHICLE, WORLD, GameConfig
from rally.meshbuild import MeshBuilder, box, cone, cylinder, gradient_sky, rock
from rally.scoring import DriftScorer, LapEvent, LapTimer, format_time
from rally.terrain import Terrain
from rally.track import Track
from rally.vehicle import Control, Vehicle


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FlatSurface:
    """A frictionless-to-reason-about test surface: flat, uniform grip."""

    def __init__(self, grip: float = 1.0, rolling: float = 1.0,
                 height: float = 0.0, limit: float = 10_000.0) -> None:
        self._grip = grip
        self._rolling = rolling
        self._height = height
        self._limit = limit

    def height_at(self, x: float, z: float) -> float:
        return self._height

    def grip_at(self, x: float, z: float) -> float:
        return self._grip

    def rolling_at(self, x: float, z: float) -> float:
        return self._rolling

    def slope_ahead(self, x, z, dx, dz, reach: float = 4.2) -> float:
        return 0.0

    def world_limit(self) -> float:
        return self._limit


def drive(vehicle: Vehicle, surface, control: Control, seconds: float,
          dt: float = 1 / 60) -> None:
    """Run the physics forward for a while."""
    for _ in range(int(seconds / dt)):
        vehicle.step(control, surface, dt)


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #
class TrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.track = Track.build(TRACK)

    def test_sample_count_matches_config(self) -> None:
        self.assertEqual(self.track.count, TRACK.samples)
        self.assertEqual(self.track.points.shape, (TRACK.samples, 3))

    def test_loop_is_closed(self) -> None:
        gap = np.linalg.norm(self.track.points[0] - self.track.points[-1])
        spacing = self.track.length / self.track.count
        self.assertLess(gap, spacing * 1.5)

    def test_samples_are_evenly_spaced(self) -> None:
        """Even spacing is what makes index/count a fair progress measure."""
        closed = np.vstack([self.track.points, self.track.points[0]])
        gaps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.assertLess(gaps.std() / gaps.mean(), 0.02)

    def test_length_is_a_plausible_rally_stage(self) -> None:
        self.assertGreater(self.track.length, 1500.0)
        self.assertLess(self.track.length, 4000.0)

    def test_tangents_are_unit_length(self) -> None:
        norms = np.linalg.norm(self.track.tangents, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_rights_are_horizontal_and_perpendicular(self) -> None:
        np.testing.assert_allclose(self.track.rights[:, 1], 0.0, atol=1e-9)
        horizontal = self.track.tangents.copy()
        horizontal[:, 1] = 0.0
        horizontal /= np.linalg.norm(horizontal, axis=1, keepdims=True)
        dots = np.sum(horizontal * self.track.rights, axis=1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-6)

    def test_nearest_index_finds_its_own_samples(self) -> None:
        for probe in (0, 37, 150, 411, TRACK.samples - 1):
            point = self.track.points[probe]
            found = self.track.nearest_index(float(point[0]), float(point[2]),
                                             hint=probe)
            self.assertEqual(found, probe)

    def test_nearest_index_recovers_without_a_useful_hint(self) -> None:
        """A bad hint must still find the right sample via the full search."""
        probe = 300
        point = self.track.points[probe]
        found = self.track.nearest_index(float(point[0]), float(point[2]), hint=0)
        self.assertEqual(found, probe)

    def test_heading_points_along_the_track(self) -> None:
        for index in (0, 100, 250, 480):
            yaw = self.track.heading_at(index)
            tangent = self.track.tangents[index]
            dot = math.sin(yaw) * tangent[0] + math.cos(yaw) * tangent[2]
            self.assertGreater(dot, 0.99)

    def test_pace_notes_use_rally_grades(self) -> None:
        notes = {self.track.pace_note(i) for i in range(self.track.count)}
        self.assertTrue(notes)
        for note in notes:
            if note == "Flat out":
                continue
            grade, side = note.split()[0], note.split()[1]
            self.assertIn(side, ("left", "right"))
            self.assertIn(int(grade), range(1, 7))

    def test_sharpest_corners_are_separated(self) -> None:
        corners = self.track.sharpest_corners(3, min_separation=24)
        self.assertEqual(len(corners), 3)
        for i, a in enumerate(corners):
            for b in corners[i + 1:]:
                wrapped = min(abs(a - b), self.track.count - abs(a - b))
                self.assertGreaterEqual(wrapped, 24)


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #
class TerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.track = Track.build(TRACK)
        cls.terrain = Terrain(cls.track, WORLD)

    def test_fields_have_grid_shape(self) -> None:
        shape = (WORLD.resolution + 1, WORLD.resolution + 1)
        self.assertEqual(self.terrain.height.shape, shape)
        self.assertEqual(self.terrain.distance.shape, shape)

    def test_distance_is_zero_on_the_centreline(self) -> None:
        """Sampled at a centreline point, distance should be within a cell."""
        point = self.track.points[120]
        d = self.terrain.distance_at(float(point[0]), float(point[2]))
        self.assertLess(d, self.terrain.cell * 1.5)

    def test_road_is_flattened_to_the_centreline_height(self) -> None:
        for index in (0, 90, 210, 400):
            point = self.track.points[index]
            ground = self.terrain.height_at(float(point[0]), float(point[2]))
            self.assertAlmostEqual(ground, float(point[1]), delta=2.0)

    def test_grip_falls_off_away_from_the_line(self) -> None:
        index = 150
        point = self.track.points[index]
        right = self.track.rights[index]

        on_line = self.terrain.grip_at(float(point[0]), float(point[2]))
        verge_x = float(point[0] + right[0] * (TRACK.road_half_width + 1.0))
        verge_z = float(point[2] + right[2] * (TRACK.road_half_width + 1.0))
        verge = self.terrain.grip_at(verge_x, verge_z)
        off_x = float(point[0] + right[0] * (TRACK.road_half_width + 25.0))
        off_z = float(point[2] + right[2] * (TRACK.road_half_width + 25.0))
        offroad = self.terrain.grip_at(off_x, off_z)

        self.assertAlmostEqual(on_line, WORLD.grip_line)
        self.assertGreater(on_line, verge)
        self.assertGreater(verge, offroad)

    def test_offroad_costs_rolling_resistance(self) -> None:
        point = self.track.points[200]
        right = self.track.rights[200]
        self.assertAlmostEqual(
            self.terrain.rolling_at(float(point[0]), float(point[2])), 1.0
        )
        off_x = float(point[0] + right[0] * 40.0)
        off_z = float(point[2] + right[2] * 40.0)
        self.assertAlmostEqual(
            self.terrain.rolling_at(off_x, off_z), WORLD.offroad_rolling
        )

    def test_sampling_outside_the_grid_is_clamped_not_crashing(self) -> None:
        far = WORLD.size  # well outside the grid
        for x, z in ((far, far), (-far, far), (far, -far), (-far, -far)):
            self.assertIsInstance(self.terrain.height_at(x, z), float)

    def test_mesh_is_well_formed(self) -> None:
        mesh = self.terrain.build_mesh()
        res = WORLD.resolution
        self.assertEqual(mesh.vertex_count, (res + 1) ** 2)
        self.assertEqual(mesh.triangle_count, res * res * 2)
        self.assertEqual(len(mesh.colors), mesh.vertex_count)
        self.assertEqual(len(mesh.normals), mesh.vertex_count)
        self.assertTrue((mesh.triangles >= 0).all())
        self.assertTrue((mesh.triangles < mesh.vertex_count).all())
        np.testing.assert_allclose(
            np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-6
        )
        self.assertTrue(((mesh.colors >= 0.0) & (mesh.colors <= 1.0)).all())


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
class VehicleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = FlatSurface()
        self.car = Vehicle()
        self.car.place(0.0, 0.0, 0.0, self.surface)

    def test_throttle_accelerates_forwards(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0), 3.0)
        self.assertGreater(self.car.v_long, 10.0)
        self.assertGreater(self.car.z, 5.0)   # yaw 0 means travelling along +Z

    def test_speed_is_capped(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0), 60.0)
        self.assertLessEqual(self.car.speed, VEHICLE.top_speed + 1e-6)

    def test_rolling_resistance_brings_the_car_to_rest(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0), 3.0)
        drive(self.car, self.surface, Control(), 30.0)
        self.assertAlmostEqual(self.car.v_long, 0.0, delta=0.2)

    def test_brake_from_a_standstill_selects_reverse(self) -> None:
        drive(self.car, self.surface, Control(brake=1.0), 2.0)
        self.assertLess(self.car.v_long, -0.5)
        self.assertGreaterEqual(self.car.v_long, -VEHICLE.reverse_speed)

    def test_steering_turns_the_car(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0), 2.0)
        before = self.car.yaw
        drive(self.car, self.surface, Control(throttle=1.0, steer=1.0), 1.0)
        self.assertGreater(self.car.yaw, before)   # +steer is a right turn

    def test_a_stationary_car_does_not_turn(self) -> None:
        drive(self.car, self.surface, Control(steer=1.0), 2.0)
        self.assertAlmostEqual(self.car.yaw, 0.0, places=6)

    def test_lateral_velocity_decays_under_grip(self) -> None:
        self.car.v_long = 20.0
        self.car.v_lat = 8.0
        drive(self.car, self.surface, Control(), 2.0)
        self.assertLess(abs(self.car.v_lat), 1.0)

    def test_handbrake_lets_the_tail_step_out(self) -> None:
        """The same corner, with and without the handbrake."""
        def corner(handbrake: bool) -> float:
            car = Vehicle()
            car.place(0.0, 0.0, 0.0, self.surface)
            drive(car, self.surface, Control(throttle=1.0), 4.0)
            drive(car, self.surface,
                  Control(throttle=0.6, steer=1.0, handbrake=handbrake), 1.2)
            return car.slip_angle

        self.assertGreater(corner(True), corner(False) * 1.5)

    def test_loose_surface_slides_more_than_grippy_surface(self) -> None:
        def corner(grip: float) -> float:
            surface = FlatSurface(grip=grip)
            car = Vehicle()
            car.place(0.0, 0.0, 0.0, surface)
            drive(car, surface, Control(throttle=1.0), 4.0)
            drive(car, surface, Control(throttle=1.0, steer=1.0), 1.5)
            return car.slip_angle

        self.assertGreater(corner(WORLD.grip_offroad), corner(WORLD.grip_line))

    def test_slip_angle_is_zero_when_tracking_straight(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0), 4.0)
        self.assertLess(self.car.slip_angle, 0.02)

    def test_gears_climb_with_speed(self) -> None:
        self.assertEqual(self.car.gear, 1)
        drive(self.car, self.surface, Control(throttle=1.0), 12.0)
        self.assertGreater(self.car.gear, 2)
        self.assertLessEqual(self.car.gear, len(VEHICLE.gear_ratios))

    def test_rpm_stays_in_range(self) -> None:
        for seconds in (0.5, 4.0, 12.0, 30.0):
            drive(self.car, self.surface, Control(throttle=1.0), seconds)
            self.assertGreaterEqual(self.car.rpm, 0.0)
            self.assertLessEqual(self.car.rpm, 1.1)

    def test_world_boundary_stops_the_car(self) -> None:
        surface = FlatSurface(limit=50.0)
        car = Vehicle()
        car.place(0.0, 0.0, 0.0, surface)
        drive(car, surface, Control(throttle=1.0), 20.0)
        self.assertLessEqual(abs(car.z), 50.0)

    def test_car_follows_the_ground_height(self) -> None:
        surface = FlatSurface(height=12.5)
        car = Vehicle()
        car.place(0.0, 0.0, 0.0, surface)
        drive(car, surface, Control(throttle=1.0), 2.0)
        self.assertAlmostEqual(car.y, 12.5, places=5)

    def test_crest_launches_and_gravity_lands_the_car(self) -> None:
        class Crest(FlatSurface):
            def slope_ahead(self, x, z, dx, dz, reach: float = 4.2) -> float:
                return 0.5 if z < 40.0 else 0.0

        surface = Crest()
        car = Vehicle()
        car.place(0.0, 0.0, 0.0, surface)
        drive(car, surface, Control(throttle=1.0), 3.0)
        self.assertTrue(car.airborne)
        drive(car, surface, Control(throttle=1.0), 4.0)
        self.assertFalse(car.airborne)
        self.assertAlmostEqual(car.y, 0.0, places=5)

    def test_place_resets_momentum(self) -> None:
        drive(self.car, self.surface, Control(throttle=1.0, steer=1.0), 4.0)
        self.car.place(10.0, 20.0, 1.2, self.surface)
        self.assertEqual((self.car.x, self.car.z, self.car.yaw), (10.0, 20.0, 1.2))
        self.assertEqual(self.car.v_long, 0.0)
        self.assertEqual(self.car.v_lat, 0.0)
        self.assertFalse(self.car.airborne)

    def test_physics_is_stable_at_a_long_frame(self) -> None:
        """A dropped frame must not send the car to infinity."""
        for _ in range(200):
            self.car.step(Control(throttle=1.0, steer=1.0), self.surface, 0.05)
        self.assertTrue(math.isfinite(self.car.x))
        self.assertTrue(math.isfinite(self.car.z))
        self.assertLessEqual(self.car.speed, VEHICLE.top_speed + 1e-6)

    def test_rear_axle_points_sit_behind_the_car(self) -> None:
        self.car.yaw = 0.0
        lx, lz, rx, rz = self.car.rear_axle_world()
        self.assertLess(lz, 0.0)
        self.assertLess(rz, 0.0)
        self.assertAlmostEqual(lx, -VEHICLE.track_width, places=5)
        self.assertAlmostEqual(rx, VEHICLE.track_width, places=5)


# --------------------------------------------------------------------------- #
# Lap timing
# --------------------------------------------------------------------------- #
class LapTimerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.track = Track.build(TRACK)

    def setUp(self) -> None:
        self.config = GameConfig(laps=3)
        self.timer = LapTimer(self.track, self.config)
        self.timer.reset(GAME.grid_sample)

    def _lap(self, timer: LapTimer, finish_time: float,
             step: int = 20) -> LapEvent:
        """Walk the car once around the loop, crossing the line at `finish_time`.

        The last index visited is 0, so the wrap - and therefore the lap time -
        lands exactly on `finish_time`.
        """
        indices = list(range(timer.index, self.track.count, step)) + [0]
        event = LapEvent.NONE
        for i, index in enumerate(indices):
            point = self.track.points[index]
            t = finish_time * (i + 1) / len(indices)
            event = timer.update(float(point[0]), float(point[2]), t)
        return event

    def test_one_loop_completes_one_lap(self) -> None:
        event = self._lap(self.timer, 60.0)
        self.assertEqual(event, LapEvent.LAP_COMPLETED)
        self.assertEqual(self.timer.lap, 1)
        self.assertEqual(len(self.timer.laps), 1)

    def test_three_loops_finish_the_stage(self) -> None:
        self.assertEqual(self._lap(self.timer, 60.0), LapEvent.LAP_COMPLETED)
        self.assertEqual(self._lap(self.timer, 120.0), LapEvent.LAP_COMPLETED)
        self.assertEqual(self._lap(self.timer, 180.0), LapEvent.STAGE_FINISHED)
        self.assertTrue(self.timer.finished)
        self.assertEqual(len(self.timer.laps), 3)

    def test_best_lap_tracks_the_quickest(self) -> None:
        self._lap(self.timer, 60.0)      # ~60s
        self._lap(self.timer, 100.0)     # ~40s
        self.assertEqual(self.timer.best, min(self.timer.laps))
        self.assertAlmostEqual(self.timer.best, 40.0, delta=1.0)

    def test_last_lap_is_the_most_recent(self) -> None:
        self._lap(self.timer, 60.0)
        self._lap(self.timer, 100.0)
        self.assertEqual(self.timer.last, self.timer.laps[-1])

    def test_crossing_the_line_backwards_removes_the_lap(self) -> None:
        self._lap(self.timer, 60.0)
        self.assertEqual(self.timer.lap, 1)
        # Reverse back over the line: progress jumps from ~0 up to ~1.
        event = LapEvent.NONE
        for index in (2, 0, self.track.count - 20):
            point = self.track.points[index]
            event = self.timer.update(float(point[0]), float(point[2]), 70.0)
        self.assertEqual(event, LapEvent.WENT_BACKWARDS)
        self.assertEqual(self.timer.lap, 0)
        self.assertEqual(self.timer.laps, [])

    def test_display_lap_is_one_based_and_capped(self) -> None:
        self.assertEqual(self.timer.display_lap, 1)
        self._lap(self.timer, 60.0)
        self.assertEqual(self.timer.display_lap, 2)
        self._lap(self.timer, 120.0)
        self._lap(self.timer, 180.0)
        self.assertEqual(self.timer.display_lap, 3)

    def test_wrong_way_needs_sustained_reversal(self) -> None:
        index = 100
        self.timer.index = index
        backwards = self.track.heading_at(index) + math.pi

        self.assertFalse(self.timer.check_wrong_way(backwards, 20.0, 0.1))
        for _ in range(20):
            flagged = self.timer.check_wrong_way(backwards, 20.0, 0.1)
        self.assertTrue(flagged)

    def test_driving_correctly_clears_the_wrong_way_flag(self) -> None:
        index = 100
        self.timer.index = index
        backwards = self.track.heading_at(index) + math.pi
        for _ in range(20):
            self.timer.check_wrong_way(backwards, 20.0, 0.1)
        forwards = self.track.heading_at(index)
        for _ in range(20):
            flagged = self.timer.check_wrong_way(forwards, 20.0, 0.1)
        self.assertFalse(flagged)

    def test_slow_reversing_is_not_flagged(self) -> None:
        self.timer.index = 100
        backwards = self.track.heading_at(100) + math.pi
        for _ in range(40):
            flagged = self.timer.check_wrong_way(backwards, 1.0, 0.1)
        self.assertFalse(flagged)


# --------------------------------------------------------------------------- #
# Drift scoring
# --------------------------------------------------------------------------- #
class DriftScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = DriftScorer(GAME)

    def _slide(self, seconds: float, slip: float = 0.5, speed: float = 25.0) -> None:
        for _ in range(int(seconds / (1 / 60))):
            self.scorer.update(slip, speed, False, 1 / 60)

    def test_sliding_builds_a_chain(self) -> None:
        self._slide(1.0)
        self.assertGreater(self.scorer.chain, 0.0)
        self.assertTrue(self.scorer.active)

    def test_small_slip_scores_nothing(self) -> None:
        self._slide(2.0, slip=0.05)
        self.assertEqual(self.scorer.chain, 0.0)

    def test_slow_slide_scores_nothing(self) -> None:
        self._slide(2.0, speed=2.0)
        self.assertEqual(self.scorer.chain, 0.0)

    def test_airborne_does_not_score(self) -> None:
        for _ in range(60):
            self.scorer.update(0.5, 25.0, True, 1 / 60)
        self.assertEqual(self.scorer.chain, 0.0)

    def test_multiplier_climbs_while_sliding(self) -> None:
        self._slide(1.0)
        self.assertEqual(self.scorer.multiplier, 1)
        self._slide(2.0)
        self.assertGreaterEqual(self.scorer.multiplier, 2)

    def test_multiplier_is_capped(self) -> None:
        self._slide(60.0)
        self.assertEqual(self.scorer.multiplier, GAME.drift_multiplier_max)

    def test_chain_banks_after_straightening_up(self) -> None:
        self._slide(2.0)
        pending = self.scorer.pending
        self.assertGreater(pending, 0.0)
        for _ in range(240):
            self.scorer.update(0.0, 25.0, False, 1 / 60)
        self.assertEqual(self.scorer.chain, 0.0)
        self.assertAlmostEqual(self.scorer.banked, pending, places=4)
        self.assertEqual(self.scorer.multiplier, 1)

    def test_brief_straightening_keeps_the_chain_alive(self) -> None:
        """Linking two corners should not break the chain."""
        self._slide(2.0)
        chain = self.scorer.chain
        for _ in range(6):
            self.scorer.update(0.0, 25.0, False, 1 / 60)
        self.assertGreater(self.scorer.chain, 0.0)
        self._slide(1.0)
        self.assertGreater(self.scorer.chain, chain)

    def test_faster_slides_score_more(self) -> None:
        slow = DriftScorer(GAME)
        fast = DriftScorer(GAME)
        for _ in range(120):
            slow.update(0.5, 12.0, False, 1 / 60)
            fast.update(0.5, 36.0, False, 1 / 60)
        self.assertGreater(fast.chain, slow.chain)

    def test_crash_out_loses_the_chain(self) -> None:
        self._slide(2.0)
        self.scorer.crash_out()
        self.assertEqual(self.scorer.chain, 0.0)
        self.assertEqual(self.scorer.banked, 0.0)
        self.assertEqual(self.scorer.multiplier, 1)

    def test_reset_clears_everything(self) -> None:
        self._slide(2.0)
        self.scorer.bank()
        self.scorer.reset()
        self.assertEqual(self.scorer.total, 0.0)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
class FormatTimeTests(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertEqual(format_time(0.0), "0:00.00")
        self.assertEqual(format_time(9.5), "0:09.50")
        self.assertEqual(format_time(61.25), "1:01.25")
        self.assertEqual(format_time(125.0), "2:05.00")

    def test_none_is_an_em_dash(self) -> None:
        self.assertEqual(format_time(None), "—")

    def test_seconds_are_always_two_digits(self) -> None:
        """A bare %.2f would render 61.25s as '1:1.25'."""
        self.assertEqual(format_time(61.0).split(":")[1], "01.00")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class MeshBuildTests(unittest.TestCase):
    def _check(self, mesh) -> None:
        self.assertGreater(mesh.vertex_count, 0)
        self.assertGreater(mesh.triangle_count, 0)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.triangles.shape[1], 3)
        self.assertEqual(len(mesh.normals), mesh.vertex_count)
        self.assertEqual(len(mesh.colors), mesh.vertex_count)
        self.assertEqual(len(mesh.uvs), mesh.vertex_count)
        self.assertTrue((mesh.triangles >= 0).all())
        self.assertTrue((mesh.triangles < mesh.vertex_count).all())
        self.assertTrue(np.isfinite(mesh.vertices).all())

    def test_primitives_are_well_formed(self) -> None:
        for mesh in (box(2, 1, 4), cylinder(0.5, 0.5, 2.0, 12),
                     cone(1.0, 3.0, 10), rock(seed=3), gradient_sky(radius=100.0)):
            with self.subTest(mesh=type(mesh).__name__):
                self._check(mesh)

    def test_box_dimensions(self) -> None:
        mesh = box(2.0, 4.0, 6.0)
        extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(extent, [2.0, 4.0, 6.0])

    def test_cone_apex_is_a_single_height(self) -> None:
        mesh = cone(1.0, 4.0, 12)
        self.assertAlmostEqual(float(mesh.vertices[:, 1].max()), 2.0, places=5)

    def test_normals_are_unit_length(self) -> None:
        for mesh in (box(), cylinder(), cone(), rock(seed=1)):
            lengths = np.linalg.norm(mesh.normals, axis=1)
            np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_rock_is_deterministic_per_seed(self) -> None:
        np.testing.assert_array_equal(rock(seed=7).vertices, rock(seed=7).vertices)
        self.assertFalse(np.array_equal(rock(seed=7).vertices, rock(seed=8).vertices))

    def test_builder_merges_and_reindexes(self) -> None:
        builder = MeshBuilder()
        a = box(1, 1, 1)
        builder.add(a, position=(0, 0, 0), color=(1.0, 0.0, 0.0))
        builder.add(a, position=(10, 0, 0), color=(0.0, 1.0, 0.0))
        merged = builder.build()

        self.assertEqual(merged.vertex_count, a.vertex_count * 2)
        self.assertEqual(merged.triangle_count, a.triangle_count * 2)
        self._check(merged)
        # The second copy must reference the second block of vertices.
        self.assertGreaterEqual(merged.triangles[a.triangle_count:].min(),
                                a.vertex_count)

    def test_builder_applies_position_and_scale(self) -> None:
        builder = MeshBuilder()
        builder.add(box(1, 1, 1), position=(5.0, 2.0, -3.0), scale=2.0)
        mesh = builder.build()
        centre = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2.0
        np.testing.assert_allclose(centre, [5.0, 2.0, -3.0], atol=1e-9)
        extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(extent, [2.0, 2.0, 2.0], atol=1e-9)

    def test_builder_rotation_preserves_normal_length(self) -> None:
        builder = MeshBuilder()
        builder.add(box(1, 2, 3), rotation=(0.3, 1.1, -0.7))
        mesh = builder.build()
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0,
                                   atol=1e-9)

    def test_empty_builder_refuses_to_build(self) -> None:
        with self.assertRaises(ValueError):
            MeshBuilder().build()


if __name__ == "__main__":
    unittest.main()


class ResyncTests(unittest.TestCase):
    """A respawn must not be mistaken for a line crossing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.track = Track.build(TRACK)

    def test_resync_does_not_award_a_lap(self) -> None:
        timer = LapTimer(self.track, GameConfig(laps=3))
        timer.reset(GAME.grid_sample)

        # Drive most of the way round, then respawn back near the start.
        for index in range(timer.index, self.track.count, 20):
            point = self.track.points[index]
            timer.update(float(point[0]), float(point[2]), 50.0)
        self.assertEqual(timer.lap, 0)

        timer.resync(5)
        point = self.track.points[7]
        event = timer.update(float(point[0]), float(point[2]), 55.0)
        self.assertEqual(event, LapEvent.NONE)
        self.assertEqual(timer.lap, 0)

    def test_resync_clears_the_wrong_way_timer(self) -> None:
        timer = LapTimer(self.track, GameConfig(laps=3))
        timer.reset(0)
        timer.index = 100
        backwards = self.track.heading_at(100) + math.pi
        for _ in range(20):
            timer.check_wrong_way(backwards, 20.0, 0.1)
        timer.resync(100)
        self.assertFalse(timer.check_wrong_way(backwards, 20.0, 0.1))


class WindingTests(unittest.TestCase):
    """Ursina is left-handed: a front face's right-hand cross product points
    *into* the surface. Anything wound the other way is culled - the ground
    silently disappears, and closed shapes render inside-out."""

    @staticmethod
    def _right_hand(mesh):
        v, t = mesh.vertices, mesh.triangles
        return np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])

    def _assert_front_faces(self, mesh, outward):
        rh = self._right_hand(mesh)
        dots = np.sum(rh * outward, axis=1)
        # Ignore degenerate slivers (cone apexes and the like).
        live = np.linalg.norm(rh, axis=1) > 1e-9
        self.assertTrue((dots[live] < 0).all(),
                        f"{int((dots[live] >= 0).sum())} faces wound the wrong way")

    def test_closed_primitives_face_outward(self) -> None:
        for name, mesh in (("box", box(1, 2, 3)), ("cylinder", cylinder()),
                           ("cone", cone()), ("rock", rock(seed=4))):
            with self.subTest(primitive=name):
                t = mesh.triangles
                centroid = mesh.vertices[t].mean(axis=1)
                # For convex shapes centred on the origin, outward is away from it.
                self._assert_front_faces(mesh, centroid)

    def test_ground_faces_up(self) -> None:
        mesh = Terrain(Track.build(TRACK), WORLD).build_mesh()
        up = np.tile([0.0, 1.0, 0.0], (mesh.triangle_count, 1))
        self._assert_front_faces(mesh, up)
