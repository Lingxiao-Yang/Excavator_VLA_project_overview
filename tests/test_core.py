"""Smoke tests for the pure-Python parts of the demo package (numpy only)."""

import math
import socket
import threading
import unittest

import numpy as np

from excavator_vla import contract as C
from excavator_vla import protocol as P
from excavator_vla.evaluation.chunk_stitch import stitch, sweep
from excavator_vla.expert.quality import score_scoop
from excavator_vla.kinematics import ExcavatorGeometry, forward_planar, solve_bucket_ik
from excavator_vla.policy.ensemble import ChunkHistory, needs_replan
from excavator_vla.policy.supervisor import ExcavationSupervisor
from excavator_vla.randomization import area_uniform_radius, sample_scene, stable_scene_seed


class ContractTest(unittest.TestCase):
    def test_dims(self):
        self.assertEqual(len(C.STATE_NAMES), 28)
        self.assertEqual(len(set(C.STATE_NAMES)), 28)
        self.assertEqual(len(C.STAGE_NAMES), 10)
        self.assertEqual(C.STAGE_ID["curl_to_hold_material"], 4)   # canonical order, not the swapped one

    def test_validate_fails_closed(self):
        with self.assertRaises(ValueError):
            C.validate_feature_names("state", C.STATE_NAMES, C.STATE_NAMES[:-1])


class ProtocolTest(unittest.TestCase):
    def test_tick_scheduler_exact_and_drift_free(self):
        s = P.PhysicsTickScheduler(10.0)
        self.assertEqual({s.next_ticks() for _ in range(10)}, {6})
        s7 = P.PhysicsTickScheduler(7.0)
        self.assertEqual(sum(s7.next_ticks() for _ in range(70)), 600)

    def test_rgb_roundtrip_over_socket(self):
        img = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
        a, b = socket.socketpair()
        t = threading.Thread(target=P.send_msg, args=(a, {"cameras": {"0": P.encode_rgb(img)}}))
        t.start()
        msg = P.recv_msg(b)
        t.join()
        a.close(); b.close()
        np.testing.assert_array_equal(P.decode_rgb(msg["cameras"]["0"]), img)

    def test_handshake_and_legacy_rejection(self):
        P.validate_handshake(P.build_handshake(10.0, "abc"))
        with self.assertRaises(ValueError):
            P.validate_step_request({"joint_velocities": [0, 0, 0, 0], "ticks": 6})

    def test_dof_resolution_by_name(self):
        self.assertEqual(P.resolve_canonical_dof_indices(["bucket", "arm", "boom", "swing"]), [3, 2, 1, 0])


class KinematicsTest(unittest.TestCase):
    def test_ik_reaches_target(self):
        geo = ExcavatorGeometry()
        target = np.array([0.0, 6.5, 0.4])
        sol = solve_bucket_ik(geo, target, np.radians([90, 40, -60, -20]), range(-140, 41, 5))
        self.assertIsNotNone(sol)
        swing, boom, arm, bucket = sol.q
        a1, a2 = boom, boom + arm
        tip = forward_planar(geo, a1, a2, a2 + bucket)["tip"]
        self.assertAlmostEqual(swing, math.atan2(target[1], target[0]), places=6)
        np.testing.assert_allclose(tip, [math.hypot(*target[:2]), target[2]], atol=1e-6)


class RandomizationTest(unittest.TestCase):
    def test_seed_depends_on_worker(self):
        self.assertNotEqual(stable_scene_seed("run", 0, worker_identity="w00"),
                            stable_scene_seed("run", 0, worker_identity="w01"))
        self.assertEqual(sample_scene("run", 3, "w01"), sample_scene("run", 3, "w01"))

    def test_area_uniform_radius_bounds(self):
        self.assertAlmostEqual(area_uniform_radius(0.0, (5.8, 7.35)), 5.8)
        self.assertAlmostEqual(area_uniform_radius(1.0, (5.8, 7.35)), 7.35)


class QualityTest(unittest.TestCase):
    def test_good_and_bad_scoops(self):
        self.assertTrue(score_scoop(40, 35, 2, 30, True, 0).accepted)
        bad = score_scoop(40, 35, 2, 1, True, 0)
        self.assertFalse(bad.accepted)
        self.assertTrue(any(f.startswith("low_bin_gain") for f in bad.failures))


class PolicyTest(unittest.TestCase):
    def test_ensemble_width1_is_plain_indexing(self):
        h = ChunkHistory(width=1)
        chunk = np.arange(200, dtype=np.float32).reshape(50, 4)
        h.add(10, chunk)
        np.testing.assert_array_equal(h.action_at(12), chunk[2])

    def test_ensemble_blends_overlapping_chunks(self):
        h = ChunkHistory(width=2, decay=0.0)
        h.add(0, np.zeros((50, 4)))
        h.add(3, np.ones((50, 4)))
        np.testing.assert_allclose(h.action_at(5), 0.5)

    def test_replan_schedule(self):
        self.assertEqual([s for s in range(10) if needs_replan(s, 3, 0, 50)], [0, 3, 6, 9])

    def test_supervisor_is_monotonic(self):
        sup = ExcavationSupervisor()
        obs = {"q": [0, 0.5, -1.0, -2.1], "tip_above_dig_target": 0.2, "load_above_unload": -3.0,
               "fill_fraction": 0.14}
        sup(np.zeros(4), obs, 3)          # dig -> curl_secure (tip close to target)
        sup(np.zeros(4), obs, 4)          # curl done + loaded -> lift_loaded
        self.assertEqual(sup.phase, "lift_loaded")
        obs["tip_above_dig_target"] = 5.0
        sup(np.zeros(4), obs, 0)          # predicted stage 0 must not rewind the phase
        self.assertNotEqual(sup.phase, "dig")


class EvaluationTest(unittest.TestCase):
    def test_chunk1_matches_per_frame(self):
        rng = np.random.default_rng(0)
        pred, true = rng.normal(size=(20, 50, 4)), rng.normal(size=(20, 50, 4))
        p, t = stitch(pred, true, 1)
        np.testing.assert_array_equal(p, pred[:, 0])
        self.assertEqual(set(sweep(pred, true, np.full(4, 0.5))), {1, 2, 3, 5, 10, 15, 20, 30, 50})


if __name__ == "__main__":
    unittest.main()
