import json
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from development.hanging_check import observe_hanging
from development.episode_demo import run_demo
from development.fresh_episode_demo import run_fresh, demonstration_boundary
from development.rack_geometry import rack_diagnostics
from tests import test_checkpoints as fixtures
from tests.test_baseline_compatibility import FakeRuntime
from tests.test_rack_recovery import RackTask
from tests.helpers import images


class HangingCheckTests(unittest.TestCase):
    def task_and_sample(self):
        task = SimpleNamespace(rack=SimpleNamespace(actor=SimpleNamespace(name="rack")),
                               scene=SimpleNamespace(step=lambda: None))
        sample = {"task_success": True, "right_gripper": 1.0,
                  "mug_contact_entities": ["rack"], "mug_position": [0.2, 0.15, 0.85]}
        return task, sample

    def test_requires_full_opening_rack_support_no_robot_and_stability(self):
        task, sample = self.task_and_sample()
        for change, expected in (({}, True), ({"right_gripper": 0.81}, False),
            ({"mug_contact_entities": ["rack", "fr_link8"]}, False),
            ({"mug_contact_entities": ["rack", "right_camera"]}, False),
            ({"mug_contact_entities": ["table"]}, False), ({"task_success": False}, False)):
            with patch("development.hanging_check.rack_diagnostics", return_value={**sample, **change}):
                r = observe_hanging(task, log=lambda e: None, ticks=3, window=3)
                self.assertEqual(r["stable_hanging"], expected)
                self.assertFalse(r["actuator_commands_written"])

    def test_motion_and_incomplete_stable_window_fail(self):
        task, sample = self.task_and_sample()
        moving = [sample, {**sample, "mug_position": [0.22, 0.15, 0.85]}, sample]
        with patch("development.hanging_check.rack_diagnostics", side_effect=moving):
            self.assertFalse(observe_hanging(task, log=lambda e: None, ticks=3, window=3)["stable_hanging"])
        transient = [sample, {**sample, "task_success": False}, sample]
        with patch("development.hanging_check.rack_diagnostics", side_effect=transient):
            self.assertFalse(observe_hanging(task, log=lambda e: None, ticks=3, window=3)["stable_hanging"])
        with self.assertRaises(ValueError): observe_hanging(task, log=lambda e: None, ticks=1, window=2)


class EpisodeDemoTests(unittest.TestCase):
    def setUp(self):
        from tests.skill_fixtures import isolate_simulator_dependencies
        isolate_simulator_dependencies(self)

    def test_fresh_policy_interruption_discards_chunk_before_recovery_without_reset(self):
        fixture = fixtures.CheckpointTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        runtime = FakeRuntime()
        task = RackTask(); task.position = np.array([0.1, 0.1, 0.88])
        class Simulator:
            def __call__(self, *args): return runtime.sim(*args)
        sim = Simulator(); sim.task = task
        self.assertTrue(demonstration_boundary(rack_diagnostics(task)))
        task.contacts.append("040_rack")
        self.assertFalse(demonstration_boundary(rack_diagnostics(task)))
        task.contacts.pop()
        video = SimpleNamespace(append=lambda *args, **kwargs: None, close=lambda: None, frames=0)
        def finish(simulator, skill, parameters, output, writer, report):
            self.assertIs(simulator, sim)
            self.assertEqual(runtime.actions, [])
            report["status"] = "stable_hanging"
            return report
        with patch("development.fresh_episode_demo.DemoVideo", return_value=video), \
             patch("development.fresh_episode_demo.finish_demo", side_effect=finish), \
             contextlib.redirect_stdout(io.StringIO()):
            report = run_fresh(runtime, sim, fixture.root / "fresh", steps=5, chunk_size=3)
        self.assertEqual(report["modal_reset_calls"], 1)
        self.assertEqual(report["modal_chunk_calls"], 1)
        self.assertEqual(report["confirmed_policy_actions"], 0)
        self.assertEqual(report["policy_source"], "fresh Modal inference")
        traces = list((fixture.root / "fresh/traces").rglob("trace.jsonl"))
        events = [json.loads(s) for s in traces[0].read_text().splitlines()]
        self.assertEqual(events[-1]["event_type"], "chunk_discarded")
        self.assertNotIn("privileged", traces[0].read_text())

    def test_records_prefix_recovery_and_check_in_one_episode_without_policy(self):
        fixture = fixtures.CheckpointTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        cp, _, _ = fixture.capture()
        cp = fixture.bank.relabel(cp.checkpoint_id, phase="approach_rack", failure_type="bad_rack_approach")
        runtime = FakeRuntime()
        class Simulator:
            task = RackTask()
            blank = np.zeros((2, 3, 3), dtype=np.uint8)
            def __call__(self, *args): return runtime.sim(*args)
        labels = []
        class Video:
            frames = 0
            def __init__(self, path): pass
            def append(self, rgb, title, detail, repeat=1):
                labels.append(title); self.frames += repeat
            def close(self): pass
        with patch("development.episode_demo.DemoVideo", Video), \
             patch("run_inference_chunked.observation_of", return_value=images()), \
             patch("development.episode_demo.observe_hanging", return_value={"stable_hanging": True}), \
             patch("run_inference_chunked.ModalPolicyClient", side_effect=AssertionError("no policy calls")):
            r = run_demo(fixture.bank, cp, Simulator(), fixture.root / "demo")
        self.assertEqual(r["status"], "stable_hanging")
        self.assertEqual(len(runtime.actions), cp.step)
        self.assertEqual(labels[:cp.step+1], ["RECORDED POLICY REPLAY"]*(cp.step+1))
        self.assertIn("DETERMINISTIC RECOVERY", labels)
        self.assertFalse(r["runtime_eligible"])
        self.assertFalse(r["automatic_detection"])
        self.assertEqual(json.loads((fixture.root / "demo/report.json").read_text())["status"], "stable_hanging")
