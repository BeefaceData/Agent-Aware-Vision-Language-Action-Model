import json
from types import SimpleNamespace
import unittest

import numpy as np

from development.hanging.prepare_replay import prepare_replay
from development.hanging.validate_entry660 import load_candidate, validate_entry
from tests import test_hanging_replay_preparation as fixtures


class Entry660Tests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.HangingReplayPreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        summary = fixture.original / "summary.json"
        data = json.loads(summary.read_text())
        data["policy_actions"] = 660
        summary.write_text(json.dumps(data))
        fixture.rows = [row for i in range(1, 661) for row in (
            dict(event="action_requested", index=i, sequence=i-1, action=fixture.state),
            dict(event="action_confirmed", index=i, sequence=i, command_state=fixture.state))]
        fixture.write_journal()
        from harness.environment.layout import CAMERAS
        np.savez(fixture.original / "observations/observation_00000660.npz",
                 **{camera: np.zeros((4, 4, 3), dtype=np.uint8) for camera in CAMERAS},
                 sequence=660, command_state=fixture.state)
        prepare_replay(fixture.original, fixture.recovered, fixture.output)
        self.path = fixture.output
        _, self.frames, self.boundary = load_candidate(self.path, allow_unverified=True)
        self.calls = []

    def environment(self, *, drift=False, fail=False, image_drift=False):
        def reset(seed):
            self.calls.append(("reset", seed))
            return SimpleNamespace(sequence=0, command_state=self.frames[0].command_state)
        def step(action):
            index = len(self.calls)
            self.calls.append(("step", action))
            if fail:
                raise RuntimeError("unconfirmed_action")
            state = (1.,)*14 if drift else self.frames[index].command_state
            images = {camera: image + (100 if image_drift else 0)
                      for camera, image in self.boundary.images.items()}
            return SimpleNamespace(sequence=index, command_state=state,
                                   images=images, measured_tcp_poses=None)
        return SimpleNamespace(reset=reset, step=step)

    def test_exact_prefix_without_grasp_or_full_state_claim(self):
        report = validate_entry(self.path, self.environment(), allow_unverified=True)
        self.assertEqual(report["executed_steps"], 660)
        self.assertEqual(len(self.calls), 661)
        self.assertEqual(report["status"], "observations_agree_unverified")
        self.assertEqual(report["grasp_validity"], "unknown")
        self.assertFalse(report["full_state_restored"])

    def test_opt_in_and_tampered_assets_precede_reset(self):
        with self.assertRaises(ValueError):
            validate_entry(self.path, self.environment())
        (self.path / "prefix.jsonl").write_text("tampered")
        with self.assertRaises(ValueError):
            validate_entry(self.path, self.environment(), allow_unverified=True)
        self.assertEqual(self.calls, [])

    def test_drift_and_unconfirmed_action_stop_without_retry(self):
        for mode in ("drift", "fail"):
            with self.subTest(mode=mode):
                self.calls.clear()
                report = validate_entry(self.path, self.environment(**{mode: True}), allow_unverified=True)
                self.assertEqual(len(self.calls), 2)
                self.assertEqual(report["status"], "execution_error" if mode == "fail" else "command_state_drift")
                self.assertEqual(report["execution_uncertain"], mode == "fail")

    def test_time_budget_stops_before_reset(self):
        times = iter((0., 2.))
        report = validate_entry(self.path, self.environment(), allow_unverified=True,
                                time_limit_seconds=1, clock=lambda: next(times))
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual(self.calls, [])

    def test_image_drift_is_separate_from_command_state_agreement(self):
        report = validate_entry(self.path, self.environment(image_drift=True), allow_unverified=True)
        self.assertEqual(report["executed_steps"], 660)
        self.assertEqual(report["command_state_max_abs_error"], 0.)
        self.assertEqual(report["status"], "image_drift")
