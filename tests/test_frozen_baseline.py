"""CPU rollout tests; no simulator or provider calls."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from development.hanging.baseline import rollout, validate_options, run


class Environment:
    def __init__(self, success_after=None, fail=False):
        self.count = 0
        self.success_after = success_after
        self.fail = fail

    @property
    def native_success(self):
        return self.success_after is not None and self.count >= self.success_after

    def reset(self, seed):
        return self.observation()

    def observation(self):
        return SimpleNamespace(sequence=self.count, command_state=(0.,)*14)

    def step(self, action):
        self.count += 1
        if self.fail:
            raise RuntimeError("response lost")
        return self.observation()


class Policy:
    def __init__(self, bad=False):
        self.requests = []
        self.bad = bad

    def reset_history(self):
        pass

    def infer(self, observation, count):
        self.requests.append(count)
        chunk = np.zeros((count, 14))
        if self.bad:
            chunk[-1, 0] = np.nan
        return chunk


class Artifacts:
    def __init__(self):
        self.sequences = []
        self.closed = False

    def save(self, observation):
        self.sequences.append(observation.sequence)

    def append(self, *args):
        pass

    def close(self):
        self.closed = True


class BaselineTests(unittest.TestCase):
    def test_invalid_camera_rejected_before_output_or_simulator_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "invalid_perception_camera_spec"):
                run("unused", output, seed=3, max_actions=8, chunk_size=8,
                    time_limit_minutes=1, perception_camera="bad")
            self.assertFalse(output.exists())

    def test_optional_camera_wired_to_recording_and_manifest(self):
        from harness.environment.layout import CAMERAS
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with patch("development.simulator.local_simulator.LocalSimulator") as simulator, \
                 patch("development.demos.episode_video.EpisodeVideo") as video, \
                 patch("harness.environment.local.LocalEnvironmentClient"), \
                 patch("harness.vla.modal_client.ModalVLAClient"), \
                 patch("development.hanging.baseline.rollout", return_value={"outcome": "action_limit"}):
                run("unused", output, seed=3, max_actions=8, chunk_size=8,
                    time_limit_minutes=1, perception_camera="default")
            self.assertEqual(simulator.call_args.kwargs["perception_cameras"][0].name,
                             "perception_camera")
            self.assertEqual(video.call_args.kwargs["cameras"], (*CAMERAS, "perception_camera"))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["policy_cameras"], list(CAMERAS))
            self.assertEqual(manifest["perception_camera"]["name"], "perception_camera")

    def execute(self, environment, policy, *, deadline=100.):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Artifacts()
            report = rollout(environment, policy, artifacts, artifacts, tmp,
                seed=3, max_actions=5, chunk_size=3, deadline=deadline, clock=lambda: 0.)
            self.assertTrue(artifacts.closed)
            self.assertEqual(json.loads((Path(tmp)/"summary.json").read_text()), report)
            return report, artifacts

    def test_budget_requests_short_final_chunk_and_records_all_actions(self):
        policy = Policy()
        report, artifacts = self.execute(Environment(), policy)
        self.assertEqual(policy.requests, [3, 2])
        self.assertEqual(report["policy_actions"], 5)
        self.assertEqual(artifacts.sequences, list(range(6)))
        self.assertEqual(report["planner_calls"], 0)

    def test_success_discards_rest_of_chunk(self):
        report, _ = self.execute(Environment(success_after=2), Policy())
        self.assertEqual(report["policy_actions"], 2)
        self.assertEqual(report["outcome"], "native_success")

    def test_success_on_last_action_is_retained(self):
        report, _ = self.execute(Environment(success_after=5), Policy())
        self.assertTrue(report["native_success"])

    def test_entire_chunk_is_validated_before_motion(self):
        env = Environment()
        report, _ = self.execute(env, Policy(bad=True))
        self.assertEqual(env.count, 0)
        self.assertEqual(report["outcome"], "execution_error")

    def test_lost_response_is_uncertain_and_not_retried(self):
        env = Environment(fail=True)
        report, _ = self.execute(env, Policy())
        self.assertEqual(env.count, 1)
        self.assertEqual(report["policy_actions"], 0)
        self.assertTrue(report["execution_uncertain"])

    def test_deadline_stops_before_policy_request(self):
        policy = Policy()
        report, _ = self.execute(Environment(), policy, deadline=0.)
        self.assertEqual(report["outcome"], "time_limit")
        self.assertEqual(policy.requests, [])

    def test_invalid_budgets_rejected(self):
        for values in ((-1, 5, 8, 30), (3, 901, 8, 30),
                       (3, 5, 25, 30), (3, 5, 8, 0), (True, 5, 8, 30)):
            with self.assertRaises(ValueError):
                validate_options(*values)
