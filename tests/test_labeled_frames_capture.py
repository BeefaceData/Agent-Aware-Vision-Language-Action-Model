"""Labelled frame capture: public frames plus separate simulator labels; never starts RoboTwin."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from development.capture.labeled_frames import mug_state, roll_out, tilt_deg
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from tests.helpers import images

UPRIGHT = [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]
ON_ITS_SIDE = [0.0, 0.0, 0.75, 0.7071068, 0.7071068, 0.0, 0.0]  # 90 degrees about x


def pose(values):
    return SimpleNamespace(p=values[:3], q=values[3:])


class FakeTask:
    def __init__(self):
        self.mug_pose = list(UPRIGHT)
        self.success = False
        self.mug = SimpleNamespace(get_pose=lambda: pose(self.mug_pose),
                                   get_functional_point=lambda index: self.mug_pose)
        self.rack = SimpleNamespace(actor=SimpleNamespace(name="rack"), get_pose=lambda: pose(UPRIGHT),
                                    get_functional_point=lambda index: UPRIGHT)
        self.middle_pos = [0.0, -0.15, 0.75, 1, 0, 0, 0]
        self.robot = SimpleNamespace(get_left_gripper_val=lambda: 1.0, get_right_gripper_val=lambda: 0.5)

    def check_success(self):
        return self.success


class FakeEnvironment:
    """Moves the fake mug up as actions run; reports success after ``success_after`` actions."""

    def __init__(self, task, success_after=None):
        self.task, self.success_after, self.actions = task, success_after, 0

    def _observation(self):
        return Observation(self.actions, images(self.actions % 255), tuple([0.0] * 14))

    def reset(self, seed):
        return self._observation()

    def step(self, action):
        self.actions += 1
        self.task.mug_pose[2] = 0.75 + 0.001 * self.actions
        self.task.success = self.success_after is not None and self.actions >= self.success_after
        return self._observation()

    @property
    def native_success(self):
        return self.task.success


class FakePolicy:
    def __init__(self):
        self.resets = 0

    def reset_history(self):
        self.resets += 1

    def infer(self, observation, requested_actions):
        return np.zeros((requested_actions, 14))


def run(task, environment, **limits):
    directory = tempfile.TemporaryDirectory()
    output = Path(directory.name)
    store = ObservationStore(output / "observations")
    policy = FakePolicy()
    result = roll_out(environment, task, policy, store, output / "labels.jsonl", seed=3,
                      contacts=lambda task: {"table"}, **limits)
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    return directory, output, result, labels, policy


def label(**overrides):
    base = {"native_success": False, "lifted_before": False, "mug": {"tilt_deg": 0.0},
            "contacts": {"left_gripper": False, "right_gripper": False, "table": True, "rack": False}}
    base["contacts"] = {**base["contacts"], **overrides.pop("contacts", {})}
    return {**base, **overrides}


class RollOutTests(unittest.TestCase):
    def test_saves_reset_and_every_nth_action_with_a_matching_label(self):
        task = FakeTask()
        directory, output, (frames, actions, stop), labels, policy = run(
            task, FakeEnvironment(task), every=4, max_actions=10, chunk_size=3)
        directory.cleanup()

        self.assertEqual((frames, actions, stop), (4, 10, "action_limit"))
        self.assertEqual([item["policy_actions"] for item in labels], [0, 4, 8, 10])
        self.assertEqual(policy.resets, 1)

    def test_labels_point_at_saved_public_observations(self):
        task = FakeTask()
        directory, output, _result, labels, _policy = run(task, FakeEnvironment(task), every=5,
                                                          max_actions=5, chunk_size=5)
        try:
            for item in labels:
                self.assertTrue((output / item["observation"]).is_file())
            with np.load(output / labels[-1]["observation"]) as archive:
                self.assertNotIn("mug", " ".join(archive.files))  # no simulator truth in public frames
        finally:
            directory.cleanup()

    def test_stops_and_records_the_final_frame_on_native_success(self):
        task = FakeTask()
        directory, _output, (frames, actions, stop), labels, _policy = run(
            task, FakeEnvironment(task, success_after=6), every=4, max_actions=50, chunk_size=4)
        directory.cleanup()

        self.assertEqual((actions, stop), (6, "native_success"))
        self.assertEqual(labels[-1]["policy_actions"], 6)
        self.assertEqual(labels[-1]["state"], "hung")

    def test_lift_is_remembered_for_later_frames(self):
        task = FakeTask()
        directory, _output, _result, labels, _policy = run(task, FakeEnvironment(task), every=20,
                                                           max_actions=60, chunk_size=20)
        directory.cleanup()

        self.assertEqual([item["lifted_before"] for item in labels], [False, False, True, True])
        self.assertEqual(labels[-1]["state"], "staged")


class MugStateTests(unittest.TestCase):
    def test_resting_before_any_lift(self):
        self.assertEqual(mug_state(label()), "resting")

    def test_back_on_the_table_after_a_lift_is_staged(self):
        self.assertEqual(mug_state(label(lifted_before=True)), "staged")

    def test_a_mug_on_its_side_is_toppled_even_while_touching_the_table(self):
        self.assertEqual(mug_state(label(lifted_before=True, mug={"tilt_deg": 90.0})), "toppled")

    def test_gripper_contacts(self):
        self.assertEqual(mug_state(label(contacts={"left_gripper": True, "table": False})), "held_left")
        self.assertEqual(mug_state(label(contacts={"right_gripper": True, "table": False})), "held_right")
        self.assertEqual(mug_state(label(contacts={"left_gripper": True, "right_gripper": True})), "held_both")
        self.assertEqual(mug_state(label(contacts={"right_gripper": True})), "gripped_on_table")


class TiltTests(unittest.TestCase):
    def test_upright_mug_has_no_tilt(self):
        self.assertAlmostEqual(tilt_deg(UPRIGHT, UPRIGHT), 0.0, places=5)

    def test_quarter_turn_onto_its_side(self):
        self.assertAlmostEqual(tilt_deg(UPRIGHT, ON_ITS_SIDE), 90.0, places=3)

    def test_tilt_is_relative_to_the_reset_orientation(self):
        self.assertAlmostEqual(tilt_deg(ON_ITS_SIDE, ON_ITS_SIDE), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
