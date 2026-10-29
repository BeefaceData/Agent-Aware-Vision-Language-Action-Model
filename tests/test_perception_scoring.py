"""Offline perception scoring: truth rules, metrics and the replay loop; no simulator or Modal."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from development.replays.perception_evaluation import (
    EVALUATED_PREDICATES, detector_phase, evaluate_capture, summarize, truth_predicates)
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from tests.helpers import images


def label(state="resting", *, tilt=0.0, height=0.0, lifted_before=False, left=False, right=False,
          table=True, sequence=0):
    return {"sequence": sequence, "policy_actions": sequence, "state": state, "lifted_before": lifted_before,
            "native_success": False,
            "mug": {"pose": [0.1, 0.2, 0.75, 1, 0, 0, 0], "tilt_deg": tilt, "height_change_m": height},
            "contacts": {"left_gripper": left, "right_gripper": right, "table": table, "rack": False},
            "observation": f"observations/observation_{sequence:08d}.npz"}


class TruthTests(unittest.TestCase):
    def test_resting_mug_is_supported_but_not_staged_or_lifted(self):
        truth = truth_predicates(label())

        self.assertEqual((truth["release_ready"], truth["mug_staged_on_table"], truth["mug_lifted"]),
                         (True, False, False))

    def test_back_on_the_table_after_a_lift_is_staged(self):
        self.assertTrue(truth_predicates(label("staged", lifted_before=True))["mug_staged_on_table"])

    def test_a_toppled_mug_is_not_release_ready(self):
        truth = truth_predicates(label("toppled", tilt=90.0, lifted_before=True))

        self.assertFalse(truth["release_ready"])
        self.assertFalse(truth["mug_staged_on_table"])

    def test_right_mug_lifted_needs_the_right_gripper_alone_off_the_table(self):
        held_right = label("held_right", right=True, table=False, height=0.05, lifted_before=True)
        held_both = label("held_both", left=True, right=True, table=False, height=0.05, lifted_before=True)

        self.assertTrue(truth_predicates(held_right)["right_mug_lifted"])
        self.assertFalse(truth_predicates(held_both)["right_mug_lifted"])

    def test_right_mug_lifted_stays_true_after_the_milestone(self):
        # The planner keeps HANG once the right arm has lifted the mug, even if
        # the mug is later put back down.
        back_on_table = label("staged", lifted_before=True)

        self.assertFalse(truth_predicates(back_on_table)["right_mug_lifted"])
        self.assertTrue(truth_predicates(back_on_table, right_lifted_before=True)["right_mug_lifted"])


class PhaseTests(unittest.TestCase):
    def test_phase_follows_the_runtime_milestones(self):
        self.assertEqual(detector_phase(label()), "grasp")
        self.assertEqual(detector_phase(label("held_left", left=True, table=False, height=0.05)), "handoff")
        self.assertEqual(detector_phase(label("held_right", right=True, lifted_before=True)), "hang")


def record(state, *, grounded=True, camera="head_camera", error=0.04, answers=None, truths=None):
    answers, truths = answers or {}, truths or {}
    return {"capture": "seed_0", "state": state, "mug_status": "grounded" if grounded else "not_detected",
            "mug_camera": camera if grounded else None,
            "mug_xy_error_m": error if grounded else None, "mug_z_offset_m": 0.0 if grounded else None,
            "predicates": {name: {"public": answers.get(name), "truth": truths.get(name, False), "reason": "r"}
                           for name in EVALUATED_PREDICATES}}


class SummaryTests(unittest.TestCase):
    def test_detection_rate_and_cameras_per_state(self):
        summary = summarize([record("held_left"), record("held_left", camera="left_camera"),
                             record("held_left", grounded=False)])

        row = summary["by_state"]["held_left"]
        self.assertEqual((row["frames"], row["mug_grounded"], row["mug_detection_rate"]), (3, 2, 0.667))
        self.assertEqual(row["mug_camera"], {"head_camera": 1, "left_camera": 1})
        self.assertEqual(row["mug_miss_reasons"], {"not_detected": 1})

    def test_precision_recall_unknown_and_false_true(self):
        rows = [record("staged", answers={"release_ready": True}, truths={"release_ready": True}),
                record("staged", answers={"release_ready": None}, truths={"release_ready": True}),
                record("toppled", answers={"release_ready": True}, truths={"release_ready": False}),
                record("held_left", answers={"release_ready": False}, truths={"release_ready": False})]

        scores = summarize(rows)["predicates"]["release_ready"]

        self.assertEqual((scores["precision"], scores["recall"]), (0.5, 0.5))
        self.assertEqual((scores["unknown_rate"], scores["false_true"]), (0.25, 1))
        self.assertEqual(summarize(rows)["toppled_reported_as_supported"],
                         {"toppled_frames": 1, "reported_supported": 1})

    def test_rates_are_none_without_cases(self):
        scores = summarize([record("resting")])["predicates"]["right_mug_lifted"]

        self.assertIsNone(scores["precision"])
        self.assertIsNone(scores["recall"])


class FakeProvider:
    """Answers like the runtime provider; records the phases it is told and what it sees."""

    def __init__(self):
        self.phases, self.seen = [], []

    def set_task_phase(self, phase):
        self.phases.append(phase)

    def observe(self, observation):
        self.seen.append(observation)
        mug = SimpleNamespace(position=(0.13, 0.24, 0.80), camera="head_camera")
        return SimpleNamespace(objects={"mug": mug}, status={"mug": "grounded"})

    def predicate_evidence(self, name, observation):
        return {"name": name, "value": None, "reason": "fake"}


class ReplayLoopTests(unittest.TestCase):
    def test_replays_frames_in_order_and_scores_against_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "seed_0"
            store = ObservationStore(capture / "observations")
            rows = [label(sequence=16), label(sequence=0)]
            for row in rows:
                store.save(Observation(row["sequence"], images(), tuple([0.0] * 14)))
            (capture / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            provider = FakeProvider()

            records = evaluate_capture(capture, provider)

        self.assertEqual([item["sequence"] for item in records], [0, 16])
        self.assertEqual(provider.phases, ["grasp", "grasp"])
        self.assertAlmostEqual(records[0]["mug_xy_error_m"], 0.05)
        self.assertAlmostEqual(records[0]["mug_z_offset_m"], 0.05)
        self.assertEqual(records[0]["predicates"]["release_ready"], {"public": None, "reason": "fake",
                                                                      "truth": True})
        # The provider only ever receives public observations, never labels.
        self.assertTrue(all(isinstance(item, Observation) for item in provider.seen))


if __name__ == "__main__":
    unittest.main()
