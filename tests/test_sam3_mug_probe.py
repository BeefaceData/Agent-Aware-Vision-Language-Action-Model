"""SAM 3 mug prompt probe: frame selection, mask grounding and scoring; no Modal or GPU."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from development.replays.sam3_mug_probe import mask_median_point, query, score
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import RGBDCamera
from tests.helpers import images


def label(sequence, *, right=False, table=True, tilt=0.0):
    return {"sequence": sequence, "native_success": False, "lifted_before": True,
            "mug": {"pose": [0.0, 0.0, 0.8, 1, 0, 0, 0], "tilt_deg": tilt, "height_change_m": 0.05},
            "contacts": {"left_gripper": False, "right_gripper": right, "table": table, "rack": False},
            "observation": f"observations/observation_{sequence:08d}.npz"}


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.detect_mug_prompt_variants = SimpleNamespace(remote=self.remote)

    def remote(self, rgb_png, *, observation_sequence, camera):
        self.calls.append((observation_sequence, camera))
        return {"variants": [{"prompt": "cup", "detections": [{"confidence": 0.3, "mask_png": b"png"}]}]}


class QueryTests(unittest.TestCase):
    def test_only_held_gripped_or_tilted_frames_are_probed_on_every_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObservationStore(Path(directory) / "observations")
            rows = [label(0), label(16, right=True, table=False)]
            for row in rows:
                store.save(Observation(row["sequence"], images(), tuple([0.0] * 14)))
            (Path(directory) / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            worker = FakeWorker()

            probe = query(directory, worker, log=lambda message: None)

        self.assertEqual(sorted(probe["frames"]), ["16"])  # frame 0 is a mug back on the table
        self.assertEqual({sequence for sequence, _camera in worker.calls}, {16})
        self.assertEqual(len(worker.calls), len(images()))


class GroundingTests(unittest.TestCase):
    def test_mask_median_point_is_in_world_coordinates(self):
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        world_to_camera = np.eye(4)
        world_to_camera[:3, 3] = (0.0, 0.0, -1.0)  # camera 1 m above the world origin along z
        frame = RGBDCamera(depth, np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]), world_to_camera)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        np.testing.assert_allclose(mask_median_point(mask, frame), (0.0, 0.0, 3.0))

    def test_a_mask_without_valid_depth_has_no_point(self):
        frame = RGBDCamera(np.zeros((2, 2), dtype=np.float32), np.eye(3), np.eye(4))

        self.assertIsNone(mask_median_point(np.ones((2, 2), dtype=bool), frame))


class ScoreTests(unittest.TestCase):
    def test_the_best_mask_per_camera_decides_and_misses_are_counted_separately(self):
        answers = [("seed_1", 16, "held_right", "head_camera", "cup", 0.35, 0.04),
                   ("seed_1", 16, "held_right", "head_camera", "cup", 0.25, 0.50),
                   ("seed_1", 32, "held_right", "head_camera", "cup", 0.15, 0.60),
                   ("seed_1", 48, "held_right", "head_camera", "cup", 0.45, 0.05)]

        rows = score(answers, missed={("seed_1", 16), ("seed_1", 32)})

        self.assertEqual(rows[("cup", 0.30)], {"found_missed": 1, "found": 2, "wrong": 0})
        self.assertEqual(rows[("cup", 0.10)], {"found_missed": 1, "found": 2, "wrong": 1})


if __name__ == "__main__":
    unittest.main()
