"""Scene-view capture tests; never starts RoboTwin."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from development.capture.scene_views import capture, depth_to_rgb


def rendered_camera(depth_mm=500.0):
    return {"rgb": np.full((2, 3, 3), 120, dtype=np.uint8),
            "depth": np.full((2, 3), depth_mm),
            "intrinsic_cv": np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]),
            "extrinsic_cv": np.eye(4)[:3],
            "cam2world_gl": np.eye(4)}


def fake_simulator(simulator_type):
    simulator = simulator_type.return_value
    simulator.setup = {}
    simulator.return_value = ({"state": np.zeros(14, dtype=np.float32)}, {})
    simulator.task.get_obs.return_value = {"observation": {
        "head_camera": rendered_camera(), "front_camera": rendered_camera()}}
    simulator.task.cameras.get_observer_rgb.return_value = np.zeros((2, 3, 3), dtype=np.uint8)
    return simulator


class SceneViewsCaptureTests(unittest.TestCase):
    def test_saves_every_rendered_camera_after_one_reset_without_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scene"
            with patch("development.simulator.local_simulator.LocalSimulator") as simulator_type:
                simulator = fake_simulator(simulator_type)
                report = capture("RoboTwin", output, 3)

            simulator_type.assert_called_once_with("RoboTwin", rgbd_enabled=True)
            simulator.assert_called_once_with("127.0.0.1", 0, "/reset", {"seed": 3})
            simulator.close.assert_called_once_with()
            for name in ("head_camera_rgb.png", "head_camera_depth.png", "front_camera_rgb.png",
                         "observer_camera_rgb.png", "overview.png", "cameras.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(json.loads((output / "cameras.json").read_text()), report)

    def test_report_marks_which_cameras_the_harness_uses(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("development.simulator.local_simulator.LocalSimulator") as simulator_type:
                fake_simulator(simulator_type)
                report = capture("RoboTwin", Path(directory) / "scene", 3)

        self.assertEqual(report["robot_actions_after_reset"], 0)
        self.assertTrue(report["cameras"]["head_camera"]["used_by_harness"])
        self.assertFalse(report["cameras"]["front_camera"]["used_by_harness"])
        self.assertFalse(report["cameras"]["observer_camera"]["used_by_harness"])
        self.assertEqual(report["cameras"]["head_camera"]["depth_range_m"], [0.5, 0.5])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("development.simulator.local_simulator.LocalSimulator") as simulator_type:
                with self.assertRaises(FileExistsError):
                    capture("RoboTwin", Path(directory), 3)
            simulator_type.assert_not_called()

    def test_missing_depth_is_drawn_black(self):
        rgb = depth_to_rgb(np.array([[0.0, 1000.0]]))

        self.assertEqual(rgb[0, 0].tolist(), [0, 0, 0])
        self.assertNotEqual(rgb[0, 1].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
