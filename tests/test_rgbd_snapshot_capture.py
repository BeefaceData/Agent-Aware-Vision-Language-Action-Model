"""Snapshot capture adapter tests; never starts RoboTwin."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from development.capture.rgbd_snapshot import capture
from tests.helpers import images


class RGBDSnapshotCaptureTests(unittest.TestCase):
    def test_captures_one_reset_observation_without_policy_or_followup_actions(self):
        payload = images()
        payload.update({
            "state": np.zeros(14, dtype=np.float32),
            "success": np.bool_(False),
            "head_camera__depth_mm": np.full((2, 3), 500, dtype=np.float32),
            "head_camera__intrinsic_cv": np.array(
                [[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]),
            "head_camera__extrinsic_cv": np.eye(4),
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            with patch("development.simulator.local_simulator.LocalSimulator") as simulator_type:
                simulator = simulator_type.return_value
                simulator.return_value = (payload, {})
                manifest = capture("RoboTwin", output, 3)

            simulator_type.assert_called_once_with("RoboTwin", rgbd_enabled=True)
            simulator.assert_called_once_with("127.0.0.1", 0, "/reset", {"seed": 3})
            simulator.close.assert_called_once_with()
            self.assertEqual(manifest["robot_actions_after_reset"], 0)
            self.assertFalse(manifest["policy_started"])
            self.assertTrue((output / "observations/observation_00000000.npz").is_file())


if __name__ == "__main__":
    unittest.main()
