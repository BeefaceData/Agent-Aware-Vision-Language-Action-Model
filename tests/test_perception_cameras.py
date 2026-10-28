"""Perception-only fixed camera: pose maths, payload, plumbing to perception; never starts RoboTwin."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from development.replays.perception_evaluation import without_cameras
from harness.environment import robotwin
from harness.environment.base import copy_images
from harness.environment.client import EnvironmentClient
from harness.environment.layout import CAMERAS
from harness.environment.perception_cameras import (
    DEFAULT_PERCEPTION_CAMERA, FixedCamera, camera_to_world, fixed_camera_payload, parse_camera_spec)
from harness.perception.observations import ObservationStore
from harness.perception.replay import load_public_observation
from harness.vla.modal_client import pack_observation
from tests.helpers import images

INTRINSIC = np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]])


class FakeSapienCamera:
    """Answers like a SAPIEN render camera: RGBA in 0-1, OpenGL positions (z < 0 in front)."""

    def __init__(self, depth_m=1.5):
        self.depth_m, self.pictures = depth_m, 0

    def take_picture(self):
        self.pictures += 1

    def get_picture(self, kind):
        if kind == "Color":
            return np.full((2, 3, 4), 0.5, dtype=np.float32)
        position = np.zeros((2, 3, 4), dtype=np.float32)
        position[..., 2] = -self.depth_m
        return position

    def get_intrinsic_matrix(self):
        return INTRINSIC

    def get_extrinsic_matrix(self):
        return np.eye(4)[:3]


def robotwin_task():
    cameras = {name: {"rgb": np.zeros((2, 3, 3), dtype=np.uint8), "depth": np.full((2, 3), 1000.),
                      "intrinsic_cv": INTRINSIC, "extrinsic_cv": np.eye(4)[:3]} for name in CAMERAS}
    return SimpleNamespace(get_obs=lambda: {"observation": cameras, "joint_action": {"vector": np.zeros(14)}})


def payload_with_camera():
    return robotwin.observation_of(robotwin_task(), np.zeros((2, 3, 3), dtype=np.uint8), False,
                                   {"perception_camera": FakeSapienCamera()})


class CameraPoseTests(unittest.TestCase):
    def test_camera_looks_at_its_target_with_the_image_upright(self):
        pose = camera_to_world((0.35, 0.40, 1.15), (0.0, -0.05, 0.80))

        forward, up = pose[:3, 0], pose[:3, 2]
        expected = np.subtract((0.0, -0.05, 0.80), (0.35, 0.40, 1.15))
        np.testing.assert_allclose(forward, expected / np.linalg.norm(expected))
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0)
        self.assertGreater(up[2], 0)
        np.testing.assert_allclose(pose[:3, 3], (0.35, 0.40, 1.15))

    def test_a_camera_looking_straight_down_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_perception_camera_pose"):
            FixedCamera("perception_camera", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 60.0)

    def test_only_known_perception_camera_names_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "unknown_perception_camera"):
            FixedCamera("head_camera", (0.3, 0.4, 1.1), (0.0, 0.0, 0.8), 60.0)


class SpecParsingTests(unittest.TestCase):
    def test_default_and_explicit_specs(self):
        self.assertIs(parse_camera_spec("default"), DEFAULT_PERCEPTION_CAMERA)

        spec = parse_camera_spec("0.3,0.5,1.2/0,-0.1,0.78/70")

        self.assertEqual((spec.position_m, spec.target_m, spec.fovy_deg),
                         ((0.3, 0.5, 1.2), (0.0, -0.1, 0.78), 70.0))

    def test_malformed_specs_fail_with_the_expected_format(self):
        for text in ("0.3,0.5/0,0,0.8/70", "0.3,0.5,1.2/0,0,0.8", "near/far/wide", "0.3,0.5,1.2/0,0,0.8/170"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "x,y,z/tx,ty,tz/fovy"):
                parse_camera_spec(text)


class PayloadTests(unittest.TestCase):
    def test_fixed_camera_renders_rgb_and_depth_in_robotwin_units(self):
        camera = FakeSapienCamera(depth_m=1.5)

        payload = fixed_camera_payload("perception_camera", camera)

        self.assertEqual(camera.pictures, 1)
        self.assertEqual(payload["perception_camera"].dtype, np.uint8)
        self.assertEqual(int(payload["perception_camera"][0, 0, 0]), 127)
        self.assertEqual(float(payload["perception_camera__depth_mm"][0, 0]), 1500.0)
        self.assertEqual(payload["perception_camera__extrinsic_cv"].shape, (3, 4))

    def test_observation_carries_the_camera_with_depth_but_the_policy_payload_does_not(self):
        observation = EnvironmentClient(lambda *args: (payload_with_camera(), {})).reset(3)

        self.assertEqual(set(observation.images), {*CAMERAS, "perception_camera"})
        self.assertAlmostEqual(float(observation.camera_frames["perception_camera"].depth_m[0, 0]), 1.5)
        packed, _ms, _kb = pack_observation(observation.images, observation.command_state)
        with np.load(io.BytesIO(packed)) as sent:
            self.assertEqual(set(sent.files), {*CAMERAS, "state"})

    def test_observations_without_the_camera_are_unchanged(self):
        payload = robotwin.observation_of(robotwin_task(), np.zeros((2, 3, 3), dtype=np.uint8), False)

        self.assertNotIn("perception_camera", payload)
        self.assertEqual(set(EnvironmentClient(lambda *args: (payload, {})).reset(3).images), set(CAMERAS))

    def test_unknown_extra_cameras_are_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "only known perception cameras"):
            copy_images({**images(), "front_camera": np.zeros((2, 3, 3), dtype=np.uint8)})


class ArchiveTests(unittest.TestCase):
    def test_saved_observation_round_trips_the_camera_and_its_depth(self):
        observation = EnvironmentClient(lambda *args: (payload_with_camera(), {})).reset(3)
        with tempfile.TemporaryDirectory() as directory:
            store = ObservationStore(Path(directory) / "observations")
            store.save(observation)

            loaded = load_public_observation(Path(directory) / "observations/observation_00000000.npz")

        self.assertIn("perception_camera", loaded.images)
        self.assertIn("perception_camera", loaded.camera_frames)

    def test_evaluator_can_score_the_same_frame_without_the_camera(self):
        observation = EnvironmentClient(lambda *args: (payload_with_camera(), {})).reset(3)

        stripped = without_cameras(observation, ("perception_camera",))

        self.assertEqual(set(stripped.images), set(CAMERAS))
        self.assertEqual(set(stripped.camera_frames), set(CAMERAS))
        self.assertIs(without_cameras(observation, ()), observation)


class LocalSimulatorTests(unittest.TestCase):
    def test_camera_is_added_after_setup_demo_on_every_reset_and_rendered_each_observation(self):
        from development.simulator.local_simulator import LocalSimulator

        events = []
        task = robotwin_task()
        task.setup_demo = lambda **kwargs: events.append("setup_demo")
        task.scene = "scene"
        task.close_env = lambda: None

        def add(scene, spec):
            events.append(("add_camera", scene, spec.name))
            return FakeSapienCamera()

        torch_stub = SimpleNamespace(enable_grad=__import__("contextlib").nullcontext)
        with patch.dict("sys.modules", {"torch": torch_stub}), \
                patch("harness.environment.robotwin.build_task", return_value=(task, {})), \
                patch("harness.environment.perception_cameras.add_fixed_camera", side_effect=add):
            simulator = LocalSimulator("RoboTwin", perception_cameras=(DEFAULT_PERCEPTION_CAMERA,))
            payload, _ = simulator("local", 0, "/reset", {"seed": 3})
            simulator("local", 0, "/reset", {"seed": 4})
            simulator.close()

        self.assertEqual(events, ["setup_demo", ("add_camera", "scene", "perception_camera")] * 2)
        self.assertIn("perception_camera__depth_mm", payload)
        self.assertEqual(simulator.perception_cameras, {})

    def test_invalid_camera_settings_are_rejected_before_the_simulator_starts(self):
        from development.simulator.local_simulator import LocalSimulator

        with patch("harness.environment.robotwin.build_task") as build:
            with self.assertRaisesRegex(ValueError, "invalid_perception_cameras"):
                LocalSimulator("RoboTwin", perception_cameras=("default",))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
