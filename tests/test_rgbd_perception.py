"""RGB-D transport and geometry checks; no simulator, policy, or GPU startup."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from development.simulator.local_simulator import LocalSimulator
from harness.environment.base import Observation
from harness.environment.client import EnvironmentClient
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import (
    RGBDCamera, camera_frames_from_archive, camera_frames_from_payload,
    require_camera_frames,
)
from harness.planner.base import PlannerContext
from harness.planner.prompt_builder import user_content
from harness.vla.modal_client import ModalVLAClient
from tests.helpers import images
from harness.environment.base import CAMERAS


def camera(*, depth=None, intrinsic=None, world_to_camera=None):
    return RGBDCamera(
        np.full((2, 3), 2.0, np.float32) if depth is None else depth,
        np.array([[2., 0., 1.], [0., 2., 0.5], [0., 0., 1.]]) if intrinsic is None else intrinsic,
        np.eye(4) if world_to_camera is None else world_to_camera,
    )


def payload():
    data = {**images(), "state": np.zeros(14, dtype=np.float32), "success": np.asarray(False)}
    for name in CAMERAS:
        data[f"{name}__depth_mm"] = np.full((2, 3), 2000., dtype=np.float32)
        data[f"{name}__intrinsic_cv"] = np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]])
        data[f"{name}__extrinsic_cv"] = np.eye(4)
    return data


class RGBDPerceptionTests(unittest.TestCase):
    def test_rgbd_enabled_run_requires_configured_camera_frames(self):
        observation = EnvironmentClient(lambda *args: (payload(), {})).reset(3)
        self.assertIs(require_camera_frames(observation, ("head_camera",)), observation)
        with self.assertRaisesRegex(ValueError, "required_rgbd_camera_frames_missing:wrist_camera"):
            require_camera_frames(observation, ("right_camera", "wrist_camera"))

    def test_metric_unprojection_uses_depth_intrinsics_and_world_transform(self):
        world_to_camera = np.eye(4)
        world_to_camera[:3, 3] = [-1., -2., -3.]
        frame = camera(world_to_camera=world_to_camera)
        np.testing.assert_allclose(frame.unproject(1, .5), (1., 2., 5.))

    def test_unprojection_uses_local_median_and_rejects_invalid_or_missing_depth(self):
        depth = np.array([[0., 1., 9.], [0., 2., 9.]], dtype=np.float32)
        frame = camera(depth=depth)
        point = frame.unproject(1, 0, radius=1)
        # Median of valid {1, 2, 9, 9} is 5.5m; robust to zero-valued invalid pixels.
        self.assertAlmostEqual(point[2], 5.5)
        with self.assertRaisesRegex(ValueError, "depth_unavailable"):
            camera(depth=np.zeros((2, 3), np.float32)).unproject(0, 0)
        with self.assertRaisesRegex(ValueError, "pixel_out_of_bounds"):
            frame.unproject(3, 0)
        with self.assertRaisesRegex(ValueError, "invalid_depth_image"):
            camera(depth=np.full((2, 3), np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "invalid_camera_intrinsic"):
            camera(intrinsic=np.diag([-1., 1., 1.]))
        with self.assertRaisesRegex(ValueError, "invalid_camera_extrinsic"):
            camera(world_to_camera=np.diag([2., 1., 1., 1.]))

    def test_local_depth_gap_flags_pixels_behind_a_nearby_surface(self):
        depth = np.full((9, 9), .74, dtype=np.float32)
        depth[4, 0] = .60
        frame = camera(depth=depth)
        self.assertGreater(frame.local_depth_gap(4, 4, search_radius=4), .06)
        depth[:] = .63
        depth[4, 0] = .61
        surface = camera(depth=depth)
        self.assertLessEqual(surface.local_depth_gap(4, 4, search_radius=4), .06)

    def test_surface_normals_tell_a_floor_from_a_wall_in_world_axes(self):
        depth = np.full((9, 9), .70, dtype=np.float32)
        mask = np.zeros((9, 9), dtype=bool)
        mask[1:8, 1:8] = True
        # Identity extrinsic: the camera looks along world z, so a constant-depth
        # surface faces up. Turned 90 degrees about x, the same surface is a wall.
        sideways = np.eye(4)
        sideways[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]

        floor = camera(depth=depth).vertical_normal_components(mask)
        wall = camera(depth=depth, world_to_camera=sideways).vertical_normal_components(mask)

        np.testing.assert_allclose(floor, 1.0)
        np.testing.assert_allclose(wall, 0.0, atol=1e-9)
        self.assertEqual(len(floor), 3 * 3)  # the 7x7 mask eroded by 2 px on each side

    def test_surface_normals_skip_pixels_beside_a_depth_jump(self):
        depth = np.full((9, 9), .70, dtype=np.float32)
        depth[:, 5:] = .90

        normal_z = camera(depth=depth).vertical_normal_components(np.ones((9, 9), dtype=bool), erode_px=0)

        self.assertEqual(len(normal_z), 7 * 5)  # columns 4 and 5 sit beside the step
        np.testing.assert_allclose(normal_z, 1.0)

    def test_environment_client_converts_millimeters_and_copies_camera_frames(self):
        raw = payload()
        client = EnvironmentClient(lambda *args: (raw, {}))
        observation = client.reset(3)
        self.assertEqual(set(observation.camera_frames), set(CAMERAS))
        self.assertEqual(float(observation.camera_frames["head_camera"].depth_m[0, 0]), 2.0)
        observation.camera_frames["head_camera"].depth_m.setflags(write=True)
        observation.camera_frames["head_camera"].depth_m[0, 0] = 10
        self.assertEqual(float(client.observe().camera_frames["head_camera"].depth_m[0, 0]), 2.0)

    def test_non_finite_or_negative_depth_pixels_become_missing_not_an_error(self):
        raw = payload()
        depth = np.full((2, 3), 2000.0, dtype=np.float32)
        depth[0, 0], depth[0, 1], depth[1, 2] = np.nan, -5.0, np.inf
        raw["head_camera__depth_mm"] = depth

        frames = camera_frames_from_payload(raw, images())

        np.testing.assert_array_equal(frames["head_camera"].depth_m,
                                      [[0.0, 0.0, 2.0], [2.0, 2.0, 0.0]])

    def test_robotwin_three_by_four_cv_extrinsic_is_homogenized(self):
        raw = payload()
        raw["head_camera__extrinsic_cv"] = np.array(
            [[1., 0., 0., .1], [0., 1., 0., .2], [0., 0., 1., .3]])
        frames = camera_frames_from_payload(raw, images())
        self.assertEqual(frames["head_camera"].world_to_camera.shape, (4, 4))
        np.testing.assert_allclose(frames["head_camera"].world_to_camera[3], [0., 0., 0., 1.])
        np.testing.assert_allclose(frames["head_camera"].unproject(1, .5), [-.1, -.2, 1.7])

    def test_simulator_observation_adapter_exports_optional_calibration_and_raw_depth_mm(self):
        from harness.environment import robotwin
        cameras = {}
        for name in CAMERAS:
            cameras[name] = {"rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                "depth": np.full((2, 3), 1500., dtype=np.float64),
                "intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4)}
        task = SimpleNamespace(get_obs=lambda: {"observation": cameras, "joint_action": {"vector": np.zeros(14)}})
        observation = robotwin.observation_of(task, np.zeros((2, 3, 3), dtype=np.uint8), False)
        self.assertEqual(float(observation["head_camera__depth_mm"][0, 0]), 1500.)
        self.assertEqual(observation["head_camera__intrinsic_cv"].shape, (3, 3))
        self.assertEqual(observation["right_camera__extrinsic_cv"].shape, (4, 4))
        self.assertEqual(observation["head_camera"].dtype, np.uint8)

    def test_legacy_rgb_observation_payload_is_unchanged_when_depth_is_disabled(self):
        from harness.environment import robotwin
        raw = {name: {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)} for name in CAMERAS}
        task = SimpleNamespace(get_obs=lambda: {"observation": raw, "joint_action": {"vector": np.zeros(14)}})
        observation = robotwin.observation_of(task, np.zeros((2, 3, 3), dtype=np.uint8), False)
        self.assertEqual(set(observation), {*CAMERAS, "state", "success"})

    def test_partial_or_misaligned_rgbd_payload_fails_closed(self):
        raw = payload()
        del raw["left_camera__extrinsic_cv"]
        client = EnvironmentClient(lambda *args: (raw, {}))
        with self.assertRaisesRegex(ValueError, "incomplete_rgbd_camera_payload"):
            client.reset(3)
        raw = payload()
        raw["head_camera__depth_mm"] = np.ones((1, 1), dtype=np.float32)
        client = EnvironmentClient(lambda *args: (raw, {}))
        with self.assertRaisesRegex(ValueError, "depth_rgb_shape_mismatch"):
            client.reset(3)

    def test_observation_store_roundtrips_rgbd_for_offline_replay(self):
        observation = Observation(7, images(), (0.,)*14,
                                  {"head_camera": camera()})
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            ref = store.save(observation)
            with np.load(store.directory / Path(ref["path"]).name, allow_pickle=False) as data:
                self.assertIn("head_camera__depth_m", data.files)
                self.assertIn("head_camera__intrinsic", data.files)
                self.assertIn("head_camera__world_to_camera", data.files)
                self.assertNotIn("right_camera__depth_m", data.files)
                restored = camera_frames_from_archive(data, images())
                self.assertEqual(float(restored["head_camera"].depth_m[0, 0]), 2.0)

    def test_observation_store_roundtrips_measured_tcp_poses_for_replay(self):
        from harness.perception.replay import load_public_observation

        poses = {"left": (0., 0., 0., 0., 0., 0., 1.),
                 "right": (1., 0., 0., 0., 0., 0., 1.)}
        observation = Observation(7, images(), (0.,)*14,
            {"head_camera": camera()}, poses)
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            ref = store.save(observation)
            restored = load_public_observation(store.directory / Path(ref["path"]).name)
        self.assertEqual(restored.measured_tcp_poses, poses)

    def test_modal_policy_receives_only_original_rgb_and_state(self):
        class Worker:
            def act_chunk(self, images_arg, state, size):
                self.received = (images_arg, state, size)
                return np.zeros((size, 14), dtype=np.float32), {}

        worker = Worker()
        observation = Observation(0, images(), (0.,)*14, {"head_camera": camera()})
        ModalVLAClient(worker).infer(observation, 2)
        image_args, state, size = worker.received
        self.assertEqual(set(image_args), set(CAMERAS))
        self.assertEqual(size, 2)
        self.assertEqual(state, (0.,)*14)
        self.assertTrue(all(image.dtype == np.uint8 for image in image_args.values()))

    def test_current_opus_prompt_does_not_serialize_internal_depth_or_calibration(self):
        frame = camera()
        poses = {"left": (0., 0., 0., 0., 0., 0., 1.),
                 "right": (1., 0., 0., 0., 0., 0., 1.)}
        context = PlannerContext(1, Observation(0, images(), (0.,)*14,
            {"head_camera": frame}, poses), (), None)
        content = user_content(context, limits={}, history=[])
        text = " ".join(block.get("text", "") for block in content if block["type"] == "text")
        self.assertNotIn("depth_m", text)
        self.assertNotIn("world_to_camera", text)
        self.assertNotIn("intrinsic", text)
        self.assertNotIn("measured_tcp_poses", text)
        self.assertNotIn("left_tcp_pose", text)

    def test_local_simulator_depth_capture_is_opt_in(self):
        from harness.environment import robotwin
        task = SimpleNamespace()
        setup = {"data_type": {"depth": False}}
        with patch.object(robotwin, "build_task", return_value=(task, setup)):
            baseline = LocalSimulator("RoboTwin")
            self.assertFalse(baseline.setup["data_type"]["depth"])
        with patch.object(robotwin, "build_task", return_value=(task, setup)):
            perception = LocalSimulator("RoboTwin", rgbd_enabled=True)
            self.assertTrue(perception.setup["data_type"]["depth"])
        with self.assertRaises(ValueError):
            LocalSimulator("RoboTwin", rgbd_enabled=1)
