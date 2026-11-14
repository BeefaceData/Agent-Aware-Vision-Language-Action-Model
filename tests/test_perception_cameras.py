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
    DEFAULT_PERCEPTION_CAMERA, SUPERVISED_PERCEPTION_CAMERA, SUPERVISED_REVERSE_CAMERA,
    supervised_camera_specs, FixedCamera, camera_to_world, fixed_camera_payload, parse_camera_spec)
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


class ComplementaryCameraTests(unittest.TestCase):
    def test_complementary_view_is_opt_in_and_uses_opposing_azimuth_without_rng(self):
        before = np.random.get_state()
        self.assertEqual(supervised_camera_specs(),(SUPERVISED_PERCEPTION_CAMERA,))
        primary,reverse = supervised_camera_specs(complementary_view=True)
        self.assertEqual(reverse,SUPERVISED_REVERSE_CAMERA)
        self.assertEqual((reverse.width,reverse.height,reverse.fovy_deg),(640,480,60.))
        self.assertEqual(primary.target_m,reverse.target_m)
        first = np.asarray(primary.position_m)[:2]-np.asarray(primary.target_m)[:2]
        second = np.asarray(reverse.position_m)[:2]-np.asarray(reverse.target_m)[:2]
        np.testing.assert_allclose(first,-second)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        for value in (1,"true",None):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"complementary_view"):
                supervised_camera_specs(complementary_view=value)

    def test_both_views_round_trip_with_calibration_and_are_excluded_from_policy(self):
        payload = payload_with_camera()
        payload.update(fixed_camera_payload("perception_reverse_camera",FakeSapienCamera(depth_m=2.)))
        current = EnvironmentClient(lambda *args:(payload,{})).reset(4)
        with tempfile.TemporaryDirectory() as directory:
            store = ObservationStore(Path(directory)/"observations")
            store.save(current)
            loaded = load_public_observation(Path(directory)/"observations/observation_00000000.npz")
        for name,expected in (("perception_camera",1.5),("perception_reverse_camera",2.)):
            self.assertIn(name,loaded.images)
            self.assertAlmostEqual(float(loaded.camera_frames[name].depth_m[0,0]),expected)
            np.testing.assert_array_equal(loaded.camera_frames[name].intrinsic,INTRINSIC)
        packed,_,_ = pack_observation(loaded.images,loaded.command_state)
        with np.load(io.BytesIO(packed)) as sent:
            self.assertEqual(set(sent.files),{*CAMERAS,"state"})
            self.assertEqual(sent["state"].shape,(14,))

    def test_provider_includes_complementary_view_in_one_bounded_request(self):
        from unittest.mock import Mock
        from development.hanging.public_provider import PublicRGBDHangingProvider,VisibleHangingReport
        payload = payload_with_camera()
        payload.update(fixed_camera_payload("perception_reverse_camera",FakeSapienCamera()))
        current = EnvironmentClient(lambda *args:(payload,{})).reset(4)
        report = VisibleHangingReport(stage="grasp",left_held=False,right_held=False,mug_off_table=False,
            transfer_complete=None,complete_visible_opening=False,opposite_handle_faces_visible=False,
            confidence=.5,task_state_confidence=.9,evidence="Synthetic empty-gripper scene")
        provider = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="test",store=Mock(),
            log=lambda _:None,maximum_requests=1)
        with patch("development.hanging.public_provider.request_tool_input",return_value=report.model_dump()) as request:
            provider.observe(current)
        labels = [block["text"] for block in request.call_args.kwargs["content"] if block["type"]=="text"]
        self.assertTrue(any("camera=perception_reverse_camera" in text for text in labels))
        self.assertEqual(provider.requests,1)
        self.assertEqual(request.call_count,1)

    def test_supervisor_and_modal_expose_opt_in_without_starting_services(self):
        import ast
        from unittest.mock import Mock
        from development.hanging.supervised import SupervisedOptions,run
        with tempfile.TemporaryDirectory() as directory:
            loader = Mock(side_effect=ValueError("synthetic_asset_check_stop"))
            report = run("unused",Path(directory)/"run",SupervisedOptions("claude-test",complementary_view=True),
                         fixture_loader=loader)
        self.assertFalse(report["simulator_started"])
        self.assertEqual([s["name"] for s in report["perception_cameras"]],
                         ["perception_camera","perception_reverse_camera"])
        self.assertEqual(report["maximum_model_requests"],161)
        tree = ast.parse(Path("modal_apps/simulator.py").read_text(encoding="utf-8"))
        entry = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="supervised_hanging")
        self.assertIn("complementary_view",[a.arg for a in entry.args.args])
        options = next(n for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                       and n.func.id=="SupervisedOptions")
        self.assertTrue(any(k.arg=="complementary_view" and isinstance(k.value,ast.Name)
                            and k.value.id=="complementary_view" for k in options.keywords))
        with self.assertRaisesRegex(ValueError,"complementary_view"):
            SupervisedOptions("claude-test",complementary_view=1).validate()


class CloseViewTests(unittest.TestCase):
    def test_close_view_is_opt_in_same_direction_and_under_vision_resize_limits(self):
        from harness.environment.perception_cameras import CLOSE_PERCEPTION_CAMERA
        self.assertEqual(supervised_camera_specs(), (SUPERVISED_PERCEPTION_CAMERA,))
        (close,) = supervised_camera_specs(view="close")
        self.assertIs(close, CLOSE_PERCEPTION_CAMERA)
        self.assertIs(parse_camera_spec("close"), CLOSE_PERCEPTION_CAMERA)
        self.assertEqual(close.name, "perception_camera")
        from harness.environment.perception_cameras import CLOSE_REVERSE_CAMERA, parse_camera_set
        primary, reverse = supervised_camera_specs(complementary_view=True, view="close")
        self.assertEqual((primary, reverse), (CLOSE_PERCEPTION_CAMERA, CLOSE_REVERSE_CAMERA))
        self.assertEqual(supervised_camera_specs(complementary_view=True)[1], SUPERVISED_REVERSE_CAMERA)
        self.assertEqual(parse_camera_set("close_pair"), (CLOSE_PERCEPTION_CAMERA, CLOSE_REVERSE_CAMERA))
        self.assertEqual(parse_camera_set("close"), (CLOSE_PERCEPTION_CAMERA,))
        self.assertEqual(reverse.name, "perception_reverse_camera")
        self.assertLessEqual(reverse.width*reverse.height, 1_150_000)
        # The two close views look at the handle from opposite sides.
        to_front = np.subtract(primary.position_m, primary.target_m)
        to_back = np.subtract(reverse.position_m, reverse.target_m)
        self.assertLess(to_front @ to_back, 0)
        # On the line from its target back to the default camera: the same side
        # of the handle as the default view, just nearer the workspace.
        to_default = np.subtract(DEFAULT_PERCEPTION_CAMERA.position_m, close.target_m)
        to_close = np.subtract(close.position_m, close.target_m)
        cosine = to_default @ to_close / np.linalg.norm(to_default) / np.linalg.norm(to_close)
        self.assertGreater(cosine, .999)
        self.assertLess(np.linalg.norm(to_close), .6)
        # Full frames go to the vision model unresized, so stay below its limits.
        self.assertLessEqual(max(close.width, close.height), 1568)
        self.assertLessEqual(close.width*close.height, 1_150_000)
        with self.assertRaisesRegex(ValueError, "perception_view"):
            supervised_camera_specs(view="near")

    def test_supervisor_and_modal_expose_close_view(self):
        import ast
        from unittest.mock import Mock
        from development.hanging.supervised import SupervisedOptions, run
        with tempfile.TemporaryDirectory() as directory:
            loader = Mock(side_effect=ValueError("synthetic_asset_check_stop"))
            report = run("unused", Path(directory)/"run", SupervisedOptions("claude-test", perception_view="close"),
                         fixture_loader=loader)
        self.assertFalse(report["simulator_started"])
        self.assertEqual(report["perception_cameras"][0]["image_wh"], [1200, 900])
        with self.assertRaisesRegex(ValueError, "perception_view"):
            SupervisedOptions("claude-test", perception_view="near").validate()
        tree = ast.parse(Path("modal_apps/simulator.py").read_text(encoding="utf-8"))
        entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "supervised_hanging")
        self.assertIn("perception_view", [a.arg for a in entry.args.args])
        options = next(n for n in ast.walk(entry) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "SupervisedOptions")
        self.assertTrue(any(k.arg == "perception_view" and isinstance(k.value, ast.Name)
                            and k.value.id == "perception_view" for k in options.keywords))


class CameraPoseTests(unittest.TestCase):
    def test_supervised_high_resolution_rgbd_preserves_lens_and_policy_contract(self):
        from harness.perception.rgbd import RGBDCamera
        spec = SUPERVISED_PERCEPTION_CAMERA
        self.assertEqual((spec.width, spec.height), (640, 480))
        self.assertEqual((DEFAULT_PERCEPTION_CAMERA.width, DEFAULT_PERCEPTION_CAMERA.height), (320, 240))
        self.assertEqual(spec.position_m, DEFAULT_PERCEPTION_CAMERA.position_m)
        self.assertEqual(spec.target_m, DEFAULT_PERCEPTION_CAMERA.target_m)
        self.assertEqual(spec.fovy_deg, 60.)
        focal = spec.height/(2*np.tan(np.deg2rad(spec.fovy_deg)/2))
        k = np.array([[focal, 0., 320.], [0., focal, 240.], [0., 0., 1.]])
        camera = RGBDCamera(np.ones((480,640), np.float32), k, np.eye(4))
        low = RGBDCamera(np.ones((240,320), np.float32),
            np.array([[focal/2, 0.,160.], [0.,focal/2,120.], [0.,0.,1.]]), np.eye(4))
        np.testing.assert_allclose(camera.unproject(400.,280.), low.unproject(200.,140.))
        rgb = {name: np.zeros((240,320,3), np.uint8) for name in CAMERAS}
        rgb["perception_camera"] = np.zeros((480,640,3), np.uint8)
        packed, _, _ = pack_observation(rgb, np.zeros(14))
        with np.load(io.BytesIO(packed)) as archive:
            self.assertEqual(set(archive.files), {*CAMERAS, "state"})
            for name in CAMERAS:
                self.assertEqual(archive[name].shape, (240,320,3))
            self.assertEqual(archive["state"].shape, (14,))

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
