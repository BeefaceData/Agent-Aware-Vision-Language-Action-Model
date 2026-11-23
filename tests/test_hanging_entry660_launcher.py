import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from development.checkpoints.bank import sha256
from development.hanging.run_entry660 import run, validate_package
from harness.environment.base import Observation
from harness.environment.perception_cameras import DEFAULT_PERCEPTION_CAMERA, describe
from tests import test_hanging_entry660 as fixtures


class Entry660LauncherTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Entry660Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        provenance = self.path / "provenance.json"
        data = json.loads(provenance.read_text())
        data["manifest"]["perception_camera"] = describe(DEFAULT_PERCEPTION_CAMERA)
        provenance.write_text(json.dumps(data))
        metadata = self.path / "records/unverified_hang_660.json"
        checkpoint = json.loads(metadata.read_text())
        checkpoint["artifact_hashes"]["provenance.json"] = sha256(provenance)
        metadata.write_text(json.dumps(checkpoint))

    def test_opt_in_and_integrity_rejected_before_native_import(self):
        output = self.path / "result"
        with patch.dict(sys.modules, {"development.simulator.local_simulator": None}):
            with self.assertRaisesRegex(ValueError, "opt_in"):
                run(self.path, output, "unused")
            (self.path / "prefix.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(self.path, output, "unused", allow_unverified=True)
        self.assertFalse(output.exists())

    def test_original_camera_is_required(self):
        # The original fixture omitted this optional camera: a different capture
        # must not silently run with the hanging recording's camera contract.
        with patch("development.hanging.run_entry660.describe", return_value={}):
            with self.assertRaisesRegex(ValueError, "original_perception_camera"):
                validate_package(self.path, allow_unverified=True)

    def test_single_run_saves_last_confirmed_observation_and_closes(self):
        boundary = self.fixture.boundary
        calls = []
        class Environment:
            def reset(self, seed):
                calls.append(seed)
                return Observation(0, boundary.images, boundary.command_state)
            def step(self, action):
                sequence = len(calls)
                calls.append(action)
                return Observation(sequence, boundary.images, boundary.command_state)
        simulator = Mock()
        create = Mock(return_value=simulator)
        video = Mock()
        video.metadata.return_value = dict(fps=10, real_time=False, frames=661)
        create_video = Mock(return_value=video)
        modules = {
            "development.simulator.local_simulator": SimpleNamespace(LocalSimulator=create),
            "harness.environment.local": SimpleNamespace(LocalEnvironmentClient=lambda *a, **k: Environment()),
            "development.demos.episode_video": SimpleNamespace(EpisodeVideo=create_video),
        }
        output = self.path / "result"
        with patch.dict(sys.modules, modules):
            report = run(self.path, output, "unused", allow_unverified=True)
        create.assert_called_once_with("unused", rgbd_enabled=True,
                                       perception_cameras=(DEFAULT_PERCEPTION_CAMERA,))
        simulator.close.assert_called_once()
        self.assertEqual(len(calls), 661)
        self.assertEqual(calls[1:], [frame.action for frame in self.fixture.frames[1:]])
        self.assertEqual(video.append.call_count, 661)
        self.assertEqual([call.args[0].sequence for call in video.append.call_args_list], list(range(661)))
        self.assertEqual(video.append.call_args_list[0].args[1], "unverified_replay_reset")
        self.assertEqual(video.append.call_args_list[-1].args[1], "unverified_recorded_action_replay")
        create_video.assert_called_once_with(output.resolve(), cameras=(
            "head_camera", "left_camera", "right_camera", "perception_camera"))
        video.close.assert_called_once()
        self.assertEqual(report["video"], dict(fps=10, real_time=False, frames=661))
        self.assertEqual(report["last_confirmed_sequence"], 660)
        self.assertTrue((output / "observations/observation_00000660.npz").is_file())
        self.assertEqual(json.loads((output / "report.json").read_text())["grasp_validity"], "unknown")
