import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from development.checkpoints.bank import sha256
from development.hanging.grounding_preflight import preflight, grounding_client, frame_crop_retry
from development.hanging.recovery import RecordedGrounder, run
from harness.perception.observations import ObservationStore
from harness.perception.replay import load_public_observation
from tests import test_hanging_entry660_launcher as fixtures
from tests.test_opus_frame_grounding import FakeClient


class ProviderRejection(Exception):
    status_code = 400
    request_id = "req_test"
    body = {"error": {"type": "invalid_request_error",
                      "message": "forced tool incompatible; sk-ant-test-secret Bearer test-token"},
            "headers": {"authorization": "must-not-be-saved"}}


class HangingGroundingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Entry660LauncherTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.package = self.fixture.path
        boundary = self.package / "boundary.npz"
        with np.load(boundary, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        values.update(perception_camera=np.zeros((4, 4, 3), dtype=np.uint8),
            perception_camera__depth_m=np.ones((4, 4), dtype=np.float32),
            perception_camera__intrinsic=np.diag([100., 100., 1.]),
            perception_camera__world_to_camera=np.eye(4))
        np.savez(boundary, **values)
        metadata = self.package / "records/unverified_hang_660.json"
        checkpoint = json.loads(metadata.read_text())
        checkpoint["artifact_hashes"]["boundary.npz"] = sha256(boundary)
        metadata.write_text(json.dumps(checkpoint))
        self.observation = load_public_observation(boundary)

    def client(self):
        return FakeClient(dict(visible=True, keypoints_px=[[0, 0], [1, 0], [0, 1]],
                               confidence=.9, description="synthetic landmarks"))

    def test_preflight_checks_real_request_shape_and_geometry_before_simulation(self):
        client = self.client()
        report = preflight(self.package, self.package / "review", "claude-test", client=client)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertEqual(report["failure"]["validation_error"],
                         "handle_surface_frame_is_not_a_reviewed_mating_frame")
        self.assertEqual(len(client.messages.requests), 2)
        for request in client.messages.requests:
            self.assertNotIn("thinking", request)
            self.assertEqual(request["extra_body"], {
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
            self.assertEqual(request["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})
            self.assertEqual(request["max_tokens"], 4096)
        self.assertFalse(report["simulator_started"])

    def test_provider_rejection_saved_sanitized_without_second_request(self):
        client = self.client()
        with patch.object(client.messages, "create", side_effect=ProviderRejection()) as create:
            report = preflight(self.package, self.package / "review", "claude-test", client=client)
        create.assert_called_once()
        self.assertEqual(report["status"], "preflight_failed")
        self.assertEqual(report["failure"]["http_status"], 400)
        self.assertEqual(report["failure"]["request_id"], "req_test")
        saved = (self.package / "review/report.json").read_text()
        for secret in ("sk-ant-test-secret", "test-token", "must-not-be-saved"):
            self.assertNotIn(secret, saved)

    def test_failed_preflight_never_imports_or_starts_simulator(self):
        failed = {"status": "preflight_failed", "failure": {"http_status": 400}}
        output = self.package / "result"
        with patch("development.hanging.recovery.preflight", return_value=failed), \
                patch.dict(sys.modules, {"development.simulator.local_simulator": None}):
            report = run(self.package, output, "unused", model="claude-test", allow_unverified=True)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertEqual(json.loads((output / "report.json").read_text())["preflight"], failed)

    def test_depth_valid_surface_reports_do_not_authorize_live_recovery(self):
        assessment = preflight(self.package, self.package / "geometry_review",
                               "claude-test", client=self.client())
        # Both numerical frames pass; the unsupported mating relation still
        # prevents simulator startup, rather than becoming a motion target.
        for target in ("rack_peg", "mug_handle"):
            self.assertEqual(assessment["targets"][target]["status"], "passed")
        with patch.dict(sys.modules, {"development.simulator.local_simulator": None}):
            result = run(self.package, self.package / "blocked_recovery", "unused",
                         model="claude-test", allow_unverified=True,
                         preflight_result=assessment)
        self.assertEqual(result["status"], "preflight_failed")
        self.assertIsNone(result["recovery"])
        self.assertNotIn("video", result)

    def test_cached_proposal_requires_identical_current_image_and_then_expires(self):
        client = self.client()
        report = preflight(self.package, self.package / "review", "claude-test", client=client)
        grounder = RecordedGrounder(client, "claude-test", "mug_handle", self.package,
            ObservationStore(self.package / "observations"), lambda event: None, report)
        grounder.locate(self.observation)
        self.assertEqual(len(client.messages.requests), 2)
        grounder.locate(self.observation)
        self.assertEqual(len(client.messages.requests), 3)
        stale = RecordedGrounder(client, "claude-test", "mug_handle", self.package,
            ObservationStore(self.package / "other_observations"), lambda event: None, report)
        changed = self.observation.copy()
        changed.images["perception_camera"][0, 0] = 255
        with self.assertRaisesRegex(ValueError, "preflight_image_no_longer_current"):
            stale.locate(changed)
        self.assertEqual(len(client.messages.requests), 3)

    def test_adaptive_response_requires_one_named_tool_despite_thinking_and_text(self):
        client = self.client()
        wrapped = grounding_client(client)
        request = dict(system="locate", tools=[{"name": "report_object_frame_keypoints"}],
                       model="claude-opus-5-5", max_tokens=350)
        valid = SimpleNamespace(type="tool_use", name="report_object_frame_keypoints", input={})
        thinking = SimpleNamespace(type="thinking")
        text = SimpleNamespace(type="text")
        with patch.object(client.messages, "create", return_value=SimpleNamespace(
                content=[thinking, text, valid])):
            self.assertEqual(wrapped.messages.create(**request).content[-1], valid)
        for content in ([thinking, text], [valid, valid],
                        [SimpleNamespace(type="tool_use", name="wrong_tool")]):
            with self.subTest(content=content), patch.object(client.messages, "create",
                    return_value=SimpleNamespace(content=content)):
                with self.assertRaisesRegex(ValueError, "exactly_one_keypoint_report"):
                    wrapped.messages.create(**request)

    def test_single_low_confidence_crop_retry_maps_back_before_depth_validation(self):
        client = self.client()
        original_points = [[0., 0.], [1., 0.], [0., 1.]]
        crop_points = [[(u+.5)*192-.5, (v+.5)*192-.5] for u, v in original_points]
        proposals = iter((
            dict(visible=True, keypoints_px=original_points, confidence=.45, description="thin peg"),
            dict(visible=True, keypoints_px=crop_points, confidence=.9, description="crop peg"),
            dict(visible=True, keypoints_px=original_points, confidence=.9, description="handle")))
        def respond(**request):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                name="report_object_frame_keypoints", input=next(proposals))])
        with patch.object(client.messages, "create", side_effect=respond) as create:
            result = preflight(self.package, self.package / "review", "claude-test", client=client)
        self.assertEqual(result["status"], "preflight_failed")
        self.assertEqual(create.call_count, 3)
        self.assertEqual(result["requests_attempted"], 3)
        row = result["targets"]["rack_peg"]
        np.testing.assert_allclose(row["proposal"]["keypoints_px"], original_points)
        self.assertEqual(row["proposal"]["image_shape_px"], [4, 4])
        self.assertEqual(row["initial_proposal"]["confidence"], .45)
        self.assertEqual(row["crop_transform"]["shown_shape_px"], [768, 768])
        self.assertTrue((self.package / "review/rack_crop_report.json").is_file())

    def test_second_low_confidence_result_stops_without_another_retry(self):
        client = self.client()
        proposals = iter(([[0., 0.], [1., 0.], [0., 1.]],
                          [[95.5, 95.5], [287.5, 95.5], [95.5, 287.5]]))
        def respond(**request):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                name="report_object_frame_keypoints", input=dict(visible=True,
                keypoints_px=next(proposals), confidence=.45, description="still uncertain"))])
        with patch.object(client.messages, "create", side_effect=respond) as create:
            result = preflight(self.package, self.package / "review", "claude-test", client=client)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(result["failure"]["validation_error"], "grounding_confidence_too_low")
        self.assertNotIn("mug_handle", result["targets"])

    def test_crop_retry_translates_nonzero_origin_without_changing_camera_calibration(self):
        from tests.test_hang_targets import make_observation
        from dataclasses import replace
        original = make_observation(660)
        image = original.images["right_camera"]
        observation = replace(original, images={**original.images, "perception_camera": image},
            camera_frames={"perception_camera": original.camera_frames["right_camera"]})
        points = np.array([[30., 30.], [40., 30.], [30., 40.]])
        first = dict(target="rack_peg", camera="perception_camera", observation_sequence=660,
                     keypoints_px=points.tolist())
        def locate(cropped):
            self.assertIsNone(cropped.camera_frames)
            self.assertEqual(cropped.images["perception_camera"].shape[:2], (768, 768))
            crop_points = (points - [6., 6.] + .5) * (768/59) - .5
            return dict(status="grounded", target="rack_peg", camera="perception_camera",
                observation_sequence=660, confidence=.9, image_shape_px=[768, 768],
                keypoints_px=crop_points.tolist())
        mapped, transform = frame_crop_retry(SimpleNamespace(locate=locate), observation,
                                             first, self.package)
        np.testing.assert_allclose(mapped["keypoints_px"], points)
        self.assertEqual(transform["box_xyxy"], [6, 6, 65, 65])
        self.assertEqual(mapped["image_shape_px"], [80, 80])
        np.testing.assert_array_equal(observation.camera_frames["perception_camera"].intrinsic,
                                      original.camera_frames["right_camera"].intrinsic)

    def test_original_depth_checks_remain_in_force_with_and_without_crop_retry(self):
        boundary = self.package / "boundary.npz"
        with np.load(boundary, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        values["perception_camera__depth_m"][:2, :2] = 1.5
        np.savez(boundary, **values)
        metadata = self.package / "records/unverified_hang_660.json"
        checkpoint = json.loads(metadata.read_text())
        checkpoint["artifact_hashes"]["boundary.npz"] = sha256(boundary)
        metadata.write_text(json.dumps(checkpoint))
        for retry in (False, True):
            with self.subTest(retry=retry):
                client = self.client()
                proposals = iter((
                    dict(visible=True, confidence=.45 if retry else .9, description="original",
                         keypoints_px=[[0., 0.], [1., 0.], [0., 1.]]),
                    dict(visible=True, confidence=.9, description="crop",
                         keypoints_px=[[95.5, 95.5], [287.5, 95.5], [95.5, 287.5]])))
                def respond(**request):
                    return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                        name="report_object_frame_keypoints", input=next(proposals))])
                with patch.object(client.messages, "create", side_effect=respond) as create:
                    result = preflight(self.package, self.package / f"review_{retry}",
                                       "claude-test", client=client)
                self.assertEqual(create.call_count, 2 if retry else 1)
                self.assertEqual(result["failure"]["validation_error"], "keypoint_depth_discontinuity")

    def test_both_targets_can_retry_once_before_simulator_startup(self):
        client = self.client()
        original = [[0., 0.], [1., 0.], [0., 1.]]
        cropped = [[95.5, 95.5], [287.5, 95.5], [95.5, 287.5]]
        proposals = iter((
            dict(visible=True, keypoints_px=original, confidence=.45, description="thin rack"),
            dict(visible=True, keypoints_px=cropped, confidence=.9, description="rack crop"),
            dict(visible=True, keypoints_px=original, confidence=.45, description="small handle"),
            dict(visible=True, keypoints_px=cropped, confidence=.9, description="handle crop")))
        def respond(**request):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                name="report_object_frame_keypoints", input=next(proposals))])
        with patch.object(client.messages, "create", side_effect=respond) as create:
            result = preflight(self.package, self.package / "review", "claude-test", client=client)
        self.assertEqual(create.call_count, 4)
        self.assertEqual(result["maximum_requests"], 4)
        self.assertEqual(result["requests_attempted"], 4)
        self.assertEqual(result["status"], "preflight_failed")
        self.assertFalse(result["simulator_started"])
        for target in ("rack_peg", "mug_handle"):
            row = result["targets"][target]
            self.assertEqual(row["status"], "passed")
            np.testing.assert_allclose(row["proposal"]["keypoints_px"], original)
        self.assertTrue((self.package / "review/mug_handle_crop_report.json").exists())

    def test_fresh_handle_crop_retry_is_used_by_seating_target_preparation(self):
        from dataclasses import replace
        from harness.perception.anchors import AnchorRegistry
        from harness.perception.grounding import RGBDPerceptionProvider
        from harness.perception.hang_targets import StaticRackFrameCache
        from harness.perception.preparation import PerceptionAnchorPreparer
        from harness.skills.schemas import SkillCommand
        current = replace(self.observation, sequence=700)
        points = [[0., 0.], [1., 0.], [0., 1.]]
        cache = StaticRackFrameCache()
        cache.cache(self.observation, dict(status="grounded", target="rack_peg",
            camera="perception_camera", observation_sequence=660, image_shape_px=[4, 4],
            confidence=.9, keypoints_px=points))
        client = self.client()
        proposals = iter((
            dict(visible=True, keypoints_px=points, confidence=.45, description="handle"),
            dict(visible=True, keypoints_px=[[95.5, 95.5], [287.5, 95.5], [95.5, 287.5]],
                 confidence=.9, description="magnified handle")))
        def respond(**request):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                name="report_object_frame_keypoints", input=next(proposals))])
        store = ObservationStore(self.package / "observations")
        grounder = RecordedGrounder(client, "claude-test", "mug_handle", self.package,
                                    store, lambda event: None)
        anchors = AnchorRegistry()
        prepare = PerceptionAnchorPreparer(RGBDPerceptionProvider(None, anchors),
            SimpleNamespace(arm_state=lambda arm: {"pose": [0., 0., 1., 1., 0., 0., 0.]}),
            anchors, hang_target_cache=cache, mug_frame_grounder=grounder,
            mug_frame_camera="perception_camera")
        with patch.object(client.messages, "create", side_effect=respond) as create:
            target = prepare(SkillCommand(action="move_to", arm="right",
                                          parameters={"target": "rack_seat"}), current)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(target.observation_sequence, 700)
        self.assertEqual(target.source, "perception")
        self.assertTrue((self.package / "mug_handle_00000700_crop_report.json").exists())
