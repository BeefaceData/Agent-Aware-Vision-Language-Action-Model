import os
from unittest.mock import Mock, patch
import unittest

from modal_apps.grounding660 import (
    grounding_files, launch, saved_crop_files, parse_observations, launch_saved_crops, DEFAULT_OBSERVATIONS,
)
from tests import test_hanging_grounding660_check as fixtures


class ModalGrounding660Tests(unittest.TestCase):
    def test_two_selected_frames_exclude_unrelated_archives(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sequence in (400, 416, 432):
                (root/f"observation_{sequence:08d}.npz").write_bytes(str(sequence).encode())
            remote = Mock(return_value={"results.json": b"{}"})
            launch_saved_crops("claude-test", root, root/"output", observations="416,432", remote=remote)
            remote.assert_called_once_with({416: b"416", 432: b"432"}, "claude-test")

    def test_sufficient_full_frame_skips_crop_but_never_authorizes_motion(self):
        import json
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        proposal = visible().model_copy(update={"complete_visible_opening": True,
            "contour_px": [(1., 1.)]*8, "material_px": [(2., 2.)]*12,
            "body_polygon_px": [(3., 3.)]*3, "peg_keypoints_px": [(4., 4.)]*3})
        response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_public_hanging",
                                                            input=proposal.model_dump())])
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
        files = saved_crop_files({660: self.fixture.archive.read_bytes()}, "claude-test", client=client)
        result = json.loads(files["results.json"])
        row = result["frames"][0]
        self.assertEqual(client.messages.create.call_count, 1)
        self.assertFalse(row["crop_needed"])
        self.assertFalse(row["crop_attempted"])
        self.assertEqual(row["requests_attempted"], 1)
        self.assertEqual(row["aperture"]["status"], "rejected")
        self.assertIsNone(row["clearance"]["clearance_validated"])
        self.assertEqual(row["recovery_eligibility"]["status"], "unavailable")
        self.assertFalse(row["motion_authorized"])

    def test_failed_full_and_crop_requests_consume_budget_without_retries(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        root = Path("data/runs/seed5_live_a209eedb8cb54725948b1e641a14b809/observations")
        if not root.exists():
            self.skipTest("saved recording unavailable")
        archives = {s: (root/f"observation_{s:08d}.npz").read_bytes() for s in (416, 432)}
        response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_public_hanging",
            input=visible().model_copy(update={"confidence": .2}).model_dump())])
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(
            side_effect=[RuntimeError("synthetic full-frame failure"), response,
                         RuntimeError("synthetic crop failure")])))
        files = saved_crop_files(archives, "claude-test", client=client)
        result = json.loads(files["results.json"])
        self.assertEqual(result["maximum_requests"], 4)
        self.assertEqual(result["requests_attempted"], 3)
        self.assertEqual(client.messages.create.call_count, 3)
        self.assertEqual(result["retries"], 0)
        first, second = result["frames"]
        self.assertEqual(first["requests_attempted"], 1)
        self.assertIn("failure", first)
        self.assertTrue(second["crop_attempted"])
        self.assertEqual(second["requests_attempted"], 2)
        for row in result["frames"]:
            self.assertFalse(row["motion_authorized"])
            self.assertIsNone(row["clearance"]["clearance_validated"])
        self.assertIn("public_grounding_failure", files["perception_events.jsonl"].decode())

    def test_aperture_candidate_does_not_replace_missing_clearance_evidence(self):
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        from modal_apps.grounding660 import saved_geometry_gates
        candidate = dict(status="candidate_not_motion_validated", motion_authorized=False)
        report = visible().model_copy(update={"complete_visible_opening": True,
            "contour_px": [(1., 1.)]*8, "material_px": [(2., 2.)]*12,
            "body_polygon_px": [(3., 3.)]*3})
        observation = SimpleNamespace(camera_frames={"perception_camera": Mock()})
        with patch("development.hanging.aperture_frame.estimate", return_value=candidate), patch(
                "development.hanging.public_provider.measured_bounds",
                side_effect=ValueError("visible_thickness_and_peg_cross_section_required")):
            result = saved_geometry_gates(observation, report)
        self.assertEqual(result["clearance"]["status"], "unavailable")
        self.assertIsNone(result["clearance"]["clearance_validated"])
        self.assertIn("visible_thickness_and_peg_cross_section_required",
                      result["recovery_eligibility"]["blockers"])
        self.assertFalse(result["motion_authorized"])

    def test_clearance_result_is_separate_from_recovery_and_motion(self):
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        from modal_apps.grounding660 import saved_geometry_gates
        report = visible().model_copy(update={"complete_visible_opening": True,
            "material_px": [(2., 2.)]*12, "body_polygon_px": [(3., 3.)]*3})
        observation = SimpleNamespace(camera_frames={"perception_camera": Mock()})
        for error, status in ((None, "validated"), ("observed_mating_clearance_insufficient", "rejected")):
            with self.subTest(status=status), patch("development.hanging.aperture_frame.estimate",
                    return_value=dict(status="candidate_not_motion_validated", motion_authorized=False)), patch(
                    "development.hanging.public_provider.measured_bounds",
                    return_value={"dimensions_validated": True},
                    side_effect=ValueError(error) if error else None):
                result = saved_geometry_gates(observation, report)
                self.assertEqual(result["clearance"]["status"], status)
                self.assertEqual(result["clearance"]["clearance_validated"], error is None)
                self.assertEqual(result["recovery_eligibility"]["status"], "unavailable")
                self.assertIn("model_geometry_requires_semantic_review",
                              result["recovery_eligibility"]["blockers"])
                self.assertFalse(result["motion_authorized"])

    def test_single_frame_selection_uploads_only_requested_archive(self):
        import tempfile
        import json
        from pathlib import Path
        self.assertEqual(parse_observations(),DEFAULT_OBSERVATIONS)
        self.assertEqual(parse_observations("464"),(464,))
        self.assertEqual(parse_observations(" 432, 464 "),(432,464))
        for value in ("464,464", "../464", "-1", "464,", " ", "1,2,3,4,5,6,7,8", None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                parse_observations(value)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"frames";source.mkdir()
            (source/"observation_00000464.npz").write_bytes(b"only selected archive")
            remote=Mock(return_value={"results.json":json.dumps({"selected_observations":[464]}).encode()})
            target=Path(directory)/"out"
            with patch.dict(os.environ,{},clear=True):
                launch_saved_crops("claude-test",source,target,observations="464",remote=remote)
            remote.assert_called_once_with({464:b"only selected archive"},"claude-test")
            self.assertTrue((target/"results.json").is_file())
            with self.assertRaises(FileExistsError):
                launch_saved_crops("claude-test",source,target,observations="464",remote=remote)
            self.assertEqual(remote.call_count,1)

    def test_saved_640_single_frame_has_two_call_cap_and_original_coordinate_mapping(self):
        import json
        import numpy as np
        from pathlib import Path
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        path=Path("data/runs/seed4_perception640_20261103_084536_510/observations/observation_00000464.npz")
        if not path.exists():self.skipTest("saved observation unavailable")
        initial=visible().model_copy(update={"body_polygon_px":None,"confidence":.2})
        angle=np.arange(16)*2*np.pi/16
        contour=np.column_stack((384+40*np.cos(angle),384+32*np.sin(angle)))
        crop=initial.model_copy(update={"contour_px":[tuple(p) for p in contour],
            "material_px":[(300.+i*2,300.) for i in range(12)]})
        def reply(report):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use",name="report_public_hanging",
                                                            input=report.model_dump())])
        client=SimpleNamespace(messages=SimpleNamespace(create=Mock(side_effect=[reply(initial),reply(crop)])),close=Mock())
        with patch("anthropic.Anthropic",return_value=client) as factory:
            files=saved_crop_files({464:path.read_bytes()},"claude-test")
        factory.assert_called_once_with(max_retries=0,timeout=45.)
        client.close.assert_called_once()
        report=json.loads(files["results.json"])
        self.assertEqual(report["maximum_requests"],2)
        self.assertEqual(report["requests_attempted"],2)
        self.assertEqual(report["retries"],0)
        self.assertEqual(report["selected_observations"],[464])
        self.assertEqual(client.messages.create.call_count,2)
        events=[json.loads(line) for line in files["perception_events.jsonl"].decode().splitlines()]
        event=next(e for e in events if e["event_type"]=="public_geometry_crop")
        left,top,right,bottom=event["crop_box"];height,width=event["shown_shape_px"]
        expected=(contour+.5)*[(right-left)/width,(bottom-top)/height]-.5+[left,top]
        np.testing.assert_allclose(report["frames"][0]["mapped_proposal"]["contour_px"],expected)
        prompt=client.messages.create.call_args_list[1].kwargs["system"]
        for term in ("THROUGH-HOLE","cup mouth","gripper fingers","mug cylinder"):
            self.assertIn(term,prompt)
        self.assertFalse(report["motion"])
        self.assertFalse(report["simulator_started"])

    def test_saved_crop_checker_is_bounded_and_never_initializes_simulator(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from tests.test_hanging_supervised import visible
        root = Path("data/runs/seed4_state_confidence_20261101_220527_553/observations")
        if not root.exists():
            self.skipTest("saved observations unavailable")
        archives = {seq: (root/f"observation_{seq:08d}.npz").read_bytes() for seq in range(416,513,16)}
        proposal = visible().model_copy(update={"body_polygon_px": [(90.,90.), (120.,90.), (120.,120.), (90.,120.)]})
        response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_public_hanging",
                                                            input=proposal.model_dump())])
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
        with patch("development.simulator.local_simulator.LocalSimulator") as simulator:
            files = saved_crop_files(archives, "claude-test", client=client)
            simulator.assert_not_called()
        report = json.loads(files["results.json"])
        self.assertEqual(report["requests_attempted"], 14)
        self.assertEqual(client.messages.create.call_count, 14)
        self.assertEqual(len(report["frames"]), 7)
        self.assertFalse(report["simulator_started"])
        self.assertFalse(report["motion"])
        self.assertEqual(len([f for f in files if f.endswith(".png")]), 7)
        self.assertTrue(all("frame_pose_world_wxyz" not in row["aperture"] for row in report["frames"]))
        for call in client.messages.create.call_args_list:
            self.assertEqual(call.kwargs["max_tokens"],4096)

    def test_saved_crop_input_validation_precedes_provider_calls(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "one_to_seven"):
            saved_crop_files({}, "claude-test", client=client)
        client.messages.create.assert_not_called()

    def setUp(self):
        self.fixture = fixtures.Grounding660CheckTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_local_upload_download_needs_no_anthropic_credentials(self):
        client = self.fixture.client()
        remote = Mock(side_effect=lambda data, model: grounding_files(data, model, client=client))
        output = self.fixture.output
        with patch.dict(os.environ, {}, clear=True):
            report = launch(self.fixture.archive, output, "explicit-test-model", remote)
        remote.assert_called_once_with(self.fixture.archive.read_bytes(), "explicit-test-model")
        self.assertEqual(len(client.messages.requests), 2)
        self.assertFalse(report["motion"])
        self.assertTrue(report["human_review_required"])
        for name in ("source.png", "rack_peg_proposal.png", "mug_handle_proposal.png", "results.json"):
            self.assertTrue((output / name).is_file())
        with self.assertRaises(FileExistsError):
            launch(self.fixture.archive, output, "explicit-test-model", remote)
        self.assertEqual(remote.call_count, 1)

    def test_secret_used_only_by_remote_checker_with_retries_disabled(self):
        client = self.fixture.client()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "synthetic-test-secret"}), \
                patch("anthropic.Anthropic", return_value=client) as factory:
            files = grounding_files(self.fixture.archive.read_bytes(), "explicit-test-model")
        factory.assert_called_once_with(api_key="synthetic-test-secret",
            base_url="https://api.anthropic.com", max_retries=0, timeout=60.)
        self.assertEqual(len(client.messages.requests), 2)
        self.assertIn("results.json", files)

    def test_invalid_input_does_not_reach_client_or_remote(self):
        remote = Mock()
        with self.assertRaisesRegex(ValueError, "explicit_model_required"):
            launch(self.fixture.archive, self.fixture.output, "", remote)
        remote.assert_not_called()
        client = self.fixture.client()
        with self.assertRaises(Exception):
            grounding_files(b"invalid archive", "explicit-test-model", client=client)
        self.assertEqual(client.messages.requests, [])
