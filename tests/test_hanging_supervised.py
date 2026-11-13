"""CPU-only supervisor/provider/launcher tests; all model and motion services fake."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from development.hanging.public_provider import (
    BoundedVisionClient, PublicRGBDHangingProvider, VisibleHangingReport,
    SurfacePixel, measured_bounds,
    PUBLIC_PROMPT,
)
from development.hanging.supervised import (
    SupervisedOptions, HangingRecoveryHook, service_preflight, run_controller, run,
    build_supervised_planner,
)
from harness.environment.base import Observation
from harness.environment.layout import CAMERAS
from harness.perception.observations import ObservationStore
from harness.perception.pose_grounding import _matrix_pose
from harness.perception.rgbd import RGBDCamera
from harness.perception.rgbd import calibration_sha256
from harness.planner.base import PlannerSelection
from harness.planner.base import PlannerContext
from harness.planner.anthropic import AnthropicPlanner
from harness.predicates.base import PredicateResult
from harness.skills.schemas import SkillCommand, SkillType
from harness.skills.executor import SkillExecutor
from harness.skills.schemas import EpisodeBudget
from tests.test_hanging_recovery_runner import PublicEnvironment
from tests import test_hanging_runtime_recovery as runtime_fixtures
from tests import test_anthropic_planner as planner_fixtures
from development.hanging import runtime


def visible(**updates):
    return VisibleHangingReport(stage="hanging", left_held=False, right_held=True,
        mug_off_table=True, transfer_complete=True, confidence=.95,
        task_state_confidence=.95,
        evidence="Synthetic visual attachment in consecutive frames", complete_visible_opening=False,
        opposite_handle_faces_visible=False, **updates)


class SupervisedHangingTests(unittest.TestCase):
    def test_deferred_takeover_still_requires_current_public_lift(self):
        environment = PublicEnvironment(x=0.)
        environment.sequence = 17
        observation = environment.observe()
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        for value, sequence in ((None, observation.sequence), (False, observation.sequence),
                                (True, observation.sequence-1)):
            with self.subTest(value=value, sequence=sequence):
                provider = Mock()
                provider.evaluate.return_value = PredicateResult(name="right_mug_lifted",
                    value=value, source="perception", observation_sequence=sequence, reason="test")
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="episode",
                    deadline=time.monotonic()+10, log=lambda _: None,
                    pause_policy_on_entry=False, require_current_lift=True)
                self.assertIsNone(hook.consider(selection, observation, remaining={}))
                provider.evaluate.assert_called_once_with("right_mug_lifted", observation, arm="right")
                provider.propose.assert_not_called()

    def test_carried_forward_lift_does_not_start_takeover(self):
        environment = PublicEnvironment(x=0.)
        environment.sequence = 40
        observation = environment.observe()
        provider = Mock()
        provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=True,
            source="perception", observation_sequence=observation.sequence,
            reason="bounded_temporal_lift_persistence")
        provider.lift_current.return_value = False
        hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
            deadline=time.monotonic()+10, log=lambda _: None)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        self.assertIsNone(hook.consider(selection, observation, remaining={}))
        provider.propose.assert_not_called()
        self.assertIsNone(hook.entry_sequence)

    def test_provider_lift_current_rejects_temporal_persistence_only(self):
        environment = PublicEnvironment(x=0.)
        observation = environment.observe()
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        provider._corroborated_count = 3
        provider._lift_motion_supported = True
        report = visible().model_copy(update={"task_state_confidence": None})
        with patch.object(provider, "observe", return_value=report):
            provider._lift_corroboration = dict(status="unknown", reason="public_lift_geometry_unavailable")
            carried = provider.evaluate("right_mug_lifted", observation)
            self.assertIs(carried.value, True)
            self.assertEqual(carried.reason, "bounded_temporal_lift_persistence")
            self.assertFalse(provider.lift_current(observation))
            provider._lift_corroboration = dict(status="confirmed", reason="lift_corroborated")
            self.assertTrue(provider.lift_current(observation))

    def test_default_takeover_requires_current_public_lift_not_raised_tcp_or_handoff_history(self):
        environment = PublicEnvironment(x=0.)
        environment.sequence = 17
        observation = environment.observe()
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        for value, source, sequence in (
            (None, "perception", observation.sequence),
            (False, "perception", observation.sequence),
            (True, "perception", observation.sequence-1),
            (True, "runtime", observation.sequence),
        ):
            with self.subTest(value=value, source=source, sequence=sequence):
                provider = Mock()
                provider.evaluate.side_effect = lambda name, obs, **kwargs: PredicateResult(
                    name=name, value=value if name == "right_mug_lifted" else True,
                    source=source, observation_sequence=sequence, reason="synthetic evidence")
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
                    deadline=time.monotonic()+10, log=lambda _: None)
                self.assertIsNone(hook.consider(selection, observation, remaining={}))
                provider.propose.assert_not_called()
                self.assertEqual(provider.evaluate.call_count, 1)
                self.assertIsNone(hook.entry_sequence)
                self.assertEqual(environment.ticks, 0)

    def test_default_blocked_takeover_is_latched_without_policy_resume_or_repeated_acquisition(self):
        environment = PublicEnvironment(x=0.)
        observation = environment.observe()
        provider = Mock()
        provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=True,
            source="perception", observation_sequence=observation.sequence, reason="confirmed lift")
        provider.propose.return_value = runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")
        provider.lift_current.return_value = True
        hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
            deadline=time.monotonic()+10, log=lambda _: None)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        remaining = dict(policy_actions=900, vla_chunks=57, skill_calls=80, analytic_ticks=12000)
        first = hook.consider(selection, observation, remaining=remaining)
        self.assertEqual(first["status"], "recovery_blocked")
        self.assertTrue(first["policy_paused"])
        self.assertFalse(first["motion_authorized"])
        self.assertEqual(first["control_owner"], "hanging_controller")
        self.assertNotIn("policy_command", first)
        self.assertIs(hook.consider(replace(selection, phase="GRASP"), observation,
                                    remaining=remaining), first)
        provider.propose.assert_called_once()
        self.assertEqual(provider.evaluate.call_count, 1)
        self.assertEqual(hook.attempts, 0)
        self.assertEqual(environment.ticks, 0)

    def test_default_takeover_keeps_deadline_and_recovery_budget_guards(self):
        environment = PublicEnvironment(x=0.)
        observation = environment.observe()
        for expired in (True, False):
            with self.subTest(expired=expired):
                provider = Mock()
                provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=True,
                    source="perception", observation_sequence=observation.sequence, reason="confirmed lift")
                provider.propose.return_value = runtime.ProposalResult("confirmed", "synthetic proposal")
                provider.lift_current.return_value = True
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
                    deadline=time.monotonic()+(-1 if expired else 10), log=lambda _: None)
                with patch("development.hanging.supervised.recover_current") as recover:
                    result = hook.consider(PlannerSelection(SkillCommand(action="vla_act"), "HANG"),
                        observation, remaining=dict(policy_actions=900, analytic_ticks=0, skill_calls=3))
                self.assertEqual(result["reason"],
                    "supervisor_episode_deadline" if expired else "recovery_budget_unavailable")
                self.assertNotIn("policy_command", result)
                recover.assert_not_called()
                self.assertEqual(environment.ticks, 0)

    def test_body_tracking_uses_measured_rotation_and_reports_slip_without_authorizing_motion(self):
        from harness.perception.object_pose import ObjectPose
        from harness.perception.pose_grounding import _pose_matrix

        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
                                            store=Mock(), log=lambda event: None)
        observation, proposal, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        report = visible(body_polygon_px=proposal.body_polygon_px)
        def sample(sequence, tcp_pose, slip=0., trusted=True, base_known=True):
            tcp = _pose_matrix(tcp_pose)
            position = tcp[:3, :3] @ np.array([0., 0., .1])+tcp[:3, 3]+np.array([slip, 0., 0.])
            pose = ObjectPose("mug", tuple(position), tuple(tcp[:3, 2]), base_known,
                .04, None, .9 if trusted else .4, trusted, "synthetic", ("perception_camera",), sequence)
            current = replace(observation, sequence=sequence, measured_tcp_poses={
                "right": tcp_pose, "left": (.5, 0., 1., 1., 0., 0., 0.)})
            with patch("development.hanging.public_provider.object_pose", return_value=(pose, "synthetic")):
                return provider._track_mug_body(current, report)

        initial = (0., 0., 1., 1., 0., 0., 0.)
        self.assertEqual(sample(1, initial)["reason"], "body_tracking_history_unavailable")
        self.assertEqual(sample(2, initial)["reason"], "insufficient_measured_tcp_motion")
        rotated = (.03, .02, 1., np.cos(.4), 0., np.sin(.4), 0.)
        evidence = sample(3, rotated)
        self.assertEqual(evidence["status"], "supported")
        self.assertAlmostEqual(evidence["position_residual_m"], 0.)
        self.assertAlmostEqual(evidence["axis_line_residual_deg"], 0.)
        slipped = sample(4, (.06, .02, 1., *rotated[3:]), slip=.05)
        self.assertEqual(slipped["status"], "inconsistent")
        self.assertFalse(slipped["motion_authorized"])
        self.assertFalse(slipped["grasp_verified"])
        self.assertFalse(provider._lift_motion_supported)
        self.assertEqual(provider._corroborated_count, 0)
        self.assertEqual(sample(5, initial, trusted=False)["status"], "unknown")
        self.assertIsNone(provider._mug_body_track)
        self.assertEqual(sample(6, initial, base_known=False)["reason"], "body_reference_end_unknown")
        self.assertIsNone(provider._mug_body_track)
        self.assertEqual(provider.requests, 0)

    def test_body_tracking_missing_polygon_and_incompatible_history_remain_unknown(self):
        from harness.perception.object_pose import ObjectPose

        observation, proposal, tcp = runtime_fixtures.RuntimeRecoveryTests().fixture()
        observation = replace(observation, measured_tcp_poses={
            "right": tuple(tcp), "left": (.5, 0., 1., 1., 0., 0., 0.)})
        report = visible(body_polygon_px=proposal.body_polygon_px)
        pose = ObjectPose("mug", (0., 0., 1.), (0., 0., 1.), True, .04, None,
                          .9, True, "synthetic", ("perception_camera",), observation.sequence)
        for change in ("missing", "stale", "duplicate", "episode", "calibration"):
            with self.subTest(change=change):
                provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
                                                    store=Mock(), log=lambda event: None)
                with patch("development.hanging.public_provider.object_pose", return_value=(pose, "synthetic")):
                    provider._track_mug_body(observation, report)
                    current = replace(observation, sequence=observation.sequence+16)
                    current_report = report
                    if change == "missing":
                        current_report = report.model_copy(update={"body_polygon_px": None})
                    elif change == "stale":
                        current = replace(current, sequence=observation.sequence+33)
                    elif change == "duplicate":
                        current = observation
                    elif change == "episode":
                        provider.episode_id = "another"
                    else:
                        camera = observation.camera_frames["perception_camera"]
                        intrinsic = camera.intrinsic.copy(); intrinsic[0, 0] += 1.
                        current = replace(current, camera_frames={"perception_camera": RGBDCamera(
                            camera.depth_m, intrinsic, camera.world_to_camera)})
                    evidence = provider._track_mug_body(current, current_report)
                self.assertEqual(evidence["status"], "unknown")
                self.assertFalse(evidence["motion_authorized"])
                self.assertFalse(provider._lift_motion_supported)

    def test_measured_tcp_rotation_preserves_surface_identity_but_slip_does_not(self):
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
                                            store=Mock(), log=lambda event: None)
        observation, _, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        grid = np.array([[x, y, .1] for x in np.linspace(-.02, .02, 9)
                         for y in np.linspace(-.01, .01, 5)])
        def sample(sequence, pose, slip=0.):
            from harness.perception.pose_grounding import _pose_matrix
            matrix = _pose_matrix(pose)
            points = grid@matrix[:3, :3].T+matrix[:3, 3]+slip
            current = replace(observation, sequence=sequence, measured_tcp_poses={
                "left": (.5, 0., 1., 1., 0., 0., 0.), "right": pose})
            return provider._track_mug_surface(current, points)
        self.assertTrue(sample(1, (0., 0., 1., 1., 0., 0., 0.)))
        self.assertTrue(sample(2, (0., 0., 1., 1., 0., 0., 0.)))
        self.assertFalse(provider._lift_motion_supported)
        self.assertEqual(provider._lift_corroboration["co_motion"], "insufficient_motion")
        rotated = (.03, .02, 1., np.cos(.4), 0., np.sin(.4), 0.)
        self.assertTrue(sample(3, rotated))
        self.assertTrue(provider._lift_motion_supported)
        self.assertAlmostEqual(provider._lift_corroboration["surface_overlap_fraction"], 1.)
        self.assertFalse(sample(4, (.06, .02, 1., *rotated[3:]), slip=.08))
        self.assertEqual(provider._lift_corroboration["reason"], "mug_tcp_motion_inconsistent")

    def test_lift_table_cache_rejects_inconsistent_planes_and_expires(self):
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
            store=Mock(), log=lambda event: None, table_plane_max_age=32)
        observation, _, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        report = visible(table_material_px=[(float(i), 0.) for i in range(6)])
        points = [np.array([x, y, 1.]) for x in (-.1, 0., .1) for y in (-.1, .1)]
        with patch("development.hanging.public_provider.read_surface", side_effect=[(p, 0.) for p in points]):
            provider._lift_table_plane(observation, report)
        missing = report.model_copy(update={"table_material_px": None})
        center, normal = provider._lift_table_plane(replace(observation, sequence=33), missing)
        np.testing.assert_allclose(normal, (0., 0., 1.))
        self.assertAlmostEqual(center[2], 1.)
        with self.assertRaisesRegex(ValueError, "table_plane_invalid"):
            provider._lift_table_plane(replace(observation, sequence=50), missing)
        with patch("development.hanging.public_provider.read_surface", side_effect=[(p, 0.) for p in points]):
            provider._lift_table_plane(observation, report)
        with patch("development.hanging.public_provider.read_surface",
                   side_effect=[(p+np.array([0., 0., .02]), 0.) for p in points]):
            with self.assertRaisesRegex(ValueError, "table_plane_inconsistent"):
                provider._lift_table_plane(replace(observation, sequence=18), report)
        self.assertIsNone(provider._table_plane)

    def test_unknown_reports_never_confirm_without_motion_or_positive_evidence(self):
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
                                            store=Mock(), log=lambda event: None)
        observation, _, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        report = visible().model_copy(update={"right_held": None, "task_state_confidence": .5})
        for sequence in (18, 34, 50):
            with patch("development.hanging.public_provider.request_tool_input", return_value=report.model_dump()):
                self.assertIsNone(provider.evaluate("right_mug_lifted", replace(observation, sequence=sequence)).value)
            self.assertEqual(provider._corroborated_count, 0)
        provider._corroborated_count = 3
        provider._lift_motion_supported = False
        with patch.object(provider, "observe", return_value=report):
            self.assertIsNone(provider.evaluate("right_mug_lifted", observation).value)

    def test_temporal_lift_resets_on_episode_calibration_age_and_gripper_open(self):
        observation, _, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        for change in ("episode", "calibration", "age", "open", "table", "separation"):
            with self.subTest(change=change):
                provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test",
                                                    store=Mock(), log=lambda event: None)
                camera = observation.camera_frames["perception_camera"]
                key = (observation.sequence, "unused", calibration_sha256(camera))
                provider._key = key
                provider._corroborated_count = 3
                provider._mug_body_track = "old body evidence"
                provider._lift_motion_supported = True
                provider._last_lift_sequence = observation.sequence
                current = replace(observation, sequence=observation.sequence+16)
                report = visible().model_copy(update={"right_held": None, "task_state_confidence": .5})
                if change == "episode":
                    provider.episode_id = "another"
                elif change == "calibration":
                    key = (key[0], key[1], "different")
                elif change == "age":
                    current = replace(current, sequence=observation.sequence+33)
                elif change == "open":
                    state = list(current.command_state); state[13] = 1.
                    current = replace(current, command_state=tuple(state))
                elif change == "table":
                    report = report.model_copy(update={"mug_off_table": False})
                else:
                    report = report.model_copy(update={"right_held": False})
                provider._update_temporal_lift(current, report, (current.sequence, key[1], key[2]))
                self.assertEqual(provider._corroborated_count, 0)
                self.assertFalse(provider._lift_motion_supported)
                self.assertIsNone(provider._mug_body_track)

    def test_seed5_saved_evidence_confirms_entry_at_416_without_authorizing_motion(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path("data/runs/seed5_live_a209eedb8cb54725948b1e641a14b809")
        if not root.exists():
            self.skipTest("saved recording unavailable")
        rows = [json.loads(line) for line in (root/"supervisor_events.jsonl").read_text().splitlines()]
        proposals = {r["observation_sequence"]: r for r in rows if r.get("event_type") == "public_hanging_proposal"}
        self.assertEqual(proposals[384]["lift_corroboration"]["reason"], "association_or_clearance_failed")
        self.assertEqual(proposals[416]["lift_corroboration"]["reason"], "table_plane_invalid")
        self.assertEqual(proposals[448]["corroborated_lift_count"], 0)
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="seed5",
                                            store=Mock(), log=lambda event: None)
        expected = {368: None, 384: None, 400: None, 416: True, 432: True,
                    448: True, 464: True, 480: False}
        for sequence, value in expected.items():
            observation = load_public_observation(root/"observations"/f"observation_{sequence:08d}.npz")
            with patch("development.hanging.public_provider.request_tool_input", return_value=proposals[sequence]["report"]):
                result = provider.evaluate("right_mug_lifted", observation)
            self.assertIs(result.value, value, (sequence, provider._lift_corroboration))
            if sequence in (384, 400):
                self.assertGreater(provider._lift_corroboration["right_tcp_distance_m"], .22)
                self.assertLess(provider._lift_corroboration["right_surface_distance_m"], .22)
            if sequence == 416:
                self.assertEqual(provider._lift_corroboration["table_plane_source"], "episode_static_cache")
                self.assertEqual(provider._lift_corroboration["co_motion"], "supported")
                body = provider._lift_corroboration["body_pose_tracking"]
                self.assertEqual(body["status"], "supported")
                self.assertLess(body["position_residual_m"], .015)
                self.assertFalse(body["motion_authorized"])
                planner = build_supervised_planner(SupervisedOptions("claude-test"),
                    planner_fixtures.Provider([planner_fixtures.response(task_phase="grasp")]))
                selection = planner.select(PlannerContext(27, observation, (result,), None,
                    remaining_budget={"policy_actions": 484, "vla_chunks": 30}))
                self.assertEqual(selection.phase, "HANG")
                environment = Mock()
                environment.arm_state.return_value = {"pose": observation.measured_tcp_poses["right"]}
                decisions = []
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="seed5",
                    deadline=time.monotonic()+60, log=decisions.append)
                with patch.object(provider, "crop_geometry", return_value=VisibleHangingReport.model_validate(proposals[sequence]["report"])):
                    decision = hook.consider(selection, observation, remaining={"policy_actions": 484,
                        "vla_chunks": 30, "skill_calls": 30, "analytic_ticks": 9000})
                self.assertEqual(decision["status"], "recovery_blocked")
                self.assertTrue(decision["policy_paused"])
                self.assertNotIn("policy_command", decision)
                self.assertEqual(decision["reason"], "complete_visible_aperture_unavailable")
                self.assertEqual(hook.attempts, 0)
                progress = next(event for event in decisions
                                if event["event_type"] == "hanging_supervisor_decision")
                self.assertEqual(progress["task_progress"], "unknown")
                self.assertIsNone(progress["failure_detected"])
            if sequence in (448, 464):
                self.assertEqual(provider._corroborated_count, 4)  # Persistence adds no positives.
                self.assertIsNone(provider._mug_body_track)

    def test_saved_seed5_material_proposals_remain_rejected_by_current_depth_checks(self):
        import json
        from development.hanging.aperture_frame import estimate
        from harness.perception.replay import load_public_observation

        recordings = Path("data/runs")
        archives = recordings/"seed5_live_a209eedb8cb54725948b1e641a14b809"/"observations"
        for review, sequence in (
            ("seed5_mug_crop_59f20f04ead34fc88f3bfdd350a57e31", 416),
            ("seed5_views_af4fd356b5074c2c9e1f87bee5aa42fc", 440),
        ):
            with self.subTest(sequence=sequence):
                path = recordings/review/"results.json"
                archive = archives/f"observation_{sequence:08d}.npz"
                if not path.exists() or not archive.exists():
                    self.skipTest("saved geometry evidence unavailable")
                frame = next(frame for frame in json.loads(path.read_text())["frames"]
                             if frame["sequence"] == sequence)
                proposal = frame["mapped_proposal"]
                observation = load_public_observation(archive)
                aperture = estimate(observation.camera_frames["perception_camera"],
                                    proposal["contour_px"], proposal["material_px"])
                self.assertEqual(aperture["status"], "rejected")
                self.assertEqual(aperture["reason"], "material_depth_discontinuity")
                self.assertFalse(aperture["motion_authorized"])

    def test_seed4_live2_missing_reports_do_not_become_positive_confirmations(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path("data/runs/seed4_live2_3f625777b3874d3f836f0bf71d937198")
        if not root.exists():
            self.skipTest("saved recording unavailable")
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="seed4",
                                            store=Mock(), log=lambda event: None)
        seen = []
        for row in map(json.loads, (root/"supervisor_events.jsonl").read_text().splitlines()):
            if row.get("event_type") != "public_hanging_proposal" or not 352 <= row["observation_sequence"] <= 480:
                continue
            sequence = row["observation_sequence"]
            observation = load_public_observation(root/"observations"/f"observation_{sequence:08d}.npz")
            with patch("development.hanging.public_provider.request_tool_input", return_value=row["report"]):
                self.assertIsNone(provider.evaluate("right_mug_lifted", observation).value)
            seen.append(sequence)
        self.assertEqual(seen, list(range(352, 481, 16)))
        self.assertEqual(provider._corroborated_count, 1)

    def test_high_resolution_tcp_fallback_excludes_rack_and_preserves_its_landmarks(self):
        image = np.zeros((480,640,3), np.uint8)
        k = np.array([[415.692,0.,320.], [0.,415.692,240.], [0.,0.,1.]])
        camera = RGBDCamera(np.ones((480,640), np.float32), k, np.eye(4))
        observation = Observation(81, {name:image for name in (*CAMERAS, "perception_camera")}, tuple(np.zeros(14)),
            {"perception_camera":camera}, {"right":(0.,0.,1.,1.,0.,0.,0.), "left":(.4,0.,1.,1.,0.,0.,0.)})
        report = visible().model_copy(update={"body_polygon_px":None, "rack_region_px":
            [(100.,180.), (140.,180.), (140.,260.), (100.,260.)],
            "peg_keypoints_px": [(100.,200.), (140.,200.), (100.,180.)], "peg_confidence": .9})
        logs=[]
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=logs.append)
        crop_report = visible().model_copy(update={"contour_px": [(10.,10.), (50.,10.), (10.,50.)],
            "peg_keypoints_px": [(1.,1.), (2.,1.), (1.,2.)], "peg_confidence": .1})
        with patch("development.hanging.public_provider.request_tool_input", return_value=crop_report.model_dump()) as request:
            result = provider.crop_geometry(observation, report)
            provider.crop_geometry(observation, report)
        request.assert_called_once()
        event = next(e for e in logs if e["event_type"]=="public_geometry_crop")
        left,top,right,bottom=event["crop_box"]
        self.assertEqual(event["crop_box"], [224,144,416,336])
        self.assertEqual(event["crop_source"], "measured_right_tcp")
        self.assertEqual(result.peg_keypoints_px, report.peg_keypoints_px)
        self.assertEqual(result.peg_confidence, report.peg_confidence)
        height,width=event["shown_shape_px"]
        expected = (np.array(crop_report.contour_px)+.5)*[(right-left)/width,(bottom-top)/height]-.5+[left,top]
        np.testing.assert_allclose(result.contour_px, expected)
        behind=replace(observation, measured_tcp_poses={"right":(0.,0.,-1.,1.,0.,0.,0.),
                                                       "left":(.4,0.,1.,1.,0.,0.,0.)})
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda e:None)
        with patch("development.hanging.public_provider.request_tool_input") as request:
            self.assertIs(provider.crop_geometry(behind, report), report)
            request.assert_not_called()

    def test_episode_static_rack_reference_survives_occlusion_not_calibration_changes(self):
        observation, proposal, tcp = runtime_fixtures.RuntimeRecoveryTests().fixture()
        observation = replace(observation, images={name:observation.images["perception_camera"]
                                                   for name in (*CAMERAS, "perception_camera")})
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode-a",
                                             store=Mock(), log=lambda e:None)
        report = visible().model_copy(update={"confidence":.2, "peg_confidence":.95,
                                              "peg_keypoints_px": [(50.,50.),(60.,50.),(50.,60.)],
                                              "peg_width_px": [(55.,48.),(55.,52.)]})
        provider._cache_rack(observation, report)
        self.assertIsNotNone(provider._rack_reference)
        later=replace(observation, sequence=33)
        provider._cache_rack(later, visible())
        self.assertEqual(provider._rack_reference[0].sequence,17)
        cached=replace(proposal, sequence=33, rack_reference_observation=observation)
        result=runtime.validate_current_geometry(later,tcp,runtime.ProposalResult("confirmed","fixture",cached),episode_id="episode-a")
        self.assertEqual(result.status,"confirmed",result.reason)
        changed=observation.camera_frames["perception_camera"]
        k=changed.intrinsic.copy();k[0,0]+=1.
        incompatible=replace(later,camera_frames={"perception_camera":RGBDCamera(changed.depth_m,k,changed.world_to_camera)})
        provider._cache_rack(incompatible,visible())
        self.assertIsNone(provider._rack_reference)
        other=PublicRGBDHangingProvider(Mock(),model="claude-test",episode_id="different",store=Mock(),log=lambda e:None)
        self.assertIsNone(other._rack_reference)
        provider._cache_rack(observation, report)
        provider.episode_id = "another-episode"
        self.assertIsNone(provider._current_rack_reference(later))

    def test_seed5_416_mug_crop_retains_review_region_without_rack_expansion(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path("data/runs/seed5_geometry_fbf0947149764aea898501f7ff7942fc")
        archive = Path("data/runs/seed5_live_a209eedb8cb54725948b1e641a14b809/observations/observation_00000416.npz")
        if not root.exists() or not archive.exists():
            self.skipTest("saved recording unavailable")
        report = VisibleHangingReport.model_validate(json.loads((root/"results.json").read_text())["frames"][0]["full_frame_proposal"])
        observation = load_public_observation(archive)
        logs = []
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="seed5",
                                             store=Mock(), log=logs.append, maximum_requests=1)
        with patch("development.hanging.public_provider.request_tool_input", return_value=report.model_dump()) as request:
            result = provider.crop_geometry(observation, report)
            provider.crop_geometry(observation, report)
        request.assert_called_once()
        event = next(e for e in logs if e["event_type"] == "public_geometry_crop")
        self.assertEqual(event["crop_box"], [118,84,306,273])
        self.assertEqual(event["padding_px"], 48.)
        self.assertEqual(event["crop_source"], "mug_body_region")
        self.assertIsNone(event["rack_reference_sequence"])
        self.assertEqual(result.peg_keypoints_px, report.peg_keypoints_px)
        self.assertIn("Rack grounding is separate", request.call_args.kwargs["system"])

    def test_mug_crop_clips_at_image_boundary_and_does_not_expand_for_rack(self):
        observation, _, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        report = visible().model_copy(update={"body_polygon_px": [(1.,1.), (8.,1.), (8.,8.), (1.,8.)],
            "rack_region_px": [(80.,80.), (90.,80.), (90.,90.)]})
        logs = []
        provider = PublicRGBDHangingProvider(Mock(), model="offline", episode_id="test", store=Mock(), log=logs.append)
        with patch("development.hanging.public_provider.request_tool_input", return_value=visible().model_dump()):
            provider.crop_geometry(observation, report)
        event = next(e for e in logs if e["event_type"] == "public_geometry_crop")
        self.assertEqual(event["crop_box"], [0,0,32,32])
        self.assertEqual(event["rack_grounding"], "separate_full_frame_or_cache")

    def test_crop_geometry_maps_pixels_to_original_depth_and_preserves_state(self):
        from development.hanging.aperture_frame import estimate
        from development.hanging.grounding_preflight import CROP_LONG_SIDE_PX
        image = np.zeros((201, 201, 3), np.uint8)
        camera = RGBDCamera(np.ones((201, 201), np.float32),
            np.array([[200., 0., 100.], [0., 200., 100.], [0., 0., 1.]]), np.eye(4))
        observation = Observation(448, {name:image for name in (*CAMERAS, "perception_camera")}, tuple(np.zeros(14)),
                                  {"perception_camera": camera})
        report = visible().model_copy(update=dict(confidence=.2, transfer_complete=None,
            body_polygon_px=[(94., 94.), (106., 94.), (106., 106.), (94., 106.)]))
        angles = np.arange(16)*2*np.pi/16
        contour = np.column_stack((100+10*np.cos(angles), 100+8*np.sin(angles)))
        material = np.rint(np.column_stack((100+17*np.cos(angles), 100+15*np.sin(angles))))
        # Body-only box [70,70,130,130], with the same pixel-center mapping.
        def shown(points):
            return [tuple(p) for p in ((np.asarray(points)-70+.5)*CROP_LONG_SIDE_PX/60-.5)]
        proposed = report.model_copy(update=dict(confidence=.95, task_state_confidence=.1,
            complete_visible_opening=True, contour_px=shown(contour), material_px=shown(material),
            body_polygon_px=shown(report.body_polygon_px), peg_keypoints_px=shown([(80.,80.), (90.,80.), (80.,70.)])))
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        with patch("development.hanging.public_provider.request_tool_input", return_value=proposed.model_dump()) as request:
            result = provider.crop_geometry(observation, report)
            self.assertIs(provider.crop_geometry(observation, report), result)
        self.assertEqual(request.call_count, 1)
        np.testing.assert_allclose(result.contour_px, contour)
        np.testing.assert_allclose(result.material_px, material)
        self.assertEqual(result.task_state_confidence, report.task_state_confidence)
        assessment = estimate(camera, result.contour_px, result.material_px)
        self.assertEqual(assessment["status"], "candidate_not_motion_validated")
        self.assertFalse(assessment["motion_authorized"])
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        invalid = proposed.model_copy(update={"contour_px": [(-1., 2.)]*8})
        with patch("development.hanging.public_provider.request_tool_input", return_value=invalid.model_dump()):
            self.assertIs(provider.crop_geometry(observation, report), report)

    def test_saved_geometry_nulls_originate_in_proposals_and_remain_unavailable(self):
        import json
        from harness.perception.replay import load_public_observation
        from development.hanging.aperture_frame import estimate
        root = Path("data/runs/seed4_state_confidence_20261101_220527_553")
        if not root.exists():
            self.skipTest("saved recording is not part of the repository")
        rows = [json.loads(line) for line in (root/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
        selected = [r for r in rows if r.get("event_type") == "public_hanging_proposal"
                    and 416 <= r["observation_sequence"] <= 512]
        self.assertEqual(len(selected), 7)
        for row in selected:
            with self.subTest(sequence=row["observation_sequence"]):
                observation = load_public_observation(root/"observations"/f'observation_{row["observation_sequence"]:08d}.npz')
                provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                                     store=Mock(), log=lambda event: None)
                with patch("development.hanging.public_provider.request_tool_input", return_value=row["report"]):
                    report = provider.observe(observation)
                for field in ("contour_px", "material_px", "peg_keypoints_px"):
                    self.assertIsNone(getattr(report, field))
                original = estimate(observation.camera_frames["perception_camera"], report.contour_px, report.material_px)
                self.assertEqual(original["reason"], "insufficient_or_invalid_pixels")
                # Simulate an honest unavailable crop response, not invented annotations.
                unavailable = report.model_copy(update={"body_polygon_px": None})
                with patch("development.hanging.public_provider.request_tool_input", return_value=unavailable.model_dump()) as request:
                    result = provider.crop_geometry(observation, report)
                    provider.crop_geometry(observation, report)
                self.assertEqual(request.call_count, 1)  # Measured TCP supplies the missing region.
                self.assertIsNone(result.contour_px)
                self.assertIsNone(result.peg_keypoints_px)
                self.assertNotEqual(estimate(observation.camera_frames["perception_camera"],
                    result.contour_px, result.material_px)["status"], "candidate_not_motion_validated")

    def test_restart_preserves_partial_artifacts_without_services(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"run"
            output.mkdir()
            (output/"episode.mp4").write_bytes(b"partial video")
            (output/"supervisor_events.jsonl").write_bytes(b"partial events")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch("development.hanging.supervised.service_preflight") as preflight, \
                    patch("anthropic.Anthropic") as sdk, \
                    patch("development.simulator.local_simulator.LocalSimulator") as simulator:
                for _ in range(2):
                    report = run("unused", output, SupervisedOptions("claude-test"))
                    self.assertEqual(report["status"], "interrupted_attempt_exists")
                    self.assertFalse(report["automatic_resume"])
                preflight.assert_not_called()
                sdk.assert_not_called()
                simulator.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_three_moderate_reports_require_rgbd_closure_and_reset(self):
        depth = np.ones((101, 101), np.float32)
        depth[40:61, 40:61] = .9
        extrinsic = np.diag([1., -1., -1., 1.])
        extrinsic[2, 3] = 2.
        camera = RGBDCamera(depth, np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]), extrinsic)
        image = np.zeros((101, 101, 3), np.uint8)
        observation = Observation(0, {name: image for name in (*CAMERAS, "perception_camera")},
            tuple(np.zeros(14)), {"perception_camera": camera},
            {"right": (0., 0., 1.1, 1., 0., 0., 0.), "left": (.4, 0., 1.1, 1., 0., 0., 0.)})
        report = visible().model_copy(update=dict(task_state_confidence=.75, confidence=.2,
            transfer_complete=None, right_gripper_closed_visible=True,
            body_polygon_px=[(44., 44.), (56., 44.), (56., 56.), (44., 56.)],
            table_material_px=[(10., 10.), (50., 10.), (90., 10.), (10., 90.), (50., 90.), (90., 90.)]))
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        self.assertTrue(provider._corroborates_lift(observation, report))
        def read(sequence, current=report, obs=observation):
            angle = sequence*.01
            poses = dict(obs.measured_tcp_poses)
            poses["right"] = (*poses["right"][:3], np.cos(angle/2), 0., 0., np.sin(angle/2))
            obs = replace(obs, measured_tcp_poses=poses)
            with patch("development.hanging.public_provider.request_tool_input", return_value=current.model_dump()):
                return provider.evaluate("right_mug_lifted", replace(obs, sequence=sequence))
        self.assertIsNone(read(16).value)
        self.assertIsNone(read(32).value)
        self.assertIsNone(read(32).value)  # Cache reread is not a third frame.
        self.assertIs(read(48).value, True)
        self.assertEqual(provider.propose(replace(observation, sequence=48), None,
                                         episode_id="episode").reason, "complete_visible_aperture_unavailable")
        intent = report.model_copy(update={"right_gripper_closed_visible": None})
        opened = list(observation.command_state)
        opened[13] = .8
        self.assertIsNone(read(64, intent, replace(observation, command_state=tuple(opened))).value)
        self.assertIsNone(read(80, intent).value)
        self.assertIsNone(read(96, intent).value)
        self.assertIs(read(112, intent).value, True)
        self.assertEqual(provider._lift_corroboration["closure_source"], "commanded_closure_not_measured")
        self.assertIsNone(read(128, intent, replace(observation, command_state=tuple(opened))).value)
        self.assertEqual(provider._corroborated_count, 0)
        for changes in ({"right_gripper_closed_visible": False}, {"right_held": False}):
            with self.subTest(changes=changes):
                seq = provider._key[0]+16
                self.assertIsNot(read(seq, report.model_copy(update=changes)).value, True)
                self.assertEqual(provider._corroborated_count, 0)
                self.assertIsNone(read(seq+16).value)
                self.assertIsNone(read(seq+32).value)
                self.assertIs(read(seq+48).value, True)
        # An occluded table can reuse its validated plane; a missing body does
        # not add a positive report and expires after the configured gap.
        seq = provider._key[0]+16
        self.assertIs(read(seq, report.model_copy(update={"table_material_px": None})).value, True)
        count = provider._corroborated_count
        missing = report.model_copy(update={"body_polygon_px": None})
        self.assertIs(read(seq+16, missing).value, True)
        self.assertEqual(provider._corroborated_count, count)
        self.assertIs(read(seq+32, missing).value, True)
        self.assertIsNone(read(seq+48, missing).value)
        wrong_arm = replace(observation, measured_tcp_poses={
            "left": observation.measured_tcp_poses["right"], "right": observation.measured_tcp_poses["left"]})
        self.assertFalse(provider._corroborates_lift(wrong_arm, report))
        changed_depth = np.ones((101, 101), np.float32)
        on_table = replace(observation, camera_frames={"perception_camera": RGBDCamera(
            changed_depth, camera.intrinsic, camera.world_to_camera)})
        self.assertFalse(provider._corroborates_lift(on_table, report))
        shifted = camera.world_to_camera.copy()
        shifted[0, 3] += .001
        recalibrated = replace(observation, camera_frames={"perception_camera": RGBDCamera(
            camera.depth_m, camera.intrinsic, shifted)})
        self.assertIsNone(read(provider._key[0]+16, obs=recalibrated).value)
        self.assertLessEqual(provider._corroborated_count, 1)

    def test_saved_416_to_512_does_not_fabricate_missing_corroboration(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path("data/runs/seed4_state_confidence_20261101_220527_553")
        if not root.exists():
            self.skipTest("saved recording is not part of the repository")
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        reports = [json.loads(line) for line in (root/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
        selected = [r for r in reports if r.get("event_type") == "public_hanging_proposal"
                    and 416 <= r["observation_sequence"] <= 512]
        self.assertEqual(len(selected), 7)
        for row in selected:
            observation = load_public_observation(root/"observations"/f'observation_{row["observation_sequence"]:08d}.npz')
            report = VisibleHangingReport.model_validate(row["report"])
            self.assertIs(report.right_held, True)
            self.assertIs(report.mug_off_table, True)
            self.assertFalse(provider._corroborates_lift(observation, report))
            with patch("development.hanging.public_provider.request_tool_input", return_value=row["report"]):
                result = provider.evaluate("right_mug_lifted", observation)
            self.assertIsNone(result.value)
        self.assertEqual(provider._corroborated_count, 0)

    def test_saved_640_filtered_component_confirms_lift_at_464(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path("data/runs/seed4_perception640_20261103_084536_510")
        if not root.exists():
            self.skipTest("saved recording unavailable")
        rows=[json.loads(line) for line in (root/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
        provider=PublicRGBDHangingProvider(Mock(),model="claude-test",episode_id="episode",store=Mock(),log=lambda e:None)
        for count,sequence in enumerate((432,448,464),1):
            observation=load_public_observation(root/"observations"/f"observation_{sequence:08d}.npz")
            report=next(row["report"] for row in rows if row.get("event_type")=="public_hanging_proposal"
                        and row["observation_sequence"]==sequence)
            self.assertLessEqual(observation.command_state[13],.05)
            self.assertIsNone(report["right_gripper_closed_visible"])
            with patch("development.hanging.public_provider.request_tool_input",return_value=report):
                result=provider.evaluate("right_mug_lifted",observation)
            self.assertEqual(result.value,True if count==3 else None)
            self.assertEqual(provider._corroborated_count,count)
            self.assertEqual(provider._lift_corroboration["closure_source"],"commanded_closure_not_measured")
            body = provider._lift_corroboration["body_pose_tracking"]
            self.assertEqual(body["status"], "unknown" if count == 1 else "supported")
            self.assertFalse(body["motion_authorized"])
            self.assertFalse(body["grasp_verified"])
            self.assertGreater(provider._lift_corroboration["minimum_mug_clearance_m"],.19)
            self.assertGreater(provider._lift_corroboration["right_association_margin_m"],.36)
            self.assertLessEqual(provider._lift_corroboration["right_tcp_distance_m"],.22)
        planner=build_supervised_planner(SupervisedOptions("claude-test"),
            planner_fixtures.Provider([planner_fixtures.response(task_phase="grasp")]))
        selection=planner.select(PlannerContext(30,observation,(result,),None,
            remaining_budget={"policy_actions":436,"vla_chunks":30}))
        self.assertEqual(selection.phase,"HANG")
        with patch.object(provider,"crop_geometry",return_value=VisibleHangingReport.model_validate(report)) as crop:
            result=provider.propose(observation,None,episode_id="episode")
            self.assertEqual(result.reason,"complete_visible_aperture_unavailable")
            crop.assert_called_once()

    def test_seed4_current_lift_advances_to_hang_without_transfer_or_geometry(self):
        # Recorded 432/448/464 facts, with a separate hypothetical state score.
        # Old scores remain geometry-only: never promote them into grasp evidence.
        environment = PublicEnvironment(x=0.)
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        client = planner_fixtures.Provider([planner_fixtures.response(task_phase="grasp")])
        planner = build_supervised_planner(SupervisedOptions("claude-test"), client)
        reports = [visible().model_copy(update=dict(confidence=confidence,
                    task_state_confidence=None, transfer_complete=transfer, left_held=left))
                   for confidence, transfer, left in ((.4, None, None), (.4, None, None), (.45, True, False))]
        for previous, current in zip(reports, reports[1:]):
            provider._previous = previous
            with patch.object(provider, "observe", return_value=current):
                self.assertIsNone(provider.evaluate("right_mug_lifted", environment.observe()).value)
        provider._previous = reports[0].model_copy(update={"task_state_confidence": .95})
        report = reports[1].model_copy(update={"task_state_confidence": .95})
        environment.sequence = 448
        observation = environment.observe()
        with patch.object(provider, "observe", return_value=report):
            lifted = provider.evaluate("right_mug_lifted", observation)
            handoff = provider.evaluate("handoff_confirmed", observation)
            self.assertIs(lifted.value, True)
            self.assertIsNone(handoff.value)
            context = PlannerContext(2, observation, (lifted, handoff), None,
                remaining_budget={"policy_actions": 900, "vla_chunks": 57})
            selection = planner.select(context)
            self.assertEqual(selection.phase, "HANG")
            hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="episode",
                                       deadline=time.monotonic()+10, log=lambda event: None)
            # Hook reaches real recovery, which must still reject missing geometry.
            with tempfile.TemporaryDirectory() as directory:
                hook.store = ObservationStore(Path(directory)/"observations")
                result = hook.consider(selection, observation,
                                      remaining={"analytic_ticks": 12000, "skill_calls": 5})
            self.assertEqual(result["reason"], "complete_visible_aperture_unavailable")
            self.assertEqual(result["stages"], {})
            self.assertEqual(environment.ticks, 0)
        for changes in ({"right_held": None}, {"right_held": False},
                        {"mug_off_table": False}, {"task_state_confidence": .4}):
            with self.subTest(changes=changes), patch.object(provider, "observe",
                    return_value=report.model_copy(update=changes)):
                self.assertIsNot(provider.evaluate("right_mug_lifted", observation).value, True)

    def test_state_confidence_does_not_authorize_low_confidence_geometry(self):
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        report = visible().model_copy(update=dict(confidence=.4, transfer_complete=None,
            complete_visible_opening=True, contour_px=[(1., 1.)]*8,
            material_px=[(2., 2.)]*12, body_polygon_px=[(3., 3.)]*3))
        provider._previous = report
        with patch.object(provider, "observe", return_value=report), patch(
                "development.hanging.public_provider.estimate") as estimate:
            self.assertIs(provider.evaluate("right_mug_lifted", SimpleNamespace(sequence=448)).value, True)
            self.assertEqual(provider.propose(SimpleNamespace(sequence=448, images=None), Mock(), episode_id="episode").reason,
                             "complete_visible_aperture_unavailable")
            estimate.assert_not_called()

    def test_provider_supplies_camera_identity_and_measured_tcp_pixels(self):
        environment = PublicEnvironment(x=0.)
        observation = environment.observe()
        frame = observation.camera_frames["perception_camera"]
        # Use calibrated rays to put anatomical right on image-left and vice versa.
        left = (*frame.unproject(20., 10.), 1., 0., 0., 0.)
        right = (*frame.unproject(10., 10.), 1., 0., 0., 0.)
        observation = replace(observation, measured_tcp_poses={"left": left, "right": right})
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        with patch("development.hanging.public_provider.request_tool_input",
                   return_value=visible().model_dump()) as request:
            provider.observe(observation)
        text = " ".join(b.get("text", "") for b in request.call_args.kwargs["content"])
        self.assertIn("robot_left_measured_TCP_uv=(20.0,10.0)", text)
        self.assertIn("robot_right_measured_TCP_uv=(10.0,10.0)", text)
        self.assertIn("never image-left/right", PUBLIC_PROMPT)
        self.assertIn("right_camera on the right wrist", PUBLIC_PROMPT)

    def test_generalized_planner_forwards_all_policy_actions_while_legacy_retains_lock(self):
        class ActionEnvironment(PublicEnvironment):
            def __init__(self):
                super().__init__(x=0.)
                self.actions = []
            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                return super().step(action)
        actions = np.arange(28, dtype=float).reshape(2, 14)/40
        for generalized in (True, False):
            with self.subTest(generalized=generalized):
                client = planner_fixtures.Provider([planner_fixtures.response(
                    parameters={"max_chunks": 1, "chunk_size": 16, "max_policy_actions": 16})])
                planner = build_supervised_planner(SupervisedOptions("claude-test"), client)
                if not generalized:
                    planner = AnthropicPlanner(planner.settings, client=client)
                environment = ActionEnvironment()
                observation = environment.observe()
                context = PlannerContext(1, observation, (), None,
                    remaining_budget={"policy_actions": 900, "vla_chunks": 57})
                selection = planner.select(context)
                self.assertEqual(selection.phase, "GRASP")
                self.assertEqual(selection.command.parameters.lock_arm, None if generalized else "right")
                policy = Mock()
                policy.infer.return_value = actions.copy()
                executor = SkillExecutor(environment, policy, log=lambda event: None,
                                         budget=EpisodeBudget(max_policy_actions=900))
                result = executor.execute(selection.command)
                self.assertEqual(result.executed_policy_actions, 2)
                forwarded = np.asarray(environment.actions)
                if generalized:
                    np.testing.assert_array_equal(forwarded, actions)
                else:
                    np.testing.assert_array_equal(forwarded[:, :7], actions[:, :7])
                    np.testing.assert_array_equal(forwarded[:, 7:], np.tile(observation.command_state[7:], (2, 1)))
                np.testing.assert_array_equal(actions, np.arange(28).reshape(2, 14)/40)

    def test_planner_summary_is_json_dictionary_for_all_confidence_and_grasp_states(self):
        import json
        provider = PublicRGBDHangingProvider(Mock(), model="claude-test", episode_id="episode",
                                             store=Mock(), log=lambda event: None)
        for confidence in (0., .3, .849, .85, .95, 1.):
            for right_held in (False, True, None):
                with self.subTest(confidence=confidence, right_held=right_held):
                    report = visible().model_copy(update={"confidence": confidence, "right_held": right_held})
                    with patch.object(provider, "observe", return_value=report):
                        summary = provider.planner_summary(Mock())
                    self.assertIsInstance(summary, dict)
                    self.assertEqual(json.loads(json.dumps(summary))["confidence"], confidence)
                    self.assertEqual(summary["stage"], report.stage)
                    self.assertIn("visual_evidence", summary)
        with patch.object(provider, "observe", return_value=visible().model_copy(update={"right_held": False})):
            result = provider.propose(Mock(), Mock(), episode_id="episode")
            self.assertEqual(result.status, "rejected")
            self.assertEqual(result.reason, "right_arm_not_holding_mug")

    def test_shared_client_records_tokens_latency_and_failed_calls_per_tool(self):
        from types import SimpleNamespace
        from development.hanging.public_provider import BoundedVisionClient
        usage = SimpleNamespace(input_tokens=1200, output_tokens=300, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_public_hanging")],
                                   usage=usage)
        inner = SimpleNamespace(messages=SimpleNamespace(create=Mock(
            side_effect=[response, RuntimeError("synthetic outage"), response])))
        shared = BoundedVisionClient(inner, model="claude-test", deadline=time.monotonic()+10)
        tool = [{"name": "report_public_hanging"}]
        shared.messages.create(tools=tool, messages=[])
        with self.assertRaises(RuntimeError):
            shared.messages.create(tools=tool, messages=[])
        shared.messages.create(tools=tool, messages=[])
        summary = shared.usage_summary()
        total = summary["total"]
        self.assertEqual((total["requests"], total["failed"]), (3, 1))
        self.assertEqual((total["input_tokens"], total["output_tokens"]), (2400, 600))
        self.assertEqual(summary["by_tool"]["report_public_hanging"]["requests"], 3)
        self.assertGreaterEqual(total["latency_seconds_total"], 0.)
        import json
        json.dumps(summary)

    def test_shared_client_caps_requests_deadline_and_no_forced_thinking_mode(self):
        call = SimpleNamespace(type="tool_use", name="report", input={})
        sdk = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(content=[call]))))
        client = BoundedVisionClient(sdk, model="claude-test", deadline=time.monotonic()+10, maximum_requests=1)
        client.messages.create(tools=[{"name": "report"}], model="ignored")
        request = sdk.messages.create.call_args.kwargs
        self.assertEqual(request["model"], "claude-test")
        self.assertEqual(request["extra_body"]["thinking"]["type"], "adaptive")
        self.assertEqual(request["tool_choice"]["type"], "auto")
        with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
            client.messages.create(tools=[{"name": "report"}])
        self.assertEqual(sdk.messages.create.call_count, 1)
        expired = BoundedVisionClient(sdk, model="claude-test", deadline=0)
        with self.assertRaises(RuntimeError):
            expired.messages.create(tools=[])

    def test_provider_requires_two_distinct_frames_and_never_fills_hidden_geometry(self):
        environment = PublicEnvironment(x=0.)
        response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="report_public_hanging",
            input=visible().model_dump(mode="json"))])
        client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
        with tempfile.TemporaryDirectory() as directory:
            provider = PublicRGBDHangingProvider(client, model="claude-test", episode_id="episode",
                store=ObservationStore(Path(directory)/"observations"), log=lambda event: None, maximum_requests=2)
            first = environment.observe()
            self.assertIsNone(provider.evaluate("handoff_confirmed", first).value)
            self.assertIsNone(provider.evaluate("handoff_confirmed", first).value)
            self.assertEqual(client.messages.create.call_count, 1)
            environment.sequence += 1
            second = environment.observe()
            self.assertIs(provider.evaluate("handoff_confirmed", second).value, True)
            result = provider.propose(second, environment.arm_state("right")["pose"], episode_id="episode")
            self.assertEqual(result.status, "unknown")
            self.assertEqual(result.reason, "complete_visible_aperture_unavailable")
            wrong = provider.propose(second, environment.arm_state("right")["pose"], episode_id="other")
            self.assertEqual(wrong.status, "rejected")
            environment.sequence += 1
            with self.assertRaisesRegex(RuntimeError, "request_budget"):
                provider.observe(environment.observe())

    def test_dimensions_are_measured_and_absent_opposite_face_has_no_nominal_fallback(self):
        intrinsic = np.array([[2000., 0., 50.], [0., 2000., 50.], [0., 0., 1.]])
        camera = RGBDCamera(np.ones((101, 101), np.float32), intrinsic, np.eye(4))
        rear_extrinsic = np.diag([-1., 1., -1., 1.])
        rear_extrinsic[2,3] = 2.
        back = RGBDCamera(np.full((101, 101), .992, np.float32), intrinsic, rear_extrinsic)
        image = np.zeros((101, 101, 3), np.uint8)
        observation = Observation(19, {name: image for name in (*CAMERAS, "perception_camera")},
            tuple(np.zeros(14)), {"perception_camera": camera, "right_camera": back})
        frame = np.eye(4)
        frame[:3, :3] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        frame[:3, 3] = [0, 0, 1]
        aperture = dict(frame_pose_world_wxyz=_matrix_pose(frame),
            uncertainty=dict(maximum_plane_residual_m=.0001),
            geometry_diagnostics=dict(centroid_inscribed_radius_m=.05))
        report = visible().model_copy(update=dict(opposite_handle_faces_visible=True,
            peg_keypoints_px=[(20., 20.), (60., 20.), (20., 60.)],
            peg_width_px=[(40., 18.), (40., 22.)],
            handle_front=SurfacePixel(camera="perception_camera", uv=(50., 30.)),
            handle_back=SurfacePixel(camera="right_camera", uv=(50., 50.-20./.992))))
        bounds = measured_bounds(observation, report, aperture)
        self.assertAlmostEqual(bounds["handle_thickness_m"], .008, places=6)
        self.assertGreaterEqual(bounds["measured_peg_radius_m"], .001)
        self.assertLess(bounds["measured_peg_radius_m"], .002)
        # A reading below the physical peg radius never enlarges the clearance.
        from development.hanging.public_provider import PHYSICAL_PEG_RADIUS_M
        self.assertGreaterEqual(bounds["peg_radius_m"], PHYSICAL_PEG_RADIUS_M)
        self.assertEqual(bounds["peg_radius_basis"], "physical_fixture_radius_floor")
        self.assertEqual(bounds["systematic_calibration_error"], "unknown")
        self.assertIn("public_registration",bounds)
        import json
        json.dumps(bounds, allow_nan=False)
        self.assertFalse(bounds["measurement_evidence"]["cad_registration_used"])
        # Two co-located views with different depths cannot establish the
        # opposing faces of an opaque handle, despite a plausible thickness.
        impossible_back = RGBDCamera(np.full((101,101),1.008,np.float32),intrinsic,np.eye(4))
        impossible = replace(observation,camera_frames={**observation.camera_frames,"right_camera":impossible_back})
        impossible_report = report.model_copy(update={"handle_back":SurfacePixel(
            camera="right_camera",uv=(50.,50.-20./1.008))})
        with self.assertRaisesRegex(ValueError,"opposite_handle_face_viewpoints_unresolved"):
            measured_bounds(impossible,impossible_report,aperture)
        # Loading an asset contract must not force a visible, directly measured
        # case through missing CAD registration assumptions.
        _,fixture,_,_=FixtureBoundsTests().fixture_case()
        direct=measured_bounds(observation,report,aperture,fixture=fixture,episode_id="episode")
        self.assertAlmostEqual(direct["handle_thickness_m"],.008,places=6)
        self.assertNotIn("fixture_provenance",direct)
        self.assertNotIn("allow_entry660_demo_override", bounds)
        with self.assertRaisesRegex(ValueError, "visible_thickness"):
            measured_bounds(observation, report.model_copy(update={"handle_back": None}), aperture)
        with self.assertRaisesRegex(ValueError, "subpixel"):
            measured_bounds(observation, report.model_copy(update={"peg_width_px": [(40., 19.8), (40., 20.2)]}), aperture)

    def test_hook_requires_phase_grasp_budget_and_allows_only_one_attempt(self):
        provider = Mock()
        provider.propose.return_value=runtime.ProposalResult("confirmed","synthetic eligible proposal")
        provider.evaluate.return_value = PredicateResult(name="handoff_confirmed", value=None,
            source="perception", observation_sequence=17, reason="unknown")
        events = []
        hook = HangingRecoveryHook(Mock(), provider, Mock(), episode_id="episode",
                                   deadline=time.monotonic()+10, log=events.append, pause_policy_on_entry=False)
        hook.environment.arm_state.return_value={"pose":np.array([0.,0.,1.,1.,0.,0.,0.])}
        command = SkillCommand(action="vla_act", parameters={"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 8})
        selection = PlannerSelection(command, "HANDOFF")
        with patch("development.hanging.supervised.recover_current") as recover:
            self.assertIsNone(hook.consider(selection, Mock(), remaining={}))
            self.assertIsNone(hook.consider(replace(selection, phase="HANG"), Mock(), remaining={}))
            recover.assert_not_called()
            provider.evaluate.return_value = provider.evaluate.return_value.model_copy(update={"value": True})
            recover.return_value = dict(status="recovery_sequence_completed_support_unverified", stages={})
            result = hook.consider(replace(selection, phase="HANG"), Mock(), remaining={"analytic_ticks": 12000, "skill_calls": 4})
            self.assertEqual(result["status"], "recovery_sequence_completed_support_unverified")
            self.assertEqual(hook.consider(replace(selection, phase="HANG"), Mock(), remaining={})["status"], "recovery_attempt_budget_exhausted")
            self.assertEqual(recover.call_count, 1)
        intervention = next(event for event in events
                            if event.get("decision") == "grounded_contact_recovery")
        self.assertIsNone(intervention["failure_detected"])
        self.assertEqual(intervention["task_progress"], "unknown")
        self.assertEqual(intervention["intervention_basis"],
                         "bounded_final_contact_skill_not_proven_vla_failure")

    def test_legacy_visibility_gap_defers_at_400_without_spending_recovery_attempt(self):
        environment=PublicEnvironment(x=0.);environment.sequence=400
        provider=Mock()
        provider.evaluate.return_value=PredicateResult(name="right_mug_lifted",value=True,source="perception",
            observation_sequence=400,reason="synthetic confirmed grasp")
        provider.propose.return_value=runtime.ProposalResult("unknown","complete_visible_aperture_unavailable")
        hook=HangingRecoveryHook(environment,provider,Mock(),episode_id="episode",deadline=time.monotonic()+10,
                                 log=lambda event:None,policy_chunk_size=16, pause_policy_on_entry=False)
        selection=PlannerSelection(SkillCommand(action="vla_act"),"HANG")
        remaining=dict(policy_actions=8,vla_chunks=1,skill_calls=4,analytic_ticks=12000)
        with patch("development.hanging.supervised.recover_current") as recover:
            result=hook.consider(selection,environment.observe(),remaining=remaining)
            self.assertEqual(result["status"],"recovery_deferred")
            fallback=SkillCommand.model_validate(result["policy_command"])
            self.assertEqual(fallback.action,SkillType.VLA_ACT)
            self.assertEqual(fallback.parameters.max_policy_actions,8)
            self.assertIsNone(fallback.parameters.lock_arm)
            self.assertEqual(hook.attempts,0)
            recover.assert_not_called()
            for sequence,budget in [(480,remaining),(481,remaining),(400,{**remaining,"policy_actions":0}),
                                    (400,{**remaining,"vla_chunks":0})]:
                # Visibility waiting is relative to entry/budget, never seed steps.
                hook.visibility_checks=0
                environment.sequence=sequence
                report=hook.consider(selection,environment.observe(),remaining=budget)
                expected="recovery_deferred" if budget["policy_actions"] and budget["vla_chunks"] else "recovery_stopped"
                if report is None:
                    self.assertFalse(budget["policy_actions"] and budget["vla_chunks"])
                else:
                    self.assertEqual(report["status"],expected)
            environment.sequence=464
            hook.next_visibility_action=0
            provider.propose.return_value=runtime.ProposalResult("confirmed","synthetic validated geometry")
            recover.return_value=dict(status="recovery_sequence_completed_support_unverified",stages={})
            result=hook.consider(selection,environment.observe(),remaining=remaining)
            recover.assert_called_once()
            self.assertEqual(hook.attempts,1)
            self.assertEqual(result["status"],"recovery_sequence_completed_support_unverified")

    def test_supervised_runner_continues_bounded_policy_when_aperture_is_unavailable(self):
        environment=PublicEnvironment(x=0.);environment.sequence=400
        provider=Mock(source="perception")
        provider.evaluate.side_effect=lambda name,obs,**kwargs:PredicateResult(name=name,
            value=True if name=="right_mug_lifted" else None,source="perception",observation_sequence=obs.sequence,
            reason="synthetic grasp; aperture unavailable")
        provider.planner_summary.return_value={}
        provider.propose.return_value=runtime.ProposalResult("unknown","complete_visible_aperture_unavailable")
        provider.lift_current.return_value=True
        planner=Mock()
        planner.select.return_value=PlannerSelection(SkillCommand(action="vla_act"),"HANG")
        vla=Mock();vla.infer.return_value=np.zeros((1,14))
        events=[]
        with tempfile.TemporaryDirectory() as directory,patch("development.hanging.supervised.recover_current") as recover:
            summary=run_controller(environment,vla,provider,ObservationStore(Path(directory)/"observations"),planner,
                SupervisedOptions("claude-test",max_actions=2,chunk_size=1,max_turns=2,policy_grace_actions=0),
                episode_id="episode",deadline=time.monotonic()+10,log=events.append)
            recover.assert_not_called()
        self.assertEqual(summary["policy_actions"],2)
        self.assertNotEqual(summary["outcome"],"recovery_blocked")
        self.assertGreater(vla.infer.call_count,0)
        decisions=[event for event in events if event.get("event_type")=="hanging_supervisor_decision"]
        self.assertTrue(any(event["decision"]=="continue_policy_and_reacquire" for event in decisions))
        self.assertFalse(any(event["decision"]=="take_over_hanging" for event in decisions))
        self.assertTrue(any(event["event_type"]=="hanging_recovery_deferred" for event in events))

    def test_legacy_saved_400_and_464_missing_geometry_defer_without_false_clearance(self):
        import json
        from harness.perception.replay import load_public_observation
        root=Path("data/runs/seed4_perception640_20261103_084536_510")
        if not (root/"supervisor_events.jsonl").exists():
            self.skipTest("saved recording unavailable")
        rows=[json.loads(line) for line in (root/"supervisor_events.jsonl").read_text().splitlines()]
        for sequence in (400,464):
            row=next(r for r in rows if r.get("event_type")=="public_hanging_proposal" and r.get("observation_sequence")==sequence)
            report=VisibleHangingReport.model_validate(row["report"])
            observation=load_public_observation(root/"observations"/f"observation_{sequence:08d}.npz")
            environment=PublicEnvironment(x=0.)
            provider=PublicRGBDHangingProvider(Mock(),model="offline",episode_id="saved",store=Mock(),log=lambda event:None)
            # Entry-state confirmation is held fixed to isolate geometry handling;
            # the aperture report and proposal rejection are the actual saved data.
            provider._corroborated_count=3
            hook=HangingRecoveryHook(environment,provider,Mock(),episode_id="saved",deadline=time.monotonic()+10,
                                     log=lambda event:None, pause_policy_on_entry=False)
            selection=PlannerSelection(SkillCommand(action="vla_act"),"HANG")
            grasp=PredicateResult(name="right_mug_lifted",value=True,source="perception",
                                  observation_sequence=sequence,reason="entry fixed for geometry regression")
            with self.subTest(sequence=sequence),patch.object(provider,"evaluate",return_value=grasp),patch.object(
                    provider,"observe",return_value=report),patch.object(provider,"crop_geometry",return_value=report),patch(
                    "development.hanging.supervised.recover_current") as recover:
                result=hook.consider(selection,observation,remaining=dict(policy_actions=900-sequence,
                    vla_chunks=20,skill_calls=10,analytic_ticks=12000))
                self.assertEqual(result["status"],"recovery_deferred")
                self.assertEqual(result["reason"],"complete_visible_aperture_unavailable")
                self.assertEqual(hook.attempts,0)
                self.assertNotIn("clearance_validated",result)
                recover.assert_not_called()

    def hand_back_hook(self, provider, environment, **options):
        return HangingRecoveryHook(environment, provider, Mock(), episode_id="episode",
            deadline=time.monotonic()+10, log=lambda event: None, pause_policy_on_entry=False,
            require_current_lift=True, hand_back_to_policy=True, **options)

    def lifted_provider(self, sequence):
        provider = Mock()
        provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=True,
            source="perception", observation_sequence=sequence, reason="synthetic grasp")
        provider.lift_current.return_value = True
        return provider

    def test_policy_keeps_control_during_grace_period_without_geometry_requests(self):
        environment = PublicEnvironment(x=0.); environment.sequence = 416
        provider = self.lifted_provider(416)
        hook = self.hand_back_hook(provider, environment, policy_grace_actions=160)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        for remaining_actions in (484, 400, 325):  # 0, 84 and 159 actions after entry
            result = hook.consider(selection, environment.observe(),
                remaining=dict(policy_actions=remaining_actions, vla_chunks=40, skill_calls=40, analytic_ticks=12000))
            self.assertEqual(result["status"], "recovery_deferred")
            self.assertEqual(result["reason"], "policy_grace_period")
            self.assertEqual(result["control_owner"], "policy")
        provider.propose.assert_not_called()
        provider.propose.return_value = runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")
        after = hook.consider(selection, environment.observe(),
            remaining=dict(policy_actions=324, vla_chunks=40, skill_calls=40, analytic_ticks=12000))
        provider.propose.assert_called_once()  # Geometry is checked only after the grace period.
        self.assertEqual(after["status"], "recovery_deferred")

    def test_stops_before_motion_hand_control_back_to_the_policy(self):
        remaining = dict(policy_actions=400, vla_chunks=30, skill_calls=30, analytic_ticks=12000)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        cases = [("rejected geometry", runtime.ProposalResult("rejected", "visible_material_not_planar")),
                 ("provider error", RuntimeError("synthetic provider failure"))]
        for name, outcome in cases:
            with self.subTest(name=name):
                environment = PublicEnvironment(x=0.); environment.sequence = 416
                provider = self.lifted_provider(416)
                if isinstance(outcome, Exception):
                    provider.propose.side_effect = outcome
                else:
                    provider.propose.return_value = outcome
                hook = self.hand_back_hook(provider, environment)
                with patch("development.hanging.supervised.recover_current") as recover:
                    first = hook.consider(selection, environment.observe(), remaining=remaining)
                    self.assertEqual(first["status"], "recovery_deferred")
                    self.assertEqual(first["reason"], "handed_back_to_policy")
                    self.assertIn("policy_command", first)
                    self.assertIsNotNone(hook.handed_back)
                    # Later turns, including a planner abort, stay with the policy.
                    for later in (selection, PlannerSelection(SkillCommand(action="abort"), "HANG")):
                        again = hook.consider(later, environment.observe(), remaining=remaining)
                        self.assertEqual(again["reason"], "handed_back_to_policy")
                    recover.assert_not_called()
        # Lift evidence lost after entry also hands back instead of ending the run.
        environment = PublicEnvironment(x=0.); environment.sequence = 416
        provider = self.lifted_provider(416)
        provider.propose.return_value = runtime.ProposalResult("unknown", "aperture_awaiting_two_frame_agreement")
        hook = self.hand_back_hook(provider, environment)
        hook.consider(selection, environment.observe(), remaining=remaining)
        provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=None,
            source="perception", observation_sequence=416, reason="public_evidence_unknown")
        lost = hook.consider(selection, environment.observe(), remaining=dict(remaining, policy_actions=360))
        self.assertEqual(lost["reason"], "handed_back_to_policy")
        self.assertEqual(hook.handed_back, "right_grasp_evidence_expired")

    def test_hand_back_yields_to_episode_limits(self):
        environment = PublicEnvironment(x=0.); environment.sequence = 416
        hook = self.hand_back_hook(self.lifted_provider(416), environment)
        hook.handed_back = "visibility_budget_exhausted"
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        self.assertIsNone(hook.consider(selection, environment.observe(),
            remaining=dict(policy_actions=0, vla_chunks=5, skill_calls=5, analytic_ticks=12000)))
        with self.assertRaisesRegex(ValueError, "hand_back_requires"):
            HangingRecoveryHook(environment, Mock(), Mock(), episode_id="e", deadline=time.monotonic()+1,
                log=lambda event: None, hand_back_to_policy=True)

    def test_awaiting_agreement_or_rack_reference_waits_like_hidden_opening(self):
        for reason in ("aperture_awaiting_two_frame_agreement", "validated_rack_reference_unavailable"):
            with self.subTest(reason=reason):
                environment=PublicEnvironment(x=0.);environment.sequence=400
                provider=Mock()
                provider.evaluate.return_value=PredicateResult(name="right_mug_lifted",value=True,
                    source="perception",observation_sequence=400,reason="synthetic grasp")
                provider.lift_current.return_value=True
                provider.propose.return_value=runtime.ProposalResult("unknown",reason)
                hook=HangingRecoveryHook(environment,provider,Mock(),episode_id="episode",
                    deadline=time.monotonic()+10,log=lambda event:None,pause_policy_on_entry=False)
                with patch("development.hanging.supervised.recover_current") as recover:
                    result=hook.consider(PlannerSelection(SkillCommand(action="vla_act"),"HANG"),
                        environment.observe(),
                        remaining=dict(policy_actions=500,vla_chunks=30,skill_calls=10,analytic_ticks=12000))
                    self.assertEqual(result["status"],"recovery_deferred")
                    self.assertIn("policy_command",result)
                    recover.assert_not_called()

    def test_visibility_wait_never_retries_rejected_geometry(self):
        environment=PublicEnvironment(x=0.);environment.sequence=400
        provider=Mock()
        provider.evaluate.return_value=PredicateResult(name="right_mug_lifted",value=True,source="perception",
            observation_sequence=400,reason="synthetic grasp")
        provider.propose.return_value=runtime.ProposalResult("rejected","visible_material_not_planar")
        hook=HangingRecoveryHook(environment,provider,Mock(),episode_id="episode",deadline=time.monotonic()+10,
                                 log=lambda event:None, pause_policy_on_entry=False)
        with patch("development.hanging.supervised.recover_current") as recover:
            result=hook.consider(PlannerSelection(SkillCommand(action="vla_act"),"HANG"),environment.observe(),
                remaining=dict(policy_actions=500,vla_chunks=30,skill_calls=10,analytic_ticks=12000))
            self.assertEqual(result["status"],"recovery_stopped")
            self.assertEqual(result["reason"],"visible_material_not_planar")
            recover.assert_not_called()

    def test_legacy_recovery_remains_available_after_four_checks_with_bounded_cooldown(self):
        for entry in (17, 416, 1200):
            with self.subTest(entry=entry):
                environment = PublicEnvironment(x=0.)
                provider = Mock()
                provider.evaluate.side_effect = lambda name, obs, **kwargs: PredicateResult(
                    name=name, value=True, source="perception", observation_sequence=obs.sequence,
                    reason="synthetic confirmed lift")
                provider.propose.side_effect = [runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")]*4 + [runtime.ProposalResult("confirmed", "synthetic geometry")]
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
                    deadline=time.monotonic()+30, log=lambda event:None, pause_policy_on_entry=False)
                selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
                def consider(offset):
                    environment.sequence = entry+offset
                    return hook.consider(selection, environment.observe(), remaining=dict(
                        policy_actions=484-offset, vla_chunks=40, skill_calls=40, analytic_ticks=12000))
                with patch("development.hanging.supervised.recover_current", return_value={"status":"synthetic_recovery"}) as recover:
                    for turn in range(4):
                        result = consider(turn*32)
                        self.assertEqual(result["status"], "recovery_deferred")
                        self.assertEqual(provider.propose.call_count, turn+1)
                        cooldown = consider(turn*32+16)
                        self.assertEqual(cooldown["reason"], "visibility_retry_cooldown")
                        self.assertFalse(cooldown["analytic_recovery_disabled"])
                        self.assertEqual(provider.propose.call_count, turn+1)
                    self.assertEqual(consider(128)["status"], "synthetic_recovery")
                    recover.assert_called_once()
                    self.assertEqual(provider.propose.call_count, 5)

    def test_missing_geometry_and_tcp_motion_do_not_become_stall_or_failure_evidence(self):
        environment = Mock()
        provider = Mock()
        provider.evaluate.side_effect = lambda name, observation, **kwargs: PredicateResult(
            name=name, value=True, source="perception", observation_sequence=observation.sequence,
            reason="synthetic confirmed lift")
        provider.propose.return_value = runtime.ProposalResult(
            "unknown", "complete_visible_aperture_unavailable")
        events = []
        hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
            deadline=time.monotonic()+30, log=events.append, pause_policy_on_entry=False)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        with patch("development.hanging.supervised.recover_current") as recovery:
            # Stationary, translating and rotating TCPs all lack task progress
            # evidence when the aperture and peg are unvalidated.
            for index, pose in enumerate((
                [0., 0., 1., 1., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0.],
                [.1, 0., 1., 1., 0., 0., 0.],
                [.1, 0., 1., 0., 0., 0., 1.],
            )):
                environment.arm_state.return_value = {"pose": pose}
                result = hook.consider(selection, SimpleNamespace(sequence=17+16*index),
                    remaining=dict(policy_actions=256-16*index, vla_chunks=20,
                                   skill_calls=20, analytic_ticks=12000))
                self.assertEqual(result["status"], "recovery_deferred")
            recovery.assert_not_called()
        decisions = [event for event in events if event["event_type"] == "hanging_supervisor_decision"]
        self.assertEqual(len(decisions), 4)
        for event in decisions:
            self.assertEqual(event["task_progress"], "unknown")
            self.assertIsNone(event["failure_detected"])
            self.assertEqual(event["decision"], "continue_policy_and_reacquire")
        self.assertEqual(hook.attempts, 0)

    def test_legacy_visibility_limits_stop_without_extra_requests_or_motion(self):
        environment = PublicEnvironment(x=0.)
        provider = Mock()
        provider.evaluate.side_effect = lambda name, obs, **kwargs: PredicateResult(name=name,
            value=True, source="perception", observation_sequence=obs.sequence, reason="fixture")
        provider.propose.return_value = runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")
        hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test", deadline=time.monotonic()+30,
            log=lambda event:None, visibility_action_budget=100, visibility_check_budget=3, pause_policy_on_entry=False)
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        for offset in (0,48,96):
            environment.sequence=offset
            result=hook.consider(selection, environment.observe(), remaining=dict(
                policy_actions=200-offset,vla_chunks=20,skill_calls=20,analytic_ticks=12000))
        self.assertEqual(result["status"], "recovery_stopped")
        self.assertTrue(result["visibility_budget_exhausted"])
        environment.sequence=97
        self.assertEqual(hook.consider(selection,environment.observe(),remaining=dict(
            policy_actions=103,vla_chunks=20,skill_calls=20,analytic_ticks=12000))["reason"],"visibility_budget_exhausted")
        self.assertEqual(provider.propose.call_count,3)
        self.assertEqual(hook.attempts,0)

    def test_confirmed_entry_loss_or_expiry_stops_without_another_policy_chunk(self):
        for value, reason in ((False, "right_grasp_lost"), (None, "right_grasp_evidence_expired")):
            with self.subTest(value=value):
                environment = PublicEnvironment(x=0.)
                provider = Mock()
                provider.evaluate.return_value = PredicateResult(name="right_mug_lifted", value=True,
                    source="perception", observation_sequence=0, reason="synthetic confirmed lift")
                provider.propose.return_value = runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")
                hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
                    deadline=time.monotonic()+30, log=lambda event:None, pause_policy_on_entry=False)
                selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
                remaining = dict(policy_actions=100, vla_chunks=10, skill_calls=10, analytic_ticks=12000)
                self.assertEqual(hook.consider(selection, environment.observe(), remaining=remaining)["status"], "recovery_deferred")
                provider.evaluate.return_value = provider.evaluate.return_value.model_copy(update={"value": value})
                result = hook.consider(selection, environment.observe(), remaining=remaining)
                self.assertEqual(result["reason"], reason)
                self.assertNotIn("policy_command", result)
                provider.propose.assert_called_once()

    def test_legacy_repeated_visibility_checks_remain_enabled_through_900_action_cap(self):
        environment = PublicEnvironment(x=0.)
        provider = Mock(source="perception")
        provider.evaluate.side_effect = lambda name, obs, **kwargs: PredicateResult(name=name,
            value=True if name=="right_mug_lifted" else None, source="perception", observation_sequence=obs.sequence, reason="unknown")
        provider.planner_summary.return_value = {}
        hook = HangingRecoveryHook(environment, provider, Mock(), episode_id="test",
            deadline=time.monotonic()+60, log=lambda event:None, pause_policy_on_entry=False)
        provider.propose.return_value = runtime.ProposalResult("unknown", "complete_visible_aperture_unavailable")
        planner = Mock()
        planner.select.return_value = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        vla = Mock()
        vla.infer.side_effect = lambda observation, size: np.zeros((size,14))
        store = Mock()
        store.save.side_effect = lambda observation: dict(sequence=observation.sequence, path="synthetic.npz")
        with patch("development.hanging.supervised.HangingRecoveryHook", return_value=hook), patch(
                "development.hanging.supervised.recover_current") as recovery:
            summary = run_controller(environment, vla, provider, store, planner,
                SupervisedOptions("claude-test",max_actions=900,chunk_size=16,max_turns=80),
                episode_id="test", deadline=time.monotonic()+60, log=lambda event:None)
        self.assertEqual(summary["policy_actions"], 900)
        self.assertEqual(summary["analytic_ticks"], 0)
        self.assertEqual(vla.infer.call_count, 57)
        self.assertEqual(vla.infer.call_args.args[1], 4)
        self.assertEqual(provider.propose.call_count, 19)
        recovery.assert_not_called()

    def test_existing_supervisor_executes_hook_and_accounts_for_recovery(self):
        environment = PublicEnvironment(x=0.)
        provider = Mock(source="perception")
        provider.evaluate.side_effect = lambda name, obs, **kw: PredicateResult(name=name, value=None,
            source="perception", observation_sequence=obs.sequence, reason="synthetic")
        provider.planner_summary.return_value = {}
        planner = Mock()
        planner.select.return_value = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        events = []
        result = dict(status="recovery_sequence_completed_support_unverified", native_success=True,
            stages={name: {"completed": True} for name in ("approach", "seat", "release", "retreat")},
            analytic_ticks=1052, post_retreat_observation_only_ticks=250)
        with tempfile.TemporaryDirectory() as directory, patch.object(HangingRecoveryHook, "consider", return_value=result):
            summary = run_controller(environment, Mock(), provider, ObservationStore(Path(directory)/"observations"),
                planner, SupervisedOptions("claude-test"), episode_id="episode", deadline=time.monotonic()+10, log=events.append)
        self.assertEqual(summary["analytic_ticks"], 1052)
        self.assertEqual(summary["skill_calls"], 4)
        self.assertEqual(summary["policy_actions"], 0)
        self.assertEqual(summary["hanging_recovery"], result)
        self.assertEqual(environment.ticks, 0)  # Hook execution itself is mocked.

    def test_frozen_policy_turn_then_real_four_stage_recovery_in_cpu_environment(self):
        environment = PublicEnvironment(x=0.)
        environment.sequence = 17
        environment.arms["right"]["gripper"] = 0.
        class Provider:
            source = "perception"
            def evaluate(self, name, observation, **kwargs):
                return PredicateResult(name=name, value=True if name == "right_mug_lifted" and observation.sequence > 17 else None,
                    source="perception", observation_sequence=observation.sequence, reason="synthetic public evidence")
            def planner_summary(self, observation):
                return {}
            def lift_current(self, observation):
                return observation.sequence > 17
            def propose(self, observation, measured_tcp, *, episode_id):
                return runtime.ProposalResult("confirmed", "synthetic proposal")
        _, proposal, _ = runtime_fixtures.RuntimeRecoveryTests().fixture()
        def validate(observation, measured_tcp, proposed, *, episode_id):
            from hashlib import sha256
            aperture = dict(status="candidate_not_motion_validated", sequence=observation.sequence,
                camera="perception_camera", calibration_sha256=calibration_sha256(observation.camera_frames["perception_camera"]),
                rgb_sha256=sha256(observation.images["perception_camera"].tobytes()).hexdigest(),
                frame_pose_world_wxyz=list(measured_tcp),
                material_points_world=[[x, y, 1.] for x in (-.02, 0., .02) for y in (-.02, 0., .02)],
                uncertainty=dict(center_sensitivity_m=.001, maximum_normal_sensitivity_deg=1.,
                                 maximum_plane_residual_m=.0005))
            return runtime.GroundingResult("confirmed", "synthetic geometry", runtime.CurrentGeometry(
                replace(proposal, mating_prior={**proposal.mating_prior, "radial_clearance_m": .1}),
                aperture, (0., 0., 1., 1., 0., 0., 0.)))
        command = SkillCommand(action="vla_act", stop="chunk_budget_exhausted",
            parameters={"max_chunks": 1, "chunk_size": 1, "max_policy_actions": 1})
        planner = Mock()
        planner.select.side_effect = [PlannerSelection(command, "GRASP"), PlannerSelection(command, "HANG")]
        vla = Mock()
        vla.infer.return_value = np.zeros((1, 14))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runtime, "validate_current_geometry", side_effect=validate), \
                patch.object(runtime, "confirm_visible_handle", return_value={"status": "confirmed_visible_subset"}):
            result = run_controller(environment, vla, Provider(), ObservationStore(Path(directory)/"observations"),
                planner, SupervisedOptions("claude-test", policy_grace_actions=0), episode_id="episode-a",
                deadline=time.monotonic()+20,
                log=lambda event: None)
        self.assertEqual(result["outcome"], "recovery_sequence_completed_support_unverified")
        self.assertEqual(result["policy_actions"], 1)
        self.assertEqual(list(result["hanging_recovery"]["stages"]), ["approach", "seat", "release", "retreat"])
        self.assertEqual(result["hanging_recovery"]["public_support"], "unknown")
        self.assertEqual(result["analytic_ticks"], environment.ticks)
        self.assertEqual(result["analytic_ticks"], result["hanging_recovery"]["analytic_ticks"])
        self.assertEqual(vla.infer.call_count, 1)
        self.assertEqual(environment.arms["right"]["gripper"], 1.)

    def test_cpu_service_preflight_and_failed_preflight_never_initialize_simulator(self):
        raw = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=SimpleNamespace(content=[
            SimpleNamespace(type="tool_use", name="report_compatibility", input={"ok": True})]))))
        shared = BoundedVisionClient(raw, model="claude-test", deadline=time.monotonic()+10)
        policy = SimpleNamespace(worker=SimpleNamespace(reset_episode=SimpleNamespace(remote=Mock()),
            act_chunk=SimpleNamespace(remote=Mock())), reset=Mock(return_value=({"ok": True}, 0.)))
        result = service_preflight(SupervisedOptions("claude-test"), shared, policy)
        self.assertFalse(result["simulator_started"])
        self.assertEqual(result["runtime_geometry"], "unvalidated")
        policy.reset.assert_called_once()
        with tempfile.TemporaryDirectory() as directory, \
                patch("anthropic.Anthropic", return_value=Mock()) as sdk, \
                patch("harness.vla.modal_client.ModalPolicyClient", return_value=policy), \
                patch("development.hanging.supervised.service_preflight", side_effect=ValueError("protocol_failed")), \
                patch("development.simulator.local_simulator.LocalSimulator") as simulator:
            failed = run("unused", Path(directory)/"run", SupervisedOptions("claude-test"))
        self.assertEqual(failed["status"], "preflight_failed")
        simulator.assert_not_called()
        sdk.assert_called_once_with(max_retries=0, timeout=45.)

    def test_launcher_validates_before_remote_and_always_downloads_video_and_rgbd(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/"modal_apps/simulator.py").read_text())
        launcher = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "supervised_hanging")
        launcher.decorator_list = []
        remote = Mock(return_value=dict(status="synthetic"))
        download = Mock()
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(Path=Path, REPO=Path(directory), time=time, GPU="T4",
                run_supervised_hanging=SimpleNamespace(remote=remote), _wait_for_volume=lambda _: True,
                _download_run=download, json=__import__("json"))
            exec(compile(ast.Module(body=[launcher], type_ignores=[]), "launcher", "exec"), namespace)
            launch = namespace["supervised_hanging"]
            with self.assertRaisesRegex(ValueError, "invalid_max_actions"):
                launch("claude-test", max_actions=901)
            remote.assert_not_called()
            launch("claude-test", seed=8, output=str(Path(directory)/"out"))
            self.assertEqual(remote.call_args.args[1]["seed"], 8)
            self.assertEqual(download.call_args.kwargs, {"skip_observations": False})
        remote_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_supervised_hanging")
        decorator = ast.unparse(remote_node.decorator_list[0])
        self.assertIn("retries=0", decorator)
        self.assertIn("max_containers=1", decorator)
        self.assertIn("anthropic", decorator)
        call=next(n for n in ast.walk(remote_node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="run")
        self.assertIn("fixture_loader",{k.arg for k in call.keywords})


class DirectPublicGeometryTests(unittest.TestCase):
    def scene(self, sequence=17, translation=0.):
        observation, proposal, tcp = runtime_fixtures.RuntimeRecoveryTests().fixture(sequence,translation)
        front = observation.camera_frames["perception_camera"]
        rear_extrinsic = np.diag([-1.,1.,-1.,1.])
        rear_extrinsic[0,3],rear_extrinsic[2,3] = translation,2.
        back = RGBDCamera(np.full(front.depth_m.shape,.98,np.float32),front.intrinsic,rear_extrinsic)
        # A real observation always carries the three policy cameras.
        observation = replace(observation,
            images={**{name: observation.images["perception_camera"] for name in CAMERAS},
                    **observation.images},
            camera_frames={**observation.camera_frames,"right_camera":back})
        report = visible().model_copy(update=dict(complete_visible_opening=True,
            contour_px=proposal.contour_px.tolist(),material_px=proposal.material_px.tolist(),
            body_polygon_px=proposal.body_polygon_px,opposite_handle_faces_visible=True,
            peg_keypoints_px=[(15.,15.),(65.,15.),(15.,30.)],peg_width_px=[(40.,13.),(40.,17.)],
            handle_front=SurfacePixel(camera="perception_camera",uv=(50.,30.)),
            handle_back=SurfacePixel(camera="right_camera",uv=(50.,50.-20./.98))))
        return observation,report,tcp

    def test_measured_public_geometry_reaches_target_preparation_without_asset_or_demo_defaults(self):
        import json
        from development.hanging.motion import aperture_targets

        targets_by_case = []
        for sequence,translation in ((17,0.),(81,.08)):
            with self.subTest(sequence=sequence):
                observation,report,tcp = self.scene(sequence,translation)
                if sequence == 81:
                    # Same synthetic rear measurement through the optional
                    # perception-only channel, not a changed policy camera.
                    images = dict(observation.images)
                    frames = dict(observation.camera_frames)
                    images["perception_reverse_camera"] = images["right_camera"]
                    frames["perception_reverse_camera"] = frames.pop("right_camera")
                    observation = replace(observation,images=images,camera_frames=frames)
                    report = report.model_copy(update={"handle_back":SurfacePixel(
                        camera="perception_reverse_camera",uv=report.handle_back.uv)})
                client = Mock()
                provider = PublicRGBDHangingProvider(client,model="offline",episode_id="direct-test",
                    store=Mock(),log=lambda _:None)
                grasp = PredicateResult(name="right_mug_lifted",value=True,source="perception",
                    observation_sequence=sequence,reason="Synthetic grasp and semantic inputs; not live perception")
                # Only semantic/entry inputs are synthetic. RGB-D measurements,
                # registration, clearance and target preparation run unmocked.
                with patch.object(provider,"observe",return_value=report),patch.object(
                        provider,"evaluate",return_value=grasp),patch.object(provider,"crop_geometry") as crop:
                    result = provider.propose(observation,tcp,episode_id="direct-test")
                self.assertEqual(result.status,"confirmed",result.reason)
                crop.assert_not_called()
                client.messages.create.assert_not_called()
                grounded = runtime.validate_current_geometry(observation,tcp,result,episode_id="direct-test")
                self.assertEqual(grounded.status,"confirmed",grounded.reason)
                geometry = grounded.geometry
                targets,evidence = aperture_targets(observation,geometry.aperture,geometry.rack_pose,tcp,
                    geometry.proposal.mating_prior,clearance_model="rss_sensitivity")
                bounds = result.proposal.mating_prior
                self.assertAlmostEqual(bounds["handle_thickness_m"],.02,places=6)
                self.assertNotIn("fixture_provenance",bounds)
                self.assertNotIn("allow_entry660_demo_override",bounds)
                self.assertFalse(bounds["measurement_evidence"]["cad_registration_used"])
                json.dumps(bounds,allow_nan=False)
                self.assertEqual(set(targets),{"rack_approach","rack_seat"})
                self.assertFalse(evidence["motion_authorized"])
                self.assertEqual(evidence["ik_planning"],"not_run_requires_simulator_planner")
                targets_by_case.append(targets)
        np.testing.assert_allclose(targets_by_case[1]["rack_seat"][:3]-targets_by_case[0]["rack_seat"][:3],
                                   [.08,0.,0.],atol=1e-7)

    def test_low_confidence_opening_needs_a_second_agreeing_frame(self):
        from harness.perception.pose_grounding import _matrix_pose
        observation,report,tcp = self.scene()
        report = report.model_copy(update={"confidence": .7})
        provider = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="direct-test",
            store=Mock(),log=lambda _:None)
        grasp = PredicateResult(name="right_mug_lifted",value=True,source="perception",
            observation_sequence=17,reason="Synthetic grasp")
        with patch.object(provider,"observe",return_value=report),patch.object(provider,"evaluate",return_value=grasp):
            first = provider.propose(observation,tcp,episode_id="direct-test")
        self.assertEqual((first.status,first.reason),("unknown","aperture_awaiting_two_frame_agreement"))
        self.assertIsNone(first.proposal)
        # Direct agreement checks: the opening moves rigidly with the measured TCP.
        def pose(x):
            frame = np.eye(4); frame[:3,3] = (x,0.,1.)
            return dict(frame_pose_world_wxyz=list(_matrix_pose(frame)))
        def tcp_at(x):
            return (x,0.,.9,1.,0.,0.,0.)
        later = lambda seq: replace(observation,sequence=seq)
        provider._aperture_candidate = None
        self.assertEqual(provider._aperture_agrees(later(20),tcp_at(0.),pose(0.))["status"],"awaiting_second_frame")
        agreed = provider._aperture_agrees(later(24),tcp_at(.02),pose(.022))
        self.assertEqual(agreed["status"],"agreed",agreed)
        self.assertEqual(provider._aperture_agrees(later(28),tcp_at(.04),pose(.052))["status"],"disagreed")
        far = provider._aperture_agrees(later(28+provider.lift_evidence_max_gap+1),tcp_at(.04),pose(.04))
        self.assertEqual(far["status"],"awaiting_second_frame")

    def test_first_validated_rack_reference_is_kept(self):
        observation,report,tcp = self.scene()
        provider = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="direct-test",
            store=Mock(),log=lambda _:None)
        provider._cache_rack(observation,report)
        kept = provider._current_rack_reference(observation)
        self.assertIsNotNone(kept)
        moved = report.model_copy(update={"peg_keypoints_px":[(16.,15.),(66.,15.),(16.,30.)]})
        provider._cache_rack(replace(observation,sequence=18),moved)
        self.assertIs(provider._current_rack_reference(replace(observation,sequence=18)),kept)
        # A low-confidence peg alone is never kept: it needs an agreeing second frame.
        fresh = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="direct-test",
            store=Mock(),log=lambda _:None)
        unsure = report.model_copy(update={"peg_confidence":.6})
        fresh._cache_rack(observation,unsure)
        self.assertIsNone(fresh._current_rack_reference(observation))
        fresh._cache_rack(replace(observation,sequence=18),unsure)
        self.assertIsNotNone(fresh._current_rack_reference(replace(observation,sequence=18)))

    def test_missing_opposing_faces_or_peg_width_never_produces_a_target_proposal(self):
        observation,report,tcp = self.scene()
        for changes in ({"handle_back":None},{"peg_width_px":None},{"opposite_handle_faces_visible":False}):
            with self.subTest(changes=changes):
                provider = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="direct-test",
                    store=Mock(),log=lambda _:None)
                grasp = PredicateResult(name="right_mug_lifted",value=True,source="perception",
                    observation_sequence=17,reason="Synthetic grasp")
                with patch.object(provider,"observe",return_value=report.model_copy(update=changes)),patch.object(
                        provider,"evaluate",return_value=grasp):
                    result = provider.propose(observation,tcp,episode_id="direct-test")
                self.assertEqual(result.status,"unknown")
                # Without a peg width no rack reference can be kept at all.
                expected = ("validated_rack_reference_unavailable" if "peg_width_px" in changes
                            else "visible_thickness_and_peg_cross_section_required")
                self.assertEqual(result.reason,expected)
                self.assertIsNone(result.proposal)


class PublicFixtureRegistrationTests(unittest.TestCase):
    def scene(self, sequence=19):
        from development.hanging.public_provider import VerifiedFixtureAssets, RACK_VISUAL_HASH, MUG_CLASS_MANIFEST_HASH
        depth = np.ones((160, 160), np.float32)
        for x, y in ((20, 20), (115, 20)):
            depth[y-3:y+4, x-3:x+4] = 1.004
        camera = RGBDCamera(depth, np.array([[1000.,0.,0.],[0.,1000.,0.],[0.,0.,1.]]), np.eye(4))
        image = np.zeros((160, 160, 3), np.uint8)
        observation = Observation(sequence, {"perception_camera": image}, tuple(np.zeros(14)),
                                  {"perception_camera": camera})
        # Synthetic asset binding isolates registration; it is not a real-asset audit.
        assets = VerifiedFixtureAssets(RACK_VISUAL_HASH, MUG_CLASS_MANIFEST_HASH)
        fixture = assets.bind(observation, episode_id="synthetic")
        report = visible().model_copy(update=dict(peg_keypoints_px=[(20.,20.),(115.,20.),(20.,60.)],
            peg_width_px=[(70.,14.),(70.,26.)]))
        return observation, report, assets, fixture

    def test_public_cross_section_registers_centerline_but_never_fabricates_cad_aperture_bounds(self):
        from development.hanging.public_provider import register_public_fixture
        observation, report, _, fixture = self.scene()
        result = register_public_fixture(observation, report, fixture, episode_id="synthetic")
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.reason, "fixture_aperture_cad_registration_unavailable")
        self.assertTrue(result.fixture.centerline_verified)
        self.assertIsNone(result.fixture.aperture_registration_error_m)
        self.assertIsNone(result.fixture.aperture_axis_error_rad)
        registration = result.fixture.public_rack_registration
        self.assertEqual(registration["observation_sequence"], observation.sequence)
        self.assertEqual(registration["episode_id"], "synthetic")
        self.assertGreater(registration["centerline_offset_m"], .0039)
        self.assertGreater(registration["position_sensitivity_m"], 0.)
        self.assertFalse(result.evidence["motion_authorized"])
        self.assertFalse(fixture.centerline_verified)  # No mutation of the asset contract.
        supplied = replace(fixture, aperture_registration_error_m=.0002, aperture_axis_error_rad=.001)
        complete = register_public_fixture(observation, report, supplied, episode_id="synthetic")
        self.assertEqual(complete.status, "confirmed")
        self.assertEqual(complete.fixture.aperture_registration_error_m, .0002)
        self.assertFalse(complete.evidence["motion_authorized"])

    def test_registration_rejects_stale_cross_episode_and_inconsistent_observed_radius(self):
        from development.hanging.public_provider import register_public_fixture
        observation, report, assets, fixture = self.scene()
        camera = observation.camera_frames["perception_camera"]
        intrinsic = camera.intrinsic.copy(); intrinsic[0,0] += 1.
        changed_camera = replace(observation,camera_frames={"perception_camera":
            RGBDCamera(camera.depth_m,intrinsic,camera.world_to_camera)})
        cases = (
            (observation, replace(fixture, sequence=18), None, None, "stale_or_incompatible"),
            (observation, replace(fixture, episode_id="other"), None, None, "stale_or_incompatible"),
            (observation, replace(fixture, peg_radius_m=.05), None, None, "radius_inconsistent"),
            (observation, fixture, (observation, report), "other", "wrong_episode"),
            (changed_camera, fixture, None, None, "stale_or_incompatible"),
            (observation, fixture, (replace(observation,sequence=20),report), "synthetic", "reference_incompatible"),
        )
        for current, contract, reference, identity, reason in cases:
            with self.subTest(reason=reason):
                result = register_public_fixture(current, report, contract, episode_id="synthetic",
                    rack_reference=reference, reference_episode_id=identity)
                self.assertEqual(result.status, "rejected")
                self.assertIn(reason, result.reason)
                self.assertFalse(result.evidence["motion_authorized"])
        for width in (None, [(70.,19.8),(70.,20.2)]):
            with self.subTest(width=width):
                result = register_public_fixture(observation, report.model_copy(update={"peg_width_px":width}),
                                                 fixture, episode_id="synthetic")
                self.assertNotEqual(result.status, "confirmed")
                self.assertNotEqual(result.evidence["rack_centerline_status"], "confirmed")

    def test_cached_registration_retains_source_sequence_and_episode(self):
        from development.hanging.public_provider import register_public_fixture
        earlier, report, assets, _ = self.scene(17)
        current = replace(earlier, sequence=33)
        fixture = assets.bind(current, episode_id="synthetic")
        result = register_public_fixture(current, report.model_copy(update={"peg_keypoints_px":None}),
            fixture, episode_id="synthetic", rack_reference=(earlier, report),
            reference_episode_id="synthetic")
        self.assertTrue(result.fixture.centerline_verified)
        self.assertEqual(result.evidence["rack_reference_sequence"], 17)
        self.assertEqual(result.fixture.sequence, 33)
        self.assertEqual(result.status, "unknown")

    def test_provider_calls_registration_and_returns_missing_cad_evidence_without_clearance(self):
        observation, report, assets, _ = self.scene()
        report = report.model_copy(update=dict(complete_visible_opening=True,
            contour_px=[(40.,40.)]*8, material_px=[(45.,45.)]*12,
            body_polygon_px=[(30.,30.),(35.,30.),(35.,35.)]))
        events = []
        provider = PublicRGBDHangingProvider(Mock(),model="offline",episode_id="synthetic",store=Mock(),
            log=events.append,fixture=assets)
        grasp = PredicateResult(name="right_mug_lifted", value=True, source="perception",
                                observation_sequence=19, reason="synthetic grasp")
        # Isolate integration after aperture fitting; actual registration uses real synthetic RGB-D.
        with patch.object(provider,"observe",return_value=report), patch.object(provider,"evaluate",return_value=grasp), patch(
                "development.hanging.public_provider.estimate",return_value={"status":"candidate_not_motion_validated"}), patch(
                "development.hanging.public_provider.measured_bounds") as bounds:
            result = provider.propose(observation, (0.,0.,1.,1.,0.,0.,0.),episode_id="synthetic")
        self.assertEqual(result.status,"unknown")
        self.assertEqual(result.reason,"fixture_aperture_cad_registration_unavailable")
        bounds.assert_not_called()
        event = next(e for e in events if e["event_type"]=="public_fixture_registration")
        self.assertEqual(event["evidence"]["rack_centerline_status"],"confirmed")
        self.assertFalse(event["evidence"]["motion_authorized"])
        import json
        json.dumps(event, allow_nan=False)

    def test_fixture_bounds_rechecks_registration_pixels_and_includes_their_uncertainty(self):
        from development.hanging.public_provider import register_public_fixture
        observation, report, _, fixture = self.scene()
        # Explicit synthetic CAD bounds test downstream wiring, not their recovery from an image.
        fixture = replace(fixture, aperture_registration_error_m=.0002, aperture_axis_error_rad=.001)
        result = register_public_fixture(observation, report, fixture, episode_id="synthetic")
        aperture = dict(status="candidate_not_motion_validated", frame_pose_world_wxyz=[0.,0.,1.,1.,0.,0.,0.],
            uncertainty=dict(center_sensitivity_m=.0001,maximum_normal_sensitivity_deg=.1,
                             maximum_plane_residual_m=.0001),
            geometry_diagnostics=dict(principal_rms_span_m=[.02,.01,.0001],centroid_inscribed_radius_m=.03))
        corridor = dict(foreground_collisions=0,minimum_radial_margin_m=.0061)
        with patch("development.hanging.public_provider.observed_corridor",return_value=corridor):
            bounds = measured_bounds(observation,report,aperture,fixture=result.fixture,episode_id="synthetic")
        self.assertIn("public_registration",bounds)
        self.assertGreater(bounds["position_uncertainty_m"],
                           bounds["public_registration"]["position_sensitivity_m"])
        self.assertTrue(bounds["clearance_validated"])
        import json
        json.dumps(bounds, allow_nan=False)
        for changes in ({"rgb_sha256":"changed"},{"episode_id":"other"},{"observation_sequence":18}):
            stale = replace(result.fixture,public_rack_registration={
                **result.fixture.public_rack_registration,**changes})
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,"registration_stale_or_incompatible"):
                measured_bounds(observation,report,aperture,fixture=stale,episode_id="synthetic")


class FixtureBoundsTests(unittest.TestCase):
    def test_verified_class_contract_binds_fresh_frames_without_inventing_registration(self):
        from development.hanging.public_provider import VerifiedFixtureAssets,MUG_CLASS_MANIFEST_HASH,RACK_VISUAL_HASH
        assets=VerifiedFixtureAssets(RACK_VISUAL_HASH,MUG_CLASS_MANIFEST_HASH)
        observation,_,_,_=self.fixture_case()
        provider=PublicRGBDHangingProvider(Mock(),model="offline",episode_id="bound",store=Mock(),
                                          log=lambda event:None,fixture=assets)
        first=provider._fixture_for(observation)
        later=provider._fixture_for(SimpleNamespace(sequence=480,camera_frames=observation.camera_frames))
        self.assertEqual((first.sequence,later.sequence),(464,480))
        self.assertTrue(first.asset_class_manifest_verified)
        self.assertTrue(first.asset_class_binding_verified)
        self.assertEqual(first.episode_id,"bound")
        self.assertIsNone(first.aperture_registration_error_m)
        self.assertIsNone(first.aperture_axis_error_rad)
        self.assertFalse(first.centerline_verified)

    def test_asset_verification_failure_precedes_model_or_simulator_initialization(self):
        loader=Mock(side_effect=ValueError("fixture_mug_asset_audit_mismatch"))
        with tempfile.TemporaryDirectory() as directory,patch("anthropic.Anthropic") as sdk,patch(
                "development.simulator.local_simulator.LocalSimulator") as simulator:
            report=run("unused",Path(directory)/"run",SupervisedOptions("claude-test"),fixture_loader=loader)
        self.assertEqual(report["status"],"preflight_failed")
        self.assertFalse(report["simulator_started"])
        sdk.assert_not_called()
        simulator.assert_not_called()

    def test_readonly_asset_verifier_checks_every_visual_collision_and_scale(self):
        import json
        import development.hanging.public_provider as pp
        root=Path("fixture_root")/"assets/objects"
        hashes={}
        metadata={}
        for i,(visual,collision,scale) in enumerate(zip(pp.MUG_CLASS_VISUAL_HASHES,pp.MUG_CLASS_COLLISION_HASHES,pp.MUG_CLASS_SCALES)):
            hashes[str(root/"039_mug/visual"/f"base{i}.glb")]=visual
            hashes[str(root/"039_mug/collision"/f"base{i}.glb")]=collision
            metadata[str(root/"039_mug"/f"model_data{i}.json")]=json.dumps({"scale":[scale]*3})
        hashes[str(root/"040_rack/visual/base0.glb")]=pp.RACK_VISUAL_HASH
        hashes[str(root/"040_rack/collision/base0.glb")]=pp.RACK_COLLISION_HASH
        metadata[str(root/"040_rack/model_data0.json")]=json.dumps({"scale":[1.1]*3})
        def file_bytes(path):return str(path).encode()
        def file_text(path):return metadata[str(path)]
        with patch.object(Path,"read_bytes",file_bytes),patch.object(Path,"read_text",file_text),patch.object(
                pp,"sha256",side_effect=lambda value:SimpleNamespace(hexdigest=lambda:hashes[value.decode()])):
            assets=pp.load_verified_fixture_assets("fixture_root")
            self.assertEqual(assets.class_manifest_hash,pp.MUG_CLASS_MANIFEST_HASH)
            hashes[str(root/"039_mug/collision/base3.glb")]="f"*64
            with self.assertRaisesRegex(ValueError,"mug_asset_audit_mismatch"):
                pp.load_verified_fixture_assets("fixture_root")

    def test_audited_topology_requires_instance_registration_and_mesh_binding(self):
        from development.hanging.public_provider import topology_slab_coverage,MUG_CLASS_CORRIDOR_CERTIFICATE
        _,fixture,aperture,_=self.fixture_case()
        fixture=replace(fixture,full_slab_coverage_verified=False,
            topology_certificate=MUG_CLASS_CORRIDOR_CERTIFICATE,asset_class_binding_verified=True,
            mug_mesh_hash="392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812",
            mug_scale=.08,aperture_registration_error_m=0.,aperture_axis_error_rad=0.)
        aperture["uncertainty"]["maximum_normal_sensitivity_deg"]=.1
        corridor=dict(foreground_collisions=0,minimum_radial_margin_m=.0061)
        result=topology_slab_coverage(fixture,aperture,corridor,.08944)
        self.assertTrue(result["full_slab_coverage_verified"])
        for changed,reason in [(replace(fixture,asset_class_binding_verified=False),"class_binding"),
                               (replace(fixture,mug_scale=.09),"class_binding"),
                               (replace(fixture,aperture_axis_error_rad=None),"registration_unverified"),
                               (replace(fixture,aperture_registration_error_m=.004),"registration_margin")]:
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):
                topology_slab_coverage(changed,aperture,corridor,.08944)
        with self.assertRaisesRegex(ValueError,"visible_clearance"):
            topology_slab_coverage(fixture,aperture,dict(foreground_collisions=0,minimum_radial_margin_m=.0049),.08944)

    def test_saved_224_visible_surface_cache_does_not_certify_hidden_slab(self):
        import json
        from harness.perception.replay import load_public_observation
        from development.hanging.aperture_frame import estimate, inset_material_samples
        from development.hanging.public_provider import fixture_axis, FixtureBounds
        root=Path("data/runs/seed4_perception640_20261103_084536_510/observations")
        review=Path("data/runs/seed4_obs464_semantic_97c95fb4a5ca4b9489c3851071345ef1/results.json")
        if not (root/"observation_00000224.npz").exists() or not review.exists():
            self.skipTest("saved recordings unavailable")
        early=load_public_observation(root/"observation_00000224.npz")
        current=load_public_observation(root/"observation_00000464.npz")
        # Coordinates returned by the authorized in-memory obs224 full-frame
        # check, not inferred from CAD or copied from a different episode.
        rack=visible().model_copy(update=dict(peg_keypoints_px=[(142.,262.),(172.,218.),(128.,240.)],
                                              peg_width_px=None,peg_confidence=.45))
        seat,tip,se,te,length=fixture_axis(early,rack.peg_keypoints_px)
        self.assertAlmostEqual(length,.08943749,places=7)
        self.assertGreater(length-se-te-.047991,0)
        fixture=FixtureBounds("saved224",464,calibration_sha256(current.camera_frames["perception_camera"]),
                              "0"*64,"0"*64,"identity placeholders; coverage unverified")
        provider=PublicRGBDHangingProvider(Mock(),model="offline",episode_id="saved224",store=Mock(),
                                          log=lambda event:None,fixture=fixture)
        provider._cache_rack(early,rack)
        self.assertIsNotNone(provider._rack_reference)
        report=VisibleHangingReport.model_validate(json.loads(review.read_text())["frames"][0]["mapped_proposal"])
        camera=current.camera_frames["perception_camera"]
        material=inset_material_samples(camera,report.contour_px,report.material_px)
        aperture=estimate(camera,report.contour_px,material)
        self.assertEqual(aperture["status"],"candidate_not_motion_validated")
        from development.hanging.public_provider import observed_corridor
        corridor=observed_corridor(current,aperture)
        self.assertEqual(corridor["axial_interval_m"],[0.,.05])
        self.assertAlmostEqual(corridor["radius_m"],.0065)
        self.assertEqual(corridor["foreground_collisions"],0)
        self.assertFalse(corridor["visibility"]["full_slab_coverage_verified"])
        with self.assertRaisesRegex(ValueError,"full_slab_coverage_unverified"):
            measured_bounds(current,report,aperture,fixture=fixture,episode_id="saved224",
                            rack_reference=provider._rack_reference)
        from development.hanging.public_provider import topology_slab_coverage,MUG_CLASS_CORRIDOR_CERTIFICATE
        # At 6.5 mm, measured sensitivity fits the all-variant CAD margin under
        # explicitly assumed exact registration. This is not evidence that the
        # live class binding or registration was independently verified.
        diagnostic=replace(fixture,topology_certificate=MUG_CLASS_CORRIDOR_CERTIFICATE,
            asset_class_binding_verified=True,mug_scale=.08,
            mug_mesh_hash="392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812",
            aperture_registration_error_m=0.,aperture_axis_error_rad=0.,centerline_verified=True)
        bounds=measured_bounds(current,report,aperture,fixture=diagnostic,episode_id="saved224",
                               rack_reference=provider._rack_reference)
        self.assertTrue(bounds["clearance_validated"])
        coverage=bounds["slab_coverage"]
        self.assertAlmostEqual(coverage["canonical_margin_m"],.004013011363462992)
        self.assertAlmostEqual(coverage["registration_displacement_m"],.003932063895006142)
        with self.assertRaisesRegex(ValueError,"aperture_registration_unverified"):
            measured_bounds(current,report,aperture,fixture=replace(diagnostic,aperture_registration_error_m=None),
                            episode_id="saved224",rack_reference=provider._rack_reference)
        with self.assertRaisesRegex(ValueError,"topology_registration_margin_insufficient"):
            measured_bounds(current,report,aperture,fixture=replace(diagnostic,aperture_registration_error_m=.0001),
                            episode_id="saved224",rack_reference=provider._rack_reference)

    def fixture_case(self):
        from development.hanging.public_provider import FixtureBounds
        camera = RGBDCamera(np.ones((30, 30), np.float32),
            np.array([[100., 0., 15.], [0., 100., 15.], [0., 0., 1.]]), np.eye(4))
        observation = SimpleNamespace(sequence=464, camera_frames={"perception_camera": camera})
        fixture = FixtureBounds("test", 464, calibration_sha256(camera), "a"*64, "b"*64,
            "synthetic CPU fixture, not verified real assets", True, True)
        aperture = dict(status="candidate_not_motion_validated",
            frame_pose_world_wxyz=[0., 0., 1., 1., 0., 0., 0.],
            uncertainty=dict(maximum_plane_residual_m=.00158, center_sensitivity_m=.001749),
            geometry_diagnostics=dict(principal_rms_span_m=[.024, .0105, .001],
                                      centroid_inscribed_radius_m=.011429))
        report = visible().model_copy(update=dict(opposite_handle_faces_visible=False,
            peg_keypoints_px=[(10.,10.), (20.,10.), (10.,20.)]))
        return observation, fixture, aperture, report

    def test_explicit_fixture_accepts_only_verified_geometry_and_positive_clearance(self):
        observation, fixture, aperture, report = self.fixture_case()
        samples = [(np.zeros(3), .0001), (np.array([.095, 0., 0.]), .0001), (np.array([0., .1, 0.]), .0001)]
        with patch("development.hanging.public_provider.read_surface", side_effect=samples), patch(
                "development.hanging.public_provider.observed_corridor",
                return_value=dict(foreground_collisions=0, hidden_space="unknown")):
            bounds = measured_bounds(observation, report, aperture, fixture=fixture, episode_id="test")
        self.assertTrue(bounds["clearance_validated"])
        self.assertAlmostEqual(bounds["radial_clearance_m"], .00418)
        self.assertAlmostEqual(bounds["axial_clearance_m"], .046809)
        from development.hanging.public_provider import MUG_CLASS_CORRIDOR_CERTIFICATE
        aperture["uncertainty"]["maximum_normal_sensitivity_deg"]=.1
        certified=replace(fixture,full_slab_coverage_verified=False,
            topology_certificate=MUG_CLASS_CORRIDOR_CERTIFICATE,asset_class_binding_verified=True,
            mug_mesh_hash="392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812",
            mug_scale=.08,aperture_registration_error_m=0.,aperture_axis_error_rad=0.)
        with patch("development.hanging.public_provider.read_surface",side_effect=samples),patch(
                "development.hanging.public_provider.observed_corridor",
                return_value=dict(foreground_collisions=0,minimum_radial_margin_m=.0061)):
            bounds=measured_bounds(observation,report,aperture,fixture=certified,episode_id="test")
        self.assertTrue(bounds["full_slab_coverage_verified"])
        self.assertTrue(bounds["clearance_validated"])

    def test_unverified_slab_stale_contract_and_missing_centerline_fail_closed(self):
        observation, fixture, aperture, report = self.fixture_case()
        for changed, reason in [(replace(fixture, full_slab_coverage_verified=False), "full_slab"),
                                (replace(fixture, sequence=463), "stale"),
                                (replace(fixture, centerline_verified=False), "centerline")]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                measured_bounds(observation, report, aperture, fixture=changed, episode_id="test")

    def test_corridor_checks_hidden_side_and_does_not_certify_empty_space(self):
        from development.hanging.public_provider import observed_corridor
        observation, _, aperture, _ = self.fixture_case()
        # Identity frame has normal +X, hidden-side sweep is -X.
        result = observed_corridor(observation, aperture)
        self.assertGreater(result["foreground_collisions"], 0)
        self.assertEqual(result["hidden_space"], "unknown")

    def test_depth_rays_behind_occluder_are_unknown_despite_zero_collisions(self):
        from development.hanging.public_provider import observed_corridor
        camera=RGBDCamera(np.full((40,40),.5,np.float32),
            np.array([[100.,0.,20.],[0.,100.,20.],[0.,0.,1.]]),np.eye(4))
        observation=SimpleNamespace(camera_frames={"perception_camera":camera})
        aperture=dict(frame_pose_world_wxyz=[0.,0.,1.,1.,0.,0.,0.])
        result=observed_corridor(observation,aperture)
        self.assertEqual(result["foreground_collisions"],0)
        self.assertEqual(result["visibility"]["observed_free_samples"],0)
        self.assertFalse(result["visibility"]["full_slab_coverage_verified"])
        # All three proposed shortcuts hold, but an opaque first return hides
        # an obstruction 20 mm behind the fitted front plane. Bounding its depth
        # does not establish a through-hole in the unobserved volume.
        self.assertGreater(result["minimum_radial_margin_m"],.005)
        self.assertLess(.047991,.08943749)
        hidden_point=np.array([-.020,0.,1.])
        self.assertLess(abs(hidden_point[0]),.047991)
        self.assertLess(np.linalg.norm(hidden_point[1:]-[0.,1.]),.0075)
        pixel=camera.intrinsic@hidden_point
        u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
        self.assertGreater(hidden_point[2],camera.depth_m[v,u])

    def test_fixture_collision_and_radial_budget_reject(self):
        observation, fixture, aperture, report = self.fixture_case()
        samples = [(np.zeros(3), .0001), (np.array([.095, 0., 0.]), .0001), (np.array([0., .1, 0.]), .0001)]
        with patch("development.hanging.public_provider.read_surface", side_effect=samples), patch(
                "development.hanging.public_provider.observed_corridor",
                return_value=dict(foreground_collisions=1)):
            with self.assertRaisesRegex(ValueError, "corridor_foreground_collision"):
                measured_bounds(observation, report, aperture, fixture=fixture, episode_id="test")
        aperture["geometry_diagnostics"]["centroid_inscribed_radius_m"] = .011
        with patch("development.hanging.public_provider.read_surface", side_effect=samples):
            with self.assertRaisesRegex(ValueError, "radial_clearance_insufficient"):
                measured_bounds(observation, report, aperture, fixture=fixture, episode_id="test")

    def test_saved_464_inset_preserves_estimator_guards_and_fixture_stays_unverified(self):
        import json
        from development.hanging.aperture_frame import estimate, inset_material_samples
        from development.hanging.public_provider import FixtureBounds, observed_corridor
        from harness.perception.replay import load_public_observation
        source = Path("data/runs/seed4_obs464_semantic_97c95fb4a5ca4b9489c3851071345ef1/results.json")
        archive = Path("data/runs/seed4_perception640_20261103_084536_510/observations/observation_00000464.npz")
        if not source.exists() or not archive.exists():
            self.skipTest("saved artifacts unavailable")
        report = VisibleHangingReport.model_validate(json.loads(source.read_text())["frames"][0]["mapped_proposal"])
        observation = load_public_observation(archive)
        camera = observation.camera_frames["perception_camera"]
        material = inset_material_samples(camera, report.contour_px, report.material_px)
        self.assertLessEqual(np.linalg.norm(np.array(material)-report.material_px, axis=1).max(), 2.)
        aperture = estimate(camera, report.contour_px, material)
        self.assertEqual(aperture["status"], "candidate_not_motion_validated", aperture)
        self.assertFalse(aperture["motion_authorized"])
        self.assertFalse(aperture["geometry_diagnostics"]["clearance_validated"])
        self.assertEqual(observed_corridor(observation, aperture)["foreground_collisions"], 0)
        fixture = FixtureBounds("saved", 464, calibration_sha256(camera), "a"*64, "b"*64, "partial mesh query")
        with self.assertRaisesRegex(ValueError, "full_slab_coverage_unverified"):
            measured_bounds(observation, report, aperture, fixture=fixture, episode_id="saved")

    def test_geometric_cache_bridges_early_low_confidence_axis_to_current_aperture(self):
        _, fixture, aperture, _ = self.fixture_case()
        camera = RGBDCamera(np.ones((40, 40), np.float32),
            np.array([[200.,0.,20.], [0.,200.,20.], [0.,0.,1.]]), np.eye(4))
        image = np.zeros((40,40,3),np.uint8)
        images = {name:image for name in (*CAMERAS,"perception_camera")}
        early = Observation(192, images, tuple(np.zeros(14)), {"perception_camera":camera})
        current = Observation(464, images, tuple(np.zeros(14)), {"perception_camera":camera})
        fixture = replace(fixture, calibration_hash=calibration_sha256(camera))
        provider = PublicRGBDHangingProvider(Mock(),model="test",episode_id="test",store=Mock(),
                                            log=lambda event:None,fixture=fixture)
        report = visible().model_copy(update=dict(confidence=.3,peg_confidence=.3,
            peg_keypoints_px=[(5.,10.),(24.,10.),(5.,30.)],peg_width_px=None))
        provider._cache_rack(early,report)
        self.assertEqual(provider._rack_reference[0].sequence,192)
        occluded = report.model_copy(update={"peg_keypoints_px":None})
        with patch("development.hanging.public_provider.observed_corridor",return_value=dict(foreground_collisions=0)):
            bounds=measured_bounds(current,occluded,aperture,fixture=fixture,episode_id="test",
                                   rack_reference=provider._rack_reference)
        self.assertTrue(bounds["clearance_validated"])
        self.assertEqual(bounds["rack_reference_sequence"],192)
        self.assertAlmostEqual(bounds["measured_axis_length_m"],.095)
        incompatible=replace(fixture,calibration_hash="bad")
        with self.assertRaisesRegex(ValueError,"incompatible"):
            measured_bounds(current,occluded,aperture,fixture=incompatible,episode_id="test",
                            rack_reference=provider._rack_reference)

    def test_saved_256_axis_cannot_prime_cache_despite_unoccluded_image(self):
        import json
        from development.hanging.public_provider import fixture_axis
        from harness.perception.replay import load_public_observation
        root=Path("data/runs/seed4_perception640_20261103_084536_510")
        if not (root/"observations/observation_00000256.npz").exists():
            self.skipTest("saved recording unavailable")
        rows=[json.loads(line) for line in (root/"supervisor_events.jsonl").read_text().splitlines()]
        row=next(r for r in rows if r.get("event_type")=="public_hanging_proposal" and r.get("observation_sequence")==256)
        early=load_public_observation(root/"observations/observation_00000256.npz")
        report=VisibleHangingReport.model_validate(row["report"])
        with self.assertRaisesRegex(ValueError,"axis_length_inconsistent"):
            fixture_axis(early,report.peg_keypoints_px)
        _,fixture,_,_=self.fixture_case()
        fixture=replace(fixture,calibration_hash=calibration_sha256(early.camera_frames["perception_camera"]))
        provider=PublicRGBDHangingProvider(Mock(),model="test",episode_id="test",store=Mock(),log=lambda event:None,fixture=fixture)
        provider._cache_rack(early,report)
        self.assertIsNone(provider._rack_reference)
