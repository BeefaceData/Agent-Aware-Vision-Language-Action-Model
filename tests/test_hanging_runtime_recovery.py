"""Synthetic observation-driven contracts, not autonomous perception validation."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import inspect
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from development.hanging import motion, runtime
from development.hanging.runtime import (
    PublicHangingProposal, ProposalResult, assess_entry, validate_current_geometry,
    recover_current,
)
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import RGBDCamera, calibration_sha256
from harness.skills.schemas import SkillCommand
from tests import test_aperture_frame as aperture_fixtures
from tests.test_hang_targets import report
from tests.test_hanging_recovery_runner import PublicEnvironment


class RuntimeRecoveryTests(unittest.TestCase):
    def test_current_narrow_aperture_uses_rss_in_recovery_preparation(self):
        observation, proposal, tcp = self.fixture()
        proposal = replace(proposal, contour_px=50+(proposal.contour_px-50)*.3,
                           mating_prior={**proposal.mating_prior, "radial_clearance_m": .0065})
        grounded = self.validate(observation, proposal, tcp)
        self.assertEqual(grounded.status, "confirmed", grounded.reason)
        geometry = grounded.geometry
        with self.assertRaisesRegex(ValueError, "clearance_insufficient"):
            motion.aperture_targets(observation, geometry.aperture, geometry.rack_pose, tcp, proposal.mating_prior)
        environment = SimpleNamespace(arm_state=lambda arm: {"pose": tcp.copy()})
        provider = Mock()
        provider.propose.return_value = ProposalResult("confirmed", "known synthetic geometry", proposal)
        events = []
        def preparation_only(env, prepare, anchors, gate, store, **kwargs):
            anchor = prepare(SkillCommand(action="move_to", arm="right", parameters={"target": "rack_approach"}), observation)
            self.assertEqual(anchor.observation_sequence, observation.sequence)
            self.assertEqual(gate.position_tolerance_m, .005)
            self.assertEqual(gate.orientation_tolerance_rad, .05)
            return dict(status="prepared", stages={})
        with patch.object(runtime, "execute_stages", side_effect=preparation_only):
            result = recover_current(environment, Mock(), episode_id="episode-a", deadline=time.monotonic()+10,
                                     provider=provider, log=events.append)
        self.assertEqual(result["entry_assessment"]["status"], "confirmed")
        evidence = next(e["evidence"] for e in events if e["event_type"] == "aperture_target_preparation")
        self.assertEqual(evidence["clearance_model"], "rss_sensitivity")
        self.assertLess(evidence["required_radial_clearance_m"], .0065)
        self.assertGreater(evidence["required_radial_clearance_m"], .005)
        uncertain = {**proposal.mating_prior, "position_uncertainty_m": .02}
        with self.assertRaisesRegex(ValueError, "clearance_insufficient"):
            motion.aperture_targets(observation, geometry.aperture, geometry.rack_pose, tcp,
                                    uncertain, clearance_model="rss_sensitivity")

    def test_rss_matches_recorded_model_and_rejects_invalid_uncertainty(self):
        assessment = dict(uncertainty=dict(center_sensitivity_m=.004318951124973484,
            maximum_plane_residual_m=.0010995578718002443, maximum_normal_sensitivity_deg=4.694103834569866))
        prior = dict(position_uncertainty_m=0., orientation_uncertainty_rad=0., engaged_length_m=.03,
                     thickness_uncertainty_m=.004, margin_m=.0005)
        radial, axial = motion.rss_clearance_budget(assessment, prior)
        self.assertAlmostEqual(radial, .005588199710455851, places=14)
        self.assertAlmostEqual(axial, .005259552376002362, places=14)
        for value in (-1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid_rss"):
                motion.rss_clearance_budget(assessment, {**prior, "position_uncertainty_m": value})
        with self.assertRaisesRegex(ValueError, "terms_required"):
            motion.rss_clearance_budget({"uncertainty": {}}, prior)

    def fixture(self, sequence=17, translation=0.):
        fixture = aperture_fixtures.ApertureFrameTests()
        fixture.setUp()
        depth = fixture.depth.copy()
        depth[72:89, 72:89] = .9
        extrinsic = np.eye(4)
        extrinsic[0, 3] = -translation
        camera = fixture.camera(depth, extrinsic)
        image = np.zeros((*depth.shape, 3), dtype=np.uint8)
        observation = Observation(sequence, {"perception_camera": image}, tuple(np.zeros(14)),
                                  {"perception_camera": camera})
        rack = {**report("rack_peg", sequence, [[50, 50], [60, 50], [50, 60]]),
                "camera": "perception_camera", "image_shape_px": list(depth.shape)}
        # These dimensions describe this synthetic fixture, not inferred priors.
        bounds = dict(handle_thickness_m=.008, thickness_uncertainty_m=.002,
            radial_clearance_m=.030, axial_clearance_m=.050, position_uncertainty_m=.001,
            orientation_uncertainty_rad=.01, engaged_length_m=.030, margin_m=.0005,
            roll_min_rad=-np.pi, roll_max_rad=np.pi, basis="synthetic geometry",
            dimensions_validated=True, dimension_basis="known synthetic fixture",
            peg_radius_m=.004, peg_length_m=.08, body_clearance_m=.005,
            minimum_handle_height_m=.01)
        proposal = PublicHangingProposal("episode-a", sequence, "perception_camera",
            sha256(image.tobytes()).hexdigest(), calibration_sha256(camera), "hanging", True,
            "synthetic multi-view held-object assessment", "public_rgbd", True, True,
            fixture.contour, fixture.material, [[74, 74], [86, 74], [86, 86], [74, 86]], rack, bounds)
        return observation, proposal, np.array([translation, 0., 1., 1., 0., 0., 0.])

    def validate(self, observation, proposal, tcp):
        return validate_current_geometry(observation, tcp, ProposalResult("confirmed", "fixture", proposal),
                                         episode_id="episode-a")

    def test_entry_distinguishes_unknown_rejected_and_confirmed_without_gripper_signal(self):
        _, proposal, _ = self.fixture()
        self.assertEqual(assess_entry(proposal).status, "confirmed")
        for changes, status in [({"right_mug_held": None}, "unknown"),
                ({"right_mug_held": False}, "rejected"), ({"task_stage": None}, "unknown"),
                ({"task_stage": "handoff"}, "rejected"), ({"source": "development_oracle"}, "rejected"),
                ({"grasp_reason": ""}, "unknown")]:
            with self.subTest(changes=changes):
                self.assertEqual(assess_entry(replace(proposal, **changes)).status, status)

    def test_geometry_accepts_different_sequences_and_current_world_translations(self):
        for sequence, translation in ((17, -.12), (81, .08), (1203, .15)):
            with self.subTest(sequence=sequence):
                observation, proposal, tcp = self.fixture(sequence, translation)
                result = self.validate(observation, proposal, tcp)
                self.assertEqual(result.status, "confirmed", result.reason)
                self.assertEqual(result.geometry.aperture["sequence"], sequence)
                self.assertAlmostEqual(result.geometry.aperture["frame_pose_world_wxyz"][0], translation)
                targets, evidence = motion.aperture_targets(observation, result.geometry.aperture,
                    result.geometry.rack_pose, tcp, proposal.mating_prior)
                self.assertEqual(set(targets), {"rack_approach", "rack_seat"})
                self.assertFalse(evidence["motion_authorized"])

    def test_wrong_episode_sequence_image_camera_and_missing_depth_block_geometry(self):
        observation, proposal, tcp = self.fixture()
        for changes, reason in [({"episode_id": "other"}, "wrong_episode"),
                ({"sequence": 16}, "stale"), ({"rgb_sha256": "bad"}, "mismatch"),
                ({"calibration_sha256": "bad"}, "mismatch"), ({"camera": "right_camera"}, "unavailable")]:
            with self.subTest(changes=changes):
                result = self.validate(observation, replace(proposal, **changes), tcp)
                self.assertNotEqual(result.status, "confirmed")
                self.assertIn(reason, result.reason)
        result = self.validate(replace(observation, camera_frames=None), proposal, tcp)
        self.assertEqual(result.status, "unknown")

    def test_aperture_size_and_measured_wrist_pose_are_not_fixed_to_one_recording(self):
        for scale, yaw in ((1., 0.), (.9, .4)):
            with self.subTest(scale=scale, yaw=yaw):
                observation, proposal, tcp = self.fixture()
                proposal = replace(proposal, contour_px=50+(proposal.contour_px-50)*scale,
                                   mating_prior={**proposal.mating_prior, "radial_clearance_m": .025})
                tcp[3:] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
                result = self.validate(observation, proposal, tcp)
                self.assertEqual(result.status, "confirmed", result.reason)
                targets, evidence = motion.aperture_targets(observation, result.geometry.aperture,
                    result.geometry.rack_pose, tcp, proposal.mating_prior)
                self.assertEqual(targets["rack_seat"].shape, (7,))
                self.assertNotAlmostEqual(evidence["selected_roll_rad"], -np.pi/4)

    def test_ambiguous_semantics_dimensions_depth_and_clearance_are_not_assumed(self):
        observation, proposal, tcp = self.fixture()
        for changes in ({"semantics_confirmed": None}, {"complete_visible_contour": False},
                        {"mating_prior": {**proposal.mating_prior, "dimensions_validated": False}}):
            self.assertEqual(self.validate(observation, replace(proposal, **changes), tcp).status, "unknown")
        invalid = replace(proposal, mating_prior={**proposal.mating_prior, "radial_clearance_m": .2})
        self.assertEqual(self.validate(observation, invalid, tcp).reason, "radial_clearance_exceeds_observed_aperture")
        invalid = replace(proposal, rack_report={**proposal.rack_report, "camera": "right_camera"})
        self.assertEqual(self.validate(observation, invalid, tcp).reason, "rack_camera_mismatch")
        camera = observation.camera_frames[proposal.camera]
        depth = camera.depth_m.copy()
        u, v = np.rint(proposal.material_px[0]).astype(int)
        depth[v, u] = 5.
        changed = RGBDCamera(depth, camera.intrinsic, camera.world_to_camera)
        result = self.validate(replace(observation, camera_frames={proposal.camera: changed}), proposal, tcp)
        self.assertEqual(result.status, "rejected")

    def test_unreachable_targets_and_unsupported_clearance_do_not_publish_motion(self):
        observation, proposal, tcp = self.fixture()
        result = self.validate(observation, proposal, tcp)
        rack = np.asarray(result.geometry.rack_pose).copy()
        rack[0] += 2
        with self.assertRaisesRegex(ValueError, "out_of_bounds"):
            motion.aperture_targets(observation, result.geometry.aperture, rack, tcp, proposal.mating_prior)
        with self.assertRaisesRegex(ValueError, "clearance_insufficient"):
            motion.aperture_targets(observation, result.geometry.aperture, result.geometry.rack_pose, tcp,
                                    {**proposal.mating_prior, "radial_clearance_m": .001})

    def test_missing_provider_returns_unknown_before_any_motion(self):
        environment = PublicEnvironment(x=0.)
        environment.sequence = 81
        with tempfile.TemporaryDirectory() as directory:
            result = recover_current(environment, ObservationStore(Path(directory)/"observations"),
                                     episode_id="episode-a", deadline=time.monotonic()+10)
        self.assertEqual(result["entry_assessment"]["status"], "unknown")
        self.assertEqual(result["reason"], "validated_visible_aperture_provider_unavailable")
        self.assertEqual(result["stages"], {})
        self.assertEqual(environment.ticks, 0)

    def test_shared_execution_has_no_recording_or_fixed_entry_dependencies(self):
        for module in (motion, runtime):
            source = inspect.getsource(module)
            for forbidden in ("entry660", "== 660", "seed ==", "RecordedGrounder", "validate_entry", "inspect_archive"):
                self.assertNotIn(forbidden, source)

    def run_wiring(self, sequence, *, reject_after_approach=False, depth_failure=False,
                   lose_alignment=False):
        """Fake geometry validation isolates sequencing; geometry has separate tests."""
        environment = PublicEnvironment(x=0.)
        environment.sequence = sequence
        environment.arms["right"]["gripper"] = 0.
        if lose_alignment:
            original_tick = environment.tick
            def tick(*args, **kwargs):
                original_tick(*args, **kwargs)
                if 0 < environment.arms["right"]["gripper"] < 1:
                    environment.arms["right"]["pose"][0] += .006
            environment.tick = tick
        calls, events = [], []
        class Provider:
            def propose(self, observation, measured_tcp, *, episode_id):
                calls.append((observation.sequence, measured_tcp.copy()))
                return ProposalResult("confirmed", "synthetic provider")
        _, proposal, _ = self.fixture(sequence)
        def validate(observation, measured_tcp, result, *, episode_id):
            if reject_after_approach and len(calls) > 1:
                return runtime.GroundingResult("unknown", "right_arm_grasp_unknown")
            camera = observation.camera_frames["perception_camera"]
            aperture = dict(status="candidate_not_motion_validated", sequence=observation.sequence,
                camera="perception_camera", rgb_sha256=sha256(observation.images["perception_camera"].tobytes()).hexdigest(),
                calibration_sha256=calibration_sha256(camera), frame_pose_world_wxyz=list(measured_tcp),
                material_points_world=[[x, y, 1.] for x in (-.02, 0., .02) for y in (-.02, 0., .02)],
                uncertainty=dict(center_sensitivity_m=.001, maximum_normal_sensitivity_deg=1.,
                                 maximum_plane_residual_m=.0005))
            synthetic = replace(proposal, sequence=observation.sequence,
                                mating_prior={**proposal.mating_prior, "radial_clearance_m": .1})
            return runtime.GroundingResult("confirmed", "synthetic geometry",
                runtime.CurrentGeometry(synthetic, aperture, (0., 0., 1., 1., 0., 0., 0.)))
        confirmation = dict(status="rejected" if depth_failure else "confirmed_visible_subset",
                            reason="insufficient_visible_handle_support" if depth_failure else "synthetic match")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runtime, "validate_current_geometry", side_effect=validate), \
                patch.object(runtime, "confirm_visible_handle", return_value=confirmation) as confirm:
            result = recover_current(environment, ObservationStore(Path(directory)/"observations"),
                episode_id="episode-a", deadline=time.monotonic()+10, provider=Provider(), log=events.append)
            checks = confirm.call_count
        return result, environment, calls, checks, events

    def test_runtime_sequences_and_measured_starts_at_multiple_observation_numbers(self):
        for sequence in (17, 81, 1203):
            with self.subTest(sequence=sequence):
                result, environment, calls, checks, events = self.run_wiring(sequence)
                self.assertEqual(list(result["stages"]), ["approach", "seat", "release", "retreat"])
                self.assertTrue(all(stage["completed"] for stage in result["stages"].values()))
                # One fit at approach; the seat reuses it after confirming the
                # handle moved rigidly with the measured TCP in a fresh image.
                self.assertEqual([call[0] for call in calls], [sequence])
                self.assertEqual(checks, 1)
                reuse = [e for e in events if e.get("event_type") == "aperture_target_reuse"]
                self.assertEqual(len(reuse), 1)
                self.assertGreater(reuse[0]["observation_sequence"], sequence)
                self.assertEqual(result["seat_settling"]["ticks"], 25)
                self.assertEqual(result["public_support"], "unknown")
                self.assertEqual(result["retention"]["status"], "unknown")
                self.assertEqual(environment.arms["right"]["gripper"], 1.)
                self.assertEqual(result["post_retreat_observation_only_ticks"], 250)
                self.assertEqual(result["analytic_ticks"], environment.ticks)

    def test_untracked_handle_prevents_seating_opening_and_retreat(self):
        # A slipped or lost grasp leaves the handle away from where the rigid
        # TCP prediction puts it, so the fresh-image confirmation rejects.
        result, environment, calls, checks, events = self.run_wiring(81, depth_failure=True)
        self.assertEqual(list(result["stages"]), ["approach"])
        self.assertEqual(result["reason"], "insufficient_visible_handle_support")
        self.assertEqual(result["stopped_at"], "seat")
        self.assertEqual(environment.arms["right"]["gripper"], 0.)
        self.assertEqual(len(calls), 1)

    def test_seat_does_not_depend_on_refitting_the_handle_near_the_rack(self):
        # Before the fix a second full fit was required for the seat; a hidden
        # opening near the rack then stopped recovery although tracking held.
        result, environment, calls, checks, events = self.run_wiring(81, reject_after_approach=True)
        self.assertEqual(list(result["stages"]), ["approach", "seat", "release", "retreat"])
        self.assertEqual(len(calls), 1)

    def test_alignment_loss_during_opening_stops_before_retreat(self):
        result, environment, calls, checks, events = self.run_wiring(81, lose_alignment=True)
        self.assertEqual(list(result["stages"]), ["approach", "seat", "release"])
        self.assertEqual(result["stages"]["release"]["stop_reason"], "rack_seat_alignment_lost")
        self.assertLess(environment.arms["right"]["gripper"], 1.)
        self.assertEqual(result["retention"]["status"], "unknown")

    def test_saved_success_keeps_exact_adapter_targets_and_roll(self):
        import json
        from development.hanging.recovery import entry660_demo_preflight
        root = Path(__file__).resolve().parents[1]
        folder = root/"data/runs/hanging_recovery_20261031T104702Z_1793443622777436500"
        if not (folder/"report.json").is_file():
            self.skipTest("ignored successful recording unavailable")
        saved = json.loads((folder/"report.json").read_text())["preflight"]
        result = entry660_demo_preflight(folder/"observations/observation_00000660.npz",
            saved["annotation"], saved["targets"]["rack_peg"]["proposal"], allow_override=True)
        for name, pose in saved["prepared_targets"].items():
            np.testing.assert_allclose(result["prepared_targets"][name], pose, atol=1e-12, rtol=0)
        self.assertEqual(result["clearance_terms"]["selected_roll_rad"], -np.pi/4)
        self.assertEqual(result["clearance_terms"], saved["clearance_terms"])
