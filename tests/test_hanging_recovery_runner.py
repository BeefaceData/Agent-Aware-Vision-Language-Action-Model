"""Focused synthetic integration; no real perception, API or simulator."""

from pathlib import Path
import tempfile
import time
import unittest
from hashlib import sha256
import numpy as np
from unittest.mock import patch

from development.hanging.recovery import (
    recover, aperture_targets, entry660_demo_preflight, confirm_visible_handle, DemoApertureProvider,
    settle_seat,
)
from harness.predicates.rack_seat_gate import RackSeatReleaseGate
from harness.perception.rgbd import calibration_sha256
from harness.perception.rgbd import RGBDCamera
from harness.environment.base import Observation
from harness.environment.layout import CAMERAS
from harness.perception.observations import ObservationStore
from tests.test_analytic_skills import AnalyticEnvironment
from tests.test_hang_targets import make_observation, report, RACK_POINTS, MUG_POINTS


class PublicEnvironment(AnalyticEnvironment):
    def observe(self):
        original = make_observation(self.sequence)
        image = original.images["right_camera"]
        return Observation(self.sequence,
            {name: image.copy() for name in (*CAMERAS, "perception_camera")},
            original.command_state,
            {"perception_camera": original.camera_frames["right_camera"]})


class HangingRecoveryRunnerTests(unittest.TestCase):
    def test_trim_one_marginal_sample_without_changing_five_mm_match_threshold(self):
        depth=np.ones((100,100),dtype=np.float32)
        intrinsic=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        camera=RGBDCamera(depth,intrinsic,np.eye(4))
        pixels=np.array([[30,30],[40,25],[50,30],[50,50],[40,55],
                         [30,50],[60,40],[65,45],[70,50],[60,60]])
        expected=np.array([camera.unproject(*uv) for uv in pixels])
        depth[49:52,69:72]+=.0052
        result=confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["status"],"confirmed_visible_subset")
        self.assertEqual(result["threshold_m"],.005)
        self.assertEqual(result["trimmed_indices"],[8])
        self.assertNotIn(8,result["visible_indices"])
        depth[59:62,59:62]+=.0052
        result=confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["status"],"rejected")
        depth[:]=1.
        depth[49:52,69:72]=1.2
        result=confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["status"],"rejected")

    def test_saved1182_single_5339_mm_outlier_is_trimmed(self):
        import json
        from harness.perception.replay import load_public_observation
        root=Path(__file__).resolve().parents[1]
        folder=root/"data/runs/hanging_recovery_20261031T101337Z_1793441617214158500"
        if not (folder/"report.json").is_file():
            self.skipTest("ignored local Entry660 recording unavailable")
        report=json.loads((folder/"report.json").read_text())
        initial=load_public_observation(folder/"observations/observation_00000660.npz")
        final=load_public_observation(folder/"observations/observation_00001182.npz")
        class Environment:
            def arm_state(self,arm):return {"pose":np.asarray(final.measured_tcp_poses[arm])}
        provider=DemoApertureProvider(initial,initial.measured_tcp_poses["right"],
            report["preflight"]["annotation"],report["preflight"]["assessment"],Environment())
        provider(final)
        self.assertEqual(provider.last_confirmation["status"],"confirmed_visible_subset")
        self.assertEqual(provider.last_confirmation["trimmed_indices"],[11])
        self.assertEqual(len(provider.last_confirmation["visible_indices"]),9)

    def test_partial_occlusion_requires_six_distributed_matches_and_rejects_background(self):
        depth = np.ones((100,100),dtype=np.float32)
        intrinsic = np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        camera = RGBDCamera(depth,intrinsic,np.eye(4))
        pixels = np.array([[30,30],[40,25],[50,30],[50,50],[40,55],[30,50],[60,40],[65,45]])
        expected = np.array([camera.unproject(*uv) for uv in pixels])
        for u,v in pixels[-2:]:depth[v-1:v+2,u-1:u+2]=.8
        result = confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["status"],"confirmed_visible_subset")
        self.assertEqual(result["visible_indices"],list(range(6)))
        self.assertEqual(result["samples"][-1]["classification"],"nearer_occluder")
        depth[29:32,29:32]=1.2
        result = confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["reason"],"visible_handle_depth_contradicts_prediction")
        depth[:]=.8
        result = confirm_visible_handle(RGBDCamera(depth,intrinsic,np.eye(4)),pixels,expected)
        self.assertEqual(result["status"],"rejected")
        self.assertEqual(result["visible_indices"],[])

    def prior(self):
        return dict(handle_thickness_m=.01, thickness_uncertainty_m=.002,
            radial_clearance_m=.2, axial_clearance_m=.2, position_uncertainty_m=.001,
            orientation_uncertainty_rad=.01, engaged_length_m=.03, margin_m=.002,
            roll_min_rad=-.2, roll_max_rad=.2, basis="synthetic fixture dimensions only")

    def assessment(self, observation, pose):
        return dict(status="candidate_not_motion_validated", sequence=observation.sequence,
            camera="perception_camera", frame_pose_world_wxyz=list(pose),
            calibration_sha256=calibration_sha256(observation.camera_frames["perception_camera"]),
            rgb_sha256=sha256(observation.images["perception_camera"].tobytes()).hexdigest(),
            uncertainty=dict(center_sensitivity_m=.001, maximum_normal_sensitivity_deg=1.))

    def run_synthetic(self, *, reject=False, aperture=False, demo=False, environment=None):
        environment = environment or PublicEnvironment(x=0.)
        environment.arms["right"]["gripper"] = 0.

        class Grounder:
            def __init__(self, client, model, target, *args):
                self.target = target

            def locate(self, observation):
                if reject:
                    raise ValueError("grounding_confidence_too_low")
                points = RACK_POINTS if self.target == "rack_peg" else MUG_POINTS
                return {**report(self.target, observation.sequence, points),
                        "camera": "perception_camera"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            store = ObservationStore(output / "observations")
            with patch("development.hanging.recovery.RecordedGrounder", Grounder):
                prior=self.prior() if aperture else None
                if demo:
                    prior["allow_entry660_demo_override"]=True
                def fresh(observation):
                    result=self.assessment(observation,environment.arm_state("right")["pose"])
                    result["uncertainty"]["maximum_plane_residual_m"]=.0005
                    return result
                result = recover(environment, object(), "test-model", output, store,
                                 deadline=time.monotonic() + 60,
                                 aperture_provider=fresh if aperture else None,
                                 mating_prior=prior)
        return result, environment

    def test_four_mm_seat_settles_and_authorizes_release_then_post_success_retreat(self):
        class FourMmSeat(PublicEnvironment):
            @property
            def native_success(self):
                return self.arms["right"]["gripper"]>.8

            def tick(self,*args,**kwargs):
                super().tick(*args,**kwargs)
                if self.goal is not None:
                    self.arms["right"]["pose"][0]=self.goal[0]+.004
        result, environment=self.run_synthetic(aperture=True,demo=True,environment=FourMmSeat(x=0.))
        self.assertEqual(result["status"],"recovery_sequence_completed_support_unverified")
        self.assertEqual(list(result["stages"]),["approach","seat","release","retreat"])
        self.assertTrue(all(stage["completed"] for stage in result["stages"].values()))
        self.assertGreaterEqual(result["seat_settling"]["ticks"],25)
        self.assertAlmostEqual(result["stages"]["seat"]["position_error_m"],.004)
        self.assertIs(result["stages"]["release"]["predicate"]["value"],True)
        self.assertTrue(result["bounded_post_success_retreat"])
        self.assertTrue(result["native_success"])
        self.assertEqual(environment.arms["right"]["gripper"],1.)

    def test_settling_is_bounded_and_failed_authorization_is_false_not_null(self):
        environment=PublicEnvironment(x=0.)
        environment.arms["right"]["gripper"]=0.
        target=environment.arm_state("right")["pose"].copy()
        target[0]+=.006
        result=settle_seat(environment,target,environment.arm_state("right")["joints"],
                           deadline=time.monotonic()+10,max_ticks=30)
        self.assertFalse(result["completed"])
        self.assertEqual(result["ticks"],30)
        self.assertEqual(environment.arms["right"]["gripper"],0.)
        gate=RackSeatReleaseGate(environment,position_tolerance_m=.005,
                                  orientation_tolerance_rad=.05,unavailable_is_false=True)
        predicate=gate.evaluate("rack_seat_aligned",environment.observe(),arm="right")
        self.assertIs(predicate.value,False)
        self.assertEqual(predicate.reason,"rack_seat_not_confirmed")

    def test_settling_preserves_tracking_and_cancellation_guards(self):
        environment=PublicEnvironment(x=0.)
        state=environment.arm_state("right")
        result=settle_seat(environment,state["pose"],state["joints"]+.3,
                           deadline=time.monotonic()+10)
        self.assertEqual(result["reason"],"tracking_error")
        self.assertEqual(result["ticks"],0)
        result=settle_seat(environment,state["pose"],state["joints"],deadline=0)
        self.assertEqual(result["reason"],"cancelled")
        self.assertEqual(result["ticks"],0)

    def test_composed_recovery_records_separate_stages_without_policy_or_oracle(self):
        result, environment = self.run_synthetic()
        self.assertEqual(result["status"], "recovery_sequence_completed_support_unverified")
        self.assertEqual(list(result["stages"]), ["approach", "seat", "release", "retreat"])
        for stage in result["stages"].values():
            self.assertTrue(stage["completed"])
            self.assertEqual(stage["executed_policy_actions"], 0)
            self.assertEqual(stage["information_source"], "public")
        self.assertEqual(result["public_support"], "unknown")
        self.assertEqual(environment.arms["right"]["gripper"], 1.)

    def test_grounding_failure_prevents_motion_and_opening(self):
        result, environment = self.run_synthetic(reject=True)
        self.assertEqual(result["reason"], "grounding_confidence_too_low")
        self.assertEqual(result["stages"], {})
        self.assertEqual(environment.ticks, 0)
        self.assertEqual(environment.arms["right"]["gripper"], 0.)

    def test_aperture_path_composes_all_four_stages_in_synthetic_environment(self):
        result, environment = self.run_synthetic(aperture=True)
        self.assertEqual(result["status"], "recovery_sequence_completed_support_unverified")
        self.assertEqual(list(result["stages"]), ["approach", "seat", "release", "retreat"])
        self.assertTrue(all(stage["completed"] for stage in result["stages"].values()))
        self.assertEqual(result["public_support"], "unknown")

    def test_aperture_targets_require_fresh_evidence_and_dimensional_bounds(self):
        environment = PublicEnvironment(x=0.)
        observation = environment.observe()
        pose = environment.arm_state("right")["pose"]
        assessment = self.assessment(observation, pose)
        with self.assertRaisesRegex(ValueError, "explicit_dimensional"):
            aperture_targets(observation, assessment, pose, pose, None)
        with self.assertRaisesRegex(ValueError, "stale_or_wrong_camera"):
            aperture_targets(observation, {**assessment, "sequence": -1}, pose, pose, self.prior())
        prior = {**self.prior(), "radial_clearance_m": .001}
        with self.assertRaisesRegex(ValueError, "clearance_insufficient"):
            aperture_targets(observation, assessment, pose, pose, prior)
        targets, evidence = aperture_targets(observation, assessment, pose, pose, self.prior())
        self.assertAlmostEqual(evidence["surface_offset_m"], -.005)
        self.assertLessEqual(abs(evidence["selected_roll_rad"]), .2)
        self.assertEqual(evidence["ik_planning"], "not_run_requires_simulator_planner")
        self.assertFalse(evidence["motion_authorized"])

    def test_entry660_demo_requires_opt_in_and_has_no_provider_calls(self):
        import json
        root = Path(__file__).resolve().parents[1]
        archive = root / ("data/runs/entry660_20261029T202054Z_1793305254857158600/"
                          "observations/observation_00000660.npz")
        folder = root / "data/runs/entry660_reviewed_demo_feasibility"
        if not archive.is_file():
            self.skipTest("ignored local Entry660 archive unavailable")
        annotation = json.loads((folder/"handle_annotation.json").read_text())
        rack = json.loads((folder/"peg_annotation.json").read_text())
        with self.assertRaisesRegex(ValueError, "override_required"):
            entry660_demo_preflight(archive, annotation, rack)
        report = entry660_demo_preflight(archive, annotation, rack, allow_override=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["requests_attempted"], 0)
        self.assertAlmostEqual(report["peg_radius_m"], .004)
        self.assertAlmostEqual(report["clearance_terms"]["surface_offset_m"], -.004)
        self.assertLess(report["clearance_terms"]["required_radial_clearance_m"],
                        report["mating_prior"]["radial_clearance_m"])
        self.assertEqual(report["clearance_terms"]["ik_planning"], "not_run_requires_simulator_planner")
        self.assertGreaterEqual(report["clearance_terms"]["roll_visibility"]["handle_above_visible_body_m"],.01)
        self.assertGreaterEqual(report["clearance_terms"]["roll_visibility"]["predicted_visible_samples"],6)

    def test_saved1158_measured_tcp_confirms_six_visible_samples_without_slip_correction(self):
        import json
        from harness.perception.replay import load_public_observation
        root=Path(__file__).resolve().parents[1]
        folder=root/"data/runs/hanging_recovery_20261030T190923Z_1793387363052902400"
        if not (folder/"report.json").is_file():
            self.skipTest("ignored local recovery recording unavailable")
        initial=load_public_observation(folder/"observations/observation_00000660.npz")
        final=load_public_observation(folder/"observations/observation_00001158.npz")
        annotation=json.loads((root/"data/runs/entry660_reviewed_demo_feasibility/handle_annotation.json").read_text())
        rack=json.loads((root/"data/runs/entry660_reviewed_demo_feasibility/peg_annotation.json").read_text())
        report=entry660_demo_preflight(folder/"observations/observation_00000660.npz",annotation,rack,allow_override=True)
        class MeasuredEnvironment:
            def arm_state(self,arm):
                return {"pose":np.asarray(final.measured_tcp_poses[arm]),
                        "commanded_pose":np.array([9,9,9,1,0,0,0])}
        provider=DemoApertureProvider(initial,initial.measured_tcp_poses["right"],annotation,
                                      report["assessment"],MeasuredEnvironment())
        result=provider(final)
        self.assertEqual(provider.last_confirmation["visible_indices"],[0,1,2,5,6,8])
        self.assertEqual(provider.last_confirmation["status"],"confirmed_visible_subset")
        self.assertEqual(result["sequence"],1158)
        self.assertEqual(result["tracking_model"],"measured_tcp_rigid_prediction_with_visible_depth_confirmation")
