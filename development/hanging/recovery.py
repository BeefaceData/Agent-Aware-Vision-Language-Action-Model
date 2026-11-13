"""One recorded-entry hanging recovery; public RGB-D targets, no oracle control."""

import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from harness.jsonl import append_jsonl
from harness.perception.anchors import Anchor, AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.opus_frame import OpusObjectFrameGrounder
from harness.perception.preparation import PerceptionAnchorPreparer
from harness.predicates.rack_seat_gate import RackSeatReleaseGate
from harness.perception.pose_grounding import _pose_matrix, _matrix_pose
from harness.perception.rgbd import calibration_sha256

from development.hanging.motion import (
    aperture_targets as current_aperture_targets, confirm_visible_handle,
    settle_seat, execute_stages, rss_clearance_budget,
)
from development.hanging.run_entry660 import validate_package
from development.hanging.validate_entry660 import validate_entry
from development.hanging.grounding_preflight import (
    preflight, grounding_client, sanitized_failure, validate_with_crop_retry,
)


def aperture_targets(observation, assessment, rack_pose, measured_tcp, prior):
    """Legacy/development adapter; dimensional demo assumptions stay out of execution."""
    budget = None
    if isinstance(prior, dict) and prior.get("allow_entry660_demo_override") is True:
        budget = rss_clearance_budget(assessment, prior)
        prior = {**prior, "peg_radius_m": .004, "peg_length_m": .08,
                 "body_clearance_m": .005, "minimum_handle_height_m": .01}
    return current_aperture_targets(observation, assessment, rack_pose, measured_tcp, prior,
                                    clearance_budget=budget)


def entry660_demo_preflight(archive, annotation, rack_report, *, allow_override=False):
    """One-entry, user-authorized dimensional hypotheses; zero provider calls."""
    from development.hanging.aperture_frame import inspect_archive
    from harness.perception.replay import load_public_observation
    if not allow_override:
        raise ValueError("explicit_entry660_demo_override_required")
    observation = load_public_observation(archive)
    assessment = inspect_archive(archive, annotation, allow_demo_override=True)
    if assessment["status"] != "candidate_not_motion_validated":
        raise ValueError("demo_aperture_rejected:"+assessment.get("reason", "unknown"))
    if observation.sequence != 660 or not observation.measured_tcp_poses:
        raise ValueError("entry660_measured_tcp_required")
    from development.hanging.aperture_frame import _inside_opening, _material_points, Limits
    camera = observation.camera_frames["perception_camera"]
    polygon = np.asarray(annotation.get("body_polygon_px"), dtype=float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3:
        raise ValueError("reviewed_visible_body_polygon_required")
    low, high = np.floor(polygon.min(axis=0)).astype(int), np.ceil(polygon.max(axis=0)).astype(int)
    pixels = np.array([[u,v] for v in range(low[1],high[1]+1,2)
                       for u in range(low[0],high[0]+1,2)])
    pixels = pixels[_inside_opening(pixels, polygon)]
    body = [camera.unproject(u,v,radius=1) for u,v in pixels
            if np.ptp(camera.depth_m[v-1:v+2,u-1:u+2])<.01]
    if len(body)<12:
        raise ValueError("visible_body_support_insufficient")
    assessment["visible_body_points_world"] = [list(point) for point in body]
    assessment["material_points_world"] = _material_points(camera, annotation["material_px"],
        Limits(), np.asarray(annotation["contour_px"])).tolist()
    rack = StaticRackFrameCache()
    rack.cache(observation, rack_report)
    # Nominal 4 mm peg radius and -4 +/-2 mm plane offset are explicit user
    # priors, not measurements. 30 mm engagement is a bounded demo hypothesis.
    radius = assessment["geometry_diagnostics"]["centroid_inscribed_radius_m"]
    prior = dict(handle_thickness_m=.008, thickness_uncertainty_m=.004,
        radial_clearance_m=radius-.004, axial_clearance_m=.03,
        position_uncertainty_m=0., orientation_uncertainty_rad=0., engaged_length_m=.03,
        margin_m=.0005, roll_min_rad=-np.pi, roll_max_rad=np.pi,
        basis="user-authorized Entry660 demo priors; systematic errors and body clearance uncalibrated",
        allow_entry660_demo_override=True)
    targets, terms = aperture_targets(observation, assessment, rack.resolve(observation),
                                    observation.measured_tcp_poses["right"], prior)
    return dict(status="passed", model="claude-opus-5-5", sequence=660,
        reviewed_aperture_demo=True, allow_entry660_demo_override=True,
        rgb_sha256=assessment["rgb_sha256"], assessment=assessment, mating_prior=prior,
        annotation=annotation, targets={"rack_peg": {"proposal": rack_report}},
        prepared_targets={key: value.tolist() for key, value in targets.items()},
        clearance_terms=terms, maximum_requests=0, requests_attempted=0,
        depth_sha256=sha256(observation.camera_frames["perception_camera"].depth_m.tobytes()).hexdigest(),
        peg_radius_m=.004, physical_support="unknown", capture_provenance="unknown")


class DemoApertureProvider:
    """Reproject reviewed points with measured TCP; verify fresh public depth."""

    def __init__(self, initial, initial_tcp, annotation, assessment, environment, log=lambda event: None):
        from development.hanging.aperture_frame import _material_points, _intersections, Limits
        self.annotation, self.assessment, self.environment = annotation, assessment, environment
        self.initial_tcp = _pose_matrix(initial_tcp)
        self.log = log
        camera = initial.camera_frames["perception_camera"]
        self.material = _material_points(camera, annotation["material_px"], Limits(),
                                         np.asarray(annotation["contour_px"]))
        frame = _pose_matrix(assessment["frame_pose_world_wxyz"])
        self.contour = _intersections(camera, np.asarray(annotation["contour_px"]),
                                     frame[:3, 3], frame[:3, 0])

    def __call__(self, observation):
        if observation.sequence == 660:
            if sha256(observation.images["perception_camera"].tobytes()).hexdigest() != self.assessment["rgb_sha256"]:
                raise ValueError("reviewed_entry_image_mismatch")
            return self.assessment
        camera = (observation.camera_frames or {}).get("perception_camera")
        if camera is None:
            raise ValueError("fresh_aperture_rgbd_missing")
        transform = _pose_matrix(self.environment.arm_state("right")["pose"]) @ np.linalg.inv(self.initial_tcp)

        def project(points):
            world = points @ transform[:3, :3].T + transform[:3, 3]
            xyz = world @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
            if np.any(xyz[:, 2] <= 0):
                raise ValueError("tracked_aperture_behind_camera")
            uv = xyz @ camera.intrinsic.T
            return uv[:, :2]/uv[:, 2, None], world

        project(self.contour)  # Keep the contour visibility guard before checking material depth.
        pixels, expected = project(self.material)
        confirmation = confirm_visible_handle(camera, pixels, expected)
        self.last_confirmation = confirmation
        self.log(dict(event_type="fresh_handle_confirmation", observation_sequence=observation.sequence,
                      measured_tcp_pose=self.environment.arm_state("right")["pose"].tolist(),
                      confirmation=confirmation))
        if confirmation["status"] != "confirmed_visible_subset":
            raise ValueError(confirmation["reason"])
        # Preserve the reviewed frame under measured rigid motion; do not fit a
        # different surface frame to a changing/occluded subset of the arch.
        frame = transform @ _pose_matrix(self.assessment["frame_pose_world_wxyz"])
        body = np.asarray(self.assessment["visible_body_points_world"])
        result = {**self.assessment, "frame_pose_world_wxyz": _matrix_pose(frame).tolist(),
                  "material_points_world": expected.tolist(),
                  "visible_body_points_world": (body @ transform[:3,:3].T+transform[:3,3]).tolist(),
                  "confirmation": confirmation,
                  "tracking_model": "measured_tcp_rigid_prediction_with_visible_depth_confirmation"}
        return {**result, "sequence": observation.sequence, "camera": "perception_camera",
                "calibration_sha256": calibration_sha256(camera),
                "rgb_sha256": sha256(observation.images["perception_camera"].tobytes()).hexdigest()}


class RecordedGrounder:
    def __init__(self, client, model, target, output, store, log, initial_reports=None):
        self.grounder = OpusObjectFrameGrounder(grounding_client(client), model, target=target,
                                               camera="perception_camera")
        self.target, self.output, self.store, self.log = target, Path(output), store, log
        self.initial_reports = initial_reports

    def locate(self, observation):
        artifact = self.store.save(observation)
        if self.initial_reports is not None:
            if (observation.sequence != 660 or sha256(
                    observation.images["perception_camera"].tobytes()).hexdigest()
                    != self.initial_reports["rgb_sha256"]):
                raise ValueError("preflight_image_no_longer_current")
            report = self.initial_reports["targets"][self.target]["proposal"]
            self.initial_reports = None
        else:
            report = self.grounder.locate(observation)
            review = {}
            report = validate_with_crop_retry(self.grounder, observation, report, self.output,
                row=review, artifact_prefix=f"{self.target}_{observation.sequence:08d}",
                on_retry=lambda: self.log(dict(event_type="public_frame_crop_retry",
                    target=self.target, observation_sequence=observation.sequence)))
            self.log(dict(event_type="public_frame_validation", target=self.target,
                          observation_sequence=observation.sequence, review=review))
        name = f"{self.target}_{observation.sequence:08d}"
        (self.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        image = Image.fromarray(observation.images["perception_camera"])
        draw = ImageDraw.Draw(image)
        for index, (u, v) in enumerate(report.get("keypoints_px", ())):
            draw.ellipse((u-3, v-3, u+3, v+3), outline="red")
            draw.text((u+4, v), str(index+1), fill="red")
        image.save(self.output / f"{name}.png")
        self.log(dict(event_type="public_frame_grounding", target=self.target,
                      observation=artifact, report=report))
        return report


def recover(environment, client, model, output, store, *, deadline, initial_reports=None,
            aperture_provider=None, mating_prior=None):
    """At most approach, seat, guarded release, retreat; first failure ends attempt."""
    output = Path(output)
    def log(event):
        append_jsonl(output / "recovery_events.jsonl", event)
    anchors = AnchorRegistry()
    cache = StaticRackFrameCache()
    demo = (mating_prior or {}).get("allow_entry660_demo_override") is True
    gate = RackSeatReleaseGate(environment, position_tolerance_m=.005 if demo else .015,
                              orientation_tolerance_rad=.05 if demo else .10,
                              unavailable_is_false=aperture_provider is not None)
    report = dict(status="grounding_failed", stages={}, public_support="unknown",
                  full_state_restored=False, capture_provenance="unknown")
    try:
        initial = environment.observe()
        rack = RecordedGrounder(client, model, "rack_peg", output, store, log, initial_reports)
        cache.cache(initial, rack.locate(initial))
        if aperture_provider is None:
            mug = RecordedGrounder(client, model, "mug_handle", output, store, log, initial_reports)
            prepare = PerceptionAnchorPreparer(
                RGBDPerceptionProvider(None, anchors), environment, anchors,
                hang_target_cache=cache, mug_frame_grounder=mug,
                mug_frame_camera="perception_camera")
        else:
            def prepare(command, observation):
                if command.action.value != "move_to":
                    return None
                targets, evidence = aperture_targets(observation, aperture_provider(observation),
                    cache.resolve(observation), environment.arm_state("right")["pose"], mating_prior)
                for name, pose in targets.items():
                    anchors.bind(name, "right", Anchor(tuple(pose), observation.sequence, source="perception"))
                log(dict(event_type="aperture_target_preparation", evidence=evidence,
                         observation_sequence=observation.sequence))
                return anchors.resolve(command.parameters.target, "right", observation)
        return {**report, **execute_stages(environment, prepare, anchors, gate, store,
            deadline=deadline, log=log, settle_after_seat=aperture_provider is not None,
            allow_post_success_retreat=demo, failure_detail=sanitized_failure)}
    except Exception as exc:
        report.update(status="recovery_stopped", error_type=type(exc).__name__)
        report["failure"] = sanitized_failure(exc)
        if isinstance(exc, ValueError):
            report["reason"] = str(exc)
    return report


def run(package, output, robotwin_root, *, model, allow_unverified=False, preflight_result=None):
    validate_package(package, allow_unverified=allow_unverified)
    if not model.startswith("claude-") or "MODEL" in model:
        raise ValueError("real_claude_model_id_required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    simulator = video = environment = store = None
    report = dict(status="initialization_error", model=model, recovery=None,
                  maximum_keypoint_requests=6, full_state_restored=False,
                  public_support="unknown", capture_provenance="unknown")
    try:
        report["preflight"] = (preflight_result if preflight_result is not None
                               else preflight(package, output / "preflight", model))
        if report["preflight"].get("status") != "passed":
            report["status"] = "preflight_failed"
            return report
        if (report["preflight"].get("model") != model
                or report["preflight"].get("sequence") != 660):
            raise ValueError("preflight_identity_mismatch")
        demo = report["preflight"].get("reviewed_aperture_demo") is True
        if demo:
            client = None
        else:
            from anthropic import Anthropic
            client = Anthropic(max_retries=0, timeout=60.)
        # Simulator imports and initialization only follow successful preflight.
        from development.simulator.local_simulator import LocalSimulator
        from development.demos.episode_video import EpisodeVideo, RecordingEnvironment
        from harness.environment.layout import CAMERAS
        from harness.environment.perception_cameras import DEFAULT_PERCEPTION_CAMERA
        from harness.perception.observations import ObservationStore

        simulator = LocalSimulator(robotwin_root, rgbd_enabled=True,
                                  perception_cameras=(DEFAULT_PERCEPTION_CAMERA,))
        video = EpisodeVideo(output, cameras=(*CAMERAS, "perception_camera"))
        environment = RecordingEnvironment(simulator, video, render_once_per_step=True)
        environment.policy_kind = "recorded_frozen_policy_prefix"
        store = ObservationStore(output / "observations")
        replay = validate_entry(package, environment, allow_unverified=allow_unverified)
        report["replay"] = replay
        if replay["status"] != "observations_agree_unverified":
            report["status"] = "replay_rejected"
        else:
            store.save(environment.observe())
            environment.policy_kind = "grounded_hanging_recovery"
            provider = None
            if demo:
                current = environment.observe()
                camera = current.camera_frames["perception_camera"]
                if (calibration_sha256(camera) != report["preflight"]["assessment"]["calibration_sha256"]
                        or sha256(camera.depth_m.tobytes()).hexdigest() != report["preflight"]["depth_sha256"]):
                    raise ValueError("reviewed_entry_depth_or_calibration_mismatch")
                provider = DemoApertureProvider(current, environment.arm_state("right")["pose"],
                    report["preflight"]["annotation"], report["preflight"]["assessment"], environment,
                    log=lambda event: append_jsonl(output/"recovery_events.jsonl", event))
            report["recovery"] = recover(environment, client, model, output, store,
                                          deadline=time.monotonic() + 600,
                                          initial_reports=report["preflight"], aperture_provider=provider,
                                          mating_prior=report["preflight"].get("mating_prior"))
            report["status"] = report["recovery"]["status"]
    except Exception as exc:
        report.update(status="run_error", error_type=type(exc).__name__)
        report["failure"] = sanitized_failure(exc)
    finally:
        if environment is not None and store is not None:
            try:
                report["last_confirmed_observation"] = store.save(environment.observe())
            except Exception as exc:
                report["final_observation_error_type"] = type(exc).__name__
        if video is not None:
            try:
                video.close()
                report["video"] = video.metadata()
            except Exception as exc:
                report["video_error_type"] = type(exc).__name__
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                report["cleanup_error_type"] = type(exc).__name__
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
