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
from harness.predicates.registry import PredicateRegistry
from harness.skills.executor import SkillExecutor
from harness.skills.schemas import EpisodeBudget, SkillCommand
from harness.perception.pose_grounding import _pose_matrix, _matrix_pose
from harness.perception.geometry import bounded_pose_waypoints, pose_error
from harness.perception.rgbd import calibration_sha256

from development.hanging.run_entry660 import validate_package
from development.hanging.validate_entry660 import validate_entry
from development.hanging.grounding_preflight import (
    preflight, grounding_client, sanitized_failure, validate_with_crop_retry,
)


class NoPolicy:
    def infer(self, *args, **kwargs):
        raise RuntimeError("recovery_uses_recorded_policy_prefix_only")

    def reset_history(self):
        raise RuntimeError("recovery_uses_recorded_policy_prefix_only")


def aperture_targets(observation, assessment, rack_pose, measured_tcp, prior):
    """Compose targets from a fresh candidate and explicit dimensional bounds.

    No dimensional defaults: the caller must supply independently justified
    priors. This function does not run IK or authorize release.
    """
    if assessment.get("status") != "candidate_not_motion_validated":
        raise ValueError("aperture_geometry_rejected")
    camera = (observation.camera_frames or {}).get(assessment.get("camera"))
    image = observation.images.get(assessment.get("camera"))
    if (camera is None or assessment.get("sequence") != observation.sequence
            or assessment.get("calibration_sha256") != calibration_sha256(camera)
            or image is None or assessment.get("rgb_sha256") != sha256(image.tobytes()).hexdigest()):
        raise ValueError("aperture_evidence_stale_or_wrong_camera")
    required = ("handle_thickness_m", "thickness_uncertainty_m", "radial_clearance_m",
                "axial_clearance_m", "position_uncertainty_m", "orientation_uncertainty_rad",
                "engaged_length_m", "margin_m", "roll_min_rad", "roll_max_rad")
    if (not isinstance(prior, dict) or not prior.get("basis")
            or any(key not in prior for key in required)):
        raise ValueError("explicit_dimensional_and_roll_bounds_required")
    values = np.array([prior[key] for key in required], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:8] < 0):
        raise ValueError("invalid_mating_bounds")
    if (prior["handle_thickness_m"] <= prior["thickness_uncertainty_m"]
            or not -np.pi <= prior["roll_min_rad"] <= prior["roll_max_rad"] <= np.pi):
        raise ValueError("invalid_mating_bounds")
    uncertainty = assessment["uncertainty"]
    demo = prior.get("allow_entry660_demo_override") is True
    position_bound = .015 + prior["position_uncertainty_m"] + uncertainty["center_sensitivity_m"]
    angle_bound = .10 + prior["orientation_uncertainty_rad"] + np.deg2rad(
        uncertainty["maximum_normal_sensitivity_deg"])
    if angle_bound >= np.pi/2:
        raise ValueError("mating_orientation_uncertainty_too_large")
    required_radial = position_bound + prior["engaged_length_m"]*np.sin(angle_bound) + prior["margin_m"]
    required_axial = (position_bound + (prior["handle_thickness_m"]
                      + prior["thickness_uncertainty_m"])/2 + prior["margin_m"])
    if demo:
        # Explicitly experimental RSS sensitivity budget, NOT an independent
        # error probability or bound on uncalibrated systematic error. Runtime
        # release uses a tighter measured gate below, never the 15 mm allowance.
        required_radial = float(np.sqrt(uncertainty["center_sensitivity_m"]**2
            + uncertainty["maximum_plane_residual_m"]**2
            + (prior["engaged_length_m"]*np.sin(np.deg2rad(
                uncertainty["maximum_normal_sensitivity_deg"])))**2)) + prior["margin_m"]
        required_axial = float(np.hypot(uncertainty["center_sensitivity_m"],
                                      prior["thickness_uncertainty_m"]/2)) + prior["margin_m"]
    if (prior["radial_clearance_m"] <= required_radial
            or prior["axial_clearance_m"] <= required_axial):
        raise ValueError("mating_clearance_insufficient")
    aperture = _pose_matrix(assessment["frame_pose_world_wxyz"])
    # The fitted normal faces the camera: the physical middle is assumed
    # half the explicitly supplied thickness behind the visible surface.
    aperture[:3, 3] -= aperture[:3, 0]*prior["handle_thickness_m"]/2
    tcp = _pose_matrix(measured_tcp)
    grasp = np.linalg.inv(tcp) @ aperture
    rack = _pose_matrix(rack_pose)
    candidates = []
    # Condition roll only inside the supplied clearance-supported interval.
    # This minimizes wrist rotation, not an assertion that IK will succeed.
    body = np.asarray(assessment.get("visible_body_points_world", []), dtype=float)
    material = np.asarray(assessment.get("material_points_world", []), dtype=float)
    for roll in np.linspace(prior["roll_min_rad"], prior["roll_max_rad"], 65):
        c, s = np.cos(roll), np.sin(roll)
        mating = rack.copy()
        mating[:3, :3] = rack[:3, :3] @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        seat = _matrix_pose(mating @ np.linalg.inv(grasp))
        approach = mating.copy()
        approach[:3, 3] += rack[:3, 0]*.05
        approach = _matrix_pose(approach @ np.linalg.inv(grasp))
        visibility = None
        if body.size and material.size:
            visibility = roll_visibility(camera, tcp, _pose_matrix(seat), body, material,
                                         mating[:3, 3], rack)
            if not visibility["accepted"]:
                continue
        for target in (approach, seat):
            if (np.linalg.norm(target[:3]-tcp[:3, 3]) > .65
                    or not (-.65 <= target[0] <= .65 and -.65 <= target[1] <= .65
                            and .70 <= target[2] <= 1.40)):
                break
        else:
            try:
                bounded_pose_waypoints(measured_tcp, approach)
                bounded_pose_waypoints(approach, seat)
            except ValueError:
                continue
            candidates.append((pose_error(measured_tcp, seat)[1], float(roll), approach, seat, visibility))
    if not candidates:
        raise ValueError("aperture_targets_out_of_bounds")
    _, roll, approach, seat, visibility = min(candidates, key=lambda item: item[0])
    return {"rack_approach": approach, "rack_seat": seat}, dict(
        selected_roll_rad=roll, surface_offset_m=-prior["handle_thickness_m"]/2,
        required_radial_clearance_m=float(required_radial),
        required_axial_clearance_m=float(required_axial),
        ik_planning="not_run_requires_simulator_planner", motion_authorized=False,
        roll_visibility=visibility)


def roll_visibility(camera, initial_tcp, target_tcp, body, material, aperture_center, rack):
    """Visible-surface roll heuristic; not a complete object collision model."""
    delta = target_tcp @ np.linalg.inv(initial_tcp)
    moved_body = body @ delta[:3, :3].T + delta[:3, 3]
    moved_material = material @ delta[:3, :3].T + delta[:3, 3]
    height = float(aperture_center[2]-moved_body[:, 2].mean())
    def project(world):
        xyz = world @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
        uv = xyz @ camera.intrinsic.T
        return uv[:, :2]/uv[:, 2, None], xyz[:, 2]
    body_uv, body_depth = project(moved_body)
    material_uv, material_depth = project(moved_material)
    visible = sum(not np.any((np.linalg.norm(body_uv-pixel, axis=1)<1.5)
                  & (body_depth<depth-.005)) for pixel, depth in zip(material_uv, material_depth))
    # Check the visible body against a bounded peg segment. Hidden surfaces
    # and rack-post collision still require runtime planning/observation.
    along = np.clip((moved_body-rack[:3, 3]) @ rack[:3, 0], 0, .08)
    distance = float(np.min(np.linalg.norm(moved_body-(rack[:3, 3]
                                      +along[:, None]*rack[:3, 0]), axis=1)))
    return dict(accepted=bool(height>=.01 and visible>=6 and distance>=.009),
        handle_above_visible_body_m=height, predicted_visible_samples=int(visible),
        visible_body_to_peg_segment_m=distance, complete_collision_model=False)


def confirm_visible_handle(camera, pixels, expected):
    """Occlusion is unavailable evidence; it is not a 3D grasp error."""
    samples, inliers = [], []
    for index, ((u, v), point) in enumerate(zip(pixels, expected)):
        row = dict(sample=index, predicted_uv=[float(u), float(v)],
                   predicted_world=point.tolist(), threshold_m=.005)
        predicted_camera = camera.world_to_camera[:3, :3] @ point + camera.world_to_camera[:3, 3]
        row["predicted_depth_m"] = float(predicted_camera[2])
        try:
            depth = camera.sample_depth(u, v, radius=1)
            actual = np.asarray(camera.unproject(u, v, radius=1))
            residual = float(np.linalg.norm(actual-point))
            x, y = np.rint((u,v)).astype(int)
            patch = camera.depth_m[max(0,y-1):y+2, max(0,x-1):x+2]
            row.update(measured_depth_m=depth, actual_world=actual.tolist(),
                       residual_m=residual, depth_difference_m=float(depth-predicted_camera[2]))
            if depth < predicted_camera[2]-.005:
                row["classification"] = "nearer_occluder"
            elif patch.shape != (3,3) or np.any(patch<=0) or np.ptp(patch)>.06:
                row["classification"] = "depth_patch_unavailable"
            elif residual <= .005:
                row["classification"] = "visible_match"
                inliers.append(index)
            else:
                row["classification"] = "contradictory_depth"
        except ValueError as exc:
            row.update(classification="depth_unavailable", reason=str(exc))
        samples.append(row)
    status, reason = "rejected", "insufficient_visible_handle_support"
    condition = None
    trimmed = []
    visible = [row for row in samples if row["classification"] in
               ("visible_match", "contradictory_depth")]
    contrary = [row for row in visible if row["classification"]=="contradictory_depth"]
    robust_cutoff = None
    if visible:
        residuals = np.array([row["residual_m"] for row in visible])
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals-median)))
        # Preserve 5 mm for every accepted match. Trim at most one marginal
        # residual in >=10 usable samples, capped at 10 mm so a background
        # surface or gross grasp error is not dismissed as raster noise.
        robust_cutoff = min(.010, max(.0055, median+3*1.4826*mad))
        if (len(visible)>=10 and len(contrary)==1 and len(inliers)>=9
                and median<=.005 and contrary[0]["residual_m"]<=robust_cutoff):
            contrary[0]["classification"]="trimmed_marginal_outlier"
            trimmed=[contrary[0]["sample"]]
    if any(row["classification"]=="contradictory_depth" for row in samples):
        reason = "visible_handle_depth_contradicts_prediction"
    elif len(inliers)>=6 and len(inliers)*2>=len(samples):
        points = expected[inliers]
        singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
        condition = float(singular[0]/max(singular[1],1e-12))
        if condition<=10 and singular[1]/np.sqrt(len(points))>=.004:
            status, reason = "confirmed_visible_subset", "measured_tcp_prediction_matches_visible_depth"
    return dict(status=status, reason=reason, samples=samples, visible_indices=inliers,
                plane_condition_number=condition, threshold_m=.005, grasp_slip_proven=False,
                trimmed_indices=trimmed, robust_outlier_cutoff_m=robust_cutoff)


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


def settle_seat(environment, target_pose, commanded_joints, *, deadline, max_ticks=250,
                log=lambda event: None):
    """Hold the existing planned joint endpoint; measure convergence every tick."""
    arm, other = environment.arm_state("right"), environment.arm_state("left")
    joints = np.asarray(commanded_joints, dtype=float)
    if joints.shape!=(6,) or not np.isfinite(joints).all():
        raise ValueError("seat_endpoint_command_missing")
    result = dict(completed=False, reason="seat_settling_not_converged", ticks=0,
                  execution_uncertain=False, position_tolerance_m=.005,
                  orientation_tolerance_rad=.05)
    stable = 0
    for _ in range(max_ticks):
        if time.monotonic()>=deadline:
            result["reason"]="cancelled"
            break
        controlled = environment.arm_state("right")
        inactive = environment.arm_state("left")
        if (np.max(np.abs(controlled["joints"]-joints))>.25
                or np.max(np.abs(inactive["joints"]-other["joints"]))>.25):
            result["reason"]="tracking_error"
            break
        try:
            environment.tick("right", joints, np.zeros(6), arm["gripper"], other)
        except Exception as exc:
            result.update(reason="execution_error",execution_uncertain=True,error_type=type(exc).__name__)
            break
        result["ticks"]+=1
        pose=environment.arm_state("right")["pose"]
        position, rotation=pose_error(pose,target_pose)
        result.update(position_error_m=float(position),orientation_error_rad=float(rotation))
        stable=stable+1 if position<=.005 and rotation<=.05 else 0
        log(dict(event_type="seat_settling_tick",tick=result["ticks"],
                 measured_ee_pose=np.asarray(pose).tolist(),position_error_m=float(position),
                 orientation_error_rad=float(rotation)))
        if result["ticks"]>=25 and stable>=5:
            result.update(completed=True,reason="completed")
            break
    log(dict(event_type="seat_settling_result",**result))
    return result


class PostReleaseRetreatEnvironment:
    """Permit only the demo's bounded retreat after benchmark success latches.

    The underlying success flag is never changed and remains in the report.
    Tracking, workspace, planning, cancellation and tick guards still execute.
    """
    def __init__(self, environment):
        self.environment=environment

    def __getattr__(self,name):
        return getattr(self.environment,name)

    @property
    def native_success(self):
        return False


def recover(environment, client, model, output, store, *, deadline, initial_reports=None,
            aperture_provider=None, mating_prior=None):
    """At most approach, seat, guarded release, retreat; first failure ends attempt."""
    output = Path(output)
    last_right_target = None
    def log(event):
        nonlocal last_right_target
        if event.get("event")=="tick_requested" and "right_position" in event:
            last_right_target=np.asarray(event["right_position"],dtype=float)
        append_jsonl(output / "recovery_events.jsonl", event)
    anchors = AnchorRegistry()
    cache = StaticRackFrameCache()
    demo = (mating_prior or {}).get("allow_entry660_demo_override") is True
    gate = RackSeatReleaseGate(environment, position_tolerance_m=.005 if demo else .015,
                              orientation_tolerance_rad=.05 if demo else .10,
                              unavailable_is_false=aperture_provider is not None)
    predicates = PredicateRegistry(rack_seat_gate=gate)
    report = dict(status="grounding_failed", stages={}, public_support="unknown",
                  full_state_restored=False, capture_provenance="unknown")
    executor = None
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
        executor = SkillExecutor(environment, NoPolicy(), log=log, anchors=anchors,
            predicates=predicates, cancelled=lambda: time.monotonic() >= deadline,
            budget=EpisodeBudget(max_skill_calls=5, max_analytic_ticks=12000))
        commands = (
            ("approach", SkillCommand(action="move_to", arm="right",
                parameters={"target": "rack_approach", "max_physics_ticks": 3000})),
            ("seat", SkillCommand(action="move_to", arm="right",
                parameters={"target": "rack_seat", "max_physics_ticks": 3000})),
            ("release", SkillCommand(action="release", arm="right",
                parameters={"authorization": "rack_seat_alignment", "ramp_ticks": 200})),
            ("retreat", SkillCommand(action="move_to", arm="right",
                parameters={"target": "rack_retreat", "max_physics_ticks": 2000})),
        )
        retreat_delta = None
        for stage, command in commands:
            if time.monotonic() >= deadline:
                report.update(status="recovery_time_limit", stopped_at=stage)
                break
            observation = environment.observe()
            if stage == "retreat":
                measured = np.asarray(environment.arm_state("right")["pose"]).copy()
                measured[:3] += retreat_delta
                anchors.bind("rack_retreat", "right", Anchor(tuple(measured), observation.sequence))
                target = anchors.resolve("rack_retreat", "right", observation)
            else:
                target = prepare(command, observation)
            if stage == "seat":
                approach = anchors.resolve("rack_approach", "right", observation)
                retreat_delta = np.asarray(approach.pose[:3]) - target.pose[:3]
            result = executor.execute(command)
            if stage=="seat" and aperture_provider is not None and result.completed and not result.execution_uncertain:
                settling=settle_seat(environment,target.pose,last_right_target,deadline=deadline,
                    max_ticks=min(250,12000-executor.executed_physics_ticks),log=log)
                executor.executed_physics_ticks+=settling["ticks"]
                report["seat_settling"]=settling
                after=None if settling["execution_uncertain"] else environment.observe()
                result=result.model_copy(update=dict(completed=settling["completed"],
                    stop_reason=settling["reason"],execution_uncertain=settling["execution_uncertain"],
                    executed_physics_ticks=result.executed_physics_ticks+settling["ticks"],
                    position_error_m=settling.get("position_error_m"),
                    orientation_error_rad=settling.get("orientation_error_rad"),
                    observation_after=None if after is None else after.sequence,
                    state_after=None if after is None else result.state_after.model_copy(
                        update={"command_state":after.command_state})))
            gate.record(command, result, target)
            report["stages"][stage] = result.model_dump(mode="json")
            if not result.execution_uncertain:
                store.save(environment.observe())
            if not result.completed or result.execution_uncertain:
                report.update(status="recovery_stopped", stopped_at=stage, reason=result.stop_reason)
                break
            if result.episode_terminal:
                if (demo and stage=="release" and result.stop_reason=="completed"
                        and environment.native_success and environment.arm_state("right")["gripper"]>=.99
                        and executor.calls<executor.budget.max_skill_calls):
                    report["native_success_at_release"]=True
                    report["bounded_post_success_retreat"]=True
                    executor.environment=PostReleaseRetreatEnvironment(environment)
                    executor.terminal=False
                    continue
                report.update(status="native_terminal", stopped_at=stage,
                              reason="remaining_stages_not_executed")
                break
        else:
            report["status"] = "recovery_sequence_completed_support_unverified"
            # Record one second after retreat without changing either arm's pose.
            right, left = environment.arm_state("right"), environment.arm_state("left")
            observed_ticks = 0
            for index in range(250):
                if time.monotonic() >= deadline:
                    break
                environment.tick("right", right["joints"], np.zeros(6), right["gripper"], left)
                observed_ticks += 1
                if (index + 1) % 25 == 0:
                    store.save(environment.observe())
            report["post_retreat_observation_only_ticks"] = observed_ticks
    except Exception as exc:
        report.update(status="recovery_stopped", error_type=type(exc).__name__)
        report["failure"] = sanitized_failure(exc)
        if isinstance(exc, ValueError):
            report["reason"] = str(exc)
    finally:
        if executor is not None:
            report["analytic_ticks"] = executor.executed_physics_ticks
        try:
            report["native_success"] = environment.native_success
        except Exception:
            report["native_success"] = None
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
