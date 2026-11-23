"""Bounded hanging geometry and execution, independent of recorded episodes."""

import time
from hashlib import sha256

import numpy as np

from harness.perception.anchors import Anchor
from harness.perception.geometry import bounded_pose_waypoints, pose_error
from harness.perception.pose_grounding import _pose_matrix, _matrix_pose
from harness.perception.rgbd import calibration_sha256
from harness.predicates.registry import PredicateRegistry
from harness.skills.executor import SkillExecutor
from harness.skills.schemas import EpisodeBudget, SkillCommand


class NoPolicy:
    def infer(self, *args, **kwargs):
        raise RuntimeError("hanging_recovery_does_not_infer_policy_actions")

    def reset_history(self):
        raise RuntimeError("hanging_recovery_does_not_infer_policy_actions")


def rss_clearance_budget(assessment, prior):
    """Local geometric RSS sensitivity, not a calibrated total-error bound.

    Preserve the recorded RSS terms while including any additional measured
    position/orientation uncertainty. Controller tracking is guarded separately;
    no 15-mm analytic endpoint allowance is treated as geometric uncertainty.
    """
    uncertainty = assessment["uncertainty"]
    try:
        center, plane, normal_deg = (float(uncertainty[key]) for key in (
            "center_sensitivity_m", "maximum_plane_residual_m", "maximum_normal_sensitivity_deg"))
        position, orientation, length, thickness_error, margin = (float(prior[key]) for key in (
            "position_uncertainty_m", "orientation_uncertainty_rad", "engaged_length_m",
            "thickness_uncertainty_m", "margin_m"))
    except (KeyError, TypeError, ValueError):
        raise ValueError("rss_uncertainty_terms_required") from None
    values = (center, plane, normal_deg, position, orientation, length, thickness_error, margin)
    if (not np.isfinite(values).all() or min(values) < 0
            or normal_deg >= 90 or orientation >= np.pi/2):
        raise ValueError("invalid_rss_uncertainty")
    radial = float(np.sqrt(center**2 + plane**2 + (length*np.sin(np.deg2rad(normal_deg)))**2
                           + position**2 + (length*np.sin(orientation))**2)) + margin
    axial = float(np.hypot(np.hypot(center, thickness_error/2), position)) + margin
    return radial, axial


def aperture_targets(observation, assessment, rack_pose, measured_tcp, prior, *, clearance_budget=None,
                     clearance_model="conservative_linear"):
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
    position_bound = .015 + prior["position_uncertainty_m"] + uncertainty["center_sensitivity_m"]
    angle_bound = .10 + prior["orientation_uncertainty_rad"] + np.deg2rad(
        uncertainty["maximum_normal_sensitivity_deg"])
    if angle_bound >= np.pi/2:
        raise ValueError("mating_orientation_uncertainty_too_large")
    required_radial = position_bound + prior["engaged_length_m"]*np.sin(angle_bound) + prior["margin_m"]
    required_axial = (position_bound + (prior["handle_thickness_m"]
                      + prior["thickness_uncertainty_m"])/2 + prior["margin_m"])
    if clearance_model == "rss_sensitivity":
        if clearance_budget is not None:
            raise ValueError("conflicting_clearance_models")
        required_radial, required_axial = rss_clearance_budget(assessment, prior)
    elif clearance_model != "conservative_linear":
        raise ValueError("unsupported_clearance_model")
    if clearance_budget is not None:
        # A caller-supplied experimental budget must remain explicit in its adapter.
        required_radial, required_axial = clearance_budget
        if not np.isfinite(clearance_budget).all() or min(clearance_budget) < 0:
            raise ValueError("invalid_clearance_budget")
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
                                         mating[:3, 3], rack, prior)
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
    evidence = dict(
        selected_roll_rad=roll, surface_offset_m=-prior["handle_thickness_m"]/2,
        required_radial_clearance_m=float(required_radial),
        required_axial_clearance_m=float(required_axial),
        ik_planning="not_run_requires_simulator_planner", motion_authorized=False,
        roll_visibility=visibility)
    if clearance_model == "rss_sensitivity":
        evidence.update(clearance_model=clearance_model,
                        uncertainty_interpretation="local_sensitivity_not_calibrated_total_error")
    return {"rack_approach": approach, "rack_seat": seat}, evidence


def roll_visibility(camera, initial_tcp, target_tcp, body, material, aperture_center, rack, prior):
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
    required = ("peg_length_m", "peg_radius_m", "body_clearance_m", "minimum_handle_height_m")
    if (any(key not in prior for key in required)
            or not np.isfinite([prior[key] for key in required]).all()
            or any(prior[key] <= 0 for key in required)):
        raise ValueError("explicit_visible_body_clearance_bounds_required")
    along = np.clip((moved_body-rack[:3, 3]) @ rack[:3, 0], 0, prior["peg_length_m"])
    distance = float(np.min(np.linalg.norm(moved_body-(rack[:3, 3]
                                      +along[:, None]*rack[:3, 0]), axis=1)))
    return dict(accepted=bool(height>=prior["minimum_handle_height_m"] and visible>=6
                              and distance>=prior["peg_radius_m"]+prior["body_clearance_m"]),
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


class BenchmarkBlindEnvironment:
    """Hide the simulator's native success checker from staged execution.

    Native success reads privileged simulator state, so it must never choose
    a branch in the public staged sequence: approach, seat, release and
    retreat run identically whether or not the checker has latched. The real
    flag is still read from the wrapped environment for the report only.
    Tracking, workspace, planning, cancellation and tick guards still execute.
    """
    def __init__(self, environment):
        self.environment=environment

    def __getattr__(self,name):
        return getattr(self.environment,name)

    @property
    def native_success(self):
        return False


def execute_stages(environment, prepare, anchors, gate, store, *, deadline,
                   log=lambda event: None, settle_after_seat=True, failure_detail=None):
    """Execute one bounded sequence from current state; stop at the first failure.

    The native success checker is reported but never consulted for control.
    """
    last_right_target = None
    def record(event):
        nonlocal last_right_target
        if event.get("event") == "tick_requested" and "right_position" in event:
            last_right_target = np.asarray(event["right_position"], dtype=float)
        log(event)
    report = dict(status="grounding_failed", stages={}, public_support="unknown")
    executor = None
    stage = None
    try:
        executor = SkillExecutor(BenchmarkBlindEnvironment(environment), NoPolicy(), log=record, anchors=anchors,
            predicates=PredicateRegistry(rack_seat_gate=gate), cancelled=lambda: time.monotonic() >= deadline,
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
            if stage=="seat" and settle_after_seat and result.completed and not result.execution_uncertain:
                settling=settle_seat(environment,target.pose,last_right_target,deadline=deadline,
                    max_ticks=min(250,12000-executor.executed_physics_ticks),log=record)
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
            if stage == "release":
                # Observed for the report only; it never changes the next stage.
                try:
                    report["native_success_at_release"] = bool(environment.native_success)
                except Exception:
                    report["native_success_at_release"] = None
            if result.episode_terminal and stage != commands[-1][0]:
                report.update(status="executor_terminal", stopped_at=stage,
                              reason="remaining_stages_not_executed")
                break
        else:
            report["status"] = "recovery_sequence_completed_support_unverified"
            # Record one second after retreat without changing either arm's pose.
            right, left = environment.arm_state("right"), environment.arm_state("left")
            observed_ticks = 0
            hold_limit = min(250, executor.budget.max_analytic_ticks-executor.executed_physics_ticks)
            for index in range(hold_limit):
                if time.monotonic() >= deadline:
                    break
                environment.tick("right", right["joints"], np.zeros(6), right["gripper"], left)
                executor.executed_physics_ticks += 1
                observed_ticks += 1
                if (index + 1) % 25 == 0:
                    store.save(environment.observe())
            report["post_retreat_observation_only_ticks"] = observed_ticks
    except Exception as exc:
        report.update(status="recovery_stopped", error_type=type(exc).__name__)
        if stage is not None:
            report["stopped_at"] = stage
        if failure_detail is not None:
            report["failure"] = failure_detail(exc)
        # Geometry/preparation errors are local typed codes, not provider messages.
        report["reason"] = str(exc) if isinstance(exc, ValueError) else "execution_error"
    finally:
        if executor is not None:
            report["analytic_ticks"] = executor.executed_physics_ticks
        try:
            report["native_success"] = environment.native_success
        except Exception:
            report["native_success"] = None
    return report
