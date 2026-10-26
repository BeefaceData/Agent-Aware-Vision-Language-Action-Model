"""Shared measured-start execution, independent of task object geometry."""

import numpy as np

from harness.perception.geometry import bounded_pose_waypoints, pose7, pose_error
from harness.skills.motion import SkillStopped, TickExecutor, validate_trajectory
from .schemas import TrackingError


# Give a planned Cartesian target up to 0.5 s of measured endpoint settling.
# Simulator physics runs at 250 Hz. A target is accepted only after several
# consecutive measurements satisfy the existing pose tolerances.
SETTLE_TICK_LIMIT = 125
SETTLE_STABLE_SAMPLES = 3
TRACKING_ERROR_LIMIT_RAD = 0.25


class AnalyticSkill:
    def execute(self, context, command):
        env, arm = context.environment, command.arm
        if not getattr(env, "supports_analytic", False):
            return "capability_unavailable"
        if context.tick_limit <= 0:
            return "episode_budget_exhausted"
        if context.cancelled():
            return "cancelled"
        if env.native_success:
            return "task_success"
        initial = env.arm_state(arm)
        release_predicate = (
            "rack_seat_aligned"
            if command.action.value == "release"
            and command.parameters.authorization == "rack_seat_alignment"
            else "release_ready")
        other = "right" if arm == "left" else "left"
        other_hold = env.arm_state(other)
        last_position = initial["joints"].copy()
        opening = False

        def guard():
            if context.cancelled(): return "cancelled"
            # Once an opening ramp is authorized, finish opening rather than
            # stop at the native benchmark's >0.8 command threshold.
            if not opening and env.native_success: return "task_success"
            if context.motion_started:
                measured = env.arm_state(arm)["joints"]
                inactive = env.arm_state(other)["joints"]
                for compared_arm, measured_joints, commanded_joints, comparison in (
                    (arm, measured, last_position, "controlled_arm"),
                    (other, inactive, other_hold["joints"], "inactive_arm"),
                ):
                    residuals = np.abs(measured_joints-commanded_joints)
                    joint_index = int(np.argmax(residuals))
                    residual = float(residuals[joint_index])
                    if residual > TRACKING_ERROR_LIMIT_RAD:
                        diagnostic = TrackingError(
                            arm=compared_arm, comparison=comparison,
                            joint_index=joint_index,
                            measured_position_rad=float(measured_joints[joint_index]),
                            commanded_position_rad=float(commanded_joints[joint_index]),
                            residual_rad=residual,
                            threshold_rad=TRACKING_ERROR_LIMIT_RAD,
                        )
                        context.tracking_error = diagnostic
                        context.log({"event_type": "tracking_error", "diagnostic": {
                            **diagnostic.model_dump(mode="json"),
                            "measured_joints_rad": measured_joints.tolist(),
                            "commanded_joints_rad": commanded_joints.tolist(),
                            "residuals_rad": residuals.tolist(),
                        }})
                        return "tracking_error"
                if opening and context.physics_ticks % 25 == 0:
                    context.observation = env.observe()
                    if context.check_predicate(release_predicate, arm) is not True:
                        return ("release_support_lost" if release_predicate == "release_ready"
                                else "rack_seat_alignment_lost")
            return None

        def advance(position, velocity, gripper):
            nonlocal last_position
            context.motion_started = context.ambiguous = True
            env.tick(arm, position, velocity, gripper, other_hold)
            context.physics_ticks += 1
            context.ambiguous = False
            last_position = np.asarray(position).copy()
            context.log({"event_type": "measured_tick", "physics_tick": context.physics_ticks,
                         "physics_seconds": context.physics_ticks / 250.0,
                         "arm": arm, "measured_joints": env.arm_state(arm)["joints"].tolist(),
                         "measured_ee_pose": env.arm_state(arm)["pose"].tolist()})

        executor = TickExecutor(min(command.parameters.max_physics_ticks, context.tick_limit),
                                advance=advance, guard=guard, log=context.log, arm=arm)
        target = None
        try:
            executor.check()
            target, gripper_target = self.prepare(context, command, initial)
            waypoints = ()
            if target is not None:
                target = pose7(target)
                if (not (-0.65 <= target[0] <= 0.65 and -0.65 <= target[1] <= 0.65
                         and 0.70 <= target[2] <= 1.40)):
                    raise SkillStopped("pose_limits")
                try:
                    waypoints = bounded_pose_waypoints(initial["pose"], target)
                except ValueError as exc:
                    if str(exc) == "pose_waypoint_limits":
                        raise SkillStopped("pose_limits") from None
                    raise
            context.log({"event_type": "analytic_start", "arm": arm,
                         "measured_start_joints": initial["joints"].tolist(),
                         "measured_start_pose": initial["pose"].tolist(),
                         "target_pose": None if target is None else list(map(float, target)),
                         "waypoints": [waypoint.tolist() for waypoint in waypoints],
                         "gripper_target": gripper_target,
                         "information_source": "development_oracle" if context.oracle_used else "public"})
            if target is not None:
                final_gripper = initial["gripper"]
                for index, waypoint in enumerate(waypoints, start=1):
                    measured = env.arm_state(arm)
                    segment_distance, segment_rotation = pose_error(measured["pose"], waypoint)
                    if segment_distance > 0.30 or segment_rotation > np.deg2rad(60):
                        raise SkillStopped("pose_limits")
                    context.log({"event_type": "analytic_waypoint_plan", "arm": arm,
                        "waypoint_index": index, "waypoint_count": len(waypoints),
                        "measured_start_joints": measured["joints"].tolist(),
                        "measured_start_pose": measured["pose"].tolist(),
                        "segment_translation_m": segment_distance,
                        "segment_rotation_rad": segment_rotation,
                        "target_pose": waypoint.tolist()})
                    try:
                        plan = env.plan(arm, waypoint)
                    except Exception as exc:
                        context.log({"event_type": "analytic_plan_result", "arm": arm,
                            "waypoint_index": index, "planner_status": "exception",
                            "planner_exception_type": type(exc).__name__[:80],
                            "trajectory_samples": 0, "result": "rejected",
                            "rejection_reason": "planner_exception"})
                        raise
                    status = plan.get("status") if isinstance(plan, dict) else None
                    planner_status = ("success" if status == "Success" else
                                      "failure" if status == "Fail" else "unknown")
                    raw_positions = plan.get("position") if isinstance(plan, dict) else None
                    try:
                        trajectory_samples = min(len(raw_positions), 100000)
                    except (TypeError, ValueError):
                        trajectory_samples = 0
                    if planner_status != "success":
                        context.log({"event_type": "analytic_plan_result", "arm": arm,
                            "waypoint_index": index, "planner_status": planner_status,
                            "trajectory_samples": trajectory_samples, "result": "rejected",
                            "rejection_reason": "planning_failure" if planner_status == "failure"
                                                else "invalid_planner_result"})
                    try:
                        positions, velocities = validate_trajectory(
                            plan, env.joint_bounds(arm), measured["joints"])
                    except SkillStopped as exc:
                        if planner_status == "success":
                            context.log({"event_type": "analytic_plan_result", "arm": arm,
                                "waypoint_index": index, "planner_status": planner_status,
                                "trajectory_samples": trajectory_samples, "result": "rejected",
                                "rejection_reason": exc.reason})
                        raise
                    context.log({"event_type": "analytic_plan_result", "arm": arm,
                        "waypoint_index": index, "planner_status": planner_status,
                        "trajectory_samples": trajectory_samples, "result": "accepted",
                        "rejection_reason": None})
                    executor.require_capacity(
                        len(positions) + (SETTLE_TICK_LIMIT if index == len(waypoints) else 0))
                    for position, velocity in zip(positions, velocities):
                        executor.tick(command.action.value, position, velocity, initial["gripper"])
                final_position = positions[-1]
            else:
                opening = gripper_target > initial["gripper"] + 1e-9
                if opening and context.check_predicate(release_predicate, arm) is not True:
                    raise SkillStopped("release_not_authorized")
                n = command.parameters.ramp_ticks
                positions = np.tile(initial["joints"], (n, 1))
                velocities = np.zeros_like(positions)
                values = np.linspace(initial["gripper"], gripper_target, n+1)[1:]
            # Include bounded settling in the budget before any drive writes.
                executor.require_capacity(len(positions)+25)
                for position, velocity, value in zip(positions, velocities, values):
                    executor.tick(command.action.value, position, velocity, value)
                final_position, final_gripper = positions[-1], float(values[-1])
            settle_limit = SETTLE_TICK_LIMIT if target is not None else 25
            stable_samples = 0
            for settle_ticks in range(1, settle_limit + 1):
                executor.tick("settle", final_position, np.zeros(6), float(final_gripper))
                if target is not None:
                    measured_pose = env.arm_state(arm)["pose"]
                    position_error, orientation_error = pose_error(measured_pose, target)
                    if position_error <= 0.015 and orientation_error <= 0.10:
                        stable_samples += 1
                        if stable_samples >= SETTLE_STABLE_SAMPLES:
                            break
                    else:
                        stable_samples = 0
            context.log({"event_type": "analytic_settling", "arm": arm,
                "ticks_used": settle_ticks, "tick_limit": settle_limit,
                "stable_samples": stable_samples,
                "position_error_m": (None if target is None else float(position_error)),
                "orientation_error_rad": (None if target is None else float(orientation_error)),
                "converged": target is not None and stable_samples >= SETTLE_STABLE_SAMPLES})
            reason = guard()
            if reason: raise SkillStopped(reason)
            final = env.arm_state(arm)
            if target is not None:
                context.position_error, context.orientation_error = pose_error(final["pose"], target)
                if context.position_error > 0.015 or context.orientation_error > 0.10:
                    return "target_not_reached"
            elif abs(final["gripper"]-gripper_target) > 0.01:
                return "gripper_target_not_reached"
            return "completed"
        except SkillStopped as exc:
            return exc.reason
        finally:
            if context.motion_started:
                # Do not overwrite a known uncertainty with a guessed state.
                uncertain = context.ambiguous
                context.ambiguous = True
                env.hold_measured()
                if not uncertain:
                    context.observation = env.observe()
                    context.ambiguous = False

    def prepare(self, context, command, initial):
        raise NotImplementedError
