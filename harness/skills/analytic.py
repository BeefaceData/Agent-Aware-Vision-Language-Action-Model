"""Shared measured-start execution, independent of task object geometry."""

import numpy as np

from harness.perception.geometry import pose7, pose_error
from harness.skill_executor import SkillStopped, TickExecutor, validate_trajectory


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
                if (np.max(np.abs(measured-last_position)) > 0.25
                        or np.max(np.abs(inactive-other_hold["joints"])) > 0.25):
                    return "tracking_error"
                if opening and context.physics_ticks % 25 == 0:
                    context.observation = env.observe()
                    if context.check_predicate("release_ready", arm) is not True:
                        return "release_support_lost"
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
            context.log({"event_type": "analytic_start", "arm": arm,
                         "measured_start_joints": initial["joints"].tolist(),
                         "measured_start_pose": initial["pose"].tolist(),
                         "target_pose": None if target is None else list(map(float, target)),
                         "gripper_target": gripper_target,
                         "information_source": "development_oracle" if context.oracle_used else "public"})
            if target is not None:
                target = pose7(target)
                if (not (-0.65 <= target[0] <= 0.65 and -0.65 <= target[1] <= 0.65
                         and 0.70 <= target[2] <= 1.40)
                        or np.linalg.norm(target[:3]-initial["pose"][:3]) > 0.30):
                    raise SkillStopped("pose_limits")
                plan = env.plan(arm, target)
                positions, velocities = validate_trajectory(plan, env.joint_bounds(arm), env.arm_state(arm)["joints"])
                values = np.full(len(positions), initial["gripper"])
            else:
                opening = gripper_target > initial["gripper"] + 1e-9
                if opening and context.check_predicate("release_ready", arm) is not True:
                    raise SkillStopped("release_not_authorized")
                n = command.parameters.ramp_ticks
                positions = np.tile(initial["joints"], (n, 1))
                velocities = np.zeros_like(positions)
                values = np.linspace(initial["gripper"], gripper_target, n+1)[1:]
            # Include bounded settling in the budget before any drive writes.
            executor.require_capacity(len(positions)+25)
            for position, velocity, value in zip(positions, velocities, values):
                executor.tick(command.action.value, position, velocity, value)
            for _ in range(25):
                executor.tick("settle", positions[-1], np.zeros(6), float(values[-1]))
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
