"""Shared bounded motion helpers used by current skills and legacy recovery.

One execution step here means one physics tick, not a policy target-action call.
Adapters supply the actuator callback and independent interruption predicate.
"""

import numpy as np


class SkillStopped(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class TickExecutor:
    def __init__(self, max_ticks, *, advance, guard, log, arm="left"):
        if type(max_ticks) is not int or max_ticks < 1:
            raise ValueError("positive integer physics-tick budget required")
        self.max_ticks = max_ticks
        if arm not in ("left", "right"):
            raise ValueError("executor arm must be left or right")
        self.arm = arm
        self.executed_ticks = 0
        self.advance, self.guard, self.log = advance, guard, log

    def check(self):
        reason = self.guard()
        if reason:
            raise SkillStopped(reason)
        if self.executed_ticks >= self.max_ticks:
            raise SkillStopped("step_budget")

    def require_capacity(self, ticks):
        self.check()
        if type(ticks) is not int or ticks < 1 or ticks > self.max_ticks - self.executed_ticks:
            raise SkillStopped("step_budget")

    def tick(self, stage, position, velocity, gripper):
        self.check()  # Before setting any target or advancing physics.
        position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        if (position.shape != (6,) or velocity.shape != (6,)
                or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not np.isfinite(gripper) or not 0 <= gripper <= 1):
            raise SkillStopped("invalid_tick_command")
        tick = self.executed_ticks + 1
        self.log({"event": "tick_requested", "tick": tick, "stage": stage,
                  f"{self.arm}_position": list(map(float, position)),
                  f"{self.arm}_velocity": list(map(float, velocity)), f"{self.arm}_gripper": float(gripper)})
        self.advance(position, velocity, gripper)
        self.executed_ticks = tick
        self.log({"event": "tick_confirmed", "tick": tick, "stage": stage})


def validate_trajectory(plan, bounds, start, *, max_joint_delta=0.15, max_velocity=5.0):
    if not isinstance(plan, dict) or plan.get("status") != "Success":
        raise SkillStopped("planning_failure")
    positions = np.asarray(plan.get("position"), dtype=float)
    velocities = np.asarray(plan.get("velocity"), dtype=float)
    bounds, start = np.asarray(bounds, dtype=float), np.asarray(start, dtype=float)
    if (positions.ndim != 2 or positions.shape[1:] != (6,) or len(positions) == 0
            or velocities.shape != positions.shape or bounds.shape != (6, 2) or start.shape != (6,)
            or not all(np.isfinite(a).all() for a in (positions, velocities, bounds, start))
            or np.any(bounds[:, 0] > bounds[:, 1])):
        raise SkillStopped("invalid_trajectory")
    if (np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1])
            or np.max(np.abs(np.diff(np.vstack([start, positions]), axis=0))) > max_joint_delta
            or np.max(np.abs(velocities)) > max_velocity):
        raise SkillStopped("trajectory_limits")
    return positions.copy(), velocities.copy()
