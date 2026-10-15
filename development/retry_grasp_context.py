"""Privileged RoboTwin adapter for the single Phase 4 regrasp candidate.

It never calls task.move/take_dense_action: every physics tick goes through the
bounded executor. Geometry/diagnostics here must not enter supervisor context.
"""

import numpy as np

from harness.skill_executor import SkillStopped, TickExecutor, validate_trajectory
from harness.trace import vector14


class RetryGraspContext:
    development_only = True

    def __init__(self, task, *, log, cancel=lambda: False, max_ticks=3000):
        if type(max_ticks) is not int or not 1 <= max_ticks <= 3000:
            raise ValueError("max_ticks must be an integer in [1, 3000]")
        self.task, self.log, self.cancel, self.max_ticks = task, log, cancel, max_ticks
        self.executor = None
        self.began_motion = False

    def begin(self, skill_limit):
        if self.executor is not None:
            raise ValueError("context is single use")
        self.executor = TickExecutor(min(skill_limit, self.max_ticks), advance=self._advance,
                                     guard=self._guard, log=self.log)
        self.right_hold = np.asarray(self.task.robot.get_right_arm_jointState()[:6], dtype=float)
        self.right_gripper = float(self.task.robot.get_right_gripper_val())
        self.left_hold = np.asarray(self.task.robot.get_left_arm_jointState()[:6], dtype=float)
        self.left_gripper = float(self.task.robot.get_left_gripper_val())
        self.bounds = self.task._arm_joint_bounds(self.task.robot.left_arm_joints)
        self.initial_mug = np.asarray(self.task.mug.get_pose().p, dtype=float).copy()

    @property
    def executed_ticks(self):
        return 0 if self.executor is None else self.executor.executed_ticks

    def command_state(self):
        return vector14(self.task.robot.get_left_arm_jointState() + self.task.robot.get_right_arm_jointState())

    def diagnostics(self):
        return {"mug_position": list(map(float, self.task.mug.get_pose().p)),
                "left_tcp": list(map(float, self.task.robot.get_left_tcp_pose())),
                "right_tcp": list(map(float, self.task.robot.get_right_tcp_pose())),
                "left_gripper": float(self.task.robot.get_left_gripper_val()),
                "task_success": bool(self.task.check_success()),
                "mug_contact_entities": sorted(self.mug_contacts())}

    def mug_contacts(self):
        touching = set()
        for contact in self.task.scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            if self.task.mug.actor.name in names and contact.points:
                touching.update(name for name in names if name != self.task.mug.actor.name)
        return touching

    def preconditions(self):
        self.executor.check()
        mug = self.initial_mug
        # Narrow initial-acquisition workspace; excludes transfer/rack/lost-mug states.
        if (not self.task.plan_success or not self.task.need_plan
                or not np.isfinite(mug).all() or not (-0.40 <= mug[0] <= 0.02)
                or not (-0.20 <= mug[1] <= 0.18) or not (0.73 <= mug[2] <= 0.86)
                or not 0 <= self.left_gripper <= 0.25
                or not 0 <= self.right_gripper <= 1
                or np.linalg.norm(np.asarray(self.task.robot.get_right_tcp_pose()[:3]) - mug) < 0.12):
            raise SkillStopped("precondition_failed")
        # Reject a mug already held between both left fingers, or touching right arm.
        touching = self.mug_contacts()
        if ("table" not in touching or {"fl_link7", "fl_link8"} <= touching
                or any(name.startswith("fr_") for name in touching)):
            raise SkillStopped("precondition_failed")
        self.command_state()  # Finite 14D controller state before any drive write.
        self.log({"event": "preconditions_passed", "privileged": self.diagnostics()})

    def _guard(self):
        if self.cancel():
            return "cancelled"
        if self.task.eval_success or self.task.check_success():
            return "task_complete"
        mug = np.asarray(self.task.mug.get_pose().p, dtype=float)
        if (not np.isfinite(mug).all() or mug[2] < 0.72
                or np.linalg.norm(mug - self.initial_mug) > 0.20):
            return "safety_violation"
        if not np.isfinite(self.command_state()).all():
            return "safety_violation"
        return None

    def _advance(self, position, velocity, gripper):
        robot = self.task.robot
        self.began_motion = True  # A failed drive write also needs a stop attempt.
        robot.set_arm_joints(position, velocity, "left")
        robot.set_arm_joints(self.right_hold, np.zeros(6), "right")
        robot.set_gripper(float(gripper), "left")
        robot.set_gripper(self.right_gripper, "right")
        self.task.scene.step()
        self.left_hold = np.asarray(position).copy()
        self.left_gripper = float(gripper)

    def stop(self):
        if self.began_motion:
            for arm in ("left", "right"):
                targets = getattr(self.task.robot, f"get_{arm}_arm_jointState")()[:6]
                self.task.robot.set_arm_joints(targets, np.zeros(6), arm)
            self.log({"event": "velocity_targets_zeroed", "physics_ticks": self.executed_ticks})

    def _pose(self, value):
        pose = np.asarray(value, dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or abs(np.linalg.norm(pose[3:]) - 1) > 0.01
                or not (-0.6 <= pose[0] <= 0.1 and -0.6 <= pose[1] <= 0.35 and 0.60 <= pose[2] <= 1.30)):
            raise SkillStopped("pose_limits")
        return pose

    def move(self, stage, pose):
        self.executor.check()
        pose = self._pose(pose)
        current = self._pose(self.task.robot.get_left_ee_pose())
        if np.linalg.norm(pose[:3] - current[:3]) > 0.40:
            raise SkillStopped("pose_limits")
        self.log({"event": "stage", "stage": stage, "target_pose": pose.tolist()})
        plan = self.task.robot.left_plan_path(pose.tolist())
        measured = self.task.robot.get_left_arm_real_jointState()[:6]
        positions, velocities = validate_trajectory(plan, self.bounds, measured)
        self.executor.require_capacity(len(positions))
        for position, velocity in zip(positions, velocities):
            self.executor.tick(stage, position, velocity, self.left_gripper)

    def offset(self, stage, distance):
        pose = self._pose(self.task.robot.get_left_ee_pose()).copy()
        pose[2] += distance
        self.move(stage, pose)

    def grasp_targets(self, backoff):
        self.executor.check()
        # Explicit privileged geometry; fixed contact choice, no pose from a model.
        pregrasp = self.task.get_grasp_pose(self.task.mug, "left", contact_point_id=0, pre_dis=backoff)
        pregrasp = self._pose(pregrasp)
        w, x, y, z = pregrasp[3:]
        local_x = np.array([1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)])
        grasp = pregrasp.copy()
        grasp[:3] += local_x * backoff
        return pregrasp, self._pose(grasp)

    def gripper(self, stage, target):
        self.executor.check()
        trajectory = self.task.set_gripper(set_tag="left", left_pos=target)
        values = np.asarray(trajectory["result"], dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise SkillStopped("invalid_gripper_trajectory")
        self.executor.require_capacity(len(values))
        self.log({"event": "stage", "stage": stage, "gripper_target": target})
        for value in values:
            self.executor.tick(stage, self.left_hold, np.zeros(6), value)

    def settle(self):
        self.executor.require_capacity(50)
        for _ in range(50):
            self.executor.tick("settle", self.left_hold, np.zeros(6), self.left_gripper)
        reason = self._guard()
        if reason:
            raise SkillStopped(reason)
