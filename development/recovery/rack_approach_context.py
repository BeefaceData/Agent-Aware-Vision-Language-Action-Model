"""Privileged rack recovery adapter, isolated from the frozen policy/runtime path."""

import numpy as np

from harness.skills.motion import SkillStopped, TickExecutor, validate_trajectory
from harness.environment.base import vector14
from development.simulator.rack_geometry import pose_matrix, rack_diagnostics, release_geometry_ready, supported_release_ready


class RackApproachContext:
    development_only = True
    skill_id = "backoff_reapproach_rack"

    def __init__(self, task, *, log, cancel=lambda: False, max_ticks=3000, frame_callback=None):
        if type(max_ticks) is not int or not 1 <= max_ticks <= 3000:
            raise ValueError("max_ticks must be an integer in [1, 3000]")
        self.task, self.log, self.cancel, self.max_ticks = task, log, cancel, max_ticks
        self.executor = None
        self.began_motion = False
        self.release_authorized = False
        self.frame_callback = frame_callback

    def begin(self, skill_limit):
        if self.executor is not None:
            raise ValueError("context is single use")
        robot = self.task.robot
        self.executor = TickExecutor(min(skill_limit, self.max_ticks), advance=self._advance,
                                     guard=self._guard, log=self.log, arm="right")
        self.left_hold = np.asarray(robot.get_left_arm_jointState()[:6], dtype=float)
        self.left_gripper = float(robot.get_left_gripper_val())
        self.right_hold = np.asarray(robot.get_right_arm_jointState()[:6], dtype=float)
        self.right_gripper = float(robot.get_right_gripper_val())
        self.bounds = self.task._arm_joint_bounds(robot.right_arm_joints)
        self.initial_attachment = self.attachment_position()

    @property
    def executed_ticks(self):
        return self.executor.executed_ticks if self.executor else 0

    def command_state(self):
        return vector14(self.task.robot.get_left_arm_jointState() + self.task.robot.get_right_arm_jointState())

    def diagnostics(self):
        return rack_diagnostics(self.task)

    def attachment_position(self):
        tcp = pose_matrix(self.task.robot.get_right_tcp_pose())
        mug = np.asarray(self.task.mug.get_pose().p, dtype=float)
        return tcp[:3, :3].T @ (mug - tcp[:3, 3])

    def preconditions(self):
        self.executor.check()
        d = self.diagnostics()
        if (not self.task.need_plan or not self.task.plan_success or not d["right_bilateral_contact"]
                or d["left_contact"] or "table" in d["mug_contact_entities"]
                or not 0 <= self.right_gripper <= 0.25 or d["mug_position"][2] <= 0.78
                or d["handle_rack_distance_m"] > 0.65):
            raise SkillStopped("precondition_failed")
        self.command_state()
        self.log({"event": "preconditions_passed", "privileged": d})

    def _guard(self):
        if self.cancel():
            return "cancelled"
        # Native success starts at an opening command >0.8. Once release is
        # authorized, finish opening and withdraw before treating it as terminal.
        if not self.release_authorized and (self.task.eval_success or self.task.check_success()):
            return "task_complete"
        mug = np.asarray(self.task.mug.get_pose().p, dtype=float)
        if not np.isfinite(mug).all() or mug[2] < 0.72 or not (-0.25 <= mug[0] <= 0.65 and -0.40 <= mug[1] <= 0.50):
            return "safety_violation"
        self.command_state()
        if not self.release_authorized:
            attachment = self.attachment_position()
            if not np.isfinite(attachment).all() or np.linalg.norm(attachment - self.initial_attachment) > 0.03:
                return "grasp_lost"
        elif self.right_gripper < 0.4:
            d = self.diagnostics()
            if not (release_geometry_ready(d) or supported_release_ready(d, self.task.rack.actor.name)):
                return "release_not_aligned"
        return None

    def _advance(self, position, velocity, gripper):
        if gripper > self.right_gripper + 1e-9 and not self.release_authorized:
            raise SkillStopped("release_not_authorized")
        robot = self.task.robot
        self.began_motion = True
        robot.set_arm_joints(self.left_hold, np.zeros(6), "left")
        robot.set_arm_joints(position, velocity, "right")
        robot.set_gripper(self.left_gripper, "left")
        robot.set_gripper(float(gripper), "right")
        self.task.scene.step()
        self.right_hold, self.right_gripper = np.asarray(position).copy(), float(gripper)

    def stop(self):
        if self.began_motion:
            for arm in ("left", "right"):
                target = getattr(self.task.robot, f"get_{arm}_arm_jointState")()[:6]
                self.task.robot.set_arm_joints(target, np.zeros(6), arm)
            self.log({"event": "velocity_targets_zeroed", "physics_ticks": self.executed_ticks})

    def _pose(self, value):
        pose_matrix(value)
        pose = np.asarray(value, dtype=float)
        if not (-0.2 <= pose[0] <= 0.65 and -0.5 <= pose[1] <= 0.5 and 0.7 <= pose[2] <= 1.4):
            raise SkillStopped("pose_limits")
        return pose.copy()

    def move(self, stage, pose):
        self.executor.check()
        pose = self._pose(pose)
        current = self._pose(self.task.robot.get_right_ee_pose())
        if np.linalg.norm(pose[:3] - current[:3]) > 0.65:
            raise SkillStopped("pose_limits")
        self.log({"event": "stage", "stage": stage, "target_pose": pose.tolist()})
        plan = self.task.robot.right_plan_path(pose.tolist())
        positions, velocities = validate_trajectory(plan, self.bounds, self.task.robot.get_right_arm_real_jointState()[:6])
        self.executor.require_capacity(len(positions))
        for position, velocity in zip(positions, velocities):
            self.executor.tick(stage, position, velocity, self.right_gripper)
            self._frame()

    def _frame(self):
        if self.frame_callback and self.executed_ticks % 25 == 0:
            self.frame_callback()

    def retreat(self, distance):
        pose = self._pose(self.task.robot.get_right_ee_pose())
        rack = np.asarray(self.task.rack.get_functional_point(0)[:3], dtype=float)
        direction = np.asarray(self.task.mug.get_pose().p, dtype=float) - rack
        norm = np.linalg.norm(direction)
        if norm < 0.01:
            raise SkillStopped("already_at_rack")
        pose[:3] += direction / norm * distance
        self.move("rack_backoff", pose)

    def rack_targets(self):
        self.executor.check()
        # Only deterministic geometry code supplies target poses. No model pose/joints.
        target = self.task.rack.get_functional_point(0)
        common = dict(actor=self.task.mug, arm_tag="right", target_pose=target,
                      functional_point_id=0, constrain="align", pre_dis_axis="fp")
        prehang = self.task.get_place_pose(**common, pre_dis=0.05)
        seated = self.task.get_place_pose(**common, pre_dis=-0.05)
        return self._pose(prehang), self._pose(seated)

    def release_when_aligned(self):
        self.executor.check()
        diagnostics = self.diagnostics()
        if not diagnostics["right_bilateral_contact"] or not release_geometry_ready(diagnostics):
            self.log({"event": "release_rejected", "privileged": diagnostics})
            raise SkillStopped("release_not_aligned")
        trajectory = self.task.set_gripper(set_tag="right", right_pos=1.0)
        values = np.asarray(trajectory["result"], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < self.right_gripper) | (values > 1)):
            raise SkillStopped("invalid_gripper_trajectory")
        self.executor.require_capacity(len(values))
        self.log({"event": "release_authorized", "privileged": diagnostics})
        self.release_authorized = True
        for value in values:
            self.executor.tick("guarded_release", self.right_hold, np.zeros(6), value)
            self._frame()

    def settle(self):
        self.executor.require_capacity(50)
        for _ in range(50):
            self.executor.tick("settle", self.right_hold, np.zeros(6), self.right_gripper)
            self._frame()
        reason = self._guard()
        if reason:
            raise SkillStopped(reason)

    def withdraw(self):
        if not self.release_authorized or self.right_gripper < 0.999:
            raise SkillStopped("withdrawal_not_authorized")
        pose = self._pose(self.task.robot.get_right_ee_pose())
        # Same fixed 10cm retreat along negative end-effector X used by the
        # native hanging_mug demonstration's move_axis='arm', z=0.1.
        pose[:3] -= 0.10 * pose_matrix(pose)[:3, 0]
        self.move("withdraw", pose)
