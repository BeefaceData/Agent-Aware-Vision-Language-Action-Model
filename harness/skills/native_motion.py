"""One validated boundary between analytic skills and RoboTwin control."""

from dataclasses import dataclass

import numpy as np

from harness.environment.base import vector14
from harness.perception.geometry import pose7


@dataclass(frozen=True)
class NativeArmState:
    joints: np.ndarray
    pose: np.ndarray
    gripper: float


class NativeMotionAdapter:
    """Read, plan and execute through the environment's native robot adapter.

    CuRobo already returns time-parameterized joint samples. The process-local
    environment applies those samples directly to RoboTwin's native drives;
    routing every sample through ``task.take_action`` would run TOPP a second
    time and change the planned trajectory.
    """

    def __init__(self, environment):
        self.environment = environment

    def state(self, arm):
        if arm not in ("left", "right"):
            raise ValueError("invalid_arm")
        raw = self.environment.arm_state(arm)
        joints = np.asarray(raw["joints"], dtype=float)
        pose = pose7(raw["pose"])
        gripper = float(raw["gripper"])
        if (joints.shape != (6,) or not np.isfinite(joints).all()
                or not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0):
            raise ValueError("invalid_robot_state")
        return NativeArmState(joints.copy(), pose.copy(), gripper)

    def joint_bounds(self, arm):
        bounds = np.asarray(self.environment.joint_bounds(arm), dtype=float)
        if (bounds.shape != (6, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])):
            raise ValueError("invalid_joint_bounds")
        return bounds.copy()

    def plan_pose(self, arm, target_pose):
        return self.environment.plan(arm, pose7(target_pose))

    @staticmethod
    def compose_qpos14(arm, position, gripper, other_state):
        position = np.asarray(position, dtype=float)
        other_joints = np.asarray(other_state.joints, dtype=float)
        values = (position, other_joints, float(gripper), float(other_state.gripper))
        if (arm not in ("left", "right") or position.shape != (6,)
                or other_joints.shape != (6,)
                or not all(np.isfinite(value).all() if isinstance(value, np.ndarray)
                           else np.isfinite(value) for value in values)
                or not 0.0 <= float(gripper) <= 1.0
                or not 0.0 <= float(other_state.gripper) <= 1.0):
            raise ValueError("invalid_qpos_update")
        controlled = (*position.tolist(), float(gripper))
        inactive = (*other_joints.tolist(), float(other_state.gripper))
        return vector14((*controlled, *inactive) if arm == "left" else (*inactive, *controlled))

    def apply_qpos_sample(self, arm, position, velocity, gripper, other_state):
        action = self.compose_qpos14(arm, position, gripper, other_state)
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError("invalid_qpos_velocity")
        native = getattr(self.environment, "apply_native_qpos", None)
        if native is not None:
            native(action, controlled_arm=arm, controlled_velocity=velocity)
            return
        # Small fake environments and legacy adapters retain this compatibility
        # boundary; production LocalEnvironmentClient implements apply_native_qpos.
        self.environment.tick(
            arm, np.asarray(position, dtype=float), velocity, float(gripper),
            {"joints": other_state.joints, "gripper": other_state.gripper},
        )

    def hold_measured(self):
        self.environment.hold_measured()
