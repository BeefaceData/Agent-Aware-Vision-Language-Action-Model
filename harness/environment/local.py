"""Native planning/tick capability around the existing process-local simulator.

No object poses or contacts are exported. Create/reset/close from the episode
owner. Use in a dedicated process because RoboTwin changes its working directory.
"""

import numpy as np

from harness.perception.geometry import pose7
from .client import EnvironmentClient


class LocalEnvironmentClient(EnvironmentClient):
    supports_analytic = True

    def __init__(self, simulator):
        super().__init__(simulator)
        self.simulator = simulator
        self._sequence = 0

    def _accept(self, raw, sequence):
        result = super()._accept(raw, sequence)
        self._sequence = sequence
        return result

    def observe(self):
        if self._observation is None or self._uncertain:
            raise RuntimeError("environment has no confirmed current observation")
        from harness.environment.robotwin import observation_of
        return self._accept(observation_of(self.simulator.task, self.simulator.blank,
                                           self.native_success), self._sequence)

    @property
    def native_success(self):
        if self._observation is None or self._uncertain:
            raise RuntimeError("environment has no confirmed current observation")
        task = self.simulator.task
        return bool(task.eval_success or getattr(task, "harness_native_success_latched", False)
                    or task.check_success())

    def arm_state(self, arm):
        if arm not in ("left", "right"):
            raise ValueError("invalid_arm")
        robot = self.simulator.task.robot
        joints = np.asarray(getattr(robot, f"get_{arm}_arm_real_jointState")()[:6], dtype=float)
        gripper = float(getattr(robot, f"get_{arm}_gripper_val")())
        if joints.shape != (6,) or not np.isfinite(joints).all() or not 0 <= gripper <= 1:
            raise ValueError("invalid_robot_state")
        return {"joints": joints, "pose": pose7(getattr(robot, f"get_{arm}_ee_pose")()),
                "gripper": gripper}  # Gripper remains a command, not aperture sensing.

    def arm_overlap_detected(self):
        """Return whether measured simulator contacts directly join both arms."""
        contacts = self.simulator.task.scene.get_contacts()

        def side(name):
            name = str(name).lower()
            if name.startswith(("fl_", "left_")):
                return "left"
            if name.startswith(("fr_", "right_")):
                return "right"
            return None

        for contact in contacts:
            bodies = getattr(contact, "bodies", ())
            if len(bodies) != 2:
                continue
            names = [getattr(getattr(body, "entity", None), "name", "")
                     for body in bodies]
            if {side(name) for name in names} == {"left", "right"}:
                return True
        return False

    def joint_bounds(self, arm):
        task = self.simulator.task
        return task._arm_joint_bounds(getattr(task.robot, f"{arm}_arm_joints"))

    def plan(self, arm, pose):
        # Native planner reads measured articulation qpos at this instant.
        return getattr(self.simulator.task.robot, f"{arm}_plan_path")(pose7(pose).tolist())

    def tick(self, arm, position, velocity, gripper, other_hold):
        self._uncertain = True
        robot = self.simulator.task.robot
        other = "right" if arm == "left" else "left"
        robot.set_arm_joints(position, velocity, arm)
        robot.set_arm_joints(other_hold["joints"], np.zeros(6), other)
        robot.set_gripper(gripper, arm)
        robot.set_gripper(other_hold["gripper"], other)
        self.simulator.task.scene.step()
        self._sequence += 1
        self._uncertain = False

    def hold_measured(self):
        # Best-effort stop target; no scene reset, teleport or extra physics tick.
        robot = self.simulator.task.robot
        for arm in ("left", "right"):
            joints = self.arm_state(arm)["joints"]
            robot.set_arm_joints(joints, np.zeros(6), arm)

    def close(self):
        self.simulator.close()
