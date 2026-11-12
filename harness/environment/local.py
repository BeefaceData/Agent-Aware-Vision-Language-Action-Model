"""Native planning/tick capability around the existing process-local simulator.

Object poses and contact details stay out of planner-facing observations.
Create/reset/close from the episode owner in a dedicated process because
RoboTwin changes its working directory.
"""

from dataclasses import dataclass, replace

import numpy as np

from harness.environment.base import vector14
from harness.perception.geometry import pose7
from .client import EnvironmentClient


@dataclass(frozen=True)
class ReplayState:
    """State-only result from a development prefix action without a camera render."""

    sequence: int
    command_state: tuple[float, ...]
    measured_tcp_poses: dict[str, tuple[float, ...]]


class LocalEnvironmentClient(EnvironmentClient):
    supports_analytic = True

    def __init__(self, simulator, *, render_once_per_step=False):
        """``render_once_per_step``: skip ``step``'s render of the pre-action state.

        The base ``step`` calls ``observe()`` for the previous sequence and the
        terminal check; here ``observe()`` renders every camera, and that frame is
        discarded. When enabled, both come from this client's own state instead,
        which ``observe()`` would have reproduced (``tick`` advances ``_sequence``
        and ``native_success`` reads the task), so the returned observations are
        the same with one render per action instead of two.
        """
        super().__init__(simulator)
        self.simulator = simulator
        self._sequence = 0
        self.render_once_per_step = render_once_per_step

    def _accept(self, raw, sequence):
        result = super()._accept(raw, sequence)
        # These are measured robot TCP poses, available from arm encoders and
        # forward kinematics on a physical robot as well. They help local
        # predicates attribute a camera-grounded mug to an arm; object pose and
        # contact state remain unavailable to the perception provider.
        measured_tcp_poses = {
            arm: tuple(self._measured_tcp_pose(arm)) for arm in ("left", "right")}
        result = replace(result, measured_tcp_poses=measured_tcp_poses)
        self._observation = result
        self._sequence = sequence
        return result.copy()

    def observe(self):
        if self._observation is None or self._uncertain:
            raise RuntimeError("environment has no confirmed current observation")
        return self._accept(self.simulator.observation(self.native_success), self._sequence)

    def _step(self, action, *, allow_after_success):
        if not self.render_once_per_step:
            return super()._step(action, allow_after_success=allow_after_success)
        if self._observation is None or self._uncertain:
            raise RuntimeError("environment has no confirmed current observation")
        if self.native_success and not allow_after_success:
            raise RuntimeError("environment is terminal")
        action = vector14(action)
        self._uncertain = True  # A failed response may still have moved the robot.
        raw, _ = self._transport(self.host, self.port, "/step", {"action": list(action)})
        observation = self._accept(raw, self._sequence + 1)
        self._uncertain = False
        return observation

    def advance(self, action):
        """Apply one action without rendering; ``observe()`` renders the new state.

        ``step`` renders twice per action (``observe`` for the previous sequence,
        then the new frame); a caller that reads only some frames uses this.
        """
        if self.native_success:
            raise RuntimeError("environment is terminal")
        self._uncertain = True  # A failed action may still have moved the robot.
        self.simulator.act(vector14(action))
        self._sequence += 1
        self._uncertain = False

    def advance_for_replay(self, action):
        """Replay one action and read joint/FK state without capturing camera frames.

        The reset observation remains cached until the next normal ``observe``;
        replay callers must compare their final RGB snapshot after replay.
        """
        if self.native_success:
            raise RuntimeError("environment is terminal")
        self._uncertain = True
        try:
            success = self.simulator.act(vector14(action))
            self._sequence += 1
            self._native_success = bool(success)
            robot = self.simulator.task.robot
            state = np.asarray(
                robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
                dtype=np.float32)
            command_state = vector14(state)
            measured_tcp_poses = {
                arm: tuple(self._measured_tcp_pose(arm)) for arm in ("left", "right")}
            self._uncertain = False
            return ReplayState(self._sequence, command_state, measured_tcp_poses)
        except BaseException:
            # A failed simulator call may have acted; preserve the uncertain latch.
            raise

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
        return {
            "joints": joints,
            # RoboTwin plans this control frame through left/right_plan_path.
            "pose": pose7(getattr(robot, f"get_{arm}_ee_pose")()),
            # Contact predicates and camera geometry use the physical gripper TCP.
            "tcp_pose": pose7(getattr(robot, f"get_{arm}_tcp_pose")()),
            "gripper": gripper,
        }  # Gripper remains a command, not aperture sensing.

    def _measured_tcp_pose(self, arm):
        state = self.arm_state(arm)
        # Small test adapters predating explicit TCP support retain their
        # planning pose as a compatibility fallback.
        return state.get("tcp_pose", state["pose"])

    def joint_bounds(self, arm):
        task = self.simulator.task
        return task._arm_joint_bounds(getattr(task.robot, f"{arm}_arm_joints"))

    def plan(self, arm, pose):
        # Native planner reads measured articulation qpos at this instant.
        return getattr(self.simulator.task.robot, f"{arm}_plan_path")(pose7(pose).tolist())

    def apply_native_qpos(self, action, *, controlled_arm, controlled_velocity):
        """Apply one qpos14 sample through RoboTwin's public action boundary."""
        self._uncertain = True
        action = vector14(action)
        velocity = np.asarray(controlled_velocity, dtype=float)
        if controlled_arm not in ("left", "right") or velocity.shape != (6,):
            raise ValueError("invalid_native_qpos_update")
        success = self.simulator.act(action)
        self._native_success = bool(success)
        self._sequence += 1
        self._uncertain = False

    def tick(self, arm, position, velocity, gripper, other_hold):
        """Compatibility wrapper for callers outside the native-motion adapter."""
        from harness.skills.native_motion import NativeArmState, NativeMotionAdapter
        other = "right" if arm == "left" else "left"
        other_state = NativeArmState(
            np.asarray(other_hold["joints"], dtype=float),
            self.arm_state(other)["pose"], float(other_hold["gripper"]))
        action = NativeMotionAdapter.compose_qpos14(arm, position, gripper, other_state)
        self.apply_native_qpos(action, controlled_arm=arm, controlled_velocity=velocity)

    def hold_measured(self):
        # Best-effort stop target; no scene reset, teleport or extra physics tick.
        robot = self.simulator.task.robot
        for arm in ("left", "right"):
            joints = self.arm_state(arm)["joints"]
            robot.set_arm_joints(joints, np.zeros(6), arm)

    def close(self):
        self.simulator.close()
