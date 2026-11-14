"""Public grasp-loss detection and Thabo's re-grasp sequence for supervised runs.

At every policy-chunk boundary the monitor checks, for one arm:

- **fingers**: the arm was commanded closed for two consecutive boundaries
  (at least one full chunk, so the fingers have finished closing) and the
  measured finger opening is at or below ``empty_opening``: the fingers
  closed on nothing;
- **camera**: this frame's public report says the hand is empty and the mug
  is on the table, at task-state confidence of at least ``camera_confidence``.
  Uncertain frames are skipped; only a frame showing the mug held or lifted
  resets the count. Replayed on the saved seed 15-19 supervised runs, 0.75
  caught all four lost grasps 4-66 actions after the loss, with no false
  alarm on the successful seed 15 (0.8 missed seed 19).

The fingers on two consecutive boundaries, or two confident camera frames,
confirm a loss. Recovery is
the sequence from ``grasp/retry-sequence``: one MOVE_TO to
``mug_recovery_clearance`` (10 cm above the mug, current wrist orientation;
the skill rises first, then travels, then descends), open the measured-empty
gripper, and give control back to the frozen policy with a fresh history.
It needs the mug visibly upright; a toppled mug is never re-grasped.

The monitor never makes its own vision request: it reads the report the
planner turn already requested for the same frame. Uses no simulator state.
"""

import numpy as np

from harness.perception.anchors import Anchor
from harness.perception.geometry import planning_pose_for_tcp_target, pose7
from harness.skills.schemas import SkillCommand, SkillType


GRIPPER_INDEX = {"left": 6, "right": 13}
COMMAND_OPEN, COMMAND_CLOSED = .5, .05
# Provisional: an empty RoboTwin gripper should settle at its commanded 0,
# while fingers stopped by the mug stay measurably open. Calibrate from the
# measured_gripper_opening values logged on every action before relying on it.
DEFAULT_EMPTY_OPENING = .02
CLEARANCE_M = .10
CONFIRMING_BOUNDARIES = 2
RECOVERY_SKILL_CALLS = 2
RECOVERY_ANALYTIC_TICKS = 3600


class GraspRecovery:
    def __init__(self, environment, executor, provider, *, log, arm="right",
                 empty_opening=DEFAULT_EMPTY_OPENING, camera_confidence=.75, max_retries=2):
        if arm not in GRIPPER_INDEX:
            raise ValueError("invalid_grasp_recovery_arm")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("invalid_grasp_retry_budget")
        if not 0 <= float(empty_opening) < .5 or not 0 < float(camera_confidence) <= 1:
            raise ValueError("invalid_grasp_monitor_threshold")
        self.environment, self.executor, self.provider, self.log = environment, executor, provider, log
        self.arm, self.empty_opening = arm, float(empty_opening)
        self.camera_confidence, self.max_retries = float(camera_confidence), max_retries
        self.retries = 0
        self.stopped = None
        self.attempts = []
        self._reset_monitor(open_seen=False)

    def _reset_monitor(self, *, open_seen):
        self.open_seen = open_seen
        self.closed_boundaries = self.finger_streak = self.camera_streak = 0
        self.last_sequence = None

    def summary(self):
        return dict(arm=self.arm, retries=self.retries, max_retries=self.max_retries,
                    empty_opening=self.empty_opening, empty_opening_calibrated=False,
                    stopped=self.stopped, attempts=self.attempts)

    def _measured_opening(self):
        read = getattr(self.environment, "measured_gripper_opening", None)
        if not callable(read):
            return None
        try:
            value = float(read(self.arm))
        except Exception:
            return None
        return value if np.isfinite(value) else None

    def check(self, observation):
        """Update the monitor at one boundary; return the loss basis or None."""
        if self.stopped is not None or observation.sequence == self.last_sequence:
            return None
        self.last_sequence = observation.sequence
        command = float(observation.command_state[GRIPPER_INDEX[self.arm]])
        if command > COMMAND_OPEN:
            self._reset_monitor(open_seen=True)
            return None
        if not (self.open_seen and command <= COMMAND_CLOSED):
            self.closed_boundaries = self.finger_streak = self.camera_streak = 0
            return None
        self.closed_boundaries += 1
        if self.closed_boundaries < 2:
            return None
        opening = self._measured_opening()
        finger_empty = opening is not None and opening <= self.empty_opening
        report = self.provider.cached_report(observation)
        held = None if report is None else getattr(report, f"{self.arm}_held")
        confidence = None if report is None else report.task_state_confidence
        camera_empty = bool(report is not None and held is False and report.mug_off_table is False
                            and confidence is not None and confidence >= self.camera_confidence)
        self.finger_streak = self.finger_streak+1 if finger_empty else 0
        contradiction = bool(report is not None and (held is True or report.mug_off_table is True))
        self.camera_streak = (self.camera_streak+1 if camera_empty
                              else 0 if contradiction else self.camera_streak)
        basis = ("measured_fingers_closed_on_nothing" if self.finger_streak >= CONFIRMING_BOUNDARIES
                 else "camera_mug_on_table_hand_empty" if self.camera_streak >= CONFIRMING_BOUNDARIES
                 else None)
        self.log(dict(event_type="grasp_monitor", observation_sequence=observation.sequence, arm=self.arm,
                      commanded_gripper=command, measured_gripper_opening=opening,
                      empty_opening=self.empty_opening, finger_empty=finger_empty,
                      camera_hand_empty_on_table=camera_empty, camera_held=held,
                      task_state_confidence=confidence, finger_streak=self.finger_streak,
                      camera_streak=self.camera_streak, loss_confirmed=basis))
        return basis

    def _stop(self, observation, reason, **details):
        self.stopped = reason
        self.log(dict(event_type="grasp_recovery_stopped", observation_sequence=observation.sequence,
                      arm=self.arm, reason=reason, retries=self.retries, **details))
        return dict(status="grasp_recovery_stopped", reason=reason)

    def recover(self, observation, basis, *, remaining):
        """Run one re-grasp sequence; returns a record, or None to wait a boundary."""
        if self.retries >= self.max_retries:
            return self._stop(observation, "grasp_retry_budget_exhausted", basis=basis)
        if (remaining.get("skill_calls", 0) < RECOVERY_SKILL_CALLS+1
                or remaining.get("analytic_ticks", 0) < RECOVERY_ANALYTIC_TICKS
                or remaining.get("policy_actions", 0) <= 0):
            return self._stop(observation, "grasp_recovery_budget_unavailable", basis=basis)
        report = self.provider.cached_report(observation)
        upright = None if report is None else report.mug_upright
        if upright is False:
            return self._stop(observation, "mug_toppled_not_regrasped", basis=basis)
        position = self.provider.mug_surface_position(observation) if upright else None
        if position is None:
            # Wait for a frame that shows an upright mug with usable depth.
            self.log(dict(event_type="grasp_recovery_waiting", observation_sequence=observation.sequence,
                          arm=self.arm, basis=basis, mug_upright=upright,
                          reason="upright_mug_position_unavailable"))
            return None
        state = self.environment.arm_state(self.arm)
        measured, measured_tcp = pose7(state["pose"]), pose7(state.get("tcp_pose", state["pose"]))
        tcp_target = np.concatenate((np.asarray(position, dtype=float)+(0., 0., CLEARANCE_M), measured_tcp[3:]))
        planning = planning_pose_for_tcp_target(measured, measured_tcp, tcp_target)
        self.executor.anchors.bind("mug_recovery_clearance", self.arm,
                                   Anchor(tuple(float(x) for x in planning), observation.sequence, "perception"))
        self.retries += 1
        record = dict(attempt=self.retries, basis=basis, observation_sequence=observation.sequence,
                      mug_surface_position=[float(x) for x in position],
                      clearance_tcp_target=[float(x) for x in tcp_target])
        self.log(dict(event_type="grasp_recovery_started", arm=self.arm, **record))
        move = self.executor.execute(SkillCommand(action=SkillType.MOVE_TO, arm=self.arm,
            parameters=dict(target="mug_recovery_clearance", orientation_mode="preserve_current",
                            max_physics_ticks=3000),
            reason=f"grasp recovery: {basis}"), confirmed_observation=observation,
            allow_recoverable_tracking_error=True)
        record["move"] = dict(completed=move.completed, stop_reason=move.stop_reason,
                              physics_ticks=move.executed_physics_ticks,
                              execution_uncertain=move.execution_uncertain)
        if not move.completed or move.execution_uncertain:
            self.attempts.append(record)
            return self._stop(observation, "clearance_move_failed", stop_reason=move.stop_reason)
        opening = self._measured_opening()
        if opening is None or opening > self.empty_opening:
            # The fingers met something on the way up; never open a hand that may hold the mug.
            record["opening_before_release"] = opening
            self.attempts.append(record)
            return self._stop(observation, "hand_not_measured_empty_before_opening", measured_opening=opening)
        self.provider.authorize_empty_hand_release(self.arm)
        try:
            release = self.executor.execute(SkillCommand(action=SkillType.SET_GRIPPER, arm=self.arm,
                parameters=dict(value=1.0, ramp_ticks=200, max_physics_ticks=600),
                reason="open the measured-empty gripper before a fresh policy grasp"))
        finally:
            self.provider.revoke_empty_hand_release(self.arm)
        record["open"] = dict(completed=release.completed, stop_reason=release.stop_reason,
                              physics_ticks=release.executed_physics_ticks)
        self.attempts.append(record)
        if not release.completed or release.execution_uncertain:
            return self._stop(observation, "gripper_open_failed", stop_reason=release.stop_reason)
        self._reset_monitor(open_seen=True)
        self.log(dict(event_type="grasp_recovery_completed", arm=self.arm, **record))
        return dict(status="grasp_recovery_completed", **record)
