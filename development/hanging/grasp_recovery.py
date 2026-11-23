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
confirm a loss. Recovery, for a visibly upright mug only:

1. lift the empty hand straight up (current wrist orientation), clear of the mug;
2. open it, only if the fingers still measure empty;
3. move the arm back to the pose it had at reset;
4. give control back to the frozen policy with a fresh history.

The first live test (seed 18, 14 Nov 2026) used Thabo's clearance pose, 10 cm
above the mug, as the hand-back point. The move and the opening worked, but
the policy restarted from that unfamiliar pose and swept the mug over. The
right arm waits at its reset pose until every handoff, so returning there
recreates the start of a normal handoff.

The monitor never makes its own vision request: it reads the report the
planner turn already requested for the same frame. Uses no simulator state.
"""

import numpy as np

from harness.perception.anchors import Anchor
from harness.perception.geometry import pose7
from harness.skills.schemas import SkillCommand, SkillType


GRIPPER_INDEX = {"left": 6, "right": 13}
COMMAND_OPEN, COMMAND_CLOSED = .5, .05
# Calibrated on policy-only seed 18 (data/calibration/finger_opening_seed18):
# a closed empty hand stops at its joint limit, 0.1818 on every action; the
# left hand holding the mug read 0.25-0.26. 0.19 sits just above the empty
# limit. A right-hand grasp has not been measured yet.
DEFAULT_EMPTY_OPENING = .19
LIFT_M = .12
WORKSPACE_TOP_Z = 1.35
CONFIRMING_BOUNDARIES = 2
RECOVERY_SKILL_CALLS = 3
START_POSE_ANCHOR = "grasp_recovery_start_pose"
RECOVERY_ANALYTIC_TICKS = 6000


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
        # Constructed right after reset: this is the arm's waiting pose.
        self.start_pose = pose7(environment.arm_state(arm)["pose"])
        self._reset_monitor(open_seen=False)

    def hand_empty(self, arm):
        """True when this arm's fingers measure closed on nothing; None if unknown."""
        if arm != self.arm:
            return None
        opening = self._measured_opening()
        return None if opening is None else opening <= self.empty_opening

    def _reset_monitor(self, *, open_seen):
        self.open_seen = open_seen
        self.closed_boundaries = self.finger_streak = self.camera_streak = 0
        self.last_sequence = None

    def summary(self):
        return dict(arm=self.arm, retries=self.retries, max_retries=self.max_retries,
                    empty_opening=self.empty_opening, empty_opening_basis="seed18_policy_only: empty 0.1818, left held 0.25",
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

    def _move(self, record, key, parameters, reason):
        current = self.environment.observe()
        if parameters.get("target") == START_POSE_ANCHOR:
            self.executor.anchors.bind(START_POSE_ANCHOR, self.arm, Anchor(
                tuple(float(x) for x in self.start_pose), current.sequence, "perception"))
        result = self.executor.execute(SkillCommand(action=SkillType.MOVE_TO, arm=self.arm,
            parameters=dict(max_physics_ticks=3000, **parameters), reason=reason),
            confirmed_observation=current, allow_recoverable_tracking_error=True)
        record[key] = dict(completed=result.completed, stop_reason=result.stop_reason,
                           physics_ticks=result.executed_physics_ticks,
                           execution_uncertain=result.execution_uncertain)
        return result.completed and not result.execution_uncertain

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
        if upright is not True:
            # Wait for a frame that shows the mug upright.
            self.log(dict(event_type="grasp_recovery_waiting", observation_sequence=observation.sequence,
                          arm=self.arm, basis=basis, mug_upright=upright, reason="mug_upright_unconfirmed"))
            return None
        state = self.environment.arm_state(self.arm)
        lift = min(LIFT_M, max(0., WORKSPACE_TOP_Z-float(pose7(state["pose"])[2])))
        self.retries += 1
        record = dict(attempt=self.retries, basis=basis, observation_sequence=observation.sequence,
                      lift_m=lift, start_pose=[float(x) for x in self.start_pose])
        self.log(dict(event_type="grasp_recovery_started", arm=self.arm, **record))
        if not self._move(record, "lift", dict(target="current_ee", offset_xyz=(0., 0., lift),
                                               orientation_mode="preserve_current"),
                          f"grasp recovery: lift the empty hand clear ({basis})"):
            self.attempts.append(record)
            return self._stop(observation, "lift_failed", stop_reason=record["lift"]["stop_reason"])
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
        if not release.completed or release.execution_uncertain:
            self.attempts.append(record)
            return self._stop(observation, "gripper_open_failed", stop_reason=release.stop_reason)
        if not self._move(record, "return", dict(target=START_POSE_ANCHOR),
                          "grasp recovery: return the arm to its reset pose for a fresh policy grasp"):
            self.attempts.append(record)
            return self._stop(observation, "return_to_start_failed", stop_reason=record["return"]["stop_reason"])
        self.attempts.append(record)
        self._reset_monitor(open_seen=True)
        self.log(dict(event_type="grasp_recovery_completed", arm=self.arm, **record))
        return dict(status="grasp_recovery_completed", **record)
