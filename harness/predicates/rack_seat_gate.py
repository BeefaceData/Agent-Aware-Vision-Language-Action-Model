"""Runtime release gate for a perception-derived rack-seat target."""

import numpy as np

from harness.perception.geometry import pose_error, pose7
from .base import PredicateResult


class RackSeatReleaseGate:
    """Authorize one release only while measured TCP remains at a grounded seat.

    This verifies geometric alignment with the target composed from public
    RGB-D and the measured grasp. It does not claim physical peg contact or
    task success; RoboTwin's task checker remains the success authority.
    """

    name = "rack_seat_aligned"

    def __init__(self, environment, *, position_tolerance_m=0.015,
                 orientation_tolerance_rad=0.10):
        self.environment = environment
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        if (not np.isfinite((self.position_tolerance_m,
                             self.orientation_tolerance_rad)).all()
                or not 0 < self.position_tolerance_m <= 0.03
                or not 0 < self.orientation_tolerance_rad <= 0.20):
            raise ValueError("invalid_rack_seat_gate_tolerances")
        self._arm = None
        self._target = None
        self._after_sequence = None

    @property
    def available(self):
        return self._arm is not None

    def record(self, command, result, target_anchor):
        """Arm the gate only after a known, successful MOVE_TO rack_seat."""
        action = command.action.value if hasattr(command.action, "value") else command.action
        valid = (action == "move_to" and command.parameters.target == "rack_seat"
                 and result is not None and result.completed
                 and not result.execution_uncertain and not result.episode_terminal
                 and result.stop_reason == "completed"
                 and target_anchor is not None and result.observation_after is not None
                 and result.position_error_m is not None
                 and result.orientation_error_rad is not None
                 and result.position_error_m <= self.position_tolerance_m
                 and result.orientation_error_rad <= self.orientation_tolerance_rad)
        if valid:
            self._arm = command.arm
            self._target = tuple(pose7(target_anchor.pose))
            self._after_sequence = result.observation_after
        else:
            self.clear()

    def clear(self):
        self._arm = self._target = self._after_sequence = None

    def evaluate(self, name, observation, *, arm="auto"):
        if name != self.name:
            return None
        value, reason = None, "rack_seat_alignment_unavailable"
        if self.available:
            if arm not in ("auto", self._arm):
                value, reason = False, "wrong_arm_for_grounded_rack_seat"
            elif observation.sequence < self._after_sequence:
                value, reason = None, "rack_seat_observation_stale"
            else:
                try:
                    actual = pose7(self.environment.arm_state(self._arm)["pose"])
                    position_error, orientation_error = pose_error(actual, self._target)
                    value = (position_error <= self.position_tolerance_m
                             and orientation_error <= self.orientation_tolerance_rad)
                    reason = "measured_tcp_at_grounded_rack_seat" if value else "tcp_left_rack_seat"
                except (KeyError, TypeError, ValueError, RuntimeError):
                    value, reason = None, "measured_tcp_unavailable"
        return PredicateResult(name=name, value=value, source="runtime",
            observation_sequence=observation.sequence, reason=reason)
