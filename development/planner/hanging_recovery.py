"""Fixed development sequence for testing analytic rack-hanging recovery."""

from harness.skills.schemas import SkillCommand
from harness.planner.base import PlannerSelection


class HangingRecoveryPlanner:
    """Approach, align the held mug handle, then release at the rack seat.

    This planner is only selected by the replay-based development-oracle route.
    Simulator-derived targets are resolved by the oracle anchor preparer and
    never enter this planner's context.
    """

    STEPS = (
        ("RACK_APPROACH", "move_to", "rack_approach"),
        ("RACK_SEAT", "move_to", "rack_seat"),
        ("RACK_RELEASE", "release", None),
    )

    def __init__(self):
        self._step = 0

    def select(self, context):
        previous = context.previous
        if previous is not None:
            if (previous.episode_terminal or not previous.completed
                    or previous.execution_uncertain):
                return self._abort("hanging_recovery_skill_failed")
            self._step += 1

        if self._step >= len(self.STEPS):
            return self._abort("hanging_recovery_sequence_complete")

        phase, action, target = self.STEPS[self._step]
        if action == "move_to":
            command = SkillCommand(action=action, arm="right",
                parameters={"target": target, "max_physics_ticks": 1600},
                reason=("Move while retaining the verified right-arm grasp"))
        else:
            command = SkillCommand(action=action, arm="right",
                parameters={"authorization": "rack_seat_alignment", "ramp_ticks": 200},
                reason="Release only after measured rack-seat alignment")
        return PlannerSelection(command, phase)

    @staticmethod
    def _abort(reason):
        return PlannerSelection(SkillCommand(action="abort", reason=reason), "FAILED")
