"""Bounded hanging_mug composition with explicit handoff and rack staging."""

from harness.skills.schemas import SkillCommand, VLAParameters
from .base import PlannerSelection
from .scripted import GraspRetryPlanner


class HandoffPlanner:
    """Acquire, complete left transfer, then stage and hang with the right arm.

    The VLA coordinates both arms during transfer. The transfer predicate is
    based on stable table support and release, without arm-home or clearance.
    """

    RIGHT_GRASP_RETRY_ACTIONS = 176

    def __init__(self, acquisition, handoff):
        self.acquisition = GraspRetryPlanner(acquisition)
        self.handoff = VLAParameters.model_validate(handoff.model_dump())
        self.stage_attempted = False
        self.right_lift_attempted = False
        self.right_lift_confirmed = False
        self.right_grasp_retries = 0
        self.right_retry_phase = None
        self.rack_stage_attempted = False
        self.rack_stage_confirmed = False
        self.hang_attempted = False
        self.left_transfer_confirmed = False

    def select(self, context):
        previous = context.previous
        if previous and (previous.episode_terminal or not previous.completed or previous.execution_uncertain):
            return self._abort("previous_skill_failed")
        if previous and previous.predicate is not None:
            if (previous.predicate.name == "left_transfer_complete"
                    and previous.predicate.value is True):
                # This is a historical phase transition. It remains verified
                # after the right arm picks the mug off the table.
                self.left_transfer_confirmed = True
        if (previous and self.hang_attempted and previous.objective_satisfied is False):
            return self._abort("rack_hang_unconfirmed")

        if not self.left_transfer_confirmed:
            if context.value("mug_lifted") is None:
                return self._abort("lift_evidence_unavailable")
            if context.value("mug_lifted") is not True:
                return self.acquisition.select(context)
            transfer_complete = context.value("left_transfer_complete")
            if transfer_complete is None:
                return self._abort("left_transfer_evidence_unavailable")
            if transfer_complete is True:
                self.left_transfer_confirmed = True
        if not self.left_transfer_confirmed:
            if self.stage_attempted:
                return self._abort("left_transfer_unconfirmed")
            self.stage_attempted = True
            return PlannerSelection(SkillCommand(action="vla_act", arm="both",
                parameters=self._parameters(lock_arm=None), stop="left_transfer_complete",
                reason="Coordinate left-arm placement and withdrawal with the right-arm approach"),
                "LEFT_TRANSFER", self.retry_count)

        if context.value("task_success") is True:
            return self._abort("task_already_complete")

        recovery = self._select_right_grasp_retry(context)
        if recovery is not None:
            return recovery

        if previous and previous.skill.value == "move_to" and self.rack_stage_attempted:
            if not previous.completed or previous.execution_uncertain:
                return self._abort("rack_staging_failed")
            self.rack_stage_confirmed = True

        if not self.right_lift_confirmed:
            right_lifted = context.value("right_mug_lifted")
            if right_lifted is None:
                return self._abort("right_lift_evidence_unavailable")
            if right_lifted is True:
                self.right_lift_confirmed = True
            elif self.right_lift_attempted:
                if (self.right_grasp_retries >= self.acquisition.settings.max_retries
                        or not self._retryable_right_grasp_failure(previous)):
                    return self._abort("right_grasp_unconfirmed")
                if context.value("release_ready") is not True:
                    return self._abort("right_grasp_retry_open_not_authorized")
                self.right_grasp_retries += 1
                self.right_retry_phase = "retreat"
                return PlannerSelection(SkillCommand(action="set_gripper", arm="right",
                    parameters={"value": 1.0},
                    reason="Open the supported or empty right hand before a bounded grasp retry"),
                    "RIGHT_RETRY_OPEN", self.retry_count)
            else:
                self.right_lift_attempted = True
                return self._right_vla_selection("RIGHT_GRASP")

        if not self.rack_stage_confirmed:
            if self.rack_stage_attempted:
                return self._abort("rack_staging_unconfirmed")
            self.rack_stage_attempted = True
            return PlannerSelection(SkillCommand(action="move_to", arm="right",
                parameters={"target": "rack_approach", "max_physics_ticks": 1000},
                reason="Carry the right-arm grasp to a fresh RGB-D rack approach point"),
                "RACK_STAGING", self.retry_count)

        if self.hang_attempted:
            return self._abort("rack_hang_unconfirmed")
        self.hang_attempted = True
        return PlannerSelection(SkillCommand(action="vla_act", arm="both",
            parameters=self._parameters(lock_arm="left"), stop="task_success",
            reason="After right grasp and grounded rack staging, attempt the constrained hang"),
            "RACK_HANG", self.retry_count)

    def _parameters(self, *, lock_arm):
        data = self.handoff.model_dump()
        data["lock_arm"] = lock_arm
        return VLAParameters.model_validate(data)

    def _select_right_grasp_retry(self, context):
        """Recover a verified, unambiguous miss through bounded existing skills."""
        phase = self.right_retry_phase
        if phase is None:
            return None
        previous = context.previous
        if phase == "retreat":
            if previous is None or previous.skill.value != "set_gripper":
                return self._abort("right_grasp_retry_open_failed")
            self.right_retry_phase = "restage"
            return PlannerSelection(SkillCommand(action="move_to", arm="right",
                parameters={"target": "current_ee", "offset_xyz": (0.0, 0.0, 0.05)},
                reason="Back off from the failed right-arm grasp using the measured pose"),
                "RIGHT_RETRY_RETREAT", self.retry_count)
        if phase == "restage":
            if previous is None or previous.skill.value != "move_to":
                return self._abort("right_grasp_retry_retreat_failed")
            self.right_retry_phase = "retry_vla"
            return PlannerSelection(SkillCommand(action="move_to", arm="right",
                parameters={"target": "mug_pregrasp"},
                reason="Re-ground the mug pregrasp from the current observation"),
                "RIGHT_RETRY_STAGE", self.retry_count)
        if phase == "retry_vla":
            if previous is None or previous.skill.value != "move_to":
                return self._abort("right_grasp_retry_staging_failed")
            self.right_retry_phase = "await_retry"
            return self._right_vla_selection("RIGHT_GRASP_RETRY")
        if phase == "await_retry":
            self.right_retry_phase = None
            if context.value("right_mug_lifted") is True:
                self.right_lift_confirmed = True
                return None
            return self._abort("right_grasp_retry_budget_exhausted")
        return self._abort("invalid_right_grasp_retry_state")

    @staticmethod
    def _retryable_right_grasp_failure(previous):
        return (previous is not None and previous.skill.value == "vla_act"
            and previous.completed and not previous.execution_uncertain
            and (previous.objective_satisfied is False
                 or (previous.stop_reason == "failure_predicate_satisfied"
                     and previous.objective_satisfied is None))
            and previous.stop_reason in {"failure_predicate_satisfied", "chunk_budget_exhausted",
                                         "policy_action_budget_exhausted"})

    def _right_vla_selection(self, phase):
        parameters = self._parameters(lock_arm="left").model_dump()
        if phase == "RIGHT_GRASP_RETRY":
            parameters["max_policy_actions"] = min(parameters["max_policy_actions"],
                                                     self.RIGHT_GRASP_RETRY_ACTIONS)
            retry_chunks = ((parameters["max_policy_actions"] + parameters["chunk_size"] - 1)
                            // parameters["chunk_size"])
            parameters["max_chunks"] = min(parameters["max_chunks"], retry_chunks)
        return PlannerSelection(SkillCommand(action="vla_act", arm="both",
            parameters=parameters, stop="right_mug_lifted",
            reason="Use the right arm to grasp and lift the staged mug"),
            phase, self.retry_count)

    @property
    def retry_count(self):
        return self.acquisition.retries + self.right_grasp_retries

    def _abort(self, reason):
        return PlannerSelection(SkillCommand(action="abort", reason=reason), "FAILED", self.retry_count)
