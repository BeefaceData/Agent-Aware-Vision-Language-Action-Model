"""Bounded hanging_mug composition with explicit handoff and rack staging."""

from harness.skills.schemas import (SkillCommand, VLAParameters,
    is_recoverable_executed_motion_failure, is_recoverable_partial_planning_failure,
    is_recoverable_trajectory_limit, is_retryable_pre_motion_planning_failure)
from .base import PlannerSelection
from .scripted import (GraspRetryPlanner, _retry_open_authorized,
                       near_grounded_retry_stage)


class HandoffPlanner:
    """Acquire, complete left transfer, then stage and hang with the right arm.

    The VLA coordinates both arms during transfer. The transfer predicate is
    based on stable table support and release, without arm-home or clearance.
    """

    RIGHT_GRASP_RETRY_ACTIONS = 176
    RIGHT_RESTAGE_MAX_PHYSICS_TICKS = 1600

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

    def resume_development_right_grasp(self):
        """Resume from a replayed, verified left-transfer boundary."""
        if self.left_transfer_confirmed or self.right_lift_attempted:
            raise ValueError("planner_already_started")
        self.left_transfer_confirmed = True
        self.stage_attempted = True

    def select(self, context):
        previous = context.previous
        if previous and (previous.episode_terminal or previous.execution_uncertain):
            return self._abort("previous_skill_failed")
        if previous and not previous.completed:
            if (not self.left_transfer_confirmed
                    and self.acquisition.phase == "APPROACH_MUG"
                    and previous.skill.value == "move_to"
                    and near_grounded_retry_stage(context, previous)):
                return self.acquisition.select(context)
            if (not self.left_transfer_confirmed
                    and self.acquisition.phase == "APPROACH_MUG"
                    and self.acquisition.reorientation_attempted
                    and previous.skill.value == "move_to"
                    and previous.stop_reason in {"trajectory_limits", "planning_failure"}):
                return self.acquisition.select(context)
            if (self.right_retry_phase == "reorientation_attempt"
                    and previous.skill.value == "move_to"
                    and previous.stop_reason in {"trajectory_limits", "planning_failure",
                                                  "tracking_error"}):
                self.right_retry_phase = "retry_restage_fallback"
                return self._right_retry_fallback_restage()
            if (self.right_retry_phase in {"retry_vla", "reorient"}
                    and previous.skill.value == "move_to"
                    and self._recoverable_right_retreat_failure(previous)):
                # A failed reverse retreat must never hand control back to the
                # VLA from a pose that may still be beside or over the mug.
                # Re-stage from the fresh measured state with the current wrist.
                self.right_retry_phase = "retry_restage_fallback"
                return self._right_retry_fallback_restage()
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
                        or not self._retryable_right_grasp_failure(
                            previous, observed_failure=right_lifted is False)):
                    return self._abort("right_grasp_unconfirmed")
                if not _retry_open_authorized(context, arm="right",
                        failed_predicate=right_lifted):
                    return self._abort("right_grasp_retry_open_not_authorized")
                self.right_grasp_retries += 1
                self.right_retry_phase = ("open_retreat_only" if self.right_grasp_retries == 1
                                          else "open_retreat_reorient")
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
            parameters=self._parameters(lock_arm=None), stop="task_success",
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
        if phase in {"open_retreat_only", "open_retreat_reorient"}:
            if previous is not None and previous.skill.value == "set_gripper":
                self.right_retry_phase = ("retry_vla" if phase == "open_retreat_only"
                                          else "reorient")
                retreat_distance_m = 0.05 if phase == "open_retreat_only" else 0.10
                return PlannerSelection(SkillCommand(action="move_to", arm="right",
                    parameters={"target": "current_ee", "path_mode": "reverse_approach",
                                "retreat_distance_m": retreat_distance_m,
                                "orientation_mode": "preserve_current",
                                "max_physics_ticks": self.RIGHT_RESTAGE_MAX_PHYSICS_TICKS},
                    reason="Retrace the measured right-arm approach path after opening the gripper"),
                    "RIGHT_RETRY_RETREAT", self.retry_count)
            else:
                return self._abort("right_grasp_retry_open_failed")
        if phase == "reorient":
            if previous is None or previous.skill.value != "move_to":
                return self._abort("right_grasp_retry_retreat_failed")
            self.right_retry_phase = "reorientation_attempt"
            return PlannerSelection(SkillCommand(action="move_to", arm="right",
                parameters={"target": "mug_pregrasp", "position_mode": "preserve_current",
                            "max_physics_ticks": self.RIGHT_RESTAGE_MAX_PHYSICS_TICKS},
                reason="Reorient the wrist after the measured retreat cleared the mug"),
                "RIGHT_RETRY_REORIENT", self.retry_count)
        if phase == "retry_restage_fallback":
            if (previous is None or previous.skill.value != "move_to"
                    or not previous.completed or previous.execution_uncertain):
                return self._abort("right_grasp_retry_fallback_failed")
            self.right_retry_phase = "await_retry"
            return self._right_vla_selection("RIGHT_GRASP_RETRY")
        if phase in {"retry_vla", "reorientation_attempt"}:
            if previous is None or previous.skill.value != "move_to":
                return self._abort("right_grasp_retry_staging_failed")
            self.right_retry_phase = "await_retry"
            return self._right_vla_selection("RIGHT_GRASP_RETRY")
        if phase == "await_retry":
            if context.value("right_mug_lifted") is True:
                self.right_retry_phase = None
                self.right_lift_confirmed = True
                return None
            if (self.right_grasp_retries < self.acquisition.settings.max_retries
                    and self._retryable_right_grasp_failure(
                        previous, observed_failure=context.value("right_mug_lifted") is False)
                    and context.value("release_ready") is True):
                self.right_grasp_retries += 1
                self.right_retry_phase = "open_retreat_reorient"
                return PlannerSelection(SkillCommand(action="set_gripper", arm="right",
                    parameters={"value": 1.0},
                    reason="Open before a second retry that re-grounds the full grasp pose"),
                    "RIGHT_RETRY_OPEN", self.retry_count)
            self.right_retry_phase = None
            return self._abort("right_grasp_retry_budget_exhausted")
        return self._abort("invalid_right_grasp_retry_state")

    def _right_retry_fallback_restage(self):
        """Recover from rejected wrist rotation using measured orientation."""
        return PlannerSelection(SkillCommand(action="move_to", arm="right",
            parameters={"target": "mug_pregrasp", "orientation_mode": "preserve_current",
                        "minimum_translation_m": 0.05,
                        "max_physics_ticks": self.RIGHT_RESTAGE_MAX_PHYSICS_TICKS},
            reason="Re-stage at the mug while preserving the measured wrist orientation"),
            "RIGHT_RETRY_RESTAGE_FALLBACK", self.retry_count)

    @staticmethod
    def _recoverable_right_retreat_failure(previous):
        """Only retry after a certain failure result supplies fresh measured state."""
        return any(check(previous) for check in (
            is_retryable_pre_motion_planning_failure,
            is_recoverable_executed_motion_failure,
            is_recoverable_partial_planning_failure,
            is_recoverable_trajectory_limit,
        ))

    @staticmethod
    def _retryable_right_grasp_failure(previous, *, observed_failure=False):
        return (previous is not None and previous.skill.value == "vla_act"
            and previous.completed and not previous.execution_uncertain
            and (previous.objective_satisfied is False
                 or (previous.objective_satisfied is None and observed_failure)
                 or (previous.stop_reason == "failure_predicate_satisfied"
                     and previous.objective_satisfied is None))
            and previous.stop_reason in {"failure_predicate_satisfied", "chunk_budget_exhausted",
                                         "policy_action_budget_exhausted"})

    def _right_vla_selection(self, phase):
        parameters = self._parameters(lock_arm=None).model_dump()
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
