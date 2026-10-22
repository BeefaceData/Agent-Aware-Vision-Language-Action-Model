"""One-episode skill turns; additive to the legacy supervision runner.

No reset, model access, object geometry, or memory writes. A trusted caller may
refresh symbolic anchors before execution, outside the planner interface.
"""

import time

from harness.planner.base import PlannerContext
from harness.skills.schemas import (SkillCommand, SkillType,
                                    is_recoverable_executed_motion_failure,
                                    is_recoverable_arm_overlap,
                                    is_recoverable_predicate_unavailable,
                                    is_recoverable_release_support_gap,
                                    is_retryable_pre_motion_planning_failure)


def grounding_reselection_feedback(target):
    if target == "rack_approach":
        return ("The previous rack pixel failed a depth consistency check. Select a visibly solid "
                "point on the rack's hanger or support bar, away from the background; if no "
                "reliable point is visible, choose ABORT.")
    return ("The previous mug pixel failed a depth consistency check. Select a visibly solid "
            "point on the lower outer mug wall, away from the hollow opening; if no reliable "
            "point is visible, choose ABORT.")
MAX_GROUNDING_RESELECTIONS = 2
HANG_GROUNDING_REJECTIONS = frozenset({
    "rack_frame_not_cached", "rack_frame_camera_unavailable", "rack_frame_camera_changed",
    "grounding_report_stale_or_wrong_camera", "grounding_confidence_too_low",
    "grounding_mask_missing_or_misaligned", "grounding_keypoint_outside_object_mask",
    "keypoint_depth_discontinuity", "detection_mask_stale_observation",
    "detection_mask_camera_unavailable", "detection_mask_ambiguous_instances",
    "detection_mask_low_confidence", "detection_mask_not_detected",
    "hang_target_not_prepared", "mug_handle_report_required",
    "grounded_frame_report_required", "frame_grounding_camera_unavailable",
    "frame_grounding_requires_uint8_rgb", "frame_grounding_structured_result_missing",
    "frame_grounding_visibility_invalid", "frame_grounding_requires_three_keypoints",
    "frame_grounding_keypoints_invalid", "frame_grounding_keypoint_out_of_bounds",
    "frame_grounding_confidence_invalid", "hang_target_out_of_reach",
})
SYMBOLIC_GROUNDING_REJECTIONS = frozenset({
    "grounding_missing_or_stale", "grounded_target_unavailable",
    "grounded_target_out_of_reach",
})
SYMBOLIC_ANCHOR_TARGETS = frozenset({
    "mug_pregrasp", "rack_approach", "rack_seat", "table_approach",
})


def symbolic_grounding_reselection_feedback(target):
    return (f"The public RGB-D provider could not prepare the symbolic {target} target "
            "from this observation. Do not repeat that analytic move without fresh "
            "grounding; use another available skill or ABORT if no safe option remains.")


class SkillOrchestrator:
    def __init__(self, executor, planner, *, log, save_observation, episode_id,
                 seed, max_turns=20, prepare=None, completion_predicate="mug_lifted", memory=None,
                 invariant_check=lambda: None):
        if type(max_turns) is not int or not 1 <= max_turns <= 1000:
            raise ValueError("invalid_planner_turn_budget")
        self.executor, self.planner = executor, planner
        self.log, self.save_observation, self.prepare = log, save_observation, prepare
        self.episode_id, self.seed, self.max_turns = episode_id, seed, max_turns
        if completion_predicate not in ("mug_lifted", "left_transfer_complete",
                                        "handoff_confirmed", "task_success"):
            raise ValueError("unsupported_completion_predicate")
        self.completion_predicate = completion_predicate
        self.memory = memory
        self.invariant_check = invariant_check
        self._started = False

    def _predicates(self, observation):
        executor = self.executor
        # Acquisition accepts either arm because Neotix drives both. Opening
        # permission is checked by the skill again for its selected arm.
        names = ("mug_grasp_confirmed", "mug_lifted", "right_mug_lifted", "handoff_confirmed", "release_ready",
                 "mug_placed_on_table", "mug_staged_on_table",
                 "left_transfer_complete", "grasp_attempt_failed", "task_success")
        if getattr(executor.predicates, "perception", None) is not None:
            names += ("mug_near_rack", "rack_contact_detected", "mug_released")
        if getattr(executor.predicates, "rack_seat_gate", None) is not None:
            names += ("rack_seat_aligned",)
        return tuple(executor.predicates.evaluate(name, observation,
            native_success=executor.environment.native_success,
            allow_oracle=executor.allow_oracle) for name in
            names)

    def run(self):
        if self._started:
            raise RuntimeError("orchestrator_is_single_episode")
        self._started = True
        executor = self.executor
        started = time.perf_counter()
        previous = None
        retries = turns = 0
        grounding_feedback = None
        grounding_camera_to_avoid = None
        grounding_reselections = 0
        phase, outcome, error_type = "APPROACH_MUG", "planner_turn_budget_exhausted", None
        success = None
        milestone = False
        acquired = False
        terminal_phase, terminal_outcome = (("COMPLETE", "task_success")
            if self.completion_predicate == "task_success" else
            (("APPROACH_RACK", "handoff_confirmed") if self.completion_predicate == "handoff_confirmed"
             else (("RIGHT_PICK_AND_HANG", "left_transfer_complete")
                   if self.completion_predicate == "left_transfer_complete"
                   else ("TRANSFER", "acquisition_confirmed"))))
        try:
            for turn in range(1, self.max_turns + 1):
                self.invariant_check()
                observation = executor.environment.observe()
                success = executor.environment.native_success
                perception = getattr(executor.predicates, "perception", None)
                configure_phase = getattr(perception, "set_task_phase", None)
                if callable(configure_phase):
                    configure_phase(phase)
                predicates = self._predicates(observation)
                visual_context = (perception.planner_summary(observation)
                                  if perception is not None else None)
                acquired |= any(p.name == "mug_lifted" and p.value is True for p in predicates)
                if success:
                    phase, outcome = "COMPLETE", "task_success"
                    break
                if self.completion_predicate != "task_success" and any(
                        p.name == self.completion_predicate and p.value is True for p in predicates):
                    milestone = True
                    phase, outcome = terminal_phase, terminal_outcome
                    break
                before_ref = self.save_observation(observation)
                retrieved = self.memory.retrieve(task="hanging_mug", phase=phase) if self.memory else None
                tick = time.perf_counter()
                selection = self.planner.select(PlannerContext(turn, observation.copy(), predicates,
                    previous.model_copy(deep=True) if previous else None,
                    remaining_budget={"planner_turns": self.max_turns-turn+1,
                        "skill_calls": executor.budget.max_skill_calls-executor.calls,
                        "policy_actions": executor.budget.max_policy_actions-executor.executed_policy_actions,
                        "vla_chunks": executor.budget.max_vla_chunks-executor.vla_chunks_requested,
                        "analytic_ticks": executor.budget.max_analytic_ticks-executor.executed_physics_ticks,
                        "grounding_reselections": max(0, MAX_GROUNDING_RESELECTIONS
                                                      - max(0, grounding_reselections-1))},
                    memory=retrieved, grounding_feedback=grounding_feedback,
                    grounding_camera_to_avoid=grounding_camera_to_avoid,
                    visual_context=visual_context))
                # Feedback is a one-turn signal; it cannot become persistent planner memory.
                grounding_feedback = None
                camera_to_avoid_this_turn = grounding_camera_to_avoid
                grounding_camera_to_avoid = None
                command = SkillCommand.model_validate(selection.command.model_dump())
                latency = time.perf_counter() - tick
                phase, retries = selection.phase, selection.retry_count
                base = dict(schema_version=2, episode=self.episode_id, seed=self.seed,
                            planner_turn=turn, phase=phase, retry_count=retries)
                # Durable intent precedes anchor preparation and all motion.
                self.log({**base, "event_type": "planner_selection", "observation_before": before_ref,
                    "predicates_before": [p.model_dump(mode="json") for p in predicates],
                    "visual_context": visual_context,
                    "memory_retrieved": retrieved.references() if retrieved else [],
                    "memory_snapshot_sha256": self.memory.sha256 if self.memory else None,
                    "command": command.model_dump(mode="json"),
                    "planner_latency_seconds": latency, "planner_metadata": selection.metadata})
                rejection_code = None
                is_grounded_pixel_move = (command.action == SkillType.MOVE_TO
                    and command.parameters.target in ("mug_pregrasp", "rack_approach")
                    and command.parameters.grounding is not None)
                if (is_grounded_pixel_move and camera_to_avoid_this_turn is not None
                        and command.parameters.grounding.camera == camera_to_avoid_this_turn):
                    rejection_code = "grounding_camera_not_changed"
                else:
                    try:
                        prepared = self.prepare(command, observation) if self.prepare else None
                    except ValueError as exc:
                        hang_target_move = (command.action == SkillType.MOVE_TO
                            and command.parameters.target in ("rack_approach", "rack_seat"))
                        if str(exc) == "grounding_pixel_behind_local_surface" and is_grounded_pixel_move:
                            rejection_code = str(exc)
                        elif hang_target_move and str(exc) in HANG_GROUNDING_REJECTIONS:
                            rejection_code = "hang_target_grounding_unavailable"
                        elif (command.action == SkillType.MOVE_TO
                              and command.parameters.target in SYMBOLIC_ANCHOR_TARGETS
                              and str(exc) in SYMBOLIC_GROUNDING_REJECTIONS):
                            rejection_code = "symbolic_target_grounding_unavailable"
                        else:
                            raise
                if rejection_code is not None:
                    turns = turn
                    grounding_reselections += 1
                    can_reselect = (grounding_reselections <= MAX_GROUNDING_RESELECTIONS
                                    and turn < self.max_turns)
                    self.log({**base, "event_type": "planner_turn_result",
                        "observation_after": before_ref, "post_condition": [], "result": None,
                        "failure_type": None if can_reselect else "grounding_reselection_budget_exhausted",
                        "grounding_rejected": True, "skill_executed": False,
                        "grounding_rejection_code": rejection_code,
                        "grounding_reselections": grounding_reselections,
                        "task_success": success, "milestone_reached": False})
                    if can_reselect:
                        if (grounding_reselections >= 2
                                and command.parameters.grounding is not None):
                            grounding_camera_to_avoid = command.parameters.grounding.camera
                            object_name = ("the mug handle in the grounded rack view" if
                                command.parameters.target in ("rack_approach", "rack_seat") else
                                "lower outer mug wall, away from the hollow opening")
                            grounding_feedback = (
                                f"The last pixel failed depth validation. Select a visibly solid point "
                                f"on the {object_name} from a different camera view. Do not use the "
                                "camera named in grounding_camera_to_avoid; choose ABORT if no "
                                "suitable alternate view is available.")
                        else:
                            grounding_feedback = ("The grounded rack target could not be prepared from "
                                "the current mug-handle RGB-D evidence. Inspect the new views; retry a "
                                "bounded grounded target only if the handle is clear, otherwise ABORT."
                                if command.parameters.target in ("rack_approach", "rack_seat") else
                                grounding_reselection_feedback(command.parameters.target)
                                if command.parameters.grounding is not None else
                                symbolic_grounding_reselection_feedback(command.parameters.target))
                        continue
                    phase, outcome = "FAILED", "grounding_reselection_budget_exhausted"
                    break
                self.invariant_check()  # Recheck after the potentially slow provider/planner call.
                target_anchor = None
                if (command.action == SkillType.MOVE_TO
                        and command.parameters.target == "rack_seat"):
                    try:
                        target_anchor = executor.anchors.resolve(
                            "rack_seat", command.arm, observation)
                    except ValueError:
                        target_anchor = prepared
                previous = executor.execute(command)
                release_gate = getattr(executor.predicates, "rack_seat_gate", None)
                if release_gate is not None:
                    release_gate.record(command, previous, target_anchor)
                turns = turn
                success = previous.task_success
                after_ref, post = None, ()
                if not previous.execution_uncertain:
                    after = executor.environment.observe()
                    after_ref, post = self.save_observation(after), self._predicates(after)
                acquired |= (previous.completed and not previous.execution_uncertain
                             and any(p.name == "mug_lifted" and p.value is True for p in post))
                milestone = (previous.completed and not previous.execution_uncertain
                             and any(p.name == self.completion_predicate and p.value is True for p in post))
                call_satisfied = (command.action == SkillType.VLA_ACT
                    and command.stop == "chunk_budget_exhausted"
                    and previous.stop_reason == "chunk_budget_exhausted") or any(
                        p.name == command.stop and p.value is True for p in post)
                unconfirmed = {"mug_lifted": "acquisition_unconfirmed",
                    "mug_staged_on_table": "left_table_placement_unconfirmed",
                    "left_transfer_complete": "left_transfer_unconfirmed",
                    "right_mug_lifted": "right_grasp_unconfirmed",
                    "handoff_confirmed": "handoff_unconfirmed", "task_success": "task_success_unconfirmed"}
                failure = (command.reason if command.action == "abort" else
                           previous.stop_reason if not previous.completed else
                           unconfirmed.get(command.stop, "skill_stop_unconfirmed")
                           if command.action == "vla_act" and not call_satisfied else None)
                retryable_plan_rejection = is_retryable_pre_motion_planning_failure(previous)
                recoverable_motion_feedback = is_recoverable_executed_motion_failure(previous)
                recoverable_arm_contact = is_recoverable_arm_overlap(previous)
                recoverable_perception_gap = (
                    is_recoverable_predicate_unavailable(previous)
                    or is_recoverable_release_support_gap(previous))
                self.log({**base, "event_type": "planner_turn_result", "observation_after": after_ref,
                    "post_condition": [p.model_dump(mode="json") for p in post],
                    "result": previous.model_dump(mode="json"), "failure_type": failure,
                    "task_success": success, "milestone_reached": milestone,
                    "retryable_pre_motion_planning_failure": retryable_plan_rejection,
                    "recoverable_executed_motion_failure": recoverable_motion_feedback,
                    "recoverable_arm_overlap": recoverable_arm_contact,
                    "recoverable_perception_gap": recoverable_perception_gap})
                recoverable_failure = (retryable_plan_rejection or recoverable_motion_feedback
                                       or recoverable_arm_contact or recoverable_perception_gap)
                if previous.episode_terminal or (not previous.completed and not recoverable_failure):
                    if success:
                        outcome = "task_success"
                    elif previous.completed and executor.calls >= executor.budget.max_skill_calls:
                        outcome = "skill_call_budget_exhausted"
                    else:
                        outcome = failure or previous.stop_reason
                    phase = "COMPLETE" if success else "FAILED"
                    break
                if recoverable_failure:
                    # Known motion/contact/perception feedback returns fresh state
                    # to the planner; motion is never auto-replayed.
                    continue
                if milestone and self.completion_predicate != "task_success":
                    phase, outcome = terminal_phase, terminal_outcome
                    break
            else:
                phase = "FAILED"
        except Exception as exc:
            # Do not include provider exception text, credentials or raw observations.
            phase, outcome, error_type = "FAILED", "controller_error", type(exc).__name__
            success = None
            milestone = False
        finally:
            # A completed subtask or turn limit cannot accidentally resume this executor.
            executor.terminal = True
        summary = dict(schema_version=2, episode=self.episode_id, seed=self.seed,
            phase=phase, outcome=outcome, task_success=success, milestone_reached=milestone,
            planner_turns=turns, retries=retries, skill_calls=executor.calls,
            vla_chunks=executor.vla_chunks_requested, policy_actions=executor.executed_policy_actions,
            analytic_ticks=executor.executed_physics_ticks,
            development_oracle=executor.allow_oracle, elapsed_seconds=time.perf_counter()-started,
            completion_predicate=self.completion_predicate,
            grounding_reselections=grounding_reselections,
            memory_snapshot_sha256=self.memory.sha256 if self.memory else None,
            error_type=error_type)
        # Preserve achieved milestones even when a later phase fails, so a
        # placement/handoff failure does not erase a successful grasp.
        summary["acquisition_confirmed"] = acquired
        self.log({"event_type": "episode_summary", **summary})
        return summary
