"""Optional Anthropic selector. Never owns simulation, execution, or policy RPC."""

import os
import time
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from harness.schemas import Model
from harness.skills.schemas import (SkillCommand, VLAParameters,
                                    is_recoverable_move_feedback,
                                    is_recoverable_arm_overlap,
                                    is_recoverable_predicate_unavailable,
                                    is_recoverable_release_support_gap)
from .base import PlannerSelection
from .prompt_builder import prompt_bundle, user_content
from .response_parser import parse_planner_response, tool_schemas, InvalidPlannerResponse


class AnthropicSettings(Model):
    # Explicit model selection: no guessed alias or silent provider fallback.
    model: Annotated[StrictStr, Field(pattern=r"^claude-[A-Za-z0-9._-]+$", max_length=100)]
    task_scope: Literal["acquisition", "full_task"] = "acquisition"
    max_tokens: Annotated[StrictInt, Field(ge=128, le=4096)] = 1024
    timeout_seconds: Annotated[StrictFloat, Field(ge=1, le=60)] = 45.0
    # messages.create only returns a proposed bounded command; execution starts
    # after a valid response, so retrying transient transport errors cannot
    # replay robot motion.
    transport_retries: Annotated[StrictInt, Field(ge=0, le=5)] = 3
    max_calls: Annotated[StrictInt, Field(ge=1, le=100)] = 100
    # Compare bounded uncertainty with the existing episode-cap behavior.
    # Only consecutive VLA feedback turns with an unknown public phase
    # milestone count; a public false result is progress evidence, not dropout.
    unknown_milestone_policy: Literal["abort_after_limit", "continue_to_episode_cap"] = "continue_to_episode_cap"
    max_consecutive_unknown_vla_feedback: Annotated[StrictInt, Field(ge=1, le=100)] = 8
    # Return fresh camera frames to Opus after each bounded policy chunk. This
    # makes grasp assessment planner-driven rather than dependent on the
    # development-only temporal grasp oracle.
    planner_feedback_chunks: Annotated[StrictInt, Field(ge=1, le=4)] = 1
    # Historical phase traces bound each initial uninterrupted policy window;
    # after it, the configured planner-feedback chunk cadence applies.
    grasp_initial_feedback_after_actions: Annotated[StrictInt, Field(ge=0, le=256)] = 0
    handoff_initial_feedback_after_actions: Annotated[StrictInt, Field(ge=0, le=512)] = 0
    hang_initial_feedback_after_actions: Annotated[StrictInt, Field(ge=0, le=256)] = 0
    # Compatibility with older oracle-gated configs; this no longer controls
    # the Opus planner's per-chunk feedback path.
    early_failure_arm: Literal["left", "right"] = "left"
    vla: VLAParameters = Field(default_factory=lambda: VLAParameters(max_chunks=3, max_policy_actions=24))


class AnthropicPlanner:
    def __init__(self, settings, *, client=None, require_pixel_grounding=False):
        self.settings = AnthropicSettings.model_validate(settings.model_dump())
        if type(require_pixel_grounding) is not bool:
            raise ValueError("pixel_grounding_mode_must_be_boolean")
        self.require_pixel_grounding = require_pixel_grounding
        self.prompt, self.prompt_hash = prompt_bundle(self.settings.task_scope)
        self.calls = self.attempts = 0
        self._initial_policy_action_budget = None
        self._feedback_phase = None
        self._feedback_phase_start_action = 0
        self.inter_arm_recovery_active = False
        self.inter_arm_recovery_moves = 0
        self.inter_arm_contact_events = 0
        self.active_phase = "grasp"
        self.unknown_vla_feedback = 0
        self.recovery_basis = None
        self.history = []
        self.failed = False
        self._owned = client is None
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY_not_set")
            from anthropic import Anthropic
            # Only the SDK receives the key. No .env/shell reads or debug logging.
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                               base_url="https://api.anthropic.com",
                               max_retries=self.settings.transport_retries,
                               timeout=self.settings.timeout_seconds)
        self.client = client

    def select(self, context):
        metadata = {"provider": "anthropic", "model": self.settings.model,
                    "prompt_sha256": self.prompt_hash,
                    "memory_snapshot_sha256": context.memory.snapshot_sha256 if context.memory else None}
        if self.failed or self.calls >= self.settings.max_calls:
            return self._abort("planner_call_budget_or_terminal", metadata)
        if context.previous and context.previous.stop_reason == "arm_overlap_detected":
            if not self.inter_arm_recovery_active:
                self.inter_arm_contact_events += 1
                self.inter_arm_recovery_active = True
                self.inter_arm_recovery_moves = 0
        if self.inter_arm_recovery_active and context.previous:
            if context.previous.arm_overlap_detected is False:
                self.inter_arm_recovery_active = False
                self.inter_arm_recovery_moves = 0
            elif context.previous.skill.value == "move_to":
                self.inter_arm_recovery_moves += 1
            if self.inter_arm_contact_events > 3 or self.inter_arm_recovery_moves >= 2:
                return self._abort("inter_arm_contact_recovery_exhausted", metadata)
        if context.previous and (context.previous.episode_terminal or context.previous.execution_uncertain
                                 or (not context.previous.completed
                                     and not is_recoverable_move_feedback(context.previous)
                                     and not is_recoverable_arm_overlap(context.previous)
                                     and not is_recoverable_predicate_unavailable(context.previous)
                                     and not is_recoverable_release_support_gap(context.previous))):
            return self._abort("previous_skill_failed", metadata)
        vla_limits = self.settings.vla.model_copy(update={
            "max_chunks": min(self.settings.vla.max_chunks,
                               self.settings.planner_feedback_chunks)})
        phase_at_turn_start = self.active_phase
        public_phase = self._public_phase_floor(context, self.settings.task_scope)
        self.active_phase = self._later_phase(self.active_phase, public_phase)
        vla_phase = self.active_phase
        phase_changed = vla_phase != phase_at_turn_start
        if phase_changed:
            self.recovery_basis = None
            self.unknown_vla_feedback = 0
        unknown_milestones = self._unknown_phase_milestones(context, vla_phase)
        if not unknown_milestones:
            self.unknown_vla_feedback = 0
        elif (not phase_changed and context.previous is not None
                and context.previous.skill.value == "vla_act"):
            self.unknown_vla_feedback += 1
        metadata.update(unknown_milestone_policy=self.settings.unknown_milestone_policy,
                        unknown_milestones=list(unknown_milestones),
                        consecutive_unknown_vla_feedback=self.unknown_vla_feedback)
        if (self.settings.unknown_milestone_policy == "abort_after_limit"
                and self.unknown_vla_feedback >= self.settings.max_consecutive_unknown_vla_feedback):
            return self._abort("public_milestone_unknown_limit", metadata)
        policy_budget_now = context.remaining_budget.get("policy_actions")
        if self._initial_policy_action_budget is None and type(policy_budget_now) is int:
            self._initial_policy_action_budget = policy_budget_now
        executed_policy_actions = (max(0, self._initial_policy_action_budget-policy_budget_now)
            if self._initial_policy_action_budget is not None and type(policy_budget_now) is int else 0)
        if self._feedback_phase != vla_phase:
            self._feedback_phase = vla_phase
            self._feedback_phase_start_action = executed_policy_actions
        phase_policy_actions = max(0, executed_policy_actions-self._feedback_phase_start_action)
        initial_feedback_actions = getattr(
            self.settings, f"{vla_phase}_initial_feedback_after_actions")
        feedback_actions = vla_limits.max_chunks * vla_limits.chunk_size
        if (self.settings.task_scope == "full_task"
                and initial_feedback_actions > phase_policy_actions):
            until_first_feedback = initial_feedback_actions-phase_policy_actions
            vla_limits = vla_limits.model_copy(update={"max_chunks": min(
                self.settings.vla.max_chunks,
                max(vla_limits.max_chunks, (until_first_feedback + vla_limits.chunk_size-1)
                    // vla_limits.chunk_size))})
        feedback_actions = vla_limits.max_chunks * vla_limits.chunk_size
        vla_limits = vla_limits.model_copy(update={"max_policy_actions": min(
            vla_limits.max_policy_actions, feedback_actions)})
        feedback_actions = vla_limits.max_policy_actions
        metadata.update(executed_policy_actions=executed_policy_actions,
                        phase_policy_actions=phase_policy_actions,
                        planner_feedback_horizon_actions=feedback_actions)
        phase_stops = self._vla_stop_options(
            vla_phase, self.settings.task_scope, self._public_predicate_names(context))
        vla_available = (context.remaining_budget.get("vla_chunks", 1) > 0
                         and context.remaining_budget.get("policy_actions", 1) > 0)
        recovery_basis = self._recovery_basis(context, vla_phase)
        analytic_skills_available = recovery_basis is not None
        limits = dict(vla={name: getattr(vla_limits, name) for name in
                           ("max_chunks", "chunk_size", "max_policy_actions")},
                      task_scope=self.settings.task_scope,
                      vla_phase=vla_phase,
                      vla_stop_predicates=list(phase_stops),
                      vla_chunks_remaining=context.remaining_budget.get("vla_chunks"),
                      policy_actions_remaining=context.remaining_budget.get("policy_actions"),
                      planner_calls_remaining=self.settings.max_calls-self.calls)
        limits["unknown_milestone_policy"] = self.settings.unknown_milestone_policy
        limits["unknown_milestones"] = list(unknown_milestones)
        limits["consecutive_unknown_vla_feedback"] = self.unknown_vla_feedback
        limits["max_consecutive_unknown_vla_feedback"] = self.settings.max_consecutive_unknown_vla_feedback
        limits["analytic_skills_available"] = analytic_skills_available
        limits["recovery_basis"] = recovery_basis
        limits["inter_arm_recovery_active"] = self.inter_arm_recovery_active
        limits["inter_arm_recovery_moves_remaining"] = max(0, 2-self.inter_arm_recovery_moves)
        perception_gap_vla_continuation = self._perception_gap_vla_continuation(
            context, vla_phase, vla_available)
        limits["perception_gap_vla_continuation"] = perception_gap_vla_continuation
        self.calls += 1  # Failed requests also consume a call.
        tick = time.perf_counter()
        try:
            response = self.client.messages.create(model=self.settings.model,
                max_tokens=self.settings.max_tokens, system=self.prompt,
                tools=[tool for tool in tool_schemas(
                           vla_limits, task_scope=self.settings.task_scope,
                           vla_stop_predicates=phase_stops,
                           release_authorized=self._public_predicate_true(context, "release_ready"),
                           rack_seat_authorized=(vla_phase == "hang" and
                               self._public_predicate_true(context, "rack_seat_aligned")))
                       if (vla_available or tool["name"] != "vla_act")
                       and (analytic_skills_available or tool["name"] in {"vla_act", "abort"})
                       and (not self.inter_arm_recovery_active
                            or tool["name"] in {"move_to", "abort"})],
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=[{"role": "user", "content": user_content(
                    context, limits=limits,
                    # A pixel rejected by trusted depth validation is neither a
                    # useful precedent nor a coordinate to repeat on reselection.
                    history=(self.history[:-1] if context.grounding_feedback else self.history[-8:]),
                    pixel_grounding=self.require_pixel_grounding)}])
            # Retain bounded diagnostics even when parsing rejects this response.
            # No raw content, thinking, error body or tool arguments are logged.
            stop_reason = response.get("stop_reason") if isinstance(response, dict) else response.stop_reason
            metadata["provider_stop_reason"] = (stop_reason if isinstance(stop_reason, str) and stop_reason in
                {"tool_use", "max_tokens", "end_turn", "stop_sequence", "pause_turn", "refusal"} else "other")
            usage = response.get("usage", {}) if isinstance(response, dict) else response.usage.model_dump()
            if isinstance(usage, dict):
                metadata["usage"] = {key: usage[key] for key in ("input_tokens", "output_tokens")
                                     if type(usage.get(key)) is int and usage[key] >= 0}
            parsed = parse_planner_response(response)
            command = parsed.command
            if perception_gap_vla_continuation:
                # When public perception loses the scene during a VLA-owned
                # grasp/transfer/hang phase, continue the frozen policy one bounded
                # chunk at a time. Do not let an ungrounded analytic action or
                # an Opus abort turn perception loss into an early stop.
                remaining_actions = context.remaining_budget.get("policy_actions")
                chunk_actions = min(vla_limits.chunk_size, vla_limits.max_policy_actions)
                if type(remaining_actions) is int:
                    chunk_actions = min(chunk_actions, remaining_actions)
                forced_limits = vla_limits.model_copy(update={
                    "max_chunks": 1,
                    "max_policy_actions": chunk_actions,
                })
                command = SkillCommand(action="vla_act", arm="both",
                    parameters=forced_limits, stop="chunk_budget_exhausted",
                    reason="Public scene perception is unavailable during a VLA-owned phase; continue one bounded chunk and reassess.")
                metadata["perception_gap_vla_continuation_applied"] = True
                metadata["perception_gap_chunk_actions"] = chunk_actions
            if (not analytic_skills_available
                    and command.action.value not in {"vla_act", "abort"}):
                raise InvalidPlannerResponse("analytic_skill_not_recovery_eligible")
            release_authorized = self._public_predicate_true(context, "release_ready")
            rack_seat_authorized = (vla_phase == "hang" and
                self._public_predicate_true(context, "rack_seat_aligned"))
            release_allowed = (release_authorized or
                (command.action.value == "release"
                 and command.parameters.authorization == "rack_seat_alignment"
                 and rack_seat_authorized))
            if (not release_allowed and
                    (command.action.value == "release"
                     or (command.action.value == "set_gripper" and command.parameters.value > 0.0))):
                raise InvalidPlannerResponse("release_not_publicly_authorized")
            if (self.inter_arm_recovery_active and context.previous
                    and context.previous.arm_overlap_detected is not False
                    and command.action.value != "abort"):
                if (command.action.value != "move_to" or command.arm != "right"
                        or command.parameters.target != "current_ee"):
                    raise InvalidPlannerResponse("inter_arm_recovery_requires_right_retreat")
                offset_norm = sum(value * value for value in command.parameters.offset_xyz) ** 0.5
                if not 0.03 <= offset_norm <= 0.10:
                    raise InvalidPlannerResponse("inter_arm_retreat_outside_bounds")
            if parsed.reason_truncated:
                metadata["reason_truncated_to_schema_limit"] = True
            claimed_phase = parsed.task_phase
            if self.settings.task_scope == "acquisition" and claimed_phase != "grasp":
                raise InvalidPlannerResponse("phase_outside_task_scope")
            phase_before = phase_at_turn_start
            public_phase = self._public_phase_floor(context, self.settings.task_scope)
            # No action has run before the first planner turn, so the task
            # cannot already be in handoff/hang absent a public milestone.
            # Later turns may use Opus's visual assessment to advance one
            # phase when perception predicates are unavailable.
            selected_phase = (self._later_phase(phase_before, public_phase)
                              if context.previous is None else
                              self._resolve_phase(phase_before, public_phase, claimed_phase))
            if (self.settings.task_scope == "full_task" and selected_phase == "hang"
                    and phase_before != "hang"
                    and not self._handoff_complete_confirmed(context)):
                # Require public transfer evidence for the first transition,
                # then retain the monotonic phase through later occlusion.
                selected_phase = "handoff"
            phase_correction = selected_phase != claimed_phase
            self.active_phase = selected_phase
            vla_phase = selected_phase
            phase_stops = self._vla_stop_options(
                vla_phase, self.settings.task_scope, self._public_predicate_names(context))
            if command.action == "move_to" and not self._target_available(
                    command.parameters.target, context, arm=command.arm, phase=vla_phase):
                raise InvalidPlannerResponse("unsupported_anchor")
            visual_context = context.visual_context or {}
            static_rack = visual_context.get("rack_static_reference", {})
            static_rack_move = (command.action.value == "move_to"
                                and command.parameters.target == "rack_approach"
                                and vla_phase == "hang"
                                and static_rack.get("world_anchor_available") is True)
            if (self.require_pixel_grounding and command.action == "move_to"
                    and command.parameters.target in ("mug_pregrasp", "rack_approach")
                    and command.parameters.grounding is None and not static_rack_move):
                raise InvalidPlannerResponse("missing_grounding_point")
            if command.action == "vla_act":
                metadata.update(allowed_vla_stops=list(phase_stops),
                                received_vla_stop=command.stop,
                                effective_vla_phase=vla_phase)
                if command.stop not in phase_stops or any(getattr(command.parameters, key) > getattr(vla_limits, key)
                        for key in ("max_chunks", "chunk_size", "max_policy_actions")):
                    raise InvalidPlannerResponse("vla_call_outside_task_phase_limits")
                if command.parameters.lock_arm is not None or command.parameters.failure_stop is not None:
                    raise InvalidPlannerResponse("vla_internal_control_not_selectable")
                # End each planner-selected VLA segment at the feedback
                # boundary. The next planner turn sees the resulting RGB
                # observations and decides whether this looks like progress,
                # a retryable miss, or a reason to use bounded analytic skills.
                # No simulator-only grasp-failure predicate is injected here.
                runtime_fields = {"max_chunks": vla_limits.max_chunks,
                                  "chunk_size": vla_limits.chunk_size,
                                  "max_policy_actions": vla_limits.max_policy_actions,
                                  "failure_arm": "left"}
                if self.settings.task_scope == "full_task" and vla_phase == "grasp":
                    # Preserve the task sequence during initial left-arm pickup.
                    # Handoff and hanging remain fully coordinated by the VLA.
                    runtime_fields["lock_arm"] = "right"
                parameters = command.parameters.model_copy(update=runtime_fields)
                command = command.model_copy(update={"parameters": parameters})
                self.attempts += 1
        except Exception as exc:
            if isinstance(exc, InvalidPlannerResponse):
                metadata["validation_error"] = exc.code
                metadata["validation_issues"] = exc.issues
            metadata.update(error_type=type(exc).__name__, provider_call=self.calls,
                            elapsed_seconds=time.perf_counter()-tick)
            return self._abort("planner_request_or_validation_failed", metadata)
        metadata.update(provider_call=self.calls, elapsed_seconds=time.perf_counter()-tick)
        metadata.update(task_phase=vla_phase,
            phase_basis=("public_milestone" if public_phase != phase_before else "planner_assessment"),
            phase_claim_corrected=phase_correction,
            phase_evidence=parsed.phase_evidence,
            phase_evidence_chars=len(parsed.phase_evidence))
        self.history.append(command.model_dump(mode="json"))
        self.history = self.history[-8:]
        # Keep phase attribution stable across VLA and analytic selections so
        # logs and task-memory retrieval describe the same active task phase.
        phase = "FAILED" if command.action == "abort" else vla_phase.upper()
        if command.action == "abort":
            self.failed = True
        elif command.action == "vla_act":
            # A recovery sequence ends when the planner hands control back to
            # the frozen policy. Another analytic sequence needs fresh failure
            # evidence, except for the known weak final hang phase.
            self.recovery_basis = None
        return PlannerSelection(command, phase, max(0, self.attempts-1), metadata)

    def _abort(self, reason, metadata):
        self.failed = True
        return PlannerSelection(SkillCommand(action="abort", reason=reason),
                                "FAILED", max(0, self.attempts-1), metadata)

    def _perception_gap_vla_continuation(self, context, phase, vla_available):
        """Continue one VLA chunk when perception is fully lost in a VLA phase."""
        if (self.settings.unknown_milestone_policy != "continue_to_episode_cap"
                or phase not in ("grasp", "handoff", "hang")
                or (self.settings.task_scope == "acquisition" and phase != "grasp")
                or not vla_available
                or self.inter_arm_recovery_active
                or self._public_predicate_true(context, "task_success")):
            return False
        previous = context.previous
        if previous is not None and previous.skill.value != "vla_act":
            # A non-VLA motion failure needs its own recovery assessment.
            return False
        visual_context = context.visual_context
        status = visual_context.get("status") if isinstance(visual_context, dict) else None
        if not isinstance(status, dict):
            return False
        required = ("mug", "rack") if phase == "hang" else ("mug", "table")
        return all(status.get(name) in ("not_detected", "unknown") for name in required)

    @staticmethod
    def _public_phase_floor(context, task_scope="acquisition"):
        """Return the minimum phase proven by public milestones; ignore oracle values."""
        if task_scope == "acquisition":
            return "grasp"
        public = {item.name: item.value for item in context.predicates
                  if item.source in ("perception", "runtime") and item.value is True}
        if (public.get("handoff_confirmed") or public.get("right_mug_lifted")
                or public.get("task_success")):
            return "hang"
        if (public.get("mug_lifted") or public.get("mug_staged_on_table")
                or public.get("left_transfer_complete")):
            return "handoff"
        return "grasp"

    @staticmethod
    def _later_phase(left, right):
        order = {"grasp": 0, "handoff": 1, "hang": 2}
        return left if order[left] >= order[right] else right

    @classmethod
    def _resolve_phase(cls, current, public_floor, claimed):
        """Apply public evidence and allow only one planner-inferred step per turn."""
        order = {"grasp": 0, "handoff": 1, "hang": 2}
        minimum = cls._later_phase(current, public_floor)
        maximum_rank = max(order[minimum], order[current] + 1)
        claimed_rank = min(order[claimed], maximum_rank)
        return ("grasp", "handoff", "hang")[max(order[minimum], claimed_rank)]

    @classmethod
    def _vla_phase(cls, context, task_scope="acquisition"):
        """Compatibility helper for callers needing the public-evidence floor."""
        return cls._public_phase_floor(context, task_scope)

    @staticmethod
    def _vla_stop_options(phase, task_scope, public_predicates=()):
        """Offer only stop conditions whose evidence is public to the planner."""
        stops = ["chunk_budget_exhausted"]
        if "mug_lifted" in public_predicates and phase in ("grasp", "handoff"):
            stops.insert(0, "mug_lifted")
        if task_scope == "full_task" and phase in ("handoff", "hang"):
            stops.append("task_success")
        return tuple(stops)

    @staticmethod
    def _public_predicate_names(context):
        # Development-oracle and unavailable results are never planner evidence,
        # even though a local runtime may use them for internal safety gates.
        return frozenset(item.name for item in context.predicates
                         if item.source in ("perception", "runtime"))

    @staticmethod
    def _public_predicate_true(context, name):
        return any(item.name == name and item.value is True
                   and item.source in ("perception", "runtime")
                   for item in context.predicates)

    @staticmethod
    def _public_predicate_value(context, name):
        return next((item.value for item in context.predicates
                     if item.name == name and item.source in ("perception", "runtime")), None)

    @classmethod
    def _unknown_phase_milestones(cls, context, phase):
        if phase == "grasp":
            return (("mug_lifted",) if cls._public_predicate_value(context, "mug_lifted") is None
                    else ())
        if phase == "handoff":
            if (cls._public_predicate_true(context, "handoff_confirmed")
                    or cls._public_predicate_true(context, "right_mug_lifted")):
                return ()
            return tuple(name for name in ("handoff_confirmed", "right_mug_lifted")
                         if cls._public_predicate_value(context, name) is None)
        return ()

    @classmethod
    def _handoff_complete_confirmed(cls, context):
        # Table staging and left-arm transfer only prove that the cup reached
        # the handoff point. HANG requires public evidence that the right arm
        # actually acquired it.
        return (cls._public_predicate_true(context, "handoff_confirmed")
                or cls._public_predicate_true(context, "right_mug_lifted"))

    def _recovery_basis(self, context, phase):
        """Offer analytic skills after a failure, or as an optional hang fallback."""
        basis = None
        if self.inter_arm_recovery_active:
            basis = "inter_arm_contact"
        elif self._public_predicate_true(context, "rack_seat_aligned"):
            basis = "rack_seat_alignment"
        elif (self.settings.task_scope == "full_task" and phase == "hang"
                and self._handoff_complete_confirmed(context)):
            basis = "hang_phase_recovery_available"
        elif self._public_predicate_true(context, "grasp_attempt_failed"):
            basis = "public_grasp_failure"
        else:
            previous = context.previous
            if previous is not None:
                if is_recoverable_arm_overlap(previous):
                    basis = "inter_arm_contact"
                elif is_recoverable_move_feedback(previous):
                    basis = "recoverable_motion_failure"
                elif is_recoverable_release_support_gap(previous):
                    basis = "release_support_lost"
            if basis is None and context.grounding_feedback:
                basis = "grounding_failure_feedback"
        if basis not in (None, "hang_phase_recovery_available"):
            self.recovery_basis = basis
        return basis or self.recovery_basis

    def _target_available(self, target, context, *, arm="left", phase=None):
        if target == "current_ee":
            return True
        if self.settings.task_scope == "acquisition":
            return target == "mug_pregrasp"
        visual = context.visual_context or {}
        status = visual.get("status", {})
        rack_reference = visual.get("rack_static_reference", {})
        cached_rack_target = (phase == "hang"
                              and rack_reference.get("world_anchor_available") is True)
        available = {"rack_approach": (status.get("rack") == "grounded"
                                       or self.require_pixel_grounding
                                       or cached_rack_target),
                     "rack_seat": cached_rack_target,
                     "table_approach": status.get("table") == "grounded",
                     "mug_pregrasp": (status.get("mug") == "grounded"
                                       or self.require_pixel_grounding)}
        return bool(available.get(target, False))

    def close(self):
        if self._owned:
            self.client.close()
