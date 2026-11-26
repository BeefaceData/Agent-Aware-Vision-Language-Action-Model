"""One serial, bounded full-task supervisor episode with public hanging recovery."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from development.hanging.public_provider import BoundedVisionClient, PublicRGBDHangingProvider
from development.hanging.grounding_preflight import sanitized_failure
from development.hanging.runtime import recover_current, ProposalResult
from harness.jsonl import append_jsonl
from harness.perception.anchors import AnchorRegistry
from harness.predicates.registry import PredicateRegistry
from harness.runtime.orchestrator import SkillOrchestrator
from harness.skills.executor import SkillExecutor
from harness.skills.schemas import EpisodeBudget, SkillType, VLAParameters, SkillCommand
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.perception.vision_request import image_block, png_bytes


# "Not yet" geometry outcomes that a later frame may resolve. Rejected geometry
# still stops: the supervisor never retries until some frame happens to pass.
WAITABLE_GEOMETRY_REASONS = frozenset({
    "complete_visible_aperture_unavailable",
    "aperture_awaiting_two_frame_agreement",   # low-confidence opening, one frame so far
    "validated_rack_reference_unavailable",    # no peg kept yet; the rack is static
})


SUPERVISED_TIME_LIMIT_MAX = 110


@dataclass(frozen=True)
class SupervisedOptions:
    model: str
    seed: int = 4
    max_actions: int = 900
    chunk_size: int = 16
    time_limit_minutes: int = 30
    max_turns: int = 80
    complementary_view: bool = False
    perception_view: str = "default"
    # Number of frozen-policy actions allowed after confirmed lift. Zero means
    # immediate analytic takeover; nonzero values are retained for ablations.
    hand_back_to_policy: bool = True
    policy_grace_actions: int = 0
    # Public grasp-loss detection plus the same re-grasp sequence for both arms.
    grasp_recovery: bool = True
    grasp_retries: int = 2
    empty_hand_opening: float = .19  # calibrated: empty 0.18, held 0.25
    # Simulator-only fallback for controlled development experiments. Public
    # perception remains the default and is always attempted first.
    development_oracle_fallback: bool = False

    def validate(self):
        from development.hanging.baseline import validate_options
        validate_options(self.seed, self.max_actions, self.chunk_size, min(self.time_limit_minutes, 75))
        # Claude latency and supervisor motion make supervised runs slower than
        # policy-only ones; 75 minutes ended a 900-action run at 880 actions.
        if type(self.time_limit_minutes) is not int or not 1 <= self.time_limit_minutes <= SUPERVISED_TIME_LIMIT_MAX:
            raise ValueError("invalid_time_limit_minutes")
        if not isinstance(self.model, str) or not self.model.startswith("claude-") or "MODEL" in self.model:
            raise ValueError("explicit_claude_model_required")
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= 80:
            raise ValueError("invalid_supervisor_turn_limit")
        if type(self.complementary_view) is not bool:
            raise ValueError("invalid_complementary_view_option")
        from harness.environment.perception_cameras import PERCEPTION_VIEWS
        if self.perception_view not in PERCEPTION_VIEWS:
            raise ValueError("invalid_perception_view_option")
        if type(self.hand_back_to_policy) is not bool:
            raise ValueError("invalid_hand_back_option")
        if type(self.policy_grace_actions) is not int or not 0 <= self.policy_grace_actions <= self.max_actions:
            raise ValueError("invalid_policy_grace_actions")
        if type(self.grasp_recovery) is not bool:
            raise ValueError("invalid_grasp_recovery_option")
        if type(self.grasp_retries) is not int or not 0 <= self.grasp_retries <= 3:
            raise ValueError("invalid_grasp_retries")
        if (isinstance(self.empty_hand_opening, bool) or not isinstance(self.empty_hand_opening, (int, float))
                or not 0 <= self.empty_hand_opening < .5):
            raise ValueError("invalid_empty_hand_opening")
        if type(self.development_oracle_fallback) is not bool:
            raise ValueError("invalid_development_oracle_fallback")


class HangingRecoveryHook:
    """Take over at confirmed lift; validate geometry before the one motion attempt.

    The supervised runner defers takeover during a bounded aperture visibility
    gap. Explicit pause-and-block callers retain their existing behavior.
    """
    def __init__(self, environment, provider, store, *, episode_id, deadline, log,
                 policy_chunk_size=16, visibility_action_budget=900, visibility_check_budget=20,
                 pause_policy_on_entry=True, require_current_lift=False,
                 hand_back_to_policy=False, policy_grace_actions=0, grasp_recovery=None,
                 grasp_recoveries=None, development_oracle_targets=None,
                 development_success_confirmation=False):
        if type(pause_policy_on_entry) is not bool or type(hand_back_to_policy) is not bool:
            raise ValueError("invalid_hanging_takeover_mode")
        if type(policy_grace_actions) is not int or policy_grace_actions < 0:
            raise ValueError("invalid_hanging_policy_grace")
        if hand_back_to_policy and pause_policy_on_entry:
            raise ValueError("hand_back_requires_policy_owned_waiting")
        self.hand_back_to_policy = hand_back_to_policy
        self.policy_grace_actions = policy_grace_actions
        self.handed_back = None
        if grasp_recovery is not None and grasp_recoveries is not None:
            raise ValueError("duplicate_grasp_recovery_configuration")
        recoveries = (() if grasp_recovery is None else (grasp_recovery,))
        if grasp_recoveries is not None:
            recoveries = tuple(grasp_recoveries)
        self.grasp_recoveries = recoveries
        if (development_oracle_targets is not None
                and not callable(development_oracle_targets)):
            raise ValueError("invalid_development_oracle_target_provider")
        if type(development_success_confirmation) is not bool:
            raise ValueError("invalid_development_success_confirmation")
        self.development_oracle_targets = development_oracle_targets
        self.development_success_confirmation = development_success_confirmation
        if type(policy_chunk_size) is not int or policy_chunk_size <= 0:
            raise ValueError("invalid_hanging_policy_chunk_size")
        if (type(visibility_action_budget) is not int or visibility_action_budget <= 0
                or type(visibility_check_budget) is not int or visibility_check_budget <= 0):
            raise ValueError("invalid_hanging_visibility_budget")
        self.environment, self.provider, self.store = environment, provider, store
        self.episode_id, self.deadline, self.log = episode_id, deadline, log
        self.attempts = 0
        self.policy_chunk_size = policy_chunk_size
        self.visibility_action_budget = visibility_action_budget
        self.visibility_check_budget = visibility_check_budget
        self.entry_sequence = self.entry_policy_remaining = None
        self.visibility_checks = 0
        self.last_visibility_sequence = None
        self.next_visibility_action = 0
        self.visibility_retry_interval = policy_chunk_size
        self.pause_policy_on_entry = pause_policy_on_entry
        self.require_current_lift = require_current_lift or pause_policy_on_entry
        self._blocked_takeover = None

    def _block_takeover(self, observation, reason, **details):
        self._decision(observation, "hanging_controller_blocked", reason,
                       control_owner="hanging_controller", policy_paused=True)
        self._blocked_takeover = dict(status="recovery_blocked", reason=reason,
            stages={}, control_owner="hanging_controller", policy_paused=True,
            motion_authorized=False, observation_sequence=observation.sequence, **details)
        return self._blocked_takeover

    def _wait_for_visibility_check(self, observation, remaining, elapsed_actions):
        if time.monotonic() >= self.deadline:
            return dict(status="recovery_stopped", reason="supervisor_episode_deadline", stages={})
        if not all(remaining.get(name, 0) > 0 for name in ("policy_actions", "vla_chunks", "skill_calls")):
            return None  # The ordinary executor/orchestrator enforces its limits.
        size = min(self.policy_chunk_size, remaining["policy_actions"],
                   self.next_visibility_action-elapsed_actions)
        command = SkillCommand(action=SkillType.VLA_ACT, stop="chunk_budget_exhausted",
            parameters=dict(max_chunks=1, chunk_size=size, max_policy_actions=size))
        self._decision(observation, "continue_policy_and_reacquire", "visibility_retry_cooldown",
                       next_policy_actions=size, next_visibility_action=self.next_visibility_action)
        return dict(status="recovery_deferred", reason="visibility_retry_cooldown",
                    stages={}, policy_command=command.model_dump(mode="json"),
                    analytic_recovery_disabled=False)

    def _decision(self, observation, decision, reason, **evidence):
        # Missing geometry and robot motion alone cannot establish a VLA stall,
        # failed placement, or successful hanging. Keep that distinct from the
        # decision to use a guarded final-contact skill.
        self.log(dict(event_type="hanging_supervisor_decision", observation_sequence=observation.sequence,
            decision=decision, reason=reason, entry_sequence=self.entry_sequence,
            visibility_checks=self.visibility_checks, recovery_attempts=self.attempts,
            task_progress="unknown", failure_detected=None,
            progress_reason="validated_temporal_handle_to_peg_measurements_unavailable",
            **evidence))

    def _policy_turn(self, observation, remaining, reason):
        """One frozen-policy chunk while the policy keeps or regains control."""
        if time.monotonic() >= self.deadline or not all(
                remaining.get(name, 0) > 0 for name in ("policy_actions", "vla_chunks", "skill_calls")):
            return None  # The ordinary executor/orchestrator limits end the episode.
        size = min(self.policy_chunk_size, remaining["policy_actions"])
        command = SkillCommand(action=SkillType.VLA_ACT, stop="chunk_budget_exhausted",
            parameters=dict(max_chunks=1, chunk_size=size, max_policy_actions=size))
        return dict(status="recovery_deferred", reason=reason, stages={},
                    policy_command=command.model_dump(mode="json"),
                    observation_sequence=observation.sequence, control_owner="policy")

    def consider(self, selection, observation, *, remaining):
        """With ``hand_back_to_policy`` the supervisor never ends an episode on its own.

        The policy keeps control for ``policy_grace_actions`` after a confirmed
        lift. Any stop that happens before motion (geometry unavailable or
        rejected, lift evidence lost, no budget, or a planner abort) hands control
        back to the frozen policy for the rest of the episode, so supervision can
        only add an intervention, never cut a policy rollout short. An executed
        recovery attempt still ends the episode with its own result.
        """
        regrasp = self._consider_regrasp(observation, remaining)
        if regrasp is not None:
            return regrasp
        if not self.hand_back_to_policy:
            return self._consider(selection, observation, remaining=remaining)
        if self.handed_back is not None:
            return self._policy_turn(observation, remaining, "handed_back_to_policy")
        if selection.command.action == SkillType.ABORT:
            report = dict(status="recovery_stopped", reason="planner_abort_overridden", stages={})
        else:
            report = self._consider(selection, observation, remaining=remaining)
        if (report is None or report.get("status") in {"recovery_deferred", "task_success"}
                or report.get("stages")):
            return report
        self.handed_back = report.get("reason") or report.get("status")
        self._decision(observation, "hand_back_to_policy", self.handed_back, control_owner="policy",
                       supervisor_report=report.get("status"))
        return self._policy_turn(observation, remaining, "handed_back_to_policy")

    def _consider_regrasp(self, observation, remaining):
        """Re-grasp after a confirmed loss; the policy then resumes from scratch."""
        if not self.grasp_recoveries or self.attempts:
            return None  # An executed hanging attempt owns the rest of the episode.
        recovery = basis = result = None
        for candidate in self.grasp_recoveries:
            candidate_basis = candidate.check(observation)
            if candidate_basis is not None:
                recovery, basis = candidate, candidate_basis
                result = candidate.recover(observation, basis, remaining=remaining)
                break
        if recovery is None or result is None or result.get("status") != "grasp_recovery_completed":
            return None  # No loss, waiting for evidence, or recovery stopped: policy keeps control.
        # The hanging supervisor starts over for the new grasp.
        self.handed_back = self._blocked_takeover = None
        self.entry_sequence = self.entry_policy_remaining = None
        self.visibility_checks = self.next_visibility_action = 0
        self.last_visibility_sequence = None
        self._decision(observation, "regrasp_hand_back_to_policy", basis, control_owner="policy",
                       grasp_retry=result.get("attempt"), grasp_arm=getattr(recovery, "arm", None))
        turn = self._policy_turn(observation, remaining, "grasp_recovery_completed")
        if turn is not None:
            turn["refresh_observation"] = True  # The arm moved since the planner's frame.
        return turn

    def _consider(self, selection, observation, *, remaining):
        if selection.command.action == SkillType.ABORT:
            return None
        if self._blocked_takeover is not None:
            return self._blocked_takeover
        if selection.metadata.get("task_phase") != "hang" and selection.phase != "HANG":
            return None
        grasp = self.provider.evaluate("right_mug_lifted", observation, arm="right")
        if grasp.value is None and not self.require_current_lift:
            grasp = self.provider.evaluate("handoff_confirmed", observation, arm="right")
        self.log(dict(event_type="recovery_entry_check", observation_sequence=observation.sequence,
                      grasp=grasp.model_dump(mode="json"), attempts=self.attempts))
        current_lift = (grasp.value is True and (not self.require_current_lift or
            (grasp.name == "right_mug_lifted" and grasp.source == "perception"
             and grasp.observation_sequence == observation.sequence)))
        if current_lift and self.require_current_lift and self.entry_sequence is None:
            # The provider stamps every result with the current sequence and may
            # carry a lift forward across a bounded gap. Entry needs this frame's
            # own corroboration; providers without that check fail closed.
            lift_current = getattr(self.provider, "lift_current", None)
            current_lift = callable(lift_current) and lift_current(observation) is True
        if not current_lift:
            if self.entry_sequence is not None:
                reason = "right_grasp_lost" if grasp.value is False else "right_grasp_evidence_expired"
                self._decision(observation, "stop", reason)
                return dict(status="recovery_stopped", reason=reason, stages={})
            return None
        if self.entry_sequence is None:
            self.entry_sequence = observation.sequence
            self.entry_policy_remaining = remaining.get("policy_actions", 0)
            # Spread the finite crop budget over the remaining episode, using
            # whole policy chunks to avoid spending all requests near entry.
            span = min(self.visibility_action_budget, self.entry_policy_remaining)
            chunks = max(1, (span+self.visibility_check_budget*self.policy_chunk_size-1)
                         // (self.visibility_check_budget*self.policy_chunk_size))
            self.visibility_retry_interval = chunks*self.policy_chunk_size
        if self.attempts >= 1:
            return dict(status="recovery_attempt_budget_exhausted", stages={})
        if time.monotonic() >= self.deadline:
            return dict(status="recovery_stopped", reason="supervisor_episode_deadline", stages={})
        if self.pause_policy_on_entry:
            self._decision(observation, "take_over_hanging", "confirmed_right_mug_lifted",
                           control_owner="hanging_controller", policy_paused=True)
        elapsed_actions = max(0, self.entry_policy_remaining-remaining.get("policy_actions", 0))
        if elapsed_actions < self.policy_grace_actions:
            # The policy may finish the hang itself: step in only if it has not
            # after the grace period. No geometry request is spent meanwhile.
            self._decision(observation, "policy_grace_period", "policy_may_complete_hang",
                           control_owner="policy", policy_actions_since_entry=elapsed_actions,
                           policy_grace_actions=self.policy_grace_actions)
            return self._policy_turn(observation, remaining, "policy_grace_period")
        if elapsed_actions < self.next_visibility_action:
            return self._wait_for_visibility_check(observation, remaining, elapsed_actions)
        if self.visibility_checks >= self.visibility_check_budget or elapsed_actions >= self.visibility_action_budget:
            return dict(status="recovery_stopped", reason="visibility_budget_exhausted", stages={},
                        visibility_budget_exhausted=True)
        # Proposal acquisition is motion-free and cached for the same frame.
        # Do not spend the single execution attempt on an early visibility gap.
        try:
            measured_tcp = self.environment.arm_state("right")["pose"]
            proposal = self.provider.propose(observation, measured_tcp, episode_id=self.episode_id)
        except Exception as exc:
            if self.development_oracle_targets is not None:
                proposal = ProposalResult("unknown", "hanging_evidence_provider_error")
                self.log(dict(event_type="development_oracle_fallback_requested",
                              observation_sequence=observation.sequence,
                              public_reason=proposal.reason,
                              public_failure=sanitized_failure(exc)))
            elif self.pause_policy_on_entry:
                return self._block_takeover(observation, "hanging_evidence_provider_error",
                                            failure=sanitized_failure(exc))
            else:
                return dict(status="recovery_stopped", reason="hanging_evidence_provider_error",
                            failure=sanitized_failure(exc), stages={})
        if not isinstance(proposal, ProposalResult):
            return dict(status="recovery_stopped", reason="invalid_provider_result", stages={})
        if (self.pause_policy_on_entry and proposal.status == "unknown"
                and self.development_oracle_targets is None):
            return self._block_takeover(observation, proposal.reason, entry_status=proposal.status)
        if proposal.status != "confirmed" and self.development_oracle_targets is None:
            if observation.sequence != self.last_visibility_sequence:
                self.visibility_checks += 1
                self.last_visibility_sequence = observation.sequence
            can_wait = (proposal.reason in WAITABLE_GEOMETRY_REASONS
                and proposal.status == "unknown"
                and elapsed_actions < self.visibility_action_budget
                and (self.visibility_checks < self.visibility_check_budget
                     or remaining.get("policy_actions", 0) <= self.policy_chunk_size)
                and remaining.get("policy_actions", 0) > 0
                and remaining.get("vla_chunks", 0) > 0
                and remaining.get("skill_calls", 0) > 0)
            if can_wait:
                self.next_visibility_action = elapsed_actions+self.visibility_retry_interval
                size = min(self.policy_chunk_size, remaining["policy_actions"],
                           self.visibility_action_budget-elapsed_actions)
                command = SkillCommand(action=SkillType.VLA_ACT, stop="chunk_budget_exhausted",
                    parameters=dict(max_chunks=1, chunk_size=size, max_policy_actions=size))
                self._decision(observation, "continue_policy_and_reacquire", proposal.reason,
                    geometry_status=proposal.status, policy_actions_since_entry=elapsed_actions,
                    next_policy_actions=size)
                return dict(status="recovery_deferred", reason=proposal.reason, stages={},
                    policy_command=command.model_dump(mode="json"),
                    observation_sequence=observation.sequence,
                    visibility_action_budget=self.visibility_action_budget,
                    visibility_check_budget=self.visibility_check_budget,
                    visibility_retry_interval=self.visibility_retry_interval,
                    policy_actions_since_entry=elapsed_actions)
            self._decision(observation, "stop", proposal.reason, geometry_status=proposal.status,
                           policy_actions_since_entry=elapsed_actions)
            return dict(status="recovery_stopped", reason=proposal.reason, stages={},
                        entry_status=proposal.status,
                        visibility_budget_exhausted=(proposal.reason in WAITABLE_GEOMETRY_REASONS))
        if proposal.status != "confirmed":
            self._decision(observation, "use_development_oracle_fallback", proposal.reason,
                           geometry_status=proposal.status,
                           information_source="development_oracle")
        # Worst-case four stages plus settling/hold fit below this reservation.
        if remaining.get("analytic_ticks", 0) < 9000 or remaining.get("skill_calls", 0) < 4:
            if self.pause_policy_on_entry:
                return self._block_takeover(observation, "recovery_budget_unavailable")
            return dict(status="recovery_budget_unavailable", stages={})
        self.attempts += 1
        self._decision(observation, "grounded_contact_recovery", "current_geometry_available",
                       geometry_status=proposal.status,
                       intervention_basis="bounded_final_contact_skill_not_proven_vla_failure")
        report = recover_current(
            self.environment, self.store, episode_id=self.episode_id,
            deadline=min(self.deadline, time.monotonic()+600), provider=self.provider,
            initial_proposal=proposal,
            development_oracle_targets=self.development_oracle_targets,
            log=self.log)
        if self.development_success_confirmation and report.get("native_success") is True:
            report["status"] = "task_success"
            report["task_success_confirmation"] = "development_oracle_native_checker"
        return report


def service_preflight(options, client, policy_client):
    """Called only by the user-launched container; never starts a simulator."""
    options.validate()
    print("Preflight: checking policy interface and selected model image/tool compatibility", flush=True)
    policy_worker = policy_client.worker
    if not all(callable(getattr(getattr(policy_worker, name, None), "remote", None))
               for name in ("reset_episode", "act_chunk")):
        raise ValueError("policy_service_interface_unavailable")
    # One CPU image/tool request checks the selected model's actual API mode.
    # It checks compatibility, not perception quality or runtime geometry.
    response = client.messages.create(model=options.model, max_tokens=4096,
        system="Call report_compatibility once. This blank image has no robot geometry.",
        tools=[{"name": "report_compatibility", "description": "CPU image/tool protocol check",
                "input_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                 "required": ["ok"], "additionalProperties": False}}],
        messages=[{"role": "user", "content": [image_block(png_bytes(np.full((32, 32, 3), 128, np.uint8)))]}])
    calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
    if len(calls) != 1 or calls[0].name != "report_compatibility" or calls[0].input.get("ok") is not True:
        raise ValueError("vision_tool_compatibility_failed")
    info, _ = policy_client.reset()
    if not isinstance(info, dict) or info.get("ok") is not True:
        raise ValueError("policy_reset_protocol_failed")
    return dict(status="passed", simulator_started=False, requests=client.requests,
                runtime_geometry="unvalidated", native_success_is_not_public_support=True)


def build_supervised_planner(options, client):
    return AnthropicPlanner(AnthropicSettings(model=options.model, task_scope="full_task",
        max_calls=options.max_turns, max_tokens=4096, transport_retries=0,
        planner_feedback_chunks=1, vla=VLAParameters(max_chunks=1, chunk_size=options.chunk_size,
                                                    max_policy_actions=options.chunk_size)),
        client=client, preserve_policy_actions=True,
        runtime_grasp_monitor=getattr(options, "public_grasp_monitor", False))


def run_controller(environment, vla, provider, store, planner, options, *, episode_id, deadline, log,
                   development_oracle_targets=None, development_oracle_predicates=None):
    """Uses the existing supervisor; caller owns reset and recording resources."""
    anchors = AnchorRegistry()
    executor = SkillExecutor(environment, vla, log=log, anchors=anchors,
        budget=EpisodeBudget(max_skill_calls=options.max_turns+4, max_analytic_ticks=12000,
                             max_policy_actions=options.max_actions,
                             max_vla_chunks=(options.max_actions+options.chunk_size-1)//options.chunk_size),
        predicates=PredicateRegistry(
            perception=provider, fallback_oracle=development_oracle_predicates),
        cancelled=lambda: time.monotonic() >= deadline,
        allow_development_oracle=development_oracle_predicates is not None)
    def prepare(command, observation):
        # Acquisition and table handoff remain frozen-policy-owned. The hooked
        # bounded pipeline is the only analytic controller enabled by this runner.
        if command.action not in (SkillType.VLA_ACT, SkillType.ABORT):
            raise ValueError("analytic_action_outside_hanging_recovery")
    def invariant():
        if time.monotonic() >= deadline:
            raise TimeoutError("supervisor_episode_deadline")
    grasp = {}
    if getattr(options, "grasp_recovery", False):
        from development.hanging.grasp_recovery import GraspRecovery
        grasp = {arm: GraspRecovery(environment, executor, provider, log=log, arm=arm,
                                   empty_opening=options.empty_hand_opening,
                                   max_retries=options.grasp_retries)
                 for arm in ("left", "right")}
        provider.measured_hand_empty = lambda arm: (
            grasp[arm].hand_empty(arm) if arm in grasp else None)
    hook = HangingRecoveryHook(environment, provider, store, episode_id=episode_id,
                               deadline=deadline, log=log, policy_chunk_size=options.chunk_size,
                               visibility_action_budget=options.max_actions,
                               pause_policy_on_entry=False, require_current_lift=True,
                               hand_back_to_policy=options.hand_back_to_policy,
                               policy_grace_actions=options.policy_grace_actions,
                               grasp_recoveries=grasp.values(),
                               development_oracle_targets=development_oracle_targets,
                               development_success_confirmation=options.development_oracle_fallback)
    summary = SkillOrchestrator(executor, planner, log=log, save_observation=store.save,
        episode_id=episode_id, seed=options.seed, max_turns=options.max_turns,
        completion_predicate="task_success", prepare=prepare, invariant_check=invariant,
        hanging_recovery=hook).run()
    if grasp and isinstance(summary, dict):
        summary["grasp_recovery"] = {
            arm: recovery.summary() for arm, recovery in grasp.items()}
    return summary


def run(robotwin_root, output, options, *, fixture_loader=None):
    options.validate()
    output = Path(output).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Infrastructure redelivery must not overwrite or resume physical state.
        return dict(status="interrupted_attempt_exists", simulator_started=False,
                    automatic_resume=False, artifacts_preserved=True, output=str(output))
    def log(event):
        append_jsonl(output/"supervisor_events.jsonl", event)
        kind = event.get("event_type")
        if kind in {"planner_selection", "hanging_entry_assessment", "fresh_handle_confirmation",
                    "hanging_recovery_result", "public_grounding_failure"}:
            progress = {key: event[key] for key in ("event_type", "observation_sequence", "phase", "status", "reason")
                        if key in event}
            if "confirmation" in event:
                progress["depth_status"] = event["confirmation"].get("status")
            if "failure" in event:
                progress["failure"] = event["failure"]
            print(json.dumps(progress), flush=True)
    report = dict(status="preflight_failed", simulator_started=False, mode="public_supervised_hanging",
                  options=options.__dict__, public_support="unknown", recovery_attempts_max=1,
                  maximum_model_requests=161, maximum_perception_requests=80)
    video = simulator = planner = client = environment = store = shared = provider = None
    started = time.monotonic()
    deadline = started+options.time_limit_minutes*60
    try:
        from harness.environment.perception_cameras import supervised_camera_specs, describe
        camera_specs = supervised_camera_specs(complementary_view=options.complementary_view,
                                               view=options.perception_view)
        report["perception_cameras"] = [describe(spec) for spec in camera_specs]
        fixture_assets = fixture_loader(robotwin_root) if fixture_loader is not None else None
        if fixture_assets is not None:
            from development.hanging.public_provider import MUG_CLASS_CORRIDOR_RADIUS_M
            report["fixture_contract"] = dict(asset_class="039_mug", static_rack_cache=True,
                corridor_radius_m=MUG_CLASS_CORRIDOR_RADIUS_M,
                class_manifest_hash=fixture_assets.class_manifest_hash,
                current_registration="unknown; must be verified before motion")
        import anthropic
        from harness.vla.modal_client import ModalVLAClient, ModalPolicyClient
        client = anthropic.Anthropic(max_retries=0, timeout=45.)
        shared = BoundedVisionClient(client, model=options.model, deadline=deadline)
        policy_client = ModalPolicyClient("robotwin-policy", "PolicyWorker")
        report["preflight"] = service_preflight(options, shared, policy_client)
        if time.monotonic() >= deadline:
            raise TimeoutError("preflight_exceeded_episode_deadline")
        # All simulator imports/initialization follow API/service preflight.
        from development.simulator.local_simulator import LocalSimulator
        from development.demos.episode_video import EpisodeVideo, RecordingEnvironment
        from harness.environment.layout import CAMERAS
        from harness.perception.observations import ObservationStore

        store = ObservationStore(output/"observations")
        simulator = LocalSimulator(robotwin_root, rgbd_enabled=True,
            perception_cameras=camera_specs)
        report["simulator_started"] = True
        video = EpisodeVideo(output, cameras=(*CAMERAS, *(spec.name for spec in camera_specs)))
        class CapturedEnvironment(RecordingEnvironment):
            def _record_policy_step(self, observation):
                result = super()._record_policy_step(observation)
                store.save(result)
                return result
        environment = CapturedEnvironment(simulator, video, render_once_per_step=True)
        store.save(environment.reset(options.seed))
        provider = PublicRGBDHangingProvider(shared, model=options.model, episode_id=output.name,
                                            store=store, log=log, fixture=fixture_assets)
        planner = build_supervised_planner(options, shared)
        development_targets = development_predicates = None
        if options.development_oracle_fallback:
            from development.simulator.skill_oracle import oracle_hang_targets, oracle_predicates
            development_targets = lambda arm: oracle_hang_targets(simulator.task, arm)
            development_predicates = oracle_predicates(simulator.task, log=log)
        report["summary"] = run_controller(
            environment, ModalVLAClient(worker=policy_client), provider, store, planner,
            options, episode_id=output.name, deadline=deadline, log=log,
            development_oracle_targets=development_targets,
            development_oracle_predicates=development_predicates)
        report["status"] = report["summary"]["outcome"]
        report["final_observation"] = store.save(environment.observe())
        report["model_requests"] = shared.requests
        report["perception_requests"] = provider.requests
    except Exception as exc:
        report.update(status="run_failed" if report["simulator_started"] else "preflight_failed",
                      failure=sanitized_failure(exc))
    finally:
        # A Modal/client cancellation derives from BaseException and bypasses
        # the Exception handler above. Do not mislabel an interrupted physical
        # episode as a preflight failure merely because its terminal summary
        # was never returned.
        mark_interrupted_physical_run(report)
        if environment is not None and store is not None:
            try:
                report["final_observation"] = store.save(environment.observe())
            except Exception as exc:
                report["final_observation_error"] = type(exc).__name__
        for resource, name in ((video, "video"), (simulator, "simulator"), (client, "client")):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    report[name+"_close_error"] = type(exc).__name__
        if video is not None:
            try:
                report["video"] = video.metadata()
            except Exception as exc:
                report["video_metadata_error"] = type(exc).__name__
        if shared is not None:
            report["model_requests"] = shared.requests
            report["claude_usage"] = shared.usage_summary()
        if provider is not None:
            report["perception_requests"] = provider.requests
        report["elapsed_seconds"] = time.monotonic()-started
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def mark_interrupted_physical_run(report):
    """Correct the untouched default status after a BaseException cancellation."""
    if report["simulator_started"] and report["status"] == "preflight_failed":
        report.update(status="interrupted", interrupted=True)
    return report
