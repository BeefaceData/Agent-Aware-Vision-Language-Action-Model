"""Identity PASS control, explicit ABORT and verified-skill infrastructure.

There is no physical skill execution, policy reset, simulator access,
or provider client here. Only the runner can report evaluator success, and that
value is written to a separate evaluation log, never to public trace context.
"""

from contextlib import nullcontext
from pathlib import Path
import time
import uuid
import warnings

from harness.legacy.assessment_log import AssessmentLogger
from harness.jsonl import append_jsonl
from harness.legacy.config import HarnessConfig
from harness.legacy.schemas import AssessmentRecord, DecisionType, FailureType, SupervisorDecision, TaskPhase
from harness.schemas import StateSummary
from harness.legacy.trace import TraceManager, TraceWriter
from harness.environment.base import vector14
from harness.legacy.safety import SafetyValidator
from harness.legacy.skill_registry import VerifiedSkillRegistry
from harness.legacy.recovery_skills import candidate_skills
from harness.legacy.recovery_skills.abort_episode import abort_decision


class PassThroughHarness:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.registry = (
            VerifiedSkillRegistry.load(config.skills.registry, candidate_skills(), config.skills)
            if config.harness.enabled and config.skills.registry is not None
            else VerifiedSkillRegistry(config.skills)
        )
        self.run_id = uuid.uuid4().hex
        self._episode_index = 0
        self._active = False
        self.trace_dir = Path(config.logging.trace_dir).resolve() / self.run_id
        self.assessments = AssessmentLogger(
            Path(config.logging.assessment_dir).resolve() / f"{self.run_id}.jsonl"
        )
        self.evaluation_path = (
            Path(config.logging.evaluation_dir).resolve() / f"{self.run_id}.jsonl"
        )
        if config.harness.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=False)
            append_jsonl(self.trace_dir / "manifest.jsonl", {
                "schema_version": 1, "run_id": self.run_id,
                "timestamp": time.time(), "config": config.model_dump(mode="json"),
                "controller": "frozen_policy", "supervisor_mode": "pass_through",
                "state_semantics": "arm_drive_targets_and_gripper_commands",
                "step_semantics": "one_policy_target_call_not_one_physics_tick",
            })

    def episode(self, *, seed: int, max_steps: int, chunk_size: int):
        if not self.config.harness.enabled:
            return nullcontext(None)
        if self._active:
            raise RuntimeError("harness already has an active episode")
        episode_id = f"episode_{self._episode_index:04d}_seed_{seed}"
        self._episode_index += 1
        return PassThroughEpisode(self, episode_id, seed, max_steps, chunk_size)


class PassThroughEpisode:
    def __init__(self, harness, episode_id, seed, max_steps, chunk_size):
        self.harness = harness
        self.episode_id = episode_id
        self.seed = seed
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.trace = TraceManager(harness.config.trace)
        self.trace.reset(episode_id)
        self.writer = TraceWriter(harness.trace_dir / episode_id)
        self._pending = None
        self._executed_steps = 0
        self._entered = False
        self._closed = False
        self._terminal = None
        self._abort_reason = None
        self.safety = SafetyValidator(
            harness.registry, harness.config.safety, harness.config.supervisor.minimum_confidence,
        )

    @property
    def aborted(self):
        return self._terminal is not None and self._terminal["reason"] == "abort"

    def request_abort(self, reason: str):
        """Trusted local control request, applied before the next target action.

        Stopping has no motion skill and needs no recovery verification.
        The runner checks aborted before posting any action or requesting a new chunk.
        """
        self._require_active()
        self._abort_reason = abort_decision(reason).reason

    def __enter__(self):
        if self._entered or self.harness._active:
            raise RuntimeError("episode cannot be entered twice or concurrently")
        self.writer.append({
            "schema_version": 1, "event_type": "episode_start",
            "episode_id": self.episode_id, "run_id": self.harness.run_id,
            "timestamp": time.time(), "seed": self.seed,
            "max_steps": self.max_steps, "requested_chunk_size": self.chunk_size,
            "task": self.harness.config.task,
        })
        self._entered = True
        self.harness._active = True
        return self

    def _require_active(self):
        if not self._entered or self._closed or self._terminal is not None:
            raise RuntimeError("episode is not accepting actions")

    def before_action(self, step, images, state, action, *, chunk_id, chunk_pos, chunk_size=None):
        self._require_active()
        if self._pending is not None or step != self._executed_steps:
            raise RuntimeError("proposal must follow the previous confirmed execution")
        if chunk_size is not None and (type(chunk_size) is not int or not 0 <= chunk_pos < chunk_size):
            raise ValueError("invalid chunk size or position")
        decision = SupervisorDecision(
            decision=DecisionType.PASS, phase=TaskPhase.UNKNOWN,
            failure_type=FailureType.NONE, confidence=1.0,
            reason="Configured PASS-only mode; no failure assessment performed.",
        )
        if self._abort_reason is not None:
            decision = abort_decision(self._abort_reason)
        checked = self.safety.reserve(decision, current_phase=TaskPhase.UNKNOWN)
        decision = checked.decision
        record = AssessmentRecord(
            run_id=self.harness.run_id, episode_id=self.episode_id,
            event_id=f"{self.episode_id}_step_{step}", event_type="decision",
            step=step, chunk_id=chunk_id, chunk_pos=chunk_pos, timestamp=time.time(),
            phase=decision.phase, failure_type=decision.failure_type,
            supervisor_confidence=decision.confidence, evidence=decision.evidence,
            policy_action=vector14(action), decision=decision.decision,
            pre_state_summary=StateSummary(command_state=vector14(state)),
            notes=decision.reason,
        )
        frame = self.trace.propose(step, images, state, action, decision)
        self._pending = record
        frame_path = self.writer.save_keyframe(frame) if frame is not None else None
        self.harness.assessments.append(record)
        self.writer.append({
            **record.model_dump(mode="json"), "keyframe_path": frame_path,
        })
        if decision.decision == DecisionType.ABORT:
            self.writer.append({
                "schema_version": 1, "event_type": "chunk_discarded",
                "run_id": self.harness.run_id, "episode_id": self.episode_id,
                "event_id": record.event_id, "step": step, "chunk_id": chunk_id,
                "chunk_pos": chunk_pos, "timestamp": time.time(),
                "discarded_actions": None if chunk_size is None else chunk_size - chunk_pos,
            })
            self._pending = None  # Proposal is deliberately not executed.
            self._terminal = {"success": False, "reason": "abort"}
            return None
        # Preserve the original object, dtype, values and queue ownership.
        return action

    def after_action(self, step, action, post_state):
        self._require_active()
        if self._pending is None or self._pending.step != step:
            raise RuntimeError("execution has no matching proposal")
        executed = vector14(action)
        if executed != self._pending.policy_action:
            raise RuntimeError("PASS-only execution differs from the policy proposal")
        record = AssessmentRecord.model_validate({
            **self._pending.model_dump(), "event_type": "action_result",
            "timestamp": time.time(), "executed_action": executed,
            "post_state_summary": StateSummary(command_state=vector14(post_state)),
            "outcome": "unknown",
            "notes": "Execution confirmed by simulator response; no intervention to grade.",
        })
        self.trace.executed(step, action, post_state)
        self._executed_steps += 1
        self.harness.assessments.append(record)
        self.writer.append(record.model_dump(mode="json"))
        self._pending = None

    def complete(self, *, success: bool):
        """Evaluator-only terminal signal; never included in trace.context()."""
        self._require_active()
        if self._pending is not None:
            raise RuntimeError("cannot complete with an unconfirmed action")
        self._terminal = {"success": bool(success), "reason": "success" if success else "horizon"}

    def __exit__(self, exc_type, exc, traceback):
        lifecycle_error = None
        if exc_type is None and self._terminal is None:
            lifecycle_error = RuntimeError("episode exited without a terminal result")
            exc_type = type(lifecycle_error)
        try:
            # Errors can occur during reset, RPC, trace writes or video output.
            # Store the type, never arbitrary exception text or environment data.
            append_jsonl(self.harness.evaluation_path, {
                "schema_version": 1, "run_id": self.harness.run_id,
                "episode_id": self.episode_id, "seed": self.seed,
                "timestamp": time.time(), "event_type": "episode_end",
                "executed_steps": self._executed_steps,
                "success": None if self._terminal is None else self._terminal["success"],
                "reason": "error" if exc_type else self._terminal["reason"],
                "error_type": exc_type.__name__ if exc_type else None,
                "pending_event_id": self._pending.event_id if self._pending else None,
            })
        except Exception as logging_error:
            if exc_type is None:
                raise
            warnings.warn(
                "Episode finalization failed: " + type(logging_error).__name__,
                RuntimeWarning,
            )
        finally:
            self._closed = True
            self.harness._active = False
        if lifecycle_error is not None:
            raise lifecycle_error
        return False  # Always propagate the original failure.
