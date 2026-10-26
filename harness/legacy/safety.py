"""Non-LLM decision admission and per-episode intervention accounting.

No simulator operation is executed here. A future executor must enforce the
admitted step budget during motion, not merely inspect a result afterward.
"""

from dataclasses import dataclass

from pydantic import ValidationError

from harness.legacy.config import SafetyOptions
from harness.legacy.schemas import DecisionType, FailureType, SkillResult, SupervisorDecision, TaskPhase
from harness.legacy.skill_registry import SkillRejected, VerifiedSkillRegistry
from harness.legacy.recovery_skills.abort_episode import abort_decision


@dataclass(frozen=True)
class SafetyResult:
    decision: SupervisorDecision
    accepted: bool
    reason: str
    max_steps: int = 0


def pass_decision(reason, phase=TaskPhase.UNKNOWN, failure=FailureType.NONE):
    return SupervisorDecision(
        decision=DecisionType.PASS, phase=phase, failure_type=failure,
        confidence=1.0, reason=reason,
    )


class SafetyValidator:
    def __init__(self, registry: VerifiedSkillRegistry, options=None, minimum_confidence=0.70):
        self.registry = registry
        self.options = options or SafetyOptions()
        if isinstance(minimum_confidence, bool) or not 0 <= minimum_confidence <= 1:
            raise ValueError("invalid minimum confidence")
        self.minimum_confidence = minimum_confidence
        self.reset_episode()

    def reset_episode(self):
        self._interventions = 0
        self._consecutive_failures = 0
        self._active = None
        self._aborted = False

    @property
    def interventions(self):
        return self._interventions

    @property
    def consecutive_failures(self):
        return self._consecutive_failures

    def _reject(self, reason, phase, *, abort=False):
        decision = abort_decision(reason, phase) if abort else pass_decision(reason, phase)
        return SafetyResult(decision, False, reason)

    def evaluate(self, request, *, current_phase: TaskPhase, safety_abort=False) -> SafetyResult:
        phase = TaskPhase(current_phase)
        if safety_abort or self._aborted:
            return self._reject("independent_safety_abort", phase, abort=True)
        if self._consecutive_failures >= self.options.max_consecutive_skill_failures:
            return self._reject("consecutive_skill_failure_limit", phase, abort=True)
        if self._active is not None:
            return self._reject("intervention_already_active", phase, abort=True)
        try:
            decision = SupervisorDecision.model_validate(
                request.model_dump(warnings=False) if isinstance(request, SupervisorDecision) else request,
            )
        except (ValidationError, TypeError, ValueError):
            return self._reject("invalid_decision_schema", phase)
        if decision.decision in (DecisionType.PASS, DecisionType.ABORT):
            return SafetyResult(decision, True, decision.decision.value)
        invalid_abort = self.options.abort_on_invalid_skill_request
        if self._interventions >= self.options.max_interventions_per_episode:
            return self._reject("intervention_limit", phase, abort=True)
        if decision.confidence < self.minimum_confidence:
            return self._reject("low_confidence", phase)
        try:
            skill = self.registry.get_verified(decision.skill_id)
            if decision.phase != phase or phase not in skill.supported_phases:
                raise SkillRejected("unsupported_phase")
            if decision.failure_type not in skill.supported_failures:
                raise SkillRejected("unsupported_failure_type")
            if skill.max_steps > self.options.max_skill_steps:
                raise SkillRejected("skill_step_budget_exceeded")
            parameters = skill.validate_parameters(decision.parameters)
        except SkillRejected as exc:
            return self._reject(str(exc), phase, abort=invalid_abort)
        except (ValidationError, ValueError, TypeError, OSError):
            return self._reject("invalid_skill_parameters_or_definition", phase, abort=invalid_abort)
        admitted = SupervisorDecision.model_validate({**decision.model_dump(), "parameters": parameters})
        return SafetyResult(admitted, True, "verified_skill", skill.max_steps)

    def reserve(self, request, *, current_phase: TaskPhase, safety_abort=False) -> SafetyResult:
        """Atomically admit and count an attempt before handing off to an executor.

        Single-threaded episode API: failed/exceptional attempts still consume
        their slot. Inspecting a decision with evaluate() never consumes a slot.
        """
        result = self.evaluate(request, current_phase=current_phase, safety_abort=safety_abort)
        if result.decision.decision == DecisionType.ABORT:
            self._aborted = True
        elif result.decision.decision == DecisionType.EXECUTE_SKILL and result.accepted:
            self._interventions += 1
            self._active = (result.decision.skill_id, result.max_steps)
        return result

    def finish_intervention(self, result: SkillResult, *, recovered: bool):
        """Use an independent evaluator's recovery label, never success_hint alone."""
        if self._active is None:
            raise RuntimeError("no reserved intervention")
        if type(recovered) is not bool:
            raise ValueError("recovered must be an independent Boolean outcome")
        try:
            result = SkillResult.model_validate(result.model_dump(warnings=False))
            skill_id, budget = self._active
            if result.skill_id != skill_id or result.executed_steps > budget:
                raise ValueError("skill result does not match the reserved ID/budget")
        except (ValueError, TypeError, AttributeError):
            self._aborted = True
            raise
        self._active = None
        self._consecutive_failures = 0 if recovered else self._consecutive_failures + 1

    def execution_failed(self):
        if self._active is None:
            raise RuntimeError("no reserved intervention")
        self._active = None
        self._consecutive_failures += 1
        # With no bounded executor yet, safe continuation after an exception
        # cannot be established. Fail closed instead of silently resuming.
        self._aborted = True
