"""ABORT is a zero-motion control decision, not a physical recovery skill."""

from harness.legacy.schemas import DecisionType, FailureType, SupervisorDecision, TaskPhase


def abort_decision(reason: str, phase=TaskPhase.UNKNOWN, failure=FailureType.UNKNOWN):
    return SupervisorDecision(
        decision=DecisionType.ABORT, phase=phase, failure_type=failure,
        confidence=1.0, reason=reason,
    )
