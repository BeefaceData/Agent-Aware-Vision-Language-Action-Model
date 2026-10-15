from .base import PredicateResult


PREDICATE_NAMES = frozenset({"mug_grasp_confirmed", "mug_lifted", "handoff_confirmed",
    "mug_near_rack", "rack_contact_detected", "mug_released", "release_ready",
    "task_success", "chunk_budget_exhausted"})


class PredicateRegistry:
    def __init__(self, hanging_mug=None):
        self.hanging_mug = hanging_mug

    def prime(self, observation, *, allow_oracle=False):
        """Capture the episode's lift reference before any skill moves."""
        if self.hanging_mug is not None:
            self.evaluate("mug_lifted", observation, allow_oracle=allow_oracle)

    def evaluate(self, name, observation, *, arm="auto", native_success=False,
                 budget_exhausted=False, allow_oracle=False):
        if name not in PREDICATE_NAMES:
            raise ValueError("unknown_predicate")
        if name in ("task_success", "chunk_budget_exhausted"):
            value = native_success if name == "task_success" else budget_exhausted
            return PredicateResult(name=name, value=bool(value), source="runtime",
                observation_sequence=observation.sequence, reason="runtime_status")
        provider = self.hanging_mug
        if provider is None or (provider.source == "development_oracle" and not allow_oracle):
            return PredicateResult(name=name, value=None, source="unavailable",
                observation_sequence=observation.sequence, reason="no_approved_evidence_provider")
        return provider.evaluate(name, observation, arm=arm)
