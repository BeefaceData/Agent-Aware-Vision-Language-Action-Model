from .base import PredicateResult


PREDICATE_NAMES = frozenset({"mug_grasp_confirmed", "mug_in_workspace", "mug_toppled", "mug_lifted", "right_mug_lifted", "handoff_confirmed",
    "mug_near_rack", "rack_contact_detected", "mug_released", "release_ready",
    "rack_seat_aligned",
    "mug_placed_on_table", "mug_staged_on_table",
    "left_transfer_complete", "grasp_attempt_failed",
    "task_success", "chunk_budget_exhausted"})


class PredicateRegistry:
    def __init__(self, hanging_mug=None, *, perception=None, rack_seat_gate=None,
                 fallback_oracle=None):
        self.hanging_mug = hanging_mug
        self.perception = perception
        self.rack_seat_gate = rack_seat_gate
        self.fallback_oracle = fallback_oracle
        self._attempt_providers = {}
        self._resolution_counts = {}
        self._fallback_counts = {}
        self._oracle_enabled = False

    @staticmethod
    def _increment(counts, key):
        counts[key] = counts.get(key, 0) + 1

    def _resolved(self, result, route):
        self._increment(self._resolution_counts, route)
        if route == "development_oracle_fallback":
            self._increment(self._fallback_counts, result.name)
        return result

    def usage_summary(self):
        """Return bounded source accounting for the completed episode."""
        return {
            "mode": ("hybrid_development" if self._oracle_enabled
                     and self.fallback_oracle is not None else "public_only"),
            "resolution_counts": dict(sorted(self._resolution_counts.items())),
            "oracle_fallbacks_by_predicate": dict(sorted(self._fallback_counts.items())),
            "oracle_fallback_total": sum(self._fallback_counts.values()),
        }

    def feedback_boundary_predicates(self):
        """Public image predicates are evaluated when control returns to planner.

        Runtime safety predicates and the development-only temporal oracle stay
        independently available; expensive RGB/RGB-D providers are never polled
        after every policy action inside a chunk.
        """
        # Only remote/public perception needs a planner feedback boundary.
        # Development oracles and local predicates can be checked at each
        # action, including when the same predicate is also a VLA stop target.
        if self.perception is None:
            return frozenset()
        return PREDICATE_NAMES - {"task_success", "chunk_budget_exhausted"}

    def prime(self, observation, *, allow_oracle=False):
        """Capture the episode's lift reference before any skill moves."""
        if (self.perception is not None or self.hanging_mug is not None
                or self.fallback_oracle is not None):
            self.evaluate("mug_lifted", observation, allow_oracle=allow_oracle)

    def begin_attempt(self, name, observation, *, arm="left", allow_oracle=False):
        self._oracle_enabled |= bool(allow_oracle)
        provider = self.perception if self.perception is not None else self.hanging_mug
        if provider is not None:
            begin = getattr(provider, "begin_attempt", None)
            if begin and begin(name, observation, arm=arm):
                self._attempt_providers[(name, arm)] = provider
                self._increment(self._resolution_counts, "public_attempt_monitor")
                return True
        fallback = self.fallback_oracle
        fallback_started = False
        if fallback is not None and allow_oracle:
            fallback_started = (fallback.begin_attempt(
                name, observation, arm=arm, allow_oracle=True)
                if isinstance(fallback, PredicateRegistry)
                else fallback.begin_attempt(name, observation, arm=arm))
        if fallback_started:
            # Keep this local temporal monitor on the provider that initialized
            # it. In development runs this also avoids invoking the public RGB-D
            # detector after every policy action merely to obtain an answer it
            # cannot currently prove.
            self._attempt_providers[(name, arm)] = fallback
            self._increment(self._resolution_counts, "development_oracle_attempt_monitor")
            return True
        if provider is None or (provider.source == "development_oracle" and not allow_oracle):
            return False
        return False

    def evaluate(self, name, observation, *, arm="auto", native_success=False,
                 budget_exhausted=False, allow_oracle=False):
        self._oracle_enabled |= bool(allow_oracle)
        if name not in PREDICATE_NAMES:
            raise ValueError("unknown_predicate")
        if name in ("task_success", "chunk_budget_exhausted"):
            value = native_success if name == "task_success" else budget_exhausted
            return self._resolved(PredicateResult(name=name, value=bool(value), source="runtime",
                observation_sequence=observation.sequence, reason="runtime_status"), "runtime")
        attempt_provider = self._attempt_providers.get((name, arm))
        if (attempt_provider is None and name in {"mug_lifted", "right_mug_lifted"}
                and arm in ("left", "right")):
            # Grasp retries need a local lift check every action. Reuse the
            # provider that owns the attempt monitor rather than polling remote
            # image perception inside a VLA chunk.
            attempt_provider = self._attempt_providers.get(("grasp_attempt_failed", arm))
        if attempt_provider is not None:
            result = attempt_provider.evaluate(name, observation, arm=arm)
            route = ("development_oracle_attempt_result"
                     if result.source == "development_oracle" else "public_attempt_result")
            return self._resolved(result, route)
        if name == "rack_seat_aligned" and self.rack_seat_gate is not None:
            result = self.rack_seat_gate.evaluate(name, observation, arm=arm)
            if result is not None:
                return self._resolved(result, "rack_seat_gate")
        provider = self.perception if self.perception is not None else self.hanging_mug
        if provider is None or (provider.source == "development_oracle" and not allow_oracle):
            result = PredicateResult(name=name, value=None, source="unavailable",
                observation_sequence=observation.sequence, reason="no_approved_evidence_provider")
        else:
            result = provider.evaluate(name, observation, arm=arm)
        if (result.value is None and self.fallback_oracle is not None and allow_oracle):
            fallback_result = (self.fallback_oracle.evaluate(
                name, observation, arm=arm, allow_oracle=True)
                if isinstance(self.fallback_oracle, PredicateRegistry)
                else self.fallback_oracle.evaluate(name, observation, arm=arm))
            if fallback_result.value is not None:
                return self._resolved(fallback_result, "development_oracle_fallback")
            return self._resolved(result, "unresolved_after_oracle")
        route = ("public_result" if result.source == "perception"
                 else "unavailable_result" if result.value is None
                 else result.source)
        return self._resolved(result, route)
