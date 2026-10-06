"""Episode-owned intervention accounting, independent of tool/executor lifetime."""


class InterventionBudget:
    def __init__(self, maximum, tool_limits):
        self.maximum = maximum
        self.tool_limits = dict(tool_limits)
        self.interventions = 0
        self.attempts = {}

    def admit(self, kind, tool=None):
        """Charge before dispatch; no completion, abort or refund operation exists."""
        reason = None
        if self.interventions >= self.maximum:
            reason = 'episode intervention limit exhausted'
        elif kind == 'recovery' and self.attempts.get(tool, 0) >= self.tool_limits.get(tool, 0):
            reason = f'recovery attempt limit exhausted: {tool}'
        if reason is None:
            self.interventions += 1
            if kind == 'recovery':
                self.attempts[tool] = self.attempts.get(tool, 0) + 1
        return dict(kind=kind, tool=tool, admitted=reason is None, reason=reason,
                    interventions=self.interventions, recovery_attempts=dict(self.attempts))
