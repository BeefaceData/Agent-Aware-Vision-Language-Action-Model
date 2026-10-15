from .analytic import AnalyticSkill


class Release(AnalyticSkill):
    def prepare(self, context, command, initial):
        return None, 1.0
