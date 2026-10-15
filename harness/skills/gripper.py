from .analytic import AnalyticSkill


class SetGripper(AnalyticSkill):
    def prepare(self, context, command, initial):
        return None, command.parameters.value
