from harness.perception.geometry import world_yaw
from .analytic import AnalyticSkill


class RotateWrist(AnalyticSkill):
    def prepare(self, context, command, initial):
        return world_yaw(initial["pose"], command.parameters.delta_yaw_deg), None
