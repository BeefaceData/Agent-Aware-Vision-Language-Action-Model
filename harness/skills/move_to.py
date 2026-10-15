import numpy as np
from .analytic import AnalyticSkill


class MoveTo(AnalyticSkill):
    def prepare(self, context, command, initial):
        parameters = command.parameters
        if parameters.target == "current_ee":
            target = initial["pose"].copy()
        else:
            anchor = context.anchors.resolve(parameters.target, command.arm, context.observation,
                                             allow_oracle=context.allow_oracle)
            context.oracle_used |= anchor.source == "development_oracle"
            target = np.asarray(anchor.pose, dtype=float)
        target[:3] += parameters.offset_xyz
        return target, None
