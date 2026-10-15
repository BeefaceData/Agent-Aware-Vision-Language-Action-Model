"""Static six-skill allowlist; environment capabilities are checked separately."""

from .schemas import SkillType
from .vla_act import Abort, VLAAct
from .move_to import MoveTo
from .rotate_wrist import RotateWrist
from .gripper import SetGripper
from .release import Release


class SkillRegistry:
    def __init__(self):
        self._implementations = {SkillType.VLA_ACT: VLAAct(), SkillType.ABORT: Abort(),
            SkillType.MOVE_TO: MoveTo(), SkillType.ROTATE_WRIST: RotateWrist(),
            SkillType.SET_GRIPPER: SetGripper(), SkillType.RELEASE: Release()}

    def get(self, action):
        try:
            return self._implementations[SkillType(action)]
        except (KeyError, ValueError):
            raise ValueError("skill_not_implemented") from None

    def available(self):
        return tuple(self._implementations)
