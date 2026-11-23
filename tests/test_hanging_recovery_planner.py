"""CPU contract tests for the replay-only hanging recovery sequence."""

import unittest
from types import SimpleNamespace

from development.planner.hanging_recovery import HangingRecoveryPlanner
from harness.skills.schemas import SkillType


def skill_result(skill, *, completed=True, terminal=False, uncertain=False):
    return SimpleNamespace(skill=SkillType(skill), completed=completed,
        episode_terminal=terminal, execution_uncertain=uncertain)


class HangingRecoveryPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = HangingRecoveryPlanner()

    def select(self, previous=None):
        return self.planner.select(SimpleNamespace(previous=previous))

    def test_retains_grasp_through_approach_and_seat_then_releases(self):
        approach = self.select()
        seat = self.select(skill_result("move_to"))
        release = self.select(skill_result("move_to"))

        self.assertEqual((approach.phase, approach.command.parameters.target),
                         ("RACK_APPROACH", "rack_approach"))
        self.assertEqual((seat.phase, seat.command.parameters.target),
                         ("RACK_SEAT", "rack_seat"))
        self.assertEqual(release.phase, "RACK_RELEASE")
        self.assertEqual(release.command.action, SkillType.RELEASE)
        self.assertEqual(release.command.parameters.authorization, "rack_seat_alignment")
        self.assertEqual([approach.command.action, seat.command.action, release.command.action],
                         [SkillType.MOVE_TO, SkillType.MOVE_TO, SkillType.RELEASE])

    def test_failed_motion_does_not_advance_to_release(self):
        failed = self.select(skill_result("move_to", completed=False))
        self.assertEqual(failed.command.action, SkillType.ABORT)
        self.assertEqual(failed.command.reason, "hanging_recovery_skill_failed")

    def test_uncertain_motion_does_not_advance_to_release(self):
        uncertain = self.select(skill_result("move_to", uncertain=True))
        self.assertEqual(uncertain.command.action, SkillType.ABORT)


if __name__ == "__main__":
    unittest.main()
