"""Public grasp-loss detection and the re-grasp sequence in supervised runs."""

from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

import numpy as np

from development.hanging.grasp_recovery import GraspRecovery, LIFT_M, START_POSE_ANCHOR
from development.hanging.supervised import HangingRecoveryHook
from harness.environment.base import Observation
from harness.perception.anchors import AnchorRegistry
from harness.planner.base import PlannerSelection
from harness.skills.schemas import SkillCommand, SkillType


POSE = np.array([.1, -.2, 1.0, 1., 0., 0., 0.])
BUDGET = dict(policy_actions=500, vla_chunks=40, skill_calls=40, analytic_ticks=12000)


def observation(sequence, right_gripper):
    state = [0.]*14
    state[13] = right_gripper
    return Observation(sequence, {}, tuple(state))


def report(*, held=None, off_table=None, confidence=None, upright=True):
    return SimpleNamespace(right_held=held, left_held=None, mug_off_table=off_table,
                           task_state_confidence=confidence, mug_upright=upright)


class FakeEnvironment:
    def __init__(self, opening=0.):
        self.opening = opening
        self.sequence = 1000

    def observe(self):
        self.sequence += 1
        return observation(self.sequence, 0.)

    def measured_gripper_opening(self, arm):
        if self.opening is None:
            raise ValueError("unavailable")
        return self.opening

    def arm_state(self, arm):
        return {"pose": POSE.copy(), "tcp_pose": POSE.copy(), "joints": np.zeros(6), "gripper": 0.}


class FakeProvider:
    def __init__(self):
        self.report = report()
        self.position = np.array([.05, .0, .78])
        self.authorized = set()
        self.authorized_during_open = None

    def cached_report(self, observation):
        return self.report

    def mug_surface_position(self, observation):
        return self.position

    def authorize_empty_hand_release(self, arm):
        self.authorized.add(arm)

    def revoke_empty_hand_release(self, arm):
        self.authorized.discard(arm)


class FakeExecutor:
    def __init__(self, provider, *, move_ok=True):
        self.anchors, self.provider, self.move_ok = AnchorRegistry(), provider, move_ok
        self.commands = []

    def execute(self, command, confirmed_observation=None, allow_recoverable_tracking_error=False):
        self.commands.append(command)
        if command.action == SkillType.MOVE_TO:
            if command.parameters.target == START_POSE_ANCHOR:
                self.anchor = self.anchors.resolve(START_POSE_ANCHOR, command.arm, confirmed_observation)
            ok = self.move_ok
        else:
            self.provider.authorized_during_open = set(self.provider.authorized)
            ok = True
        return SimpleNamespace(completed=ok, stop_reason="completed" if ok else "planning_failure",
                               executed_physics_ticks=120, execution_uncertain=False)


def monitor(opening=0., **options):
    environment, provider = FakeEnvironment(opening), FakeProvider()
    executor = FakeExecutor(provider)
    events = []
    grasp = GraspRecovery(environment, executor, provider, log=events.append, **options)
    return grasp, environment, provider, executor, events


def run_boundaries(grasp, grippers, start=0):
    return [grasp.check(observation(start+16*i, value)) for i, value in enumerate(grippers)]


class DetectionTests(unittest.TestCase):
    def test_closed_on_nothing_confirms_after_two_measured_boundaries(self):
        grasp, *_ , events = monitor(opening=.1818)
        results = run_boundaries(grasp, [1., 0., 0., 0.])
        # open, closing chunk (fingers may still move), first empty reading, confirmation
        self.assertEqual(results, [None, None, None, "measured_fingers_closed_on_nothing"])
        self.assertEqual(events[-1]["measured_gripper_opening"], .1818)

    def test_fingers_stopped_by_the_mug_never_trigger(self):
        grasp, *_ = monitor(opening=.25)
        self.assertEqual(run_boundaries(grasp, [1.]+[0.]*10), [None]*11)

    def test_no_monitoring_before_the_arm_was_seen_open(self):
        grasp, *_ = monitor(opening=0.)
        self.assertEqual(run_boundaries(grasp, [0.]*6), [None]*6)

    def test_camera_confirms_loss_when_fingers_are_unavailable(self):
        grasp, _, provider, *_ = monitor(opening=None)
        provider.report = report(held=False, off_table=False, confidence=.85)
        self.assertEqual(run_boundaries(grasp, [1., 0., 0., 0.])[-1], "camera_mug_on_table_hand_empty")

    def test_low_confidence_or_held_camera_report_does_not_trigger(self):
        for camera in (report(held=False, off_table=False, confidence=.6),
                       report(held=None, off_table=False, confidence=.9),
                       report(held=True, off_table=True, confidence=.95)):
            with self.subTest(camera=camera):
                grasp, _, provider, *_ = monitor(opening=None)
                provider.report = camera
                self.assertEqual(run_boundaries(grasp, [1.]+[0.]*6), [None]*7)

    def test_uncertain_frame_is_skipped_but_a_held_frame_resets(self):
        sequence = [report(held=False, off_table=False, confidence=.85),
                    report(held=None, off_table=False, confidence=.6),
                    report(held=False, off_table=False, confidence=.75)]
        grasp, _, provider, *_ = monitor(opening=None)
        results = []
        for i, (gripper, camera) in enumerate(zip([1., 0., 0., 0., 0.], [report()]*2+sequence)):
            provider.report = camera
            results.append(grasp.check(observation(16*i, gripper)))
        self.assertEqual(results[-1], "camera_mug_on_table_hand_empty")
        grasp, _, provider, *_ = monitor(opening=None)
        sequence[1] = report(held=True, off_table=True, confidence=.9)
        results = []
        for i, (gripper, camera) in enumerate(zip([1., 0., 0., 0., 0.], [report()]*2+sequence)):
            provider.report = camera
            results.append(grasp.check(observation(16*i, gripper)))
        self.assertEqual(results, [None]*5)

    def test_reopening_resets_the_streak(self):
        grasp, environment, *_ = monitor(opening=0.)
        self.assertEqual(run_boundaries(grasp, [1., 0., 0., 1., 0., 0.]), [None]*6)


class RecoveryTests(unittest.TestCase):
    def test_lifts_opens_then_returns_to_the_reset_pose(self):
        grasp, environment, provider, executor, events = monitor(opening=0.)
        environment.arm_state = lambda arm: {"pose": np.array([.3, -.1, .95, 1., 0., 0., 0.]),
                                             "tcp_pose": POSE.copy()}
        result = grasp.recover(observation(400, 0.), "measured_fingers_closed_on_nothing", remaining=BUDGET)
        self.assertEqual(result["status"], "grasp_recovery_completed")
        lift, release, back = executor.commands
        self.assertEqual((lift.action, lift.arm, lift.parameters.target, lift.parameters.offset_xyz,
                          lift.parameters.orientation_mode),
                         (SkillType.MOVE_TO, "right", "current_ee", (0., 0., LIFT_M), "preserve_current"))
        self.assertEqual((release.action, release.parameters.value), (SkillType.SET_GRIPPER, 1.))
        self.assertEqual((back.action, back.parameters.target), (SkillType.MOVE_TO, START_POSE_ANCHOR))
        # The reset pose recorded when the monitor was built, not the current one.
        np.testing.assert_allclose(executor.anchor.pose, POSE)
        self.assertEqual(provider.authorized_during_open, {"right"})
        self.assertEqual(provider.authorized, set())
        self.assertEqual(grasp.retries, 1)
        self.assertTrue(grasp.open_seen)
        self.assertIn("grasp_recovery_completed", [e["event_type"] for e in events])

    def test_toppled_mug_is_never_regrasped(self):
        grasp, _, provider, executor, _ = monitor()
        provider.report = report(upright=False)
        result = grasp.recover(observation(400, 0.), "basis", remaining=BUDGET)
        self.assertEqual(result["reason"], "mug_toppled_not_regrasped")
        self.assertEqual(executor.commands, [])
        self.assertIsNone(grasp.check(observation(416, 0.)))

    def test_unclear_upright_waits_without_motion(self):
        for camera in (report(upright=None), None):
            with self.subTest(camera=camera):
                grasp, _, provider, executor, _ = monitor()
                provider.report = camera
                self.assertIsNone(grasp.recover(observation(400, 0.), "basis", remaining=BUDGET))
                self.assertEqual((executor.commands, grasp.retries, grasp.stopped), ([], 0, None))

    def test_never_opens_a_hand_that_is_not_measured_empty(self):
        grasp, environment, provider, executor, _ = monitor(opening=.25)
        result = grasp.recover(observation(400, 0.), "camera_mug_on_table_hand_empty", remaining=BUDGET)
        self.assertEqual(result["reason"], "hand_not_measured_empty_before_opening")
        self.assertEqual([c.action for c in executor.commands], [SkillType.MOVE_TO])

    def test_failed_lift_stops_recovery_before_opening(self):
        grasp, _, provider, executor, _ = monitor()
        executor.move_ok = False
        self.assertEqual(grasp.recover(observation(400, 0.), "basis", remaining=BUDGET)["reason"],
                         "lift_failed")
        self.assertEqual(len(executor.commands), 1)

    def test_lift_never_leaves_the_workspace(self):
        grasp, environment, _, executor, _ = monitor()
        environment.arm_state = lambda arm: {"pose": np.array([.1, -.2, 1.30, 1., 0., 0., 0.])}
        grasp.recover(observation(400, 0.), "basis", remaining=BUDGET)
        self.assertAlmostEqual(executor.commands[0].parameters.offset_xyz[2], .05)

    def test_hand_empty_reads_the_fingers(self):
        for opening, expected in ((.1818, True), (.25, False), (1., False), (None, None)):
            with self.subTest(opening=opening):
                grasp, environment, *_ = monitor(opening=opening)
                self.assertIs(grasp.hand_empty("right"), expected)
                self.assertIsNone(grasp.hand_empty("left"))

    def test_retry_and_budget_limits(self):
        grasp, *_ , executor, _ = monitor(max_retries=1)
        grasp.recover(observation(400, 0.), "basis", remaining=BUDGET)
        self.assertEqual(grasp.recover(observation(500, 0.), "basis", remaining=BUDGET)["reason"],
                         "grasp_retry_budget_exhausted")
        grasp, *_ = monitor()
        self.assertEqual(grasp.recover(observation(400, 0.), "basis",
                                       remaining={**BUDGET, "analytic_ticks": 100})["reason"],
                         "grasp_recovery_budget_unavailable")


class HookTests(unittest.TestCase):
    def hook(self, grasp):
        return HangingRecoveryHook(Mock(), Mock(), Mock(), episode_id="e", deadline=time.monotonic()+60,
                                   log=lambda event: None, policy_chunk_size=16, pause_policy_on_entry=False,
                                   hand_back_to_policy=True, policy_grace_actions=160, grasp_recovery=grasp)

    def test_completed_regrasp_hands_a_fresh_chunk_to_the_policy(self):
        grasp = Mock()
        grasp.check.return_value = "measured_fingers_closed_on_nothing"
        grasp.recover.return_value = dict(status="grasp_recovery_completed", attempt=1)
        hook = self.hook(grasp)
        hook.handed_back, hook.entry_sequence = "right_grasp_evidence_expired", 380
        selection = PlannerSelection(SkillCommand(action="vla_act"), "HANG")
        result = hook.consider(selection, observation(400, 0.), remaining=BUDGET)
        self.assertEqual(result["status"], "recovery_deferred")
        self.assertTrue(result["refresh_observation"])
        self.assertEqual(SkillCommand.model_validate(result["policy_command"]).action, SkillType.VLA_ACT)
        self.assertIsNone(hook.handed_back)
        self.assertIsNone(hook.entry_sequence)

    def test_no_loss_leaves_the_hanging_supervisor_unchanged(self):
        grasp = Mock()
        grasp.check.return_value = None
        hook = self.hook(grasp)
        hook.handed_back = "x"
        result = hook.consider(PlannerSelection(SkillCommand(action="vla_act"), "HANG"),
                               observation(400, 0.), remaining=BUDGET)
        self.assertEqual(result["reason"], "handed_back_to_policy")
        grasp.recover.assert_not_called()

    def test_left_loss_uses_left_recovery_before_right_monitor(self):
        left, right = Mock(arm="left"), Mock(arm="right")
        left.check.return_value = "camera_mug_on_table_hand_empty"
        left.recover.return_value = dict(status="grasp_recovery_completed", attempt=1)
        right.check.return_value = None
        hook = HangingRecoveryHook(Mock(), Mock(), Mock(), episode_id="e",
            deadline=time.monotonic()+60, log=lambda event: None,
            policy_chunk_size=16, pause_policy_on_entry=False,
            hand_back_to_policy=True, policy_grace_actions=0,
            grasp_recoveries=(left, right))

        result = hook.consider(
            PlannerSelection(SkillCommand(action="vla_act"), "GRASP"),
            observation(400, 0.), remaining=BUDGET)

        self.assertEqual(result["reason"], "grasp_recovery_completed")
        left.recover.assert_called_once()
        right.check.assert_not_called()

    def test_handoff_monitors_right_arm_without_left_arm_starvation(self):
        left, right = Mock(arm="left"), Mock(arm="right")
        left.check.return_value = "measured_fingers_closed_on_nothing"
        left.recover.return_value = None
        right.check.return_value = "measured_fingers_closed_on_nothing"
        right.recover.return_value = dict(
            status="grasp_recovery_completed", attempt=1)
        hook = HangingRecoveryHook(
            Mock(), Mock(), Mock(), episode_id="e",
            deadline=time.monotonic()+60, log=lambda event: None,
            policy_chunk_size=16, pause_policy_on_entry=False,
            hand_back_to_policy=True, policy_grace_actions=0,
            grasp_recoveries=(left, right))

        result = hook.consider(
            PlannerSelection(SkillCommand(action="vla_act"), "HANDOFF"),
            observation(400, 0.), remaining=BUDGET)

        self.assertEqual(result["reason"], "grasp_recovery_completed")
        left.check.assert_not_called()
        right.check.assert_called_once()
        right.recover.assert_called_once()


class ProviderReleaseTests(unittest.TestCase):
    def test_empty_hand_release_answers_without_a_vision_request(self):
        from development.hanging.public_provider import PublicRGBDHangingProvider
        client = Mock()
        provider = PublicRGBDHangingProvider(client, model="claude-test", episode_id="e",
                                             store=Mock(), log=lambda event: None)
        current = observation(5, 0.)
        provider.authorize_empty_hand_release("right")
        result = provider.evaluate("release_ready", current, arm="right")
        self.assertEqual((result.value, result.reason), (True, "measured_empty_hand"))
        provider.revoke_empty_hand_release("right")
        provider.measured_hand_empty = lambda arm: True
        lifted = provider.evaluate("right_mug_lifted", current, arm="right")
        self.assertEqual((lifted.value, lifted.reason), (False, "measured_fingers_empty"))
        provider.measured_hand_empty = None
        self.assertIsNone(provider.evaluate("release_ready", current, arm="right").value)
        self.assertIsNone(provider.cached_report(current))
        client.assert_not_called()


class MeasuredOpeningTests(unittest.TestCase):
    def test_reads_the_base_finger_joint_on_the_command_scale(self):
        from harness.environment.local import LocalEnvironmentClient
        joint = object()
        robot = SimpleNamespace(right_gripper=[(joint, 1., 0.)], right_gripper_scale=[-.01, .045],
                                right_entity=SimpleNamespace(get_qpos=lambda: np.array([0., .0175]),
                                                             get_active_joints=lambda: [object(), joint]),
                                right_gripper_val=0.)
        environment = LocalEnvironmentClient.__new__(LocalEnvironmentClient)
        environment.simulator = SimpleNamespace(task=SimpleNamespace(robot=robot))
        self.assertAlmostEqual(environment.measured_gripper_opening("right"), .5)
        with self.assertRaises(ValueError):
            environment.measured_gripper_opening("middle")


if __name__ == "__main__":
    unittest.main()
