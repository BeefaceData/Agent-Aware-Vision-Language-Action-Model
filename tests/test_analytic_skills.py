"""Focused R2/R3 contract checks; physical operating ranges still need live trials."""

import unittest
import numpy as np

from harness.environment.base import Observation
from harness.skills import SkillExecutor
from harness.skills.move_to import MoveTo
from harness.skills.native_motion import NativeArmState, NativeMotionAdapter
from harness.skills.motion import subsample_trajectory
from harness.predicates.registry import PredicateRegistry
from harness.predicates.hanging_mug import HangingMugPredicates, MugEvidence
from harness.perception.anchors import Anchor, AnchorRegistry
from harness.perception.geometry import reverse_approach_waypoints
from harness.perception.preparation import PlannerPixelAnchorPreparer
from harness.perception.rgbd import RGBDCamera
from harness.skills.schemas import SkillCommand
from harness.vla.modal_client import ModalVLAClient
from tests.test_skills import Runtime, COMMAND
from tests.helpers import images
from tests.scenario_replay import scenario


class AnalyticEnvironment:
    supports_analytic = True

    def __init__(self, x=-0.2):
        self.sequence = self.ticks = self.holds = 0
        self.arms = {arm: {"joints": np.full(6, 0.2 if arm == "left" else 0.4),
                          "pose": np.array([x, 0., 0.9, 1., 0., 0., 0.]), "gripper": 0.5}
                     for arm in ("left", "right")}
        self.plan_status = "Success"
        self.goal = None
        self.goals = []
        self.writes = []
        self.success_on_open = False

    @property
    def native_success(self):
        return self.success_on_open and self.arms["left"]["gripper"] > 0.8

    def observe(self):
        # Intentionally different commanded vs measured joints.
        return Observation(self.sequence, images(0), (1.0,)*14)

    def arm_state(self, arm):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k,v in self.arms[arm].items()}

    def joint_bounds(self, arm): return np.tile([-3., 3.], (6, 1))

    def plan(self, arm, pose):
        self.goal = np.asarray(pose).copy()
        self.goals.append(self.goal.copy())
        self.start = self.arms[arm]["joints"].copy()
        positions = np.linspace(self.start, self.start+0.01, 4)
        return {"status": self.plan_status, "position": positions, "velocity": np.zeros_like(positions)}

    def tick(self, arm, position, velocity, gripper, other_hold):
        self.writes.append((arm, position.copy(), other_hold["joints"].copy()))
        self.arms[arm]["joints"] = position.copy()
        self.arms[arm]["gripper"] = float(gripper)
        if self.goal is not None and np.max(abs(position-self.start-0.01)) < 1e-6:
            self.arms[arm]["pose"] = self.goal.copy()
        self.ticks += 1
        self.sequence += 1

    def hold_measured(self): self.holds += 1

    def step(self, action):
        self.sequence += 1
        return self.observe()

    def step_after_success(self, action):
        return self.step(action)


class RetryPlanEnvironment(AnalyticEnvironment):
    def __init__(self):
        super().__init__(x=-0.2)
        self.plan_calls = 0

    def plan(self, arm, pose):
        self.plan_calls += 1
        self.plan_status = "Fail" if self.plan_calls == 1 else "Success"
        return super().plan(arm, pose)


class FailPlanAtCallEnvironment(AnalyticEnvironment):
    def __init__(self, fail_at):
        super().__init__(x=-0.2)
        self.fail_at = fail_at
        self.plan_calls = 0

    def plan(self, arm, pose):
        self.plan_calls += 1
        self.plan_status = "Fail" if self.plan_calls == self.fail_at else "Success"
        return super().plan(arm, pose)


class FailInitialRetreatAndLongFallbackEnvironment(AnalyticEnvironment):
    def __init__(self):
        super().__init__(x=-0.2)
        self.plan_calls = 0

    def plan(self, arm, pose):
        self.plan_calls += 1
        self.plan_status = "Fail" if self.plan_calls in (3, 4) else "Success"
        return super().plan(arm, pose)


class RetreatOrientationDriftEnvironment(AnalyticEnvironment):
    """Model small measured wrist drift between successive retreat segments."""

    def __init__(self):
        super().__init__(x=-0.2)
        self.drift_pending = False

    def plan(self, arm, pose):
        self.drift_pending = True
        return super().plan(arm, pose)

    def tick(self, arm, position, velocity, gripper, other_hold):
        super().tick(arm, position, velocity, gripper, other_hold)
        if (self.drift_pending
                and np.max(abs(position-self.start-0.01)) < 1e-6):
            self.arms[arm]["pose"][3:] = [np.cos(0.05), np.sin(0.05), 0., 0.]
            self.drift_pending = False


class ShortRetreatOnlyEnvironment(AnalyticEnvironment):
    """Reject the long fallback while allowing a distinct local clearance move."""

    def plan(self, arm, pose):
        target = np.asarray(pose)
        if np.linalg.norm(target[:3] - self.arms[arm]["pose"][:3]) > 0.09:
            self.goal = target.copy()
            self.goals.append(self.goal.copy())
            return {"status": "Fail", "position": [], "velocity": []}
        return super().plan(arm, pose)


class ShortMoveOnlyEnvironment(AnalyticEnvironment):
    """A long fallback is infeasible while a local retreat remains feasible."""

    def plan(self, arm, pose):
        if np.linalg.norm(np.asarray(pose)[:3] - self.arms[arm]["pose"][:3]) > 0.10:
            self.goal = np.asarray(pose).copy()
            self.goals.append(self.goal.copy())
            return {"status": "Fail", "position": [], "velocity": []}
        return super().plan(arm, pose)


class InexactRetreatEnvironment(AnalyticEnvironment):
    def tick(self, arm, position, velocity, gripper, other_hold):
        super().tick(arm, position, velocity, gripper, other_hold)
        if self.goal is not None:
            self.arms[arm]["pose"] = self.goal.copy()
            self.arms[arm]["pose"][0] -= 0.018
            angle = 0.075
            self.arms[arm]["pose"][3:] = [np.cos(angle), 0.0, 0.0, np.sin(angle)]


class EndpointLagEnvironment(AnalyticEnvironment):
    """CPU-only endpoint profile: settle off-target, then converge."""

    def __init__(self, *, off_target_commands=6, fixed_error=None):
        super().__init__()
        self.off_target_commands = off_target_commands
        self.fixed_error = fixed_error
        self.endpoint_commands = 0

    def tick(self, arm, position, velocity, gripper, other_hold):
        super().tick(arm, position, velocity, gripper, other_hold)
        at_endpoint = (self.goal is not None
                       and np.max(abs(position-self.start-0.01)) < 1e-6)
        if not at_endpoint:
            return
        self.endpoint_commands += 1
        if self.fixed_error is not None:
            position_error, orientation_error = self.fixed_error
        elif self.endpoint_commands <= self.off_target_commands:
            position_error, orientation_error = 0.02, 0.16
        else:
            return
        pose = self.goal.copy()
        pose[0] += position_error
        pose[3:] = [np.cos(orientation_error/2), 0.0, 0.0,
                    np.sin(orientation_error/2)]
        self.arms[arm]["pose"] = pose


class RGBDAnalyticEnvironment(AnalyticEnvironment):
    """CPU analytic controller fixture with same-frame RGB-D camera data."""

    def observe(self):
        camera = RGBDCamera(np.full((2, 3), 0.5, dtype=np.float32),
            np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]),
            np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                      [0., 0., 1., -.5], [0., 0., 0., 1.]]))
        return Observation(self.sequence, images(0), (1.0,) * 14,
                           {"head_camera": camera})


def evidence(sequence, **kwargs):
    data = dict(observation_sequence=sequence, mug_position=(-0.2,0.0,0.75),
                mug_in_task_workspace=True,
                left_tcp=(-0.2,0.0,0.8), right_tcp=(0.2,0.0,0.8),
                left_contact=False, right_contact=False, left_bilateral=False, right_bilateral=False,
                left_gripper=0.0, right_gripper=0.0, table_contact=True,
                rack_contact=False, rack_distance_m=0.4,
                stage_distance_m=0.0)
    data.update(kwargs)
    return MugEvidence(**data)


def executor(env, *, predicates=None, oracle=False, anchors=None):
    worker = Runtime()
    logs = []
    return SkillExecutor(env, ModalVLAClient(worker), log=logs.append,
                             predicates=predicates, anchors=anchors,
                             allow_development_oracle=oracle), worker, logs


MOVE = {"action": "move_to", "arm": "left", "parameters": {"offset_xyz": [0.02, 0.0, 0.0]}}


class AnalyticTests(unittest.TestCase):
    def test_native_motion_composes_canonical_qpos14_for_either_arm(self):
        inactive = NativeArmState(
            joints=np.arange(6, dtype=float),
            pose=np.array([0., 0., .9, 1., 0., 0., 0.]),
            gripper=0.75,
        )
        controlled = np.arange(10, 16, dtype=float)

        left = NativeMotionAdapter.compose_qpos14("left", controlled, 0.25, inactive)
        right = NativeMotionAdapter.compose_qpos14("right", controlled, 0.25, inactive)

        np.testing.assert_allclose(left, (*controlled, .25, *inactive.joints, .75))
        np.testing.assert_allclose(right, (*inactive.joints, .75, *controlled, .25))

    def test_native_trajectory_subsampling_preserves_endpoints(self):
        positions = np.linspace(0.0, 1.0, 100)[:, None] * np.ones((1, 6))
        velocities = np.zeros_like(positions)

        sampled_positions, sampled_velocities = subsample_trajectory(
            positions, velocities, max_waypoints=25)

        self.assertEqual(sampled_positions.shape, (25, 6))
        np.testing.assert_allclose(sampled_positions[0], positions[0])
        np.testing.assert_allclose(sampled_positions[-1], positions[-1])
        np.testing.assert_allclose(sampled_velocities, 0.0)

    def test_reverse_retreat_reanchors_each_waypoint_to_measured_wrist(self):
        env = RetreatOrientationDriftEnvironment()
        run, _, logs = executor(env)
        run.measured_tcp_history["right"].extend((
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.26, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.24, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.20, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))

        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.05,
                           "orientation_mode": "preserve_current"}})

        self.assertEqual(result.stop_reason, "completed")
        planned = [event for event in logs
                   if event.get("event_type") == "analytic_waypoint_plan"]
        self.assertGreater(len(planned), 1)
        for event in planned:
            np.testing.assert_allclose(event["target_pose"][3:],
                                       event["measured_start_pose"][3:], atol=1e-12)
        clearance = next(event for event in logs
                         if event.get("event_type") == "retreat_clearance_check")
        self.assertTrue(clearance["satisfied"])
        self.assertGreaterEqual(clearance["minimum_clearance_m"], 0.05)

    def test_rgbd_grounded_analytic_grasp_and_lift_composition(self):
        env = RGBDAnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        run, _, _ = executor(env, anchors=anchors)
        prepare = PlannerPixelAnchorPreparer(env, anchors)
        move_to_mug = {
            "action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [.5, .5]}},
        }

        # A planner-selected image pixel is resolved through this observation's
        # depth and measured TCP before the ordinary analytic motion executes.
        grounded_anchor = prepare(SkillCommand.model_validate(move_to_mug), env.observe())
        self.assertEqual(grounded_anchor.source, "perception")
        staged = run.execute(move_to_mug)
        self.assertEqual(staged.stop_reason, "completed")
        grounded_pose = env.goals[-1].copy()

        closed = run.execute({"action": "set_gripper", "arm": "left",
                              "parameters": {"value": 0.0, "ramp_ticks": 50}})
        self.assertEqual(closed.stop_reason, "completed")
        self.assertAlmostEqual(env.arm_state("left")["gripper"], 0.0)

        lifted = run.execute({"action": "move_to", "arm": "left",
                              "parameters": {"target": "current_ee",
                                             "offset_xyz": [0.0, 0.0, .06]}})
        self.assertEqual(lifted.stop_reason, "completed")
        np.testing.assert_allclose(env.goals[-1][:3],
                                   grounded_pose[:3] + np.array([0.0, 0.0, .06]))

    def test_two_start_poses_and_measured_joints_not_commanded_start(self):
        for x in (-0.2, 0.1):
            env = AnalyticEnvironment(x)
            run, _, logs = executor(env)
            result = run.execute(MOVE)
            self.assertEqual(result.stop_reason, "completed")
            self.assertAlmostEqual(env.goal[0], x+0.02)
            np.testing.assert_allclose(env.writes[0][1], 0.2)
            np.testing.assert_allclose(env.writes[0][2], 0.4)
            self.assertEqual(result.executed_physics_ticks, 7)
            self.assertEqual(result.position_error_m, 0.0)
            plan_result = next(event for event in logs
                               if event.get("event_type") == "analytic_plan_result")
            self.assertEqual(plan_result["planner_status"], "success")
            self.assertEqual(plan_result["trajectory_samples"], 4)
            self.assertEqual(plan_result["result"], "accepted")

    def test_pose_settling_waits_for_measured_convergence_and_stability(self):
        env = EndpointLagEnvironment(off_target_commands=6)
        run, _, logs = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.executed_physics_ticks, 12)  # 4 plan + 8 settle
        settling = next(event for event in logs
                        if event.get("event_type") == "analytic_settling")
        self.assertEqual(settling["ticks_used"], 8)
        self.assertEqual(settling["stable_samples"], 3)
        self.assertTrue(settling["converged"])

    def test_recorded_seed4_endpoint_miss_stays_a_failure_without_convergence(self):
        recorded = scenario("seed4_retreat_endpoint_tracking_miss")
        env = EndpointLagEnvironment(
            fixed_error=(recorded["position_error_m"], recorded["orientation_error_rad"]))
        run, _, logs = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, recorded["expected_outcome"])
        self.assertAlmostEqual(result.position_error_m, recorded["position_error_m"])
        self.assertAlmostEqual(result.orientation_error_rad, recorded["orientation_error_rad"])
        settling = next(event for event in logs
                        if event.get("event_type") == "analytic_settling")
        self.assertEqual(settling["tick_limit"], 125)
        self.assertEqual(settling["ticks_used"], 125)
        self.assertFalse(settling["converged"])

    def test_long_pose_move_uses_one_native_plan_from_measured_start(self):
        env = AnalyticEnvironment(x=-0.2)
        start_pose = env.arms["left"]["pose"].copy()
        target_pose = np.array([0.30, 0.10, 1.0,
                                0.20, 0.70, 0.65, 0.10])
        target_pose[3:] /= np.linalg.norm(target_pose[3:])
        anchors = AnchorRegistry()
        anchors.bind("test_anchor", "left", Anchor(tuple(target_pose), 0))
        run, _, logs = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "left",
                              "parameters": {"target": "test_anchor"}})
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(len(env.goals), 1)
        planned = [event for event in logs if event.get("event_type") == "analytic_waypoint_plan"]
        self.assertEqual([e["waypoint_index"] for e in planned], [1])
        self.assertEqual([e["waypoint_count"] for e in planned], [1])
        np.testing.assert_allclose(planned[0]["measured_start_pose"], start_pose)
        self.assertAlmostEqual(result.position_error_m, 0.0)
        self.assertAlmostEqual(result.orientation_error_rad, 0.0)

    def test_long_target_is_delegated_to_native_planner(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        anchors.bind("too_far", "left", Anchor((0.50, 0., 0.9, 1., 0., 0., 0.), 0))
        run, _, _ = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "left",
                              "parameters": {"target": "too_far"}})
        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(env.goals[-1][:3], [0.50, 0.0, 0.9])

    def test_grounded_recovery_rises_then_moves_above_mug_before_descending(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        target = (-0.05, 0.10, 0.85, 1.0, 0.0, 0.0, 0.0)
        anchors.bind("mug_recovery_clearance", "right", Anchor(target, 0))
        run, _, logs = executor(env, anchors=anchors)

        result = run.execute({
            "action": "move_to",
            "arm": "right",
            "parameters": {
                "target": "mug_recovery_clearance",
                "orientation_mode": "preserve_current",
            },
        })

        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(env.goals[0][:3], [-0.2, 0.0, 0.95])
        np.testing.assert_allclose(env.goals[-1][:3], target[:3])
        np.testing.assert_allclose(
            [goal[3:] for goal in env.goals],
            [[1.0, 0.0, 0.0, 0.0]] * len(env.goals),
        )
        self.assertTrue(all(
            abs(goal[2] - 0.95) < 1e-9 for goal in env.goals[:-1]))
        prepared = next(event for event in logs
                        if event.get("event_type") == "grounded_recovery_path_prepared")
        self.assertEqual(prepared["waypoint_count"], len(env.goals))

    def test_grounded_recovery_does_not_double_rise_above_higher_safe_target(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        target = (-0.05, 0.10, 0.96, 1.0, 0.0, 0.0, 0.0)
        anchors.bind("mug_recovery_clearance", "right", Anchor(target, 0))
        run, _, _ = executor(env, anchors=anchors)

        result = run.execute({
            "action": "move_to",
            "arm": "right",
            "parameters": {
                "target": "mug_recovery_clearance",
                "orientation_mode": "preserve_current",
            },
        })

        self.assertEqual(result.stop_reason, "completed")
        self.assertTrue(all(abs(goal[2] - target[2]) < 1e-9 for goal in env.goals))

    def test_minimum_translation_guard_rejects_short_retreat_before_planning(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        anchors.bind("near_pregrasp", "left", Anchor((-.16, 0., .9, 1., 0., 0., 0.), 0))
        run, _, _ = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "left",
                              "parameters": {"target": "near_pregrasp",
                                             "minimum_translation_m": 0.05}})
        self.assertEqual(result.stop_reason, "minimum_translation_not_met")
        self.assertEqual(env.goals, [])
        self.assertEqual(env.writes, [])

    def test_reverse_approach_retreat_follows_measured_path(self):
        env = AnalyticEnvironment(x=-0.2)
        run, _, _ = executor(env)
        run.measured_tcp_history["right"].extend((
            (-0.30, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.25, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.2, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(env.goal[:3], [-0.265, 0.0, 0.9])
        np.testing.assert_allclose(env.goal[3:], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(env.goals), 4)
        np.testing.assert_allclose([goal[0] for goal in env.goals],
                                   [-0.22, -0.24, -0.26, -0.265])
        np.testing.assert_allclose([goal[1:] for goal in env.goals],
                                   [[0.0, 0.9, 1.0, 0.0, 0.0, 0.0]] * 4)

    def test_reverse_approach_preserves_current_orientation_throughout_retreat(self):
        env = AnalyticEnvironment(x=-0.2)
        run, _, logs = executor(env)
        half_angle = np.deg2rad(15)
        preapproach_orientation = np.array(
            [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
        run.measured_tcp_history["right"].extend((
            (-0.30, 0.0, 0.9, *preapproach_orientation),
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.25, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.20, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))

        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})

        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(len(env.goals), 4)
        np.testing.assert_allclose([goal[3:] for goal in env.goals],
                                   [[1.0, 0.0, 0.0, 0.0]] * 4)
        np.testing.assert_allclose(env.goals[-1][:3], [-0.265, 0.0, 0.9])
        self.assertFalse(any(event.get("event_type") ==
                             "preapproach_orientation_return_prepared" for event in logs))

    def test_reverse_approach_without_history_or_grounding_fails_explicitly(self):
        env = AnalyticEnvironment(x=-0.2)
        run, _, logs = executor(env)
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "preapproach_target_unavailable")
        self.assertEqual(env.goals, [])
        self.assertTrue(any(event.get("event_type") == "native_preapproach_unavailable"
                            for event in logs))

    def test_reverse_approach_uses_fresh_grounded_preapproach_when_history_is_short(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        anchors.bind("mug_pregrasp", "right",
                     Anchor((-.32, 0., .9, 1., 0., 0., 0.), 0))
        run, _, logs = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(env.goal, [-.32, 0., .9, 1., 0., 0., 0.])
        selected = next(event for event in logs
                        if event.get("event_type") == "fresh_preapproach_selected")
        self.assertEqual(selected["source"], "perception")
        self.assertAlmostEqual(selected["distance_m"], 0.12)

    def test_fresh_preapproach_must_provide_requested_clearance(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                env = AnalyticEnvironment(x=-0.2)
                anchors = AnchorRegistry()
                anchors.bind("mug_pregrasp", arm,
                             Anchor((-.19, 0., .9, 1., 0., 0., 0.), 0))
                run, _, logs = executor(env, anchors=anchors)
                result = run.execute({"action": "move_to", "arm": arm,
                    "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                                   "retreat_distance_m": 0.05,
                                   "orientation_mode": "preserve_current"}})

                self.assertEqual(result.stop_reason, "preapproach_clearance_unavailable")
                self.assertEqual(env.goals, [])

    def test_reverse_approach_accepts_clearance_despite_small_endpoint_error(self):
        env = InexactRetreatEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        anchors.bind("mug_pregrasp", "left",
                     Anchor((-.32, 0., .9, 1., 0., 0., 0.), 0))
        run, _, logs = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "left",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "completed")
        self.assertGreater(result.position_error_m, 0.015)
        self.assertGreater(result.orientation_error_rad, 0.10)
        clearance = next(event for event in logs
                         if event.get("event_type") == "retreat_clearance_check")
        self.assertTrue(clearance["satisfied"])
        self.assertGreaterEqual(clearance["measured_clearance_m"], 0.065)

    def test_reverse_approach_accepts_five_centimeter_clearance_with_endpoint_error(self):
        env = InexactRetreatEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        anchors.bind("mug_pregrasp", "right",
                     Anchor((-.32, 0., .9, 1., 0., 0., 0.), 0))
        run, _, logs = executor(env, anchors=anchors)
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.05,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "completed")
        clearance = next(event for event in logs
                         if event.get("event_type") == "retreat_clearance_check")
        self.assertTrue(clearance["satisfied"])
        self.assertGreaterEqual(clearance["measured_clearance_m"], 0.05)

    def test_measured_reverse_retreat_plans_requested_clearance(self):
        env = InexactRetreatEnvironment(x=-0.2)
        run, _, logs = executor(env)
        run.measured_tcp_history["right"].extend((
            (-0.26, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.24, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.20, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))

        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.05,
                           "orientation_mode": "preserve_current"}})

        self.assertEqual(result.stop_reason, "completed")
        planned = [event for event in logs
                   if event.get("event_type") == "analytic_waypoint_plan"]
        planned_clearance = np.linalg.norm(
            np.asarray(planned[-1]["target_pose"][:3])
            - np.asarray(planned[0]["measured_start_pose"][:3]))
        self.assertAlmostEqual(planned_clearance, 0.05, places=5)
        clearance = next(event for event in logs
                         if event.get("event_type") == "retreat_clearance_check")
        self.assertEqual(clearance["minimum_clearance_m"], 0.05)
        self.assertGreaterEqual(clearance["measured_clearance_m"], 0.05)
        self.assertTrue(clearance["satisfied"])

    def test_reverse_approach_uses_net_clearance_not_path_length(self):
        current = (0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0)
        curved = (
            (-0.08, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.04, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.02, 0.9, 1.0, 0.0, 0.0, 0.0),
            current,
        )
        waypoints = reverse_approach_waypoints(curved, current, 0.065)
        self.assertAlmostEqual(np.linalg.norm(np.asarray(waypoints[-1][:3]) - current[:3]),
                               0.065, places=6)

        loop = [(-0.015, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
                (0.0, 0.015, 0.9, 1.0, 0.0, 0.0, 0.0),
                (0.015, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
                (0.0, -0.015, 0.9, 1.0, 0.0, 0.0, 0.0), current]
        with self.assertRaisesRegex(ValueError, "approach_history_clearance_unavailable"):
            reverse_approach_waypoints(loop, current, 0.065)

    def test_reverse_approach_plan_failure_is_returned_to_supervisor(self):
        env = RetryPlanEnvironment()
        anchors = AnchorRegistry()
        anchors.bind("mug_pregrasp", "right",
                     Anchor((-.32, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0), 0))
        run, _, logs = executor(env, anchors=anchors)
        run.measured_tcp_history["right"].extend((
            (-0.31, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.25, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.2, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(result.stop_reason, "planning_failure")
        self.assertEqual(sum(event.get("event_type") == "analytic_plan_result"
                             and event.get("result") == "rejected" for event in logs), 1)
        self.assertFalse(any("fallback" in event.get("event_type", "") for event in logs))

    def test_partial_reverse_retreat_failure_stops_without_inventing_new_path(self):
        env = FailPlanAtCallEnvironment(fail_at=3)
        run, _, logs = executor(env)
        run.measured_tcp_history["right"].extend((
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.26, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.24, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.20, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))

        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.05,
                           "orientation_mode": "preserve_current"}})

        self.assertEqual(result.stop_reason, "planning_failure")
        self.assertEqual(sum(event.get("event_type") == "analytic_waypoint_plan"
                             for event in logs), 3)
        self.assertFalse(any("fallback" in event.get("event_type", "") for event in logs))

    def test_rejected_retreat_does_not_step_down_to_another_target(self):
        env = FailInitialRetreatAndLongFallbackEnvironment()
        run, _, logs = executor(env)
        run.measured_tcp_history["right"].extend((
            (-0.28, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.26, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.24, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.22, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
            (-0.20, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0),
        ))

        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                           "retreat_distance_m": 0.05,
                           "orientation_mode": "preserve_current"}})

        self.assertEqual(result.stop_reason, "planning_failure")
        self.assertFalse(any("fallback" in event.get("event_type", "") for event in logs))

    def test_rejected_pregrasp_plan_is_not_retried_in_short_segments(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                env = ShortRetreatOnlyEnvironment(x=-0.2)
                anchors = AnchorRegistry()
                anchors.bind("mug_pregrasp", arm,
                             Anchor((-.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0), 0))
                run, _, logs = executor(env, anchors=anchors)
                result = run.execute({"action": "move_to", "arm": arm,
                    "parameters": {"target": "current_ee", "path_mode": "reverse_approach",
                                   "retreat_distance_m": 0.1,
                                   "orientation_mode": "preserve_current"}})

                self.assertEqual(result.stop_reason, "planning_failure")
                self.assertEqual(len(env.goals), 1)
                selected = next(event for event in logs
                                if event.get("event_type") == "fresh_preapproach_selected")
                self.assertEqual(selected["source"], "perception")
                self.assertFalse(any("fallback" in event.get("event_type", "")
                                     for event in logs))

    def test_retreat_and_reorientation_can_be_planned_as_separate_moves(self):
        env = AnalyticEnvironment(x=-0.2)
        anchors = AnchorRegistry()
        target = np.array([-0.10, -0.08, 0.98,
                           np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])
        anchors.bind("mug_pregrasp", "left", Anchor(tuple(target), 0))
        run, _, _ = executor(env, anchors=anchors)

        retreat = run.execute({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "minimum_translation_m": 0.065,
                           "orientation_mode": "preserve_current"}})
        self.assertEqual(retreat.stop_reason, "completed")
        np.testing.assert_allclose(env.goal[:3], target[:3])
        np.testing.assert_allclose(env.goal[3:], [1., 0., 0., 0.])

        # Anchors are observation-scoped; refresh the same symbolic target at
        # the post-retreat observation before asking for wrist reorientation.
        anchors.bind("mug_pregrasp", "left", Anchor(tuple(target), env.sequence))
        reorient = run.execute({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "position_mode": "preserve_current"}})
        self.assertEqual(reorient.stop_reason, "completed")
        np.testing.assert_allclose(env.goal[:3], target[:3])
        np.testing.assert_allclose(env.goal[3:], target[3:])

    def test_bad_plan_and_insufficient_tick_capacity_do_not_write(self):
        for failed_plan in (True, False):
            env = AnalyticEnvironment()
            if failed_plan: env.plan_status = "Fail"
            run, _, logs = executor(env)
            result = run.execute({**MOVE, "parameters": {"max_physics_ticks": 10}})
            self.assertEqual(result.stop_reason, "planning_failure" if failed_plan else "step_budget")
            self.assertEqual(env.writes, [])
            self.assertEqual(env.holds, 0)
            plan_result = next(event for event in logs
                               if event.get("event_type") == "analytic_plan_result")
            self.assertEqual(plan_result["planner_status"], "failure" if failed_plan else "success")
            self.assertEqual(plan_result["trajectory_samples"], 4)
            self.assertEqual(plan_result["result"], "rejected" if failed_plan else "accepted")
            self.assertEqual(plan_result["rejection_reason"], "planning_failure" if failed_plan else None)

    def test_relative_rotation_and_policy_history_refresh(self):
        env = AnalyticEnvironment()
        run, worker, _ = executor(env)
        run.execute(COMMAND)
        result = run.execute({"action": "rotate_wrist", "arm": "right",
                              "parameters": {"delta_yaw_deg": 20.0}})
        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(env.goal[3:], [np.cos(np.pi/18),0,0,np.sin(np.pi/18)])
        run.execute(COMMAND)
        self.assertEqual(worker.policy_resets, 2)

    def test_opening_requires_evidence_for_both_gripper_entry_points(self):
        for command in ({"action": "release", "arm": "left"},
                        {"action": "set_gripper", "arm": "left", "parameters": {"value": 1.0}}):
            env = AnalyticEnvironment()
            run, _, _ = executor(env)
            result = run.execute(command)
            self.assertEqual(result.stop_reason, "release_not_authorized")
            self.assertEqual(env.ticks, 0)

    def test_authorized_open_finishes_even_when_native_success_starts_at_point_eight(self):
        env = AnalyticEnvironment()
        env.success_on_open = True
        gates = PredicateRegistry(HangingMugPredicates(lambda obs: evidence(obs.sequence), source="development_oracle"))
        run, _, _ = executor(env, predicates=gates, oracle=True)
        result = run.execute({"action": "release", "arm": "left", "parameters": {"ramp_ticks": 50}})
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(env.arms["left"]["gripper"], 1.0)
        self.assertEqual(result.executed_physics_ticks, 75)
        self.assertTrue(result.task_success)
        self.assertEqual(result.information_source, "development_oracle")

    def test_unknown_support_during_known_release_returns_fresh_state_for_reassessment(self):
        from harness.predicates.base import PredicateResult
        from harness.skills.schemas import is_recoverable_release_support_gap

        class SupportDropout:
            source = "perception"

            def evaluate(self, name, observation, *, arm="auto"):
                value = True if observation.sequence == 0 else None
                return PredicateResult(name=name, value=value, source="perception",
                    observation_sequence=observation.sequence,
                    reason="supported_before_open" if value else "temporary_mask_dropout")

        commands = (
            {"action": "release", "arm": "left", "parameters": {"ramp_ticks": 50}},
            {"action": "set_gripper", "arm": "left",
             "parameters": {"value": 1.0, "ramp_ticks": 50}},
        )
        for command in commands:
            env = AnalyticEnvironment()
            run, _, _ = executor(env, predicates=PredicateRegistry(perception=SupportDropout()))
            result = run.execute(command)
            self.assertEqual(result.stop_reason, "release_support_lost")
            self.assertFalse(result.completed)
            self.assertFalse(result.episode_terminal)
            self.assertFalse(result.execution_uncertain)
            self.assertEqual(result.executed_physics_ticks, 25)
            self.assertGreater(result.observation_after, result.observation_before)
            self.assertIsNone(result.predicate.value)
            self.assertTrue(is_recoverable_release_support_gap(result))
            self.assertFalse(is_recoverable_release_support_gap(
                result.model_copy(update={"execution_uncertain": True})))
            self.assertFalse(is_recoverable_release_support_gap(
                result.model_copy(update={"information_source": "development_oracle"})))
            self.assertFalse(is_recoverable_release_support_gap(
                result.model_copy(update={"predicate": result.predicate.model_copy(update={"value": False})})))

    def test_actual_loss_of_release_support_remains_terminal(self):
        from harness.predicates.base import PredicateResult

        class LostSupport:
            source = "perception"

            def evaluate(self, name, observation, *, arm="auto"):
                return PredicateResult(name=name, value=True if observation.sequence == 0 else False,
                    source="perception", observation_sequence=observation.sequence,
                    reason="support_lost")

        env = AnalyticEnvironment()
        run, _, _ = executor(env, predicates=PredicateRegistry(perception=LostSupport()))
        result = run.execute({"action": "release", "arm": "left",
                              "parameters": {"ramp_ticks": 50}})
        self.assertEqual(result.stop_reason, "release_support_lost")
        self.assertTrue(result.episode_terminal)

    def test_tracking_error_stops_without_claiming_completion(self):
        env = AnalyticEnvironment()
        tick = env.tick
        def slip(*args):
            tick(*args)
            env.arms["left"]["joints"] += 0.3
        env.tick = slip
        run, _, logs = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertEqual(env.ticks, 3)
        self.assertTrue(result.episode_terminal)
        self.assertIsNotNone(result.tracking_error)
        self.assertEqual(result.tracking_error.arm, "left")
        self.assertEqual(result.tracking_error.comparison, "controlled_arm")
        self.assertEqual(result.tracking_error.joint_index, 0)
        self.assertAlmostEqual(result.tracking_error.measured_position_rad, 0.5066667)
        self.assertAlmostEqual(result.tracking_error.commanded_position_rad, 0.2066667)
        self.assertAlmostEqual(result.tracking_error.residual_rad, 0.3)
        diagnostic = next(e for e in logs if e.get("event_type") == "tracking_error")
        self.assertEqual(len(diagnostic["diagnostic"]["residuals_rad"]), 6)
        self.assertEqual(diagnostic["diagnostic"]["consecutive_ticks"], 3)

    def test_authorized_right_reorientation_tracking_error_returns_measured_state(self):
        env = AnalyticEnvironment()
        tick = env.tick

        def slip(*args):
            tick(*args)
            env.arms["right"]["joints"][2] += 0.251

        env.tick = slip
        run, _, _ = executor(env)
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "offset_xyz": [0.02, 0.0, 0.0]}},
            allow_recoverable_tracking_error=True)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertFalse(result.episode_terminal)
        self.assertFalse(result.execution_uncertain)
        self.assertEqual(result.tracking_error.arm, "right")
        self.assertEqual(result.tracking_error.comparison, "controlled_arm")
        self.assertIsNotNone(result.state_after)
        from harness.runtime.orchestrator import recovery_flags
        self.assertTrue(recovery_flags(result)["recoverable_controlled_tracking_error"])

    def test_single_tracking_residual_spike_does_not_stop_motion(self):
        env = AnalyticEnvironment()
        tick = env.tick
        calls = 0
        def one_tick_spike(*args):
            nonlocal calls
            tick(*args)
            calls += 1
            if calls == 1:
                env.arms["left"]["joints"][0] += 0.251
        env.tick = one_tick_spike
        run, _, logs = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "completed")
        self.assertFalse(any(event.get("event_type") == "tracking_error" for event in logs))

    def test_severe_tracking_residual_stops_immediately(self):
        env = AnalyticEnvironment()
        tick = env.tick
        def severe_slip(*args):
            tick(*args)
            env.arms["left"]["joints"][0] += 0.51
        env.tick = severe_slip
        run, _, _ = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertEqual(env.ticks, 1)

    def test_tracking_error_identifies_motion_in_inactive_arm(self):
        env = AnalyticEnvironment()
        tick = env.tick
        def inactive_slip(*args):
            tick(*args)
            env.arms["right"]["joints"][4] += 0.3
        env.tick = inactive_slip
        run, _, _ = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertEqual(result.tracking_error.arm, "right")
        self.assertEqual(result.tracking_error.comparison, "inactive_arm")
        self.assertEqual(result.tracking_error.joint_index, 4)

    def test_inactive_arm_tracking_error_remains_terminal_even_if_recovery_was_enabled(self):
        env = AnalyticEnvironment()
        tick = env.tick

        def inactive_slip(*args):
            tick(*args)
            env.arms["left"]["joints"][4] += 0.3

        env.tick = inactive_slip
        run, _, _ = executor(env)
        result = run.execute({"action": "move_to", "arm": "right",
            "parameters": {"target": "current_ee", "offset_xyz": [0.02, 0.0, 0.0]}},
            allow_recoverable_tracking_error=True)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertTrue(result.episode_terminal)
        self.assertEqual(result.tracking_error.comparison, "inactive_arm")


class PredicateTests(unittest.TestCase):
    def test_workspace_loss_is_reported_as_false(self):
        gate = HangingMugPredicates(
            lambda obs: evidence(obs.sequence, mug_in_task_workspace=False),
            source="development_oracle")
        result = gate.evaluate("mug_in_workspace", Observation(12, images(0), (0.,)*14))
        self.assertFalse(result.value)
        self.assertEqual(result.source, "development_oracle")

    def test_table_stage_requires_stable_support_and_two_observations(self):
        frames = {
            0: evidence(0, table_contact=True, left_gripper=1.0, stage_distance_m=0.03),
            1: evidence(1, table_contact=True, left_gripper=1.0, stage_distance_m=0.03,
                right_contact=True),
            2: evidence(2, table_contact=True, left_gripper=1.0, stage_distance_m=0.03),
            3: evidence(3, table_contact=True, left_gripper=1.0, stage_distance_m=0.03),
        }
        gate = HangingMugPredicates(lambda obs: frames[obs.sequence], source="development_oracle")
        def check(n): return gate.evaluate("mug_staged_on_table",
            Observation(n, images(0), (0.,)*14)).value
        self.assertFalse(check(0))
        self.assertFalse(check(1))  # Right contact invalidates staging evidence.
        self.assertFalse(check(2))
        self.assertTrue(check(3))

    def test_mug_staging_does_not_require_arm_clearance(self):
        frames = {
            n: evidence(n, table_contact=True, left_gripper=1.0, stage_distance_m=0.11)
            for n in (0, 1)
        }
        gate = HangingMugPredicates(lambda obs: frames[obs.sequence], source="development_oracle")
        def check(n, name):
            return gate.evaluate(name, Observation(n, images(0), (0.,)*14)).value
        self.assertFalse(check(0, "mug_placed_on_table"))
        self.assertTrue(check(1, "mug_placed_on_table"))
        self.assertTrue(check(1, "mug_staged_on_table"))

    def test_left_transfer_requires_only_stable_table_support(self):
        frames = {
            0: evidence(0, table_contact=True, left_gripper=1.0, stage_distance_m=0.03),
            1: evidence(1, table_contact=True, left_gripper=1.0, stage_distance_m=0.03),
        }
        gate = HangingMugPredicates(lambda obs: frames[obs.sequence], source="development_oracle")
        def check(n): return gate.evaluate("left_transfer_complete",
            Observation(n, images(0), (0.,)*14)).value
        self.assertFalse(check(0))  # Placement evidence has only one frame.
        self.assertTrue(check(1))  # Two-frame placement; arm position is irrelevant.

    def test_grasp_failure_monitor_stops_after_closed_no_lift_window(self):
        gate = HangingMugPredicates(lambda obs: evidence(obs.sequence, left_gripper=0.0),
                                    source="development_oracle")
        observation = lambda n: Observation(n, images(0), (0.,)*14)
        gate.evaluate("mug_lifted", observation(0))
        self.assertTrue(gate.begin_attempt("grasp_attempt_failed", observation(0), arm="left"))
        for n in range(1, 80):
            gate.evaluate("mug_lifted", observation(n))
            self.assertFalse(gate.evaluate("grasp_attempt_failed", observation(n)).value)
        gate.evaluate("mug_lifted", observation(80))
        self.assertTrue(gate.evaluate("grasp_attempt_failed", observation(80)).value)

    def test_recorded_fast_lift_scenario_does_not_trigger_failure_monitor(self):
        # Seed-3 recordings showed a successful retry lifting within 27 closed
        # observations. Keep that measured boundary as a CPU regression case.
        def recorded_success(obs):
            lifted = obs.sequence >= 26
            return evidence(obs.sequence, left_contact=lifted, left_bilateral=lifted,
                table_contact=not lifted, mug_position=(-0.2, 0.0, 0.80 if lifted else 0.75),
                left_gripper=0.0)

        gate = HangingMugPredicates(recorded_success, source="development_oracle")
        observation = lambda n: Observation(n, images(0), (0.,)*14)
        gate.evaluate("mug_lifted", observation(0))
        self.assertTrue(gate.begin_attempt("grasp_attempt_failed", observation(0), arm="left"))
        for n in range(1, 28):
            lifted = gate.evaluate("mug_lifted", observation(n)).value
        self.assertTrue(lifted)
        self.assertFalse(gate.evaluate("grasp_attempt_failed", observation(27)).value)

    def test_grasp_failure_monitor_stops_after_debounced_contact_loss(self):
        def failed_contact(obs):
            # Contact acquired after begin_attempt belongs to this VLA call.
            touching = 1 <= obs.sequence < 3
            return evidence(obs.sequence, left_contact=touching,
                            left_bilateral=touching, left_gripper=0.0)

        gate = HangingMugPredicates(failed_contact, source="development_oracle")
        observation = lambda n: Observation(n, images(0), (0.,) * 14)
        gate.evaluate("mug_lifted", observation(0))
        self.assertTrue(gate.begin_attempt(
            "grasp_attempt_failed", observation(0), arm="left"))
        for n in range(1, 6):
            gate.evaluate("mug_lifted", observation(n))
            self.assertFalse(gate.evaluate(
                "grasp_attempt_failed", observation(n)).value)
        gate.evaluate("mug_lifted", observation(6))
        result = gate.evaluate("grasp_attempt_failed", observation(6))
        self.assertTrue(result.value)
        self.assertEqual(result.reason, "closed_gripper_lost_mug_contact")

    def test_grasp_failure_monitor_ignores_loss_of_inherited_analytic_contact(self):
        def inherited_contact(obs):
            touching = obs.sequence < 3
            return evidence(obs.sequence, left_contact=touching,
                            left_bilateral=touching, left_gripper=0.0)

        gate = HangingMugPredicates(inherited_contact, source="development_oracle")
        observation = lambda n: Observation(n, images(0), (0.,) * 14)
        gate.evaluate("mug_lifted", observation(0))
        self.assertTrue(gate.begin_attempt(
            "grasp_attempt_failed", observation(0), arm="left"))

        for n in range(1, 12):
            gate.evaluate("mug_lifted", observation(n))
            result = gate.evaluate("grasp_attempt_failed", observation(n))
            self.assertFalse(result.value)
        self.assertNotEqual(result.reason, "closed_gripper_lost_mug_contact")

    def test_grasp_failure_monitor_stops_closed_hand_far_from_mug(self):
        gate = HangingMugPredicates(
            lambda obs: evidence(obs.sequence, right_gripper=0.0,
                                 right_tcp=(0.10, 0.0, 0.75)),
            source="development_oracle")
        observation = lambda n: Observation(n, images(0), (0.,) * 14)
        gate.evaluate("right_mug_lifted", observation(0))
        self.assertTrue(gate.begin_attempt(
            "grasp_attempt_failed", observation(0), arm="right"))
        for n in range(1, 8):
            gate.evaluate("right_mug_lifted", observation(n))
            self.assertFalse(gate.evaluate(
                "grasp_attempt_failed", observation(n)).value)
        gate.evaluate("right_mug_lifted", observation(8))
        result = gate.evaluate("grasp_attempt_failed", observation(8))
        self.assertTrue(result.value)
        self.assertEqual(result.reason, "closed_gripper_outside_grasp_region")

    def test_empty_grasp_temporal_confirmation_lift_and_handoff(self):
        frames = {
            0: evidence(0),
            1: evidence(1, left_contact=True, left_bilateral=True, table_contact=False,
                        mug_position=(-0.2,0.0,0.80)),
            2: evidence(2, left_contact=True, left_bilateral=True, table_contact=False,
                        mug_position=(-0.2,0.0,0.81)),
            3: evidence(3, left_contact=True, left_bilateral=True, right_contact=True,
                        right_bilateral=True, table_contact=False, mug_position=(-0.2,0.0,0.81)),
            4: evidence(4, right_contact=True, right_bilateral=True, table_contact=False,
                        mug_position=(-0.2,0.0,0.81)),
        }
        gate = HangingMugPredicates(lambda obs: frames[obs.sequence], source="development_oracle")
        def check(n, name): return gate.evaluate(name, Observation(n, images(0), (0.,)*14)).value
        self.assertFalse(check(0, "mug_grasp_confirmed"))
        self.assertFalse(check(1, "mug_lifted"))
        self.assertFalse(check(1, "mug_lifted"))  # Duplicate reads cannot confirm.
        self.assertTrue(check(2, "mug_lifted"))
        self.assertFalse(check(3, "handoff_confirmed"))
        self.assertTrue(check(4, "handoff_confirmed"))

    def test_predicate_stops_mid_chunk_with_explicit_oracle_opt_in(self):
        env = AnalyticEnvironment()
        def sample(obs):
            return evidence(obs.sequence, left_contact=obs.sequence>0,
                            left_bilateral=obs.sequence>0)
        gates = PredicateRegistry(HangingMugPredicates(sample, source="development_oracle"))
        run, worker, _ = executor(env, predicates=gates, oracle=True)
        result = run.execute({"action": "vla_act", "stop": "mug_grasp_confirmed",
                              "parameters": {"chunk_size": 8}})
        self.assertEqual(result.stop_reason, "predicate_satisfied")
        self.assertEqual((result.executed_policy_actions, result.discarded_actions), (2, 6))
        self.assertFalse(result.episode_terminal)
        self.assertEqual(result.information_source, "development_oracle")
        self.assertTrue(result.objective_satisfied)
        run, worker, _ = executor(AnalyticEnvironment(), predicates=gates, oracle=False)
        result = run.execute({"action": "vla_act", "stop": "mug_lifted"})
        self.assertEqual(result.stop_reason, "predicate_unavailable")
        self.assertEqual(worker.chunks, 0)

    def test_public_perception_stop_is_not_polled_inside_a_vla_chunk(self):
        from harness.predicates.base import PredicateResult

        class PublicProvider:
            source = "perception"

            def __init__(self):
                self.calls = 0

            def evaluate(self, name, observation, *, arm="auto"):
                self.calls += 1
                return PredicateResult(name=name, value=False, source="perception",
                    observation_sequence=observation.sequence, reason="test_public_frame")

        provider = PublicProvider()
        gates = PredicateRegistry(perception=provider)
        run, worker, _ = executor(AnalyticEnvironment(), predicates=gates)
        # Executor construction primes one reset observation. The VLA chunk
        # must not issue another public perception query per action.
        self.assertEqual(provider.calls, 1)
        result = run.execute({"action": "vla_act", "stop": "mug_lifted",
            "parameters": {"max_chunks": 1, "chunk_size": 4,
                           "max_policy_actions": 4}})
        self.assertEqual(result.executed_policy_actions, 4)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(worker.chunks, 1)
