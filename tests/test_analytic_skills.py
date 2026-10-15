"""Focused R2/R3 contract checks; physical operating ranges still need live trials."""

import unittest
import numpy as np

from harness.environment.base import Observation
from harness.skills import SkillExecutor
from harness.predicates.registry import PredicateRegistry
from harness.predicates.hanging_mug import HangingMugPredicates, MugEvidence
from harness.vla.modal_client import ModalVLAClient
from tests.test_skills import Runtime, COMMAND
from tests.helpers import images


class AnalyticEnvironment:
    supports_analytic = True

    def __init__(self, x=-0.2):
        self.sequence = self.ticks = self.holds = 0
        self.arms = {arm: {"joints": np.full(6, 0.2 if arm == "left" else 0.4),
                          "pose": np.array([x, 0., 0.9, 1., 0., 0., 0.]), "gripper": 0.5}
                     for arm in ("left", "right")}
        self.plan_status = "Success"
        self.goal = None
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


def evidence(sequence, **kwargs):
    data = dict(observation_sequence=sequence, mug_position=(-0.2,0.0,0.75),
                left_tcp=(-0.2,0.0,0.8), right_tcp=(0.2,0.0,0.8),
                left_contact=False, right_contact=False, left_bilateral=False, right_bilateral=False,
                left_gripper=0.0, right_gripper=0.0, table_contact=True,
                rack_contact=False, rack_distance_m=0.4)
    data.update(kwargs)
    return MugEvidence(**data)


def executor(env, *, predicates=None, oracle=False):
    worker = Runtime()
    logs = []
    return SkillExecutor(env, ModalVLAClient(worker), log=logs.append,
                             predicates=predicates, allow_development_oracle=oracle), worker, logs


MOVE = {"action": "move_to", "arm": "left", "parameters": {"offset_xyz": [0.02, 0.0, 0.0]}}


class AnalyticTests(unittest.TestCase):
    def test_two_start_poses_and_measured_joints_not_commanded_start(self):
        for x in (-0.2, 0.1):
            env = AnalyticEnvironment(x)
            run, _, _ = executor(env)
            result = run.execute(MOVE)
            self.assertEqual(result.stop_reason, "completed")
            self.assertAlmostEqual(env.goal[0], x+0.02)
            np.testing.assert_allclose(env.writes[0][1], 0.2)
            np.testing.assert_allclose(env.writes[0][2], 0.4)
            self.assertEqual(result.executed_physics_ticks, 29)
            self.assertEqual(result.position_error_m, 0.0)

    def test_bad_plan_and_insufficient_tick_capacity_do_not_write(self):
        for failed_plan in (True, False):
            env = AnalyticEnvironment()
            if failed_plan: env.plan_status = "Fail"
            run, _, _ = executor(env)
            result = run.execute({**MOVE, "parameters": {"max_physics_ticks": 10}})
            self.assertEqual(result.stop_reason, "planning_failure" if failed_plan else "step_budget")
            self.assertEqual(env.writes, [])
            self.assertEqual(env.holds, 0)

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

    def test_tracking_error_stops_without_claiming_completion(self):
        env = AnalyticEnvironment()
        tick = env.tick
        def slip(*args):
            tick(*args)
            env.arms["left"]["joints"] += 0.3
        env.tick = slip
        run, _, _ = executor(env)
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "tracking_error")
        self.assertEqual(env.ticks, 1)
        self.assertTrue(result.episode_terminal)


class PredicateTests(unittest.TestCase):
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
