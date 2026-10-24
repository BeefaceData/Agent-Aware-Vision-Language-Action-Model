import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from development.recovery.manual_skill import run_manual
from development.recovery.retry_grasp_context import RetryGraspContext
from harness.skills.motion import SkillStopped, TickExecutor, validate_trajectory
from harness.legacy.recovery_skills.retry_grasp import RetryGrasp
from tests import test_checkpoints as checkpoint_fixtures


class FakeRobot:
    def __init__(self):
        self.state = [0.0] * 14
        self.state[13] = 1.0
        self.left_arm_joints = []
        self.ee = [-0.2, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]
        self.writes = []
        self.plan_status = "Success"

    def get_left_arm_jointState(self): return self.state[:7]
    def get_right_arm_jointState(self): return self.state[7:]
    def get_left_arm_real_jointState(self): return self.state[:7]
    def get_left_gripper_val(self): return self.state[6]
    def get_right_gripper_val(self): return self.state[13]
    def get_left_ee_pose(self): return self.ee.copy()
    def get_left_tcp_pose(self): return self.ee.copy()
    def get_right_tcp_pose(self): return [0.3, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0]

    def set_arm_joints(self, position, velocity, arm):
        self.writes.append((arm, list(position), list(velocity)))
        start = 0 if arm == "left" else 7
        self.state[start:start+6] = list(position)

    def set_gripper(self, value, arm):
        self.state[6 if arm == "left" else 13] = value

    def left_plan_path(self, pose):
        return {"status": self.plan_status, "position": np.zeros((2, 6)), "velocity": np.zeros((2, 6))}


class FakeTask:
    def __init__(self):
        self.robot = FakeRobot()
        self.position = np.array([-0.2, 0.0, 0.75])
        self.mug = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=self.position), actor=SimpleNamespace(name="mug"))
        self.scene = SimpleNamespace(step=self.step, get_contacts=lambda: [SimpleNamespace(points=[1], bodies=[
            SimpleNamespace(entity=SimpleNamespace(name="mug")),
            SimpleNamespace(entity=SimpleNamespace(name="table"))])])
        self.plan_success, self.need_plan, self.eval_success = True, True, False
        self.ticks = 0
        self.fail_at = None

    def check_success(self): return self.eval_success
    def _arm_joint_bounds(self, joints): return np.tile([-3.14, 3.14], (6, 1))
    def set_gripper(self, set_tag, left_pos):
        return {"result": np.linspace(self.robot.get_left_gripper_val(), left_pos, 3)}
    def get_grasp_pose(self, mug, arm, contact_point_id, pre_dis):
        return [-0.2, 0.0, 0.85, 1.0, 0.0, 0.0, 0.0]

    def step(self):
        if self.ticks == self.fail_at:
            raise RuntimeError("SECRET_MUST_NOT_APPEAR")
        self.ticks += 1


class RecoveryExecutionTests(unittest.TestCase):
    def setUp(self):
        from tests.skill_fixtures import isolate_simulator_dependencies
        isolate_simulator_dependencies(self)

    def run_skill(self, task=None, **kwargs):
        task = task or FakeTask()
        events = []
        context = RetryGraspContext(task, log=events.append, **kwargs)
        result = RetryGrasp().execute(context, {})
        return task, context, events, result

    def test_real_skill_order_and_exact_tick_accounting(self):
        task, ctx, events, result = self.run_skill()
        self.assertEqual(result.termination_reason, "completed")
        self.assertIsNone(result.success_hint)
        self.assertEqual(result.executed_steps, 64)  # 2 gripper ramps, 4 plans, 50 settle ticks.
        self.assertEqual(task.ticks, result.executed_steps)
        self.assertEqual([e["stage"] for e in events if e["event"] == "stage"],
                         ["open", "retreat", "pregrasp", "approach", "close", "lift"])
        self.assertEqual(len([e for e in events if e["event"] == "tick_confirmed"]), task.ticks)
        self.assertEqual(result.metadata["step_unit"], "physics_tick")
        self.assertFalse(RetryGrasp().runtime_observation_compatible)
        self.assertFalse(RetryGrasp().returns_to_policy)
        for arm, _, velocity in task.robot.writes:
            if arm == "right": self.assertEqual(velocity, [0.0] * 6)

    def test_parameters_rejected_before_context_begin(self):
        for params in ({"arm": "right"}, {"backoff_scale": 0.1}, {"lift_distance": float("nan")},
                       {"action": [0.0] * 14}, {"backoff_scale": "0.03"}):
            ctx = RetryGraspContext(FakeTask(), log=lambda x: None)
            with self.assertRaises(ValidationError):
                RetryGrasp().execute(ctx, params)
            self.assertIsNone(ctx.executor)

    def test_unsupported_preconditions_issue_no_drive_writes(self):
        for change in ("open", "dropped", "rack", "planner", "held", "unsupported"):
            task = FakeTask()
            if change == "open": task.robot.state[6] = 1.0
            if change == "dropped": task.position[2] = 0.5
            if change == "rack": task.position[0] = 0.3
            if change == "planner": task.plan_success = False
            if change == "unsupported": task.scene.get_contacts = lambda: []
            if change == "held":
                task.scene.get_contacts = lambda: [SimpleNamespace(points=[1], bodies=[
                    SimpleNamespace(entity=SimpleNamespace(name="mug")),
                    SimpleNamespace(entity=SimpleNamespace(name=name))]) for name in ("fl_link7", "fl_link8")]
            _, _, _, result = self.run_skill(task)
            self.assertIn(result.termination_reason, ("precondition_failed", "safety_violation"))
            self.assertEqual(task.ticks, 0)
            self.assertEqual(task.robot.writes, [])

    def test_budget_refuses_whole_stage_and_never_overruns(self):
        task, _, _, result = self.run_skill(max_ticks=4)
        self.assertEqual(result.termination_reason, "step_budget")
        self.assertEqual(task.ticks, 3)  # Open complete, two-tick retreat will not fit.
        self.assertEqual(result.executed_steps, 3)

    def test_cancellation_and_task_success_stop_before_next_tick(self):
        task = FakeTask()
        _, _, _, result = self.run_skill(task, cancel=lambda: task.ticks >= 1)
        self.assertEqual(result.termination_reason, "cancelled")
        self.assertEqual(task.ticks, 1)
        task = FakeTask()
        task.check_success = lambda: task.ticks >= 2
        _, _, _, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "task_complete")
        self.assertEqual(task.ticks, 2)

    def test_planning_failure_stops_without_motion_from_failed_plan(self):
        task = FakeTask()
        task.robot.plan_status = "Fail"
        _, _, events, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "planning_failure")
        self.assertEqual(task.ticks, 3)
        self.assertTrue(all(e["stage"] == "open" for e in events if e["event"] == "tick_confirmed"))

    def test_exception_keeps_requested_unconfirmed_and_never_claims_success(self):
        task = FakeTask()
        task.fail_at = 1
        _, _, events, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "execution_error")
        self.assertEqual(result.executed_steps, 1)
        self.assertEqual(events[-2]["event"], "tick_requested")
        self.assertEqual(events[-1]["event"], "velocity_targets_zeroed")
        self.assertNotIn("SECRET_MUST_NOT_APPEAR", result.model_dump_json())
        self.assertEqual(task.robot.writes[-1][2], [0.0]*6)

    def test_logging_failure_prevents_next_actuation(self):
        task = FakeTask()
        def log(event):
            if event["event"] == "tick_requested": raise OSError("disk full")
        ctx = RetryGraspContext(task, log=log)
        result = RetryGrasp().execute(ctx, {})
        self.assertEqual(result.termination_reason, "execution_error")
        self.assertEqual(task.ticks, 0)
        self.assertEqual(task.robot.writes, [])

    def test_invalid_trajectories_rejected_before_motion(self):
        bounds = np.tile([-3.14, 3.14], (6, 1))
        valid = {"status": "Success", "position": np.zeros((2, 6)), "velocity": np.zeros((2, 6))}
        for updates in ({"position": np.zeros((2, 14))}, {"position": np.zeros((0, 6))},
                        {"position": np.full((2, 6), float("nan"))},
                        {"position": np.ones((2, 6))}, {"velocity": np.ones((2, 6))*6},
                        {"position": np.ones((2, 6))*4}, {"status": "Fail"}):
            with self.assertRaises(SkillStopped): validate_trajectory({**valid, **updates}, bounds, np.zeros(6))

    def test_direct_executor_budget_checks_before_callback(self):
        calls, logs = [], []
        ex = TickExecutor(1, advance=lambda *args: calls.append(args), guard=lambda: None, log=logs.append)
        ex.tick("test", np.zeros(6), np.zeros(6), 0.0)
        with self.assertRaises(SkillStopped): ex.tick("test", np.zeros(6), np.zeros(6), 0.0)
        self.assertEqual(len(calls), 1)

    def test_invalid_tick_command_never_reaches_actuator(self):
        calls = []
        ex = TickExecutor(3, advance=lambda *args: calls.append(args), guard=lambda: None, log=lambda x: None)
        for q, v, g in ((np.zeros(14), np.zeros(6), 0.0),
                        (np.zeros(6), np.full(6, float("nan")), 0.0),
                        (np.zeros(6), np.zeros(6), 1.1)):
            with self.assertRaises(SkillStopped): ex.tick("invalid", q, v, g)
        self.assertEqual(calls, [])

    def test_mug_leaving_local_region_interrupts_before_next_tick(self):
        task = FakeTask()
        def advance():
            task.ticks += 1
            task.position[0] -= 0.3
        task.scene.step = advance
        _, _, _, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "safety_violation")
        self.assertEqual(result.executed_steps, 1)

    def test_geometry_offset_uses_contact_orientation(self):
        task = FakeTask()
        ctx = RetryGraspContext(task, log=lambda x: None)
        ctx.begin(3000)
        pre, grasp = ctx.grasp_targets(0.05)
        np.testing.assert_allclose(grasp[:3]-pre[:3], [0.05, 0.0, 0.0])

    def test_context_single_use_and_hash_covers_adapter(self):
        task, ctx, _, _ = self.run_skill()
        with self.assertRaises(ValueError): RetryGrasp().execute(ctx, {})
        skill = RetryGrasp()
        self.assertTrue(all(p.exists() for p in skill.dependency_files))
        self.assertIn(Path('development/recovery/retry_grasp_context.py').resolve(), skill.dependency_files)
        self.assertEqual(len(skill.code_hash()), 64)

    def test_manual_rejects_unlabeled_checkpoint_without_reset(self):
        fixture = checkpoint_fixtures.CheckpointTests()
        fixture.setUp()
        try:
            cp, _, _ = fixture.capture()
            with self.assertRaises(ValueError):
                run_manual(fixture.bank, cp, None, fixture.root / "attempt")
            self.assertFalse((fixture.root / "attempt").exists())
        finally: fixture.doCleanups()

    def test_manual_drift_blocks_skill_and_writes_report(self):
        fixture = checkpoint_fixtures.CheckpointTests()
        fixture.setUp()
        try:
            cp, _, _ = fixture.capture()
            cp = fixture.bank.relabel(cp.checkpoint_id, phase="grasp_mug", failure_type="missed_grasp")
            from development.checkpoints.schemas import ReplayReport
            drift = ReplayReport(checkpoint_id=cp.checkpoint_id, status="drifted", expected_steps=cp.step,
                                 executed_steps=0, reason="test reset drift")
            with patch("development.recovery.manual_skill.ReplayEnvironment.restore_checkpoint", return_value=drift):
                result = run_manual(fixture.bank, cp, None, fixture.root / "attempt")
            self.assertEqual(result["status"], "restore_rejected")
            self.assertFalse((fixture.root / "attempt/execution.jsonl").exists())
            self.assertEqual(json.loads((fixture.root / "attempt/report.json").read_text())["recovery_outcome"], "unknown")
        finally: fixture.doCleanups()

    def test_manual_completion_keeps_outcome_unknown_and_never_calls_policy(self):
        fixture = checkpoint_fixtures.CheckpointTests()
        fixture.setUp()
        try:
            cp, _, _ = fixture.capture()
            cp = fixture.bank.relabel(cp.checkpoint_id, phase="grasp_mug", failure_type="missed_grasp")
            from development.checkpoints.schemas import ReplayReport
            from tests.helpers import images
            matched = ReplayReport(checkpoint_id=cp.checkpoint_id, status="matched", expected_steps=cp.step,
                                   executed_steps=cp.step, reason="fixture")
            simulator = SimpleNamespace(task=FakeTask(), blank=np.zeros((2, 3, 3), dtype=np.uint8))
            with patch("development.recovery.manual_skill.ReplayEnvironment.restore_checkpoint", return_value=matched), \
                    patch("harness.environment.robotwin.observation_of", return_value=images()), \
                    patch("harness.vla.modal_client.ModalPolicyClient", side_effect=AssertionError("no cloud call")):
                report = run_manual(fixture.bank, cp, simulator, fixture.root / "attempt")
            self.assertEqual(report["status"], "invoked")
            self.assertEqual(report["recovery_outcome"], "unknown")
            self.assertFalse(report["runtime_eligible"])
            self.assertFalse(report["policy_resumed"])
            self.assertTrue((fixture.root / "attempt/before.npz").exists())
            self.assertTrue((fixture.root / "attempt/after.npz").exists())
            self.assertIsNone(report["result"]["success_hint"])
        finally: fixture.doCleanups()

    def test_local_simulator_uses_baseline_reset_and_target_calls(self):
        from development.simulator.local_simulator import LocalSimulator
        from unittest.mock import Mock
        import contextlib
        task = Mock(eval_success=False)
        task.check_success.return_value = False
        torch_stub = SimpleNamespace(enable_grad=contextlib.nullcontext)
        with patch.dict("sys.modules", {"torch": torch_stub}), \
                patch("harness.environment.robotwin.build_task", return_value=(task, {"task_name": "hanging_mug"})), \
                patch("harness.environment.robotwin.observation_of", return_value={"sentinel": True}):
            simulator = LocalSimulator("RoboTwin")
            with self.assertRaises(ValueError): simulator("local", 0, "/step", {"action": [0.0]*14})
            simulator("local", 0, "/reset", {"seed": 7})
            task.setup_demo.assert_called_once_with(task_name="hanging_mug", seed=7, is_test=True)
            observation, _ = simulator("local", 0, "/step", {"action": [0.0]*14})
            np.testing.assert_array_equal(task.take_action.call_args.args[0], np.zeros(14))
            self.assertTrue(observation["sentinel"])
            simulator.close()
            task.close_env.assert_called_once()
