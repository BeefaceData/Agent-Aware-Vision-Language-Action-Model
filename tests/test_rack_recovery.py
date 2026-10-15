import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from development.capture_rack_failure import capture
from development.checkpoint_bank import CheckpointBank, load_prefix
from development.rack_approach_context import RackApproachContext
from development.rack_geometry import premature_release_candidate, rack_diagnostics, release_geometry_ready, supported_release_ready
from harness.skill_executor import SkillStopped
from harness.legacy.recovery_skills.backoff_reapproach_rack import BackoffReapproachRack
from tests.test_baseline_compatibility import FakeRuntime
from tests.test_recovery_execution import FakeRobot
from tests import test_checkpoints as checkpoint_fixtures


class RackRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.state[6], self.state[13] = 1.0, 0.0
        self.right_arm_joints = []
        self.right_ee = [-0.05, -0.12, 0.91, 1.0, 0.0, 0.0, 0.0]
        self.pending = None

    def get_right_ee_pose(self): return self.right_ee.copy()
    def get_right_tcp_pose(self): return self.right_ee.copy()
    def get_right_arm_real_jointState(self): return self.state[7:]

    def right_plan_path(self, pose):
        self.pending = pose
        return {"status": self.plan_status, "position": np.zeros((2, 6)), "velocity": np.zeros((2, 6))}


class RackTask:
    def __init__(self):
        self.robot = RackRobot()
        self.position = np.array([-0.05, -0.12, 0.86])
        self.rotation = [1.0, 0.0, 0.0, 0.0]
        self.mug = SimpleNamespace(actor=SimpleNamespace(name="mug"),
            get_pose=lambda: SimpleNamespace(p=self.position),
            get_functional_point=lambda i: [*self.position, *self.rotation])
        self.rack = SimpleNamespace(actor=SimpleNamespace(name="rack"), get_pose=lambda: SimpleNamespace(p=np.array([0.2, 0.15, 0.84])),
            get_functional_point=lambda i: [0.2, 0.15, 0.94, 1.0, 0.0, 0.0, 0.0])
        self.contacts = ["fr_link7", "fr_link8"]
        self.scene = SimpleNamespace(step=self.step, get_contacts=lambda: [SimpleNamespace(points=[1], bodies=[
            SimpleNamespace(entity=SimpleNamespace(name="mug")),
            SimpleNamespace(entity=SimpleNamespace(name=name))]) for name in self.contacts])
        self.need_plan = self.plan_success = True
        self.eval_success = False
        self.ticks = 0
        self.slip = False
        self.bad_target = False

    def _arm_joint_bounds(self, joints): return np.tile([-3.14, 3.14], (6, 1))
    def check_success(self): return self.eval_success
    def get_place_pose(self, **kwargs):
        if self.bad_target: return self.robot.right_ee.copy()
        return [0.2, 0.15, 0.99 - kwargs["pre_dis"], 1.0, 0.0, 0.0, 0.0]
    def set_gripper(self, set_tag, right_pos):
        return {"result": np.linspace(self.robot.get_right_gripper_val(), right_pos, 3)}

    def step(self):
        self.ticks += 1
        if self.robot.pending is not None:
            self.robot.right_ee = list(self.robot.pending)
        self.position = np.array(self.robot.right_ee[:3]) - [0, 0, 0.05]
        if self.slip: self.position[0] += 0.06


class RackRecoveryTests(unittest.TestCase):
    def setUp(self):
        from tests.skill_fixtures import isolate_simulator_dependencies
        isolate_simulator_dependencies(self)

    def run_skill(self, task=None, **kwargs):
        task = task or RackTask()
        events = []
        context = RackApproachContext(task, log=events.append, **kwargs)
        result = BackoffReapproachRack().execute(context, {})
        return task, context, events, result

    def test_approach_alignment_precede_any_opening(self):
        task, _, events, result = self.run_skill()
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.executed_steps, task.ticks)
        stages = [e["stage"] for e in events if e["event"] == "stage"]
        self.assertEqual(stages, ["rack_backoff", "rack_approach", "handle_alignment", "withdraw"])
        release = next(i for i,e in enumerate(events) if e["event"] == "release_authorized")
        self.assertTrue(all(e["right_gripper"] == 0 for e in events[:release] if e["event"] == "tick_requested"))
        self.assertTrue(any(e.get("right_gripper", 0) > 0 for e in events[release:]))
        self.assertTrue(result.metadata["release_authorized"])
        self.assertIsNone(result.success_hint)
        self.assertEqual(task.robot.state[6], 1.0)  # Inactive left gripper preserved.

    def test_native_success_during_release_does_not_skip_full_opening_and_withdrawal(self):
        task = RackTask()
        task.check_success = lambda: task.robot.state[13] > 0.8
        _, _, events, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(task.robot.state[13], 1.0)
        self.assertTrue(any(e.get("stage") == "withdraw" for e in events))
        self.assertTrue(any(e.get("stage") == "settle" for e in events))

    def test_withdrawal_requires_authorized_full_release(self):
        context = RackApproachContext(RackTask(), log=lambda e: None)
        context.begin(3000)
        for authorized, gripper in ((False, 1.0), (True, 0.8)):
            context.release_authorized, context.right_gripper = authorized, gripper
            with self.assertRaises(SkillStopped): context.withdraw()
        self.assertEqual(context.task.robot.writes, [])

    def test_never_release_when_motion_completed_but_still_far(self):
        task = RackTask()
        task.bad_target = True
        _, _, events, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "release_not_aligned")
        self.assertFalse(result.metadata["release_authorized"])
        self.assertEqual(task.robot.state[13], 0)
        self.assertFalse(any(e.get("right_gripper", 0) > 0 for e in events))

    def test_release_requires_correct_position_orientation_and_height(self):
        task = RackTask()
        task.position = np.array([0.2, 0.15, 0.94])
        valid = rack_diagnostics(task)
        self.assertTrue(release_geometry_ready(valid))
        for changes in ({"handle_xy_error_m": 0.1}, {"handle_frame_error_rad": 0.5},
                        {"mug_functional_pose": [0.2, 0.15, 0.8, 1, 0, 0, 0]},
                        {"handle_xy_error_m": float("nan")}):
            self.assertFalse(release_geometry_ready({**valid, **changes}))

    def test_supported_seating_requires_rack_contact_and_bounded_corridor(self):
        task = RackTask()
        task.rack.get_pose = lambda: SimpleNamespace(p=np.array([0.2, 0.15, 0.74]))
        task.position = np.array([0.2, 0.15, 0.87])
        task.contacts.append("rack")
        d = rack_diagnostics(task)
        self.assertFalse(release_geometry_ready(d))  # 7cm below tip, after entry gate.
        self.assertTrue(supported_release_ready(d, "rack"))
        for change in ({"mug_contact_entities": ["fr_link7", "fr_link8"]},
                       {"mug_contact_entities": ["rack", "table"]},
                       {"handle_xy_error_m": 0.019}, {"handle_frame_error_rad": 0.26},
                       {"mug_functional_pose": [0.24, 0.15, 0.87, 1, 0, 0, 0]},
                       {"mug_functional_pose": [0.2, 0.15, 0.96, 1, 0, 0, 0]},
                       {"mug_functional_pose": [0.2, 0.15, 0.85, 1, 0, 0, 0]}):
            self.assertFalse(supported_release_ready({**d, **change}, "rack"))
        # A supported seated state cannot bypass the initial release gate.
        context = RackApproachContext(task, log=lambda e: None); context.begin(3000)
        with self.assertRaises(SkillStopped): context.release_when_aligned()
        self.assertEqual(task.robot.writes, [])

    def test_empty_gripper_or_left_hand_or_table_support_rejected(self):
        for contacts in ([], ["fr_link7"], ["fr_link7", "fr_link8", "fl_link7"],
                         ["fr_link7", "fr_link8", "table"]):
            task = RackTask()
            task.contacts = contacts
            _, _, _, result = self.run_skill(task)
            self.assertEqual(result.termination_reason, "precondition_failed")
            self.assertEqual(task.ticks, 0)
            self.assertEqual(task.robot.writes, [])

    def test_grasp_slip_stops_before_opening(self):
        task = RackTask()
        task.slip = True
        _, _, _, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "grasp_lost")
        self.assertEqual(task.ticks, 1)
        self.assertEqual(task.robot.state[13], 0)

    def test_alignment_loss_during_early_release_stops_further_opening(self):
        task = RackTask()
        advance = task.step
        def move_during_release():
            advance()
            if task.robot.state[13] > 0:
                task.position[0] += 0.04
        task.scene.step = move_during_release
        task.set_gripper = lambda **kwargs: {"result": [0.0, 0.1, 0.2, 1.0]}
        _, _, events, result = self.run_skill(task)
        self.assertEqual(result.termination_reason, "release_not_aligned")
        self.assertTrue(result.metadata["release_authorized"])
        self.assertEqual(task.robot.state[13], 0.1)
        self.assertFalse(any(e.get("right_gripper", 0) > 0.1 for e in events))

    def test_planner_failure_budget_cancel_and_task_completion(self):
        task = RackTask()
        task.robot.plan_status = "Fail"
        self.assertEqual(self.run_skill(task)[-1].termination_reason, "planning_failure")
        self.assertEqual(task.ticks, 0)
        self.assertEqual(self.run_skill(max_ticks=1)[-1].termination_reason, "step_budget")
        self.assertEqual(self.run_skill(cancel=lambda: True)[-1].termination_reason, "cancelled")
        task = RackTask()
        task.eval_success = True
        self.assertEqual(self.run_skill(task)[-1].termination_reason, "task_complete")
        self.assertEqual(task.ticks, 0)

    def test_direct_opening_without_authorization_is_rejected(self):
        context = RackApproachContext(RackTask(), log=lambda e: None)
        context.begin(3000)
        with self.assertRaises(SkillStopped): context._advance(np.zeros(6), np.zeros(6), 1.0)
        self.assertEqual(context.task.robot.writes, [])

    def test_bad_parameters_fail_before_context_or_gpu_work(self):
        for parameters in ({"backoff_scale": 1.0}, {"open_gripper": True}, {"joint_targets": [0.0]*14}):
            with self.assertRaises(ValidationError): BackoffReapproachRack().execute(None, parameters)
        with self.assertRaises(ValueError): BackoffReapproachRack().execute(None, {})
        skill = BackoffReapproachRack()
        self.assertTrue(skill.candidate_only)
        self.assertFalse(skill.runtime_observation_compatible)
        self.assertEqual(len(skill.code_hash()), 64)

    def test_capture_cue_distinguishes_far_release_from_already_lost_mug(self):
        task = RackTask()
        state = task.robot.state
        proposed = state.copy(); proposed[13] = 0.2
        d = rack_diagnostics(task)
        self.assertTrue(premature_release_candidate(d, state, proposed))
        self.assertFalse(premature_release_candidate({**d, "right_bilateral_contact": False}, state, proposed))
        self.assertFalse(premature_release_candidate({**d, "handle_rack_distance_m": 0.05}, state, proposed))
        self.assertFalse(premature_release_candidate(d, state, state))

    def test_capture_stops_before_proposal_and_keeps_oracle_outside_public_trace(self):
        with tempfile.TemporaryDirectory() as td:
            bank = CheckpointBank(td)
            runtime = FakeRuntime()
            class Sim:
                task = RackTask()
                def __call__(self, *args): return runtime.sim(*args)
            with contextlib.redirect_stdout(io.StringIO()):
                result = capture(runtime, Sim(), bank, steps=5, video=False)
            self.assertEqual(result["reason"], "premature_release_candidate")
            self.assertEqual(runtime.actions, [])
            cp = bank.get(result["checkpoint_id"])
            self.assertEqual(cp.step, 0)
            self.assertEqual(len(load_prefix(bank, cp)), 1)
            self.assertEqual(cp.failure_type.value, "unknown")
            public = bank.resolve_artifact(cp.trace_path).read_text()
            self.assertNotIn("rack_midpoint", public)
            traces = list(Path(td).rglob("trace.jsonl"))
            events = [json.loads(line) for line in traces[0].read_text().splitlines()]
            self.assertEqual(events[-1]["event_type"], "chunk_discarded")
            self.assertFalse(any("privileged" in e for e in events))

    def test_manual_allowlist_dispatches_rack_candidate_and_keeps_outcome_unknown(self):
        from development.manual_skill import run_manual
        from development.schemas import ReplayReport
        from tests.helpers import images
        fixture = checkpoint_fixtures.CheckpointTests()
        fixture.setUp()
        try:
            cp, _, _ = fixture.capture()
            cp = fixture.bank.relabel(cp.checkpoint_id, phase="hang_mug", failure_type="bad_rack_approach")
            matched = ReplayReport(checkpoint_id=cp.checkpoint_id, status="matched", expected_steps=cp.step,
                                   executed_steps=cp.step, reason="fixture")
            simulator = SimpleNamespace(task=RackTask(), blank=np.zeros((2, 3, 3), dtype=np.uint8))
            with patch("development.manual_skill.ReplayEnvironment.restore_checkpoint", return_value=matched), \
                    patch("run_inference_chunked.observation_of", return_value=images()), \
                    patch("run_inference_chunked.ModalPolicyClient", side_effect=AssertionError("no policy call")):
                report = run_manual(fixture.bank, cp, simulator, fixture.root / "rack_manual",
                                    skill_id="backoff_reapproach_rack")
            self.assertEqual(report["status"], "invoked")
            self.assertEqual(report["result"]["skill_id"], "backoff_reapproach_rack")
            self.assertEqual(report["recovery_outcome"], "unknown")
            self.assertFalse(report["runtime_eligible"])
            self.assertFalse(report["policy_resumed"])
        finally: fixture.doCleanups()
