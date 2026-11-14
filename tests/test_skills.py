"""Focused R1 checks; no GPU, Modal request, or broad new test framework."""

import json
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from harness.environment.client import EnvironmentClient
from harness.predicates.base import PredicateResult
from harness.predicates.registry import PredicateRegistry
from harness.skills import SkillExecutor
from harness.skills.schemas import EpisodeBudget
from harness.skills.vla_act import VLAAct
from harness.vla.modal_client import ModalVLAClient
from tests.helpers import images


class Runtime:
    def __init__(self, *, success_at=None, malformed=False, step_error=False):
        self.resets = self.policy_resets = self.chunks = 0
        self.actions, self.inputs = [], []
        self.success_at, self.malformed, self.step_error = success_at, malformed, step_error

    def transport(self, host, port, path, payload):
        if path == "/reset":
            self.resets += 1
        else:
            self.actions.append(payload["action"])
            if self.step_error:
                raise ConnectionError("response lost after execution")
        n = len(self.actions)
        return {**images(n), "state": np.full(14, n, dtype=float),
                "success": np.asarray(self.success_at is not None and n >= self.success_at),
                "contacts": "PRIVATE", "mug_pose": "PRIVATE"}, {}

    def reset(self):
        self.policy_resets += 1
        return {"ok": True, "chunk_limit": 24}, 0.0

    def act_chunk(self, images, state, size):
        self.inputs.append(tuple(state))
        self.chunks += 1
        actions = np.full((size, 14), self.chunks / 10)
        if self.malformed:
            actions[-1, 0] = float("nan")
        return actions, {}


def setup_runtime(**kwargs):
    runtime = Runtime(**kwargs)
    environment = EnvironmentClient(runtime.transport)
    environment.reset(3)
    logs = []
    executor = SkillExecutor(environment, ModalVLAClient(runtime), log=logs.append)
    return runtime, executor, logs


COMMAND = {"action": "vla_act", "parameters": {
    "max_chunks": 1, "chunk_size": 4, "max_policy_actions": 2}}


class SkillTests(unittest.TestCase):
    def test_failure_monitor_discards_unexecuted_actions_from_current_chunk(self):
        class ClosedGripperMonitor:
            source = "perception"

            def begin_attempt(self, name, observation, *, arm):
                return name == "grasp_attempt_failed" and arm == "right"

            def evaluate(self, name, observation, *, arm="auto"):
                failed = name == "grasp_attempt_failed" and observation.sequence >= 80
                return PredicateResult(name=name, value=failed, source="perception",
                    observation_sequence=observation.sequence,
                    reason="closed_without_lift" if failed else "monitor_pending")

        runtime = Runtime()
        environment = EnvironmentClient(runtime.transport)
        environment.reset(3)
        logs = []
        executor = SkillExecutor(environment, ModalVLAClient(runtime), log=logs.append,
            predicates=PredicateRegistry(perception=ClosedGripperMonitor()))

        result = executor.execute({"action": "vla_act", "stop": "mug_lifted",
            "parameters": {"failure_stop": "grasp_attempt_failed", "failure_arm": "right",
                           "max_chunks": 1, "chunk_size": 24, "max_policy_actions": 4}})

        self.assertEqual(result.stop_reason, "failure_predicate_satisfied")
        self.assertEqual(result.executed_policy_actions, 80)
        self.assertEqual(result.vla_chunks_requested, 4)
        self.assertEqual(result.discarded_actions, 16)
        self.assertEqual(len(runtime.actions), 80)
        discarded = [event["discarded_actions"] for event in logs
                     if event.get("event_type") == "chunk_discarded"]
        self.assertEqual(discarded, [16])

    def test_monitored_lift_stops_the_unbounded_grasp_attempt(self):
        class LiftMonitor:
            source = "perception"

            def begin_attempt(self, name, observation, *, arm):
                return name == "grasp_attempt_failed" and arm == "right"

            def evaluate(self, name, observation, *, arm="auto"):
                lifted = name == "right_mug_lifted" and observation.sequence >= 5
                return PredicateResult(name=name, value=lifted, source="perception",
                    observation_sequence=observation.sequence,
                    reason="lift_confirmed" if lifted else "monitor_pending")

        runtime = Runtime()
        environment = EnvironmentClient(runtime.transport)
        environment.reset(3)
        executor = SkillExecutor(environment, ModalVLAClient(runtime), log=lambda _: None,
            predicates=PredicateRegistry(perception=LiftMonitor()))

        result = executor.execute({"action": "vla_act", "stop": "right_mug_lifted",
            "parameters": {"failure_stop": "grasp_attempt_failed", "failure_arm": "right",
                           "max_chunks": 1, "chunk_size": 24, "max_policy_actions": 1}})

        self.assertEqual(result.stop_reason, "predicate_satisfied")
        self.assertEqual(result.executed_policy_actions, 5)
        self.assertEqual(result.discarded_actions, 19)
        self.assertEqual(len(runtime.actions), 5)

    def test_episode_ceiling_is_only_a_watchdog_when_grasp_evidence_stays_unresolved(self):
        class PendingMonitor:
            source = "perception"

            def begin_attempt(self, name, observation, *, arm):
                return name == "grasp_attempt_failed" and arm == "right"

            def evaluate(self, name, observation, *, arm="auto"):
                return PredicateResult(name=name, value=False, source="perception",
                    observation_sequence=observation.sequence, reason="monitor_pending")

        runtime = Runtime()
        environment = EnvironmentClient(runtime.transport)
        environment.reset(3)
        executor = SkillExecutor(environment, ModalVLAClient(runtime), log=lambda _: None,
            predicates=PredicateRegistry(perception=PendingMonitor()),
            budget=EpisodeBudget(max_policy_actions=4, max_vla_chunks=8))

        result = executor.execute({"action": "vla_act", "stop": "right_mug_lifted",
            "parameters": {"failure_stop": "grasp_attempt_failed", "failure_arm": "right",
                           "max_chunks": 1, "chunk_size": 24, "max_policy_actions": 1}})

        self.assertEqual(result.stop_reason, "grasp_monitor_watchdog_exhausted")
        self.assertTrue(result.episode_terminal)
        self.assertEqual(result.executed_policy_actions, 4)

    def test_monitored_grasp_does_not_fall_back_to_an_action_cap_without_monitor(self):
        runtime, executor, _ = setup_runtime()

        result = executor.execute({"action": "vla_act", "stop": "right_mug_lifted",
            "parameters": {"failure_stop": "grasp_attempt_failed", "failure_arm": "right",
                           "max_chunks": 1, "chunk_size": 24, "max_policy_actions": 1}})

        self.assertEqual(result.stop_reason, "failure_monitor_unavailable")
        self.assertTrue(result.episode_terminal)
        self.assertEqual(runtime.chunks, 0)
        self.assertEqual(runtime.actions, [])

    def test_confirmed_planner_observation_skips_executor_refresh_and_traces_preflight(self):
        runtime = Runtime()
        environment = EnvironmentClient(runtime.transport)
        environment.reset(3)
        observe_calls = 0
        original_observe = environment.observe

        def observe():
            nonlocal observe_calls
            observe_calls += 1
            return original_observe()

        environment.observe = observe
        logs = []
        executor = SkillExecutor(environment, ModalVLAClient(runtime), log=logs.append)
        confirmed = environment.observe()
        calls_before_execute = observe_calls

        executor.execute({"action": "abort", "parameters": {}},
                         confirmed_observation=confirmed)

        self.assertEqual(observe_calls, calls_before_execute)
        preflight = [event for event in logs if event["event_type"].startswith("skill_preflight_")]
        self.assertEqual([event["event_type"] for event in preflight], [
            "skill_preflight_started", "skill_preflight_completed"])
        self.assertEqual(preflight[0]["observation_source"], "planner_confirmed_observation")
        self.assertEqual(preflight[1]["observation"], confirmed.sequence)
        self.assertEqual(logs[-2]["event_type"], "skill_requested")

    def test_failed_environment_refresh_is_logged_before_any_skill_dispatch(self):
        _, executor, logs = setup_runtime()

        def fail_observation():
            raise TimeoutError("simulated observation timeout")

        executor.environment.observe = fail_observation
        with self.assertRaises(TimeoutError):
            executor.execute({"action": "abort", "parameters": {}})

        event_types = [event["event_type"] for event in logs]
        self.assertEqual(event_types, ["skill_preflight_started", "skill_preflight_failed"])
        self.assertNotIn("skill_requested", event_types)
        self.assertEqual(executor.calls, 0)

    def test_vla_act_records_distinct_measured_tcp_poses(self):
        history = {"left": deque(maxlen=8), "right": deque(maxlen=8)}
        context = SimpleNamespace(measured_tcp_history=history)
        pose = (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)
        observation = SimpleNamespace(measured_tcp_poses={"left": pose, "right": pose})
        VLAAct._remember_tcp_poses(context, observation)
        VLAAct._remember_tcp_poses(context, observation)
        self.assertEqual(tuple(history["right"]), (pose,))
        self.assertEqual(tuple(history["left"]), (pose,))

    def test_two_calls_hand_back_without_scene_reset_or_stale_tail(self):
        runtime, executor, logs = setup_runtime()
        first, second = executor.execute(COMMAND), executor.execute(COMMAND)
        self.assertFalse(first.episode_terminal)
        self.assertFalse(second.episode_terminal)
        self.assertEqual((runtime.resets, runtime.policy_resets, runtime.chunks), (1, 1, 2))
        self.assertEqual([a[0] for a in runtime.actions], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual((first.discarded_actions, second.discarded_actions), (2, 2))
        self.assertEqual(runtime.inputs[1], (2.0,) * 14)
        self.assertNotIn("PRIVATE", json.dumps(logs))

    def test_action_and_inference_timing_are_traced_per_request(self):
        runtime, executor, logs = setup_runtime()
        executor.execute(COMMAND)
        events = [event for event in logs if event["event_type"] in {
            "chunk_received", "action_requested", "action_confirmed"}]
        received = [event for event in events if event["event_type"] == "chunk_received"]
        requested = [event for event in events if event["event_type"] == "action_requested"]
        confirmed = [event for event in events if event["event_type"] == "action_confirmed"]
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["returned_actions"], 4)
        self.assertGreaterEqual(received[0]["inference_ms"], 0)
        self.assertEqual([event["action_index"] for event in requested], [1, 2])
        self.assertEqual([event["action_index"] for event in confirmed], [1, 2])
        self.assertTrue(all(event["environment_step_ms"] >= 0 for event in confirmed))
        self.assertTrue(all("requested_at_monotonic_seconds" in event for event in requested))

    def test_vla_arm_locking_is_not_supported(self):
        runtime, executor, logs = setup_runtime()
        command = {"action": "vla_act", "parameters": {
            "max_chunks": 1, "chunk_size": 3, "max_policy_actions": 3,
            "lock_arm": "left"}}
        with self.assertRaisesRegex(ValueError, "lock_arm"):
            executor.execute(command)
        self.assertEqual(runtime.inputs, [])
        self.assertEqual(logs, [])

    def test_native_success_stops_before_the_next_chunk_action(self):
        runtime, executor, _ = setup_runtime(success_at=1)
        result = executor.execute(COMMAND)
        self.assertEqual((result.stop_reason, result.discarded_actions),
                         ("task_success", 3))
        self.assertTrue(result.episode_terminal)
        self.assertTrue(result.native_success_seen)
        self.assertEqual(result.post_success_actions, 0)
        with self.assertRaises(RuntimeError): executor.execute(COMMAND)
        self.assertEqual(len(runtime.actions), 1)

    def test_native_success_discards_the_rest_of_the_received_chunk(self):
        runtime, executor, logs = setup_runtime(success_at=1)
        result = executor.execute({"action": "vla_act", "stop": "chunk_budget_exhausted",
            "parameters": {"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 32}})

        self.assertEqual(result.stop_reason, "task_success")
        self.assertTrue(result.episode_terminal)
        self.assertTrue(result.native_success_seen)
        self.assertEqual(result.post_success_actions, 0)
        self.assertEqual(result.executed_policy_actions, 1)
        self.assertEqual(result.vla_chunks_requested, 1)
        self.assertEqual(result.discarded_actions, 7)
        event_types = [event["event_type"] for event in logs]
        self.assertIn("native_success_candidate", event_types)
        self.assertNotIn("native_success_follow_through_complete", event_types)
        with self.assertRaises(RuntimeError):
            executor.execute({"action": "vla_act", "stop": "chunk_budget_exhausted",
                "parameters": {"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 32}})

    def test_invalid_calls_have_no_side_effects_and_abort_is_no_motion(self):
        runtime, executor, _ = setup_runtime()
        for command in ({"action": "move_to"}, {**COMMAND, "arm": "left"},
                        {**COMMAND, "stop": "unknown_predicate"},
                        {"action": "vla_act", "parameters": {"joint_targets": [0] * 14}}):
            with self.assertRaises(ValueError): executor.execute(command)
        result = executor.execute({"action": "abort", "parameters": {}})
        self.assertEqual(result.stop_reason, "aborted")
        self.assertEqual((runtime.chunks, runtime.policy_resets, runtime.actions), (0, 0, []))

    def test_bad_chunk_rejected_before_any_motion(self):
        runtime, executor, _ = setup_runtime(malformed=True)
        result = executor.execute(COMMAND)
        self.assertEqual(result.stop_reason, "execution_error")
        self.assertTrue(result.episode_terminal)
        self.assertEqual(runtime.actions, [])

    def test_lost_response_is_not_retried_or_reported_as_confirmed(self):
        runtime, executor, _ = setup_runtime(step_error=True)
        result = executor.execute(COMMAND)
        self.assertEqual(len(runtime.actions), 1)
        self.assertEqual(result.executed_policy_actions, 0)
        self.assertIsNone(result.state_after)
        self.assertIsNone(result.task_success)
        self.assertTrue(result.episode_terminal)

    def test_policy_action_budget_exhaustion_hands_back_without_ending_episode(self):
        runtime = Runtime()
        env = EnvironmentClient(runtime.transport)
        env.reset(3)
        executor = SkillExecutor(env, ModalVLAClient(runtime), log=lambda e: None,
                                     budget=EpisodeBudget(max_policy_actions=1))
        result = executor.execute(COMMAND)
        self.assertEqual(result.stop_reason, "policy_action_budget_exhausted")
        self.assertEqual(len(runtime.actions), 1)
        self.assertTrue(result.completed)
        self.assertFalse(result.episode_terminal)

    def test_vla_chunk_budget_exhaustion_hands_back_without_ending_episode(self):
        runtime = Runtime()
        env = EnvironmentClient(runtime.transport)
        env.reset(3)
        executor = SkillExecutor(env, ModalVLAClient(runtime), log=lambda e: None,
            budget=EpisodeBudget(max_vla_chunks=1, max_policy_actions=20))
        result = executor.execute({"action": "vla_act", "parameters": {
            "max_chunks": 1, "chunk_size": 1, "max_policy_actions": 4}})
        self.assertEqual(result.stop_reason, "chunk_budget_exhausted")
        self.assertTrue(result.completed)
        self.assertFalse(result.episode_terminal)
