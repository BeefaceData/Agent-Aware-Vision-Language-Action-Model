"""Focused R1 checks; no GPU, Modal request, or broad new test framework."""

import json
import unittest

import numpy as np

from harness.environment.client import EnvironmentClient
from harness.skills import SkillExecutor
from harness.skills.schemas import EpisodeBudget
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

    def test_policy_timing_details_are_numeric_safe_and_do_not_leak_to_next_chunk(self):
        runtime, executor, logs = setup_runtime()
        original = runtime.act_chunk

        def timed_chunk(*args):
            actions, _ = original(*args)
            timing = {"pack_ms": 1., "modal_rpc_ms": 8., "cloud_inference_ms": 4.,
                      "modal_overhead_ms": -1., "modal_compute_ms": float("nan"),
                      "cloud_postprocess_ms": True, "provider_error": "PRIVATE_PAYLOAD"}
            return actions, timing if runtime.chunks == 1 else {}

        runtime.act_chunk = timed_chunk
        executor.execute(COMMAND)
        executor.execute(COMMAND)
        received = [row for row in logs if row["event_type"] == "chunk_received"]
        self.assertEqual(received[0]["policy_timing"],
                         {"pack_ms": 1., "modal_rpc_ms": 8., "cloud_inference_ms": 4.})
        self.assertEqual(received[1]["policy_timing"], {})
        reset = next(row for row in logs if row["event_type"] == "policy_history_reset")
        self.assertEqual(reset["reset_ms"], 0.)
        self.assertNotIn("PRIVATE_PAYLOAD", json.dumps(logs, allow_nan=False))

    def test_vla_can_hold_one_arm_at_last_confirmed_command(self):
        runtime, executor, logs = setup_runtime()
        command = {"action": "vla_act", "parameters": {
            "max_chunks": 1, "chunk_size": 3, "max_policy_actions": 3,
            "lock_arm": "left"}}
        executor.execute(command)
        actions = np.asarray(runtime.actions, dtype=float)
        # The locked arm follows only confirmed command state. The first action
        # holds the reset target; the next follows the acknowledged state.
        np.testing.assert_allclose(actions[0][:7], 0.0)
        np.testing.assert_allclose(actions[0][7:], 0.1)
        np.testing.assert_allclose(actions[1][:7], 1.0)
        np.testing.assert_allclose(actions[1][7:], 0.1)
        json.dumps(logs)  # Locked actions remain JSON-safe for durable event logs.

    def test_native_success_runs_only_within_remaining_action_budget(self):
        runtime, executor, _ = setup_runtime(success_at=1)
        result = executor.execute(COMMAND)
        self.assertEqual((result.stop_reason, result.discarded_actions),
                         ("policy_action_budget_exhausted", 2))
        self.assertTrue(result.episode_terminal)
        self.assertTrue(result.native_success_seen)
        self.assertEqual(result.post_success_actions, 1)
        with self.assertRaises(RuntimeError): executor.execute(COMMAND)
        self.assertEqual(len(runtime.actions), 2)

    def test_task_success_stop_allows_fixed_vla_hang_follow_through(self):
        runtime, executor, logs = setup_runtime(success_at=1)
        result = executor.execute({"action": "vla_act", "stop": "chunk_budget_exhausted",
            "parameters": {"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 32}})

        self.assertEqual(result.stop_reason, "task_success")
        self.assertTrue(result.episode_terminal)
        self.assertTrue(result.native_success_seen)
        self.assertEqual(result.post_success_actions, 16)
        self.assertEqual(result.executed_policy_actions, 17)
        self.assertEqual(result.vla_chunks_requested, 3)
        self.assertEqual(result.discarded_actions, 7)
        event_types = [event["event_type"] for event in logs]
        self.assertIn("native_success_candidate", event_types)
        self.assertIn("native_success_follow_through_complete", event_types)
        with self.assertRaises(RuntimeError):
            executor.execute({"action": "vla_act", "stop": "chunk_budget_exhausted",
                "parameters": {"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 32}})

    def test_inter_arm_contact_guard_stops_chunk_and_returns_recoverable_feedback(self):
        runtime, executor, logs = setup_runtime()
        executor.environment.arm_overlap_detected = lambda: True

        result = executor.execute(COMMAND)

        self.assertEqual(result.stop_reason, "arm_overlap_detected")
        self.assertFalse(result.episode_terminal)
        self.assertFalse(result.completed)
        self.assertTrue(result.arm_overlap_detected)
        self.assertEqual((len(runtime.actions), result.discarded_actions), (1, 3))
        from harness.skills.schemas import is_recoverable_arm_overlap
        self.assertTrue(is_recoverable_arm_overlap(result))
        guard = next(event for event in logs
                     if event.get("event_type") == "inter_arm_contact_guard")
        self.assertTrue(guard["stopped"])

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
