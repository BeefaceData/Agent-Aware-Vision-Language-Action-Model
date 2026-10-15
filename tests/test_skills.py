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

    def test_native_success_discards_tail_and_prevents_further_execution(self):
        runtime, executor, _ = setup_runtime(success_at=1)
        result = executor.execute(COMMAND)
        self.assertEqual((result.stop_reason, result.discarded_actions), ("task_success", 3))
        self.assertTrue(result.episode_terminal)
        with self.assertRaises(RuntimeError): executor.execute(COMMAND)
        self.assertEqual(len(runtime.actions), 1)

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

    def test_episode_budget_caps_total_actions(self):
        runtime = Runtime()
        env = EnvironmentClient(runtime.transport)
        env.reset(3)
        executor = SkillExecutor(env, ModalVLAClient(runtime), log=lambda e: None,
                                     budget=EpisodeBudget(max_policy_actions=1))
        result = executor.execute(COMMAND)
        self.assertEqual(result.stop_reason, "episode_budget_exhausted")
        self.assertEqual(len(runtime.actions), 1)
        self.assertTrue(result.episode_terminal)
