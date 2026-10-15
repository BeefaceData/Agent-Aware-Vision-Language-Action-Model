import unittest

import numpy as np

from harness.config import TraceOptions
from harness.trace import TraceManager, vector14
from tests.helpers import decision, images


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = TraceManager(TraceOptions(
            state_history=3, action_history=2, keyframes=2, keyframe_interval=2,
        ))
        self.trace.reset("episode")

    def test_bounded_histories_all_decisions_and_reset(self):
        for step in range(6):
            self.trace.propose(step, images(step), np.full(14, step), np.zeros(14), decision())
            self.trace.executed(step, np.zeros(14), np.full(14, step + 1))
        context = self.trace.context()
        self.assertEqual([s.step for s in context.states], [4, 5, 6])
        self.assertEqual([s.step for s in context.actions], [4, 5])
        self.assertEqual([s.step for s in context.keyframes], [2, 4])
        self.assertEqual(len(context.decisions), 6)
        self.trace.reset("next")
        context = self.trace.context()
        self.assertEqual(context.episode_id, "next")
        for name in ("states", "actions", "decisions", "keyframes"):
            self.assertEqual(getattr(context, name), ())

    def test_input_and_context_mutation_cannot_corrupt_history(self):
        rgb, state, action, choice = images(), np.zeros(14), np.ones(14), decision()
        frame = self.trace.propose(0, rgb, state, action, choice)
        rgb["head_camera"][:] = 90
        state[:] = 90
        action[:] = 90
        choice.parameters["unexpected"] = 90
        frame.images["head_camera"][:] = 91
        context = self.trace.context()
        self.assertEqual(context.states[0].command_state, (0.0,) * 14)
        self.assertEqual(context.actions[0].policy_action, (1.0,) * 14)
        self.assertTrue(np.all(context.keyframes[0].images["head_camera"] == 0))
        self.assertEqual(context.decisions[0].parameters, {})
        context.keyframes[0].images["head_camera"][:] = 92
        context.decisions[0].parameters["unexpected"] = 92
        self.assertTrue(np.all(self.trace.context().keyframes[0].images["head_camera"] == 0))
        self.assertEqual(self.trace.context().decisions[0].parameters, {})

    def test_ordering_and_execution_confirmation(self):
        with self.assertRaises(ValueError):
            self.trace.executed(0, np.zeros(14), np.zeros(14))
        self.trace.propose(0, images(), np.zeros(14), np.ones(14), decision())
        self.assertIsNone(self.trace.context().actions[0].executed_action)
        self.trace.executed(0, np.ones(14), np.zeros(14))
        with self.assertRaises(ValueError):
            self.trace.executed(0, np.ones(14), np.zeros(14))
        with self.assertRaises(ValueError):
            self.trace.propose(0, images(), np.zeros(14), np.ones(14), decision())

    def test_vector_shape_type_and_camera_allowlist(self):
        for vector in (np.zeros((1, 14)), np.zeros(13), np.zeros(14, dtype=bool), np.full(14, np.inf)):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                vector14(vector)
        for rgb in ({**images(), "success": True}, {"head_camera": np.zeros((1, 1, 3))},
                    {**images(), "head_camera": np.zeros((1, 1, 4), dtype=np.uint8)}):
            with self.assertRaises(ValueError):
                self.trace.propose(0, rgb, np.zeros(14), np.ones(14), decision())
