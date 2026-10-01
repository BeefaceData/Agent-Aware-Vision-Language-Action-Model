"""Complete synthetic episodes through the baseline's public interface."""

import unittest

from episode_harness import run_episode
from replay_adapters import ReplayPolicy, successful_replay


class SuccessfulReplayTests(unittest.TestCase):
    def test_fixture_executes_known_actions_and_preserves_success_evidence(self):
        fixture = successful_replay()

        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder)

        self.assertEqual(fixture.policy.observations,
                         ['item visible', 'item held'])
        self.assertEqual(fixture.environment.actions,
                         [('reach', 0.25), ('place', 0.75)])
        self.assertEqual(fixture.recorder.observations,
                         ['item visible', 'item held', 'item placed'])
        self.assertEqual([step for step, _, _ in fixture.recorder.steps], [1, 2])
        self.assertTrue(fixture.recorder.finalized)
        self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason,
                          outcome.sum_rewards, outcome.artifacts),
                         (True, 2, 'success', 1.0, {}))
        self.assertGreaterEqual(outcome.rollout_seconds, 0.0)

    def test_second_complete_run_reproduces_logical_result(self):
        fixture = successful_replay()
        runs = []
        for _ in range(2):
            outcome = run_episode(fixture.config, fixture.policy,
                                  fixture.environment, fixture.recorder)
            runs.append((tuple(fixture.environment.actions),
                         tuple(fixture.policy.observations),
                         tuple(fixture.recorder.steps),
                         tuple(fixture.recorder.observations),
                         outcome.success, outcome.steps, outcome.stop_reason,
                         outcome.sum_rewards, outcome.artifacts))

        # rollout_seconds measures elapsed wall time and is intentionally omitted.
        self.assertEqual(runs[0], runs[1])

    def test_divergent_action_cannot_receive_scripted_success(self):
        fixture = successful_replay()
        divergent = ReplayPolicy((('item visible', ('retreat', 0.25)),))

        with self.assertRaisesRegex(AssertionError, 'Unexpected action at step 1'):
            run_episode(fixture.config, divergent, fixture.environment,
                        fixture.recorder)

        self.assertEqual(fixture.environment.actions, [('retreat', 0.25)])
        self.assertFalse(fixture.recorder.finalized)


if __name__ == '__main__':
    unittest.main()
