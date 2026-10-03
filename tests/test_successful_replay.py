"""Complete synthetic episodes through the baseline's public interface."""

import unittest

from episode_harness import run_episode
from replay_adapters import ReplayPolicy, successful_replay


class SuccessfulReplayTests(unittest.TestCase):
    def test_fixture_executes_known_actions_and_preserves_success_evidence(self):
        fixture = successful_replay()

        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder)

        self.assertEqual([p.observation for p in fixture.policy.observations],
                         ['item visible', 'item held'])
        self.assertEqual(fixture.environment.actions,
                         [('reach', 0.25), ('place', 0.75)])
        self.assertEqual([p.observation for p in fixture.recorder.observations],
                         ['item visible', 'item held', 'item placed'])
        self.assertEqual([step for step, _, _, _, _ in fixture.recorder.steps], [1, 2])
        self.assertTrue(fixture.recorder.finalized)
        self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason,
                          outcome.sum_rewards, outcome.artifacts),
                         (True, 2, 'success', 1.0, {}))
        self.assertGreaterEqual(outcome.rollout_seconds, 0.0)

    def test_second_complete_run_reproduces_logical_result(self):
        fixture = successful_replay()
        runs = []
        episode_ids = []
        for _ in range(2):
            outcome = run_episode(fixture.config, fixture.policy,
                                  fixture.environment, fixture.recorder)
            episode_ids.append(outcome.episode_id)
            runs.append((tuple(fixture.environment.actions),
                         tuple(p.observation for p in fixture.policy.observations),
                         tuple((step, source.observation, action,
                                result.observation.observation)
                               for step, source, action, result, _
                               in fixture.recorder.steps),
                         tuple(p.observation for p in fixture.recorder.observations),
                         outcome.success, outcome.steps, outcome.stop_reason,
                         outcome.sum_rewards, outcome.artifacts))

        # rollout_seconds measures elapsed wall time and is intentionally omitted.
        self.assertEqual(runs[0], runs[1])
        self.assertNotEqual(episode_ids[0], episode_ids[1])

    def test_divergent_action_cannot_receive_scripted_success(self):
        fixture = successful_replay()
        divergent = ReplayPolicy((('item visible', ('retreat', 0.25)),))

        with self.assertRaisesRegex(AssertionError, 'Unexpected action at step 1'):
            run_episode(fixture.config, divergent, fixture.environment,
                        fixture.recorder)

        self.assertEqual(fixture.environment.actions, [('retreat', 0.25)])
        self.assertTrue(fixture.recorder.finalized)


if __name__ == '__main__':
    unittest.main()
