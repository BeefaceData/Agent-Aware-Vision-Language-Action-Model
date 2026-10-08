"""Public pre-start replacement decisions and complete replay."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, PreStartFailure, run_episode
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_replacement import next_pre_start_replacement
from evaluation_schedule import build_evaluation_schedule
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        fields = declaration()
        fields['seeds']['scheduling'] = 701
        fields['allocation'] = {f'task_{task}': {'pairs': 1}
                                for task in range(10)}
        self.protocol = EvaluationProtocol.freeze(self.directory / 'protocol.json',
                                                   **fields)
        self.rows = build_evaluation_schedule(self.protocol)
        self.row = self.rows[0]

    def pre_start(self):
        class BrokenPolicy(ReplayPolicy):
            def reset(self):
                raise RuntimeError('policy reset unavailable')

        with self.assertRaises(RuntimeError) as caught:
            run_episode(EpisodeConfig(seed=17, max_steps=2),
                        BrokenPolicy((('visible', 'reach'),)),
                        ReplayEnvironment(17, 'visible', ()), ReplayRecorder())
        return caught.exception.episode_interruption

    def test_eligible_failure_preserves_origin_and_replays_new_attempt(self):
        original = self.pre_start()
        plan = next_pre_start_replacement(
            self.protocol, self.rows, self.row, [(self.row['attempt_id'], original)])
        self.assertNotEqual(plan['attempt_id'], self.row['attempt_id'])
        self.assertEqual(plan['prior_attempts'], [{
            'attempt_id': self.row['attempt_id'], 'episode_id': original.episode_id,
            'stop_reason': original.stop_reason, 'stage': 'policy_reset',
            'environment_seed': 17}])
        outcome = run_episode(
            EpisodeConfig(seed=plan['schedule_entry']['environment_seed'], max_steps=2),
            ReplayPolicy((('visible', 'reach'),)),
            ReplayEnvironment(17, 'visible',
                              (('reach', ReplayStep('held', 1.0, True, True, False)),)),
            ReplayRecorder())
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.steps, 1)
        self.assertNotEqual(outcome.episode_id, original.episode_id)
        with self.assertRaisesRegex(ProtocolError, 'exhausted'):
            next_pre_start_replacement(self.protocol, self.rows, self.row,
                                       [(self.row['attempt_id'], original),
                                        (plan['attempt_id'], self.pre_start())])

    def test_post_start_and_conflicting_start_cannot_be_replaced(self):
        original = self.pre_start()
        completed = run_episode(
            EpisodeConfig(seed=17, max_steps=1),
            ReplayPolicy((('visible', 'reach'),)),
            ReplayEnvironment(17, 'visible',
                              (('reach', ReplayStep('held', 0.0, False, True, False)),)),
            ReplayRecorder())
        with self.assertRaisesRegex(ProtocolError, 'cannot be replaced'):
            next_pre_start_replacement(self.protocol, self.rows, self.row,
                                       [(self.row['attempt_id'], completed)])
        for evidence in (
                replace(original, pre_start_failure=None),
                replace(original, steps=1),
                replace(original, pre_start_failure=PreStartFailure(
                    'policy_reset', 99, False, False))):
            with self.subTest(evidence=evidence):
                with self.assertRaisesRegex(ProtocolError, 'not an eligible'):
                    next_pre_start_replacement(self.protocol, self.rows, self.row,
                                               [(self.row['attempt_id'], evidence)])

    def test_frozen_rule_and_chain_identity_are_enforced(self):
        original = self.pre_start()
        with self.assertRaisesRegex(ProtocolError, 'identity or order'):
            next_pre_start_replacement(self.protocol, self.rows, self.row,
                                       [('same-episode-new-label', original)])
        changed = declaration()
        changed['outcomes'].pop('pre_start_replacement')
        changed['seeds']['scheduling'] = 701
        changed['allocation'] = {f'task_{task}': {'pairs': 1}
                                 for task in range(10)}
        no_rule = EvaluationProtocol.freeze(self.directory / 'without-rule.json',
                                            **changed)
        no_rule_rows = build_evaluation_schedule(no_rule)
        with self.assertRaisesRegex(ProtocolError, 'frozen pre-start'):
            next_pre_start_replacement(no_rule, no_rule_rows, no_rule_rows[0],
                                       [(no_rule_rows[0]['attempt_id'], original)])


if __name__ == '__main__':
    unittest.main()
