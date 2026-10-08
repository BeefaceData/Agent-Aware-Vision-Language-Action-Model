"""Public schedule scoring with complete replay and mixed attempt outcomes."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, PreStartFailure, run_episode
from evaluation_outcomes import score_evaluation_outcomes
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_schedule import build_evaluation_schedule
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration


class EvaluationOutcomeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fields = declaration()
        fields['seeds']['scheduling'] = 701
        fields['allocation'] = {f'task_{task}': {'pairs': 1}
                                for task in range(10)}
        self.protocol = EvaluationProtocol.freeze(Path(temporary.name) / 'protocol.json',
                                                   **fields)
        self.rows = build_evaluation_schedule(self.protocol)

    def episode(self, *, success=False, terminated=False):
        step = ReplayStep('held', 1.0 if success else 0.0,
                          success, terminated, False)
        return run_episode(
            EpisodeConfig(seed=17, max_steps=1),
            ReplayPolicy((('visible', 'reach'),)),
            ReplayEnvironment(17, 'visible', (('reach', step),)),
            ReplayRecorder())

    def interruption(self, reason):
        class FailingEnvironment(ReplayEnvironment):
            def step(self, action):
                raise TimeoutError('controller did not return')

        with self.assertRaises(TimeoutError) as raised:
            run_episode(EpisodeConfig(seed=17, max_steps=1),
                        ReplayPolicy((('visible', 'reach'),)),
                        FailingEnvironment(17, 'visible', ()), ReplayRecorder())
        return replace(raised.exception.episode_interruption, stop_reason=reason)

    def test_mixed_attempts_keep_post_start_failures_in_denominator(self):
        first = self.rows[:7]
        pre_start = replace(self.interruption('infrastructure_failure'),
                            last_observation=None,
                            pre_start_failure=PreStartFailure(
                                'environment_reset', 17, True, False))
        attempts = {row['attempt_id']: replace(pre_start, episode_id=row['attempt_id'])
                    for row in self.rows[7:]}
        attempts.update({
            first[0]['attempt_id']: self.episode(success=True, terminated=True),
            first[1]['attempt_id']: self.episode(terminated=True),
            first[2]['attempt_id']: self.episode(),  # action horizon exhausted
            first[3]['attempt_id']: self.interruption('controller_failure'),
            first[4]['attempt_id']: self.interruption('timeout'),
            first[5]['attempt_id']: self.interruption('intervention_budget_exhausted'),
        })
        report = score_evaluation_outcomes(self.protocol, self.rows, attempts)
        self.assertEqual((report['scheduled'], report['valid_starts'],
                          report['successes'], report['post_start_failures'],
                          report['missing']), (30, 6, 1, 5, 1))
        self.assertIsNone(report['success_rate'])
        self.assertEqual([item['status'] for item in report['dispositions'][:7]],
                         ['success'] + ['post_start_failure'] * 5 + ['missing'])
        attempts[first[6]['attempt_id']] = replace(
            pre_start, episode_id=first[6]['attempt_id'])
        complete = score_evaluation_outcomes(self.protocol, self.rows, attempts)
        self.assertEqual(complete['success_rate'], 1 / 6)

    def test_pre_start_and_unscoreable_evidence_remain_visible(self):
        first = self.rows[:3]
        pre_start = replace(self.interruption('infrastructure_failure'),
                            last_observation=None,
                            pre_start_failure=PreStartFailure(
                                'environment_reset', 17, True, False))
        incomplete = replace(self.episode(success=True, terminated=True),
                             artifact_status='incomplete')
        report = score_evaluation_outcomes(self.protocol, self.rows, {
            first[0]['attempt_id']: pre_start,
            first[1]['attempt_id']: incomplete,
        })
        self.assertEqual((report['valid_starts'], report['pre_start_exclusions'],
                          report['unscoreable'], report['missing']), (1, 1, 1, 28))
        self.assertIsNone(report['success_rate'])
        self.assertEqual([item['status'] for item in report['dispositions'][:3]],
                         ['pre_start_exclusion', 'unscoreable', 'missing'])
        with self.assertRaisesRegex(ProtocolError, 'outside sealed schedule'):
            score_evaluation_outcomes(self.protocol, self.rows, {'other': incomplete})


if __name__ == '__main__':
    unittest.main()
