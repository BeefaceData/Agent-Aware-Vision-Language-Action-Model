"""Primary acceptance decisions from complete public replay episodes."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_acceptance import report_simulation_acceptance
from evaluation_protocol import EvaluationProtocol
from evaluation_schedule import build_evaluation_schedule
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration, member


class SimulationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def schedule(self, pairs):
        fields = declaration()
        fields['seeds'] = {'pairing': list(range(17, 17 + pairs)),
                           'scheduling': 701, 'bootstrap': 991}
        fields['allocation'] = {f'task_{task}': {'pairs': pairs}
                                for task in range(10)}
        fields['splits']['held_out'].extend(member(task, 2)
                                            for task in range(10))
        protocol = EvaluationProtocol.freeze(self.directory / 'protocol.json',
                                             **fields)
        return protocol, build_evaluation_schedule(protocol)

    def episode(self, row, success):
        seed = row['environment_seed']
        outcome = run_episode(
            EpisodeConfig(seed=seed, max_steps=1),
            ReplayPolicy((('visible', 'reach'),)),
            ReplayEnvironment(seed, 'visible',
                              (('reach', ReplayStep('held', float(success),
                                                   success, True, False)),)),
            ReplayRecorder())
        return replace(outcome, episode_id=row['attempt_id'])

    def test_exactly_ten_points_passes_with_positive_lower_endpoint(self):
        protocol, rows = self.schedule(5)
        # Five tasks each gain one success in five trials. The winning state
        # varies under paired resampling, so the primary interval has width.
        attempts = {}
        for row in rows:
            success = (row['task_id'] < 5 and row['state_id'] == 1 and
                       row['repetition'] == 0 and
                       row['condition'] == 'fixed_memory')
            attempts[row['attempt_id']] = self.episode(row, success)
        report = report_simulation_acceptance(protocol, rows, attempts)
        self.assertEqual(report['observed_gain_pp'], 10)
        self.assertGreater(report['lower_endpoint_pp'], 0)
        self.assertLess(report['lower_endpoint_pp'], 10)
        self.assertEqual(report['decision'], 'pass')

    def test_zero_lower_endpoint_fails_even_at_ten_points(self):
        protocol, rows = self.schedule(2)
        attempts = {row['attempt_id']: self.episode(
            row, row['task_id'] < 2 and row['state_id'] == 1 and
            row['condition'] == 'fixed_memory') for row in rows}
        report = report_simulation_acceptance(protocol, rows, attempts)
        self.assertEqual(report['observed_gain_pp'], 10)
        self.assertEqual(report['lower_endpoint_pp'], 0)
        self.assertEqual(report['decision'], 'fail')

    def test_relative_gain_over_ten_percent_does_not_replace_points(self):
        protocol, rows = self.schedule(5)
        attempts = {}
        for row in rows:
            success = (row['repetition'] < 3 or
                       (row['task_id'] < 4 and row['repetition'] == 3 and
                        row['condition'] == 'fixed_memory'))
            attempts[row['attempt_id']] = self.episode(row, success)
        report = report_simulation_acceptance(protocol, rows, attempts)
        self.assertAlmostEqual(report['observed_gain_pp'], 8)
        self.assertGreater(report['lower_endpoint_pp'], 0)
        self.assertEqual(report['decision'], 'fail')
        self.assertIn('below 10 percentage points', report['reasons'][0])

    def test_missing_evidence_is_inconclusive(self):
        protocol, rows = self.schedule(2)
        attempts = {row['attempt_id']: self.episode(row, False)
                    for row in rows[1:]}
        report = report_simulation_acceptance(protocol, rows, attempts)
        self.assertEqual(report['decision'], 'inconclusive')
        self.assertIsNone(report['lower_endpoint_pp'])
        self.assertTrue(any('missing' in reason for reason in report['reasons']))

    def test_degenerate_interval_is_inconclusive(self):
        protocol, rows = self.schedule(2)
        attempts = {row['attempt_id']: self.episode(row, True)
                    for row in rows}
        report = report_simulation_acceptance(protocol, rows, attempts)
        self.assertEqual(report['decision'], 'inconclusive')
        self.assertEqual(report['lower_endpoint_pp'], 0)
        self.assertTrue(any('degenerate' in reason for reason in report['reasons']))


if __name__ == '__main__':
    unittest.main()
