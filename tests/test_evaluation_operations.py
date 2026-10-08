"""Operational report uses the public episode and sealed schedule interfaces."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_operations import report_operational_metrics
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_schedule import build_evaluation_schedule
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration


class OperationalReportTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fields = declaration()
        fields['seeds']['scheduling'] = 701
        fields['allocation'] = {f'task_{task}': {'pairs': 1}
                                for task in range(10)}
        self.protocol = EvaluationProtocol.freeze(
            Path(temporary.name) / 'protocol.json', **fields)
        self.rows = build_evaluation_schedule(self.protocol)

    def episode(self, row, success):
        outcome = run_episode(
            EpisodeConfig(seed=row['environment_seed'], max_steps=1),
            ReplayPolicy((('visible', 'reach'),)),
            ReplayEnvironment(row['environment_seed'], 'visible',
                              (('reach', ReplayStep('held', float(success),
                                                   success, True, False)),)),
            ReplayRecorder())
        return replace(outcome, episode_id=row['attempt_id'])

    def test_same_attempt_population_and_unknowns(self):
        attempts = {row['attempt_id']: self.episode(row, row['condition'] == 'baseline')
                    for row in self.rows}
        operational = {}
        for row in self.rows:
            if row['condition'] == 'baseline':
                operational[row['attempt_id']] = {
                    'episode_id': row['attempt_id'], 'interventions': 0,
                    'human_assistance': 0, 'model_calls': 0,
                    'reported_cost_subtotals': {}, 'unknown_cost_calls': 0}
            elif row['condition'] == 'no_memory':
                operational[row['attempt_id']] = {
                    'episode_id': row['attempt_id'], 'interventions': 1,
                    'human_assistance': 0, 'model_calls': 2,
                    'reported_cost_subtotals': {'USD': .25},
                    'unknown_cost_calls': 1}
        report = report_operational_metrics(self.protocol, self.rows,
                                            attempts, operational)
        self.assertEqual(len(report['attempts']), len(self.rows))
        for condition in ('baseline', 'no_memory', 'fixed_memory'):
            group = report['conditions'][condition]
            self.assertEqual(group['attempts'], 10)
            self.assertEqual(group['successes'], 10 if condition == 'baseline' else 0)
            self.assertEqual(group['complete_totals']['executed_actions'], 10)
        self.assertEqual(report['conditions']['baseline']['total_cost_by_currency'], {})
        self.assertEqual(report['conditions']['no_memory']['reported_cost_subtotals'],
                         {'USD': 2.5})
        self.assertIsNone(report['conditions']['no_memory']['total_cost_by_currency'])
        self.assertEqual(report['conditions']['no_memory']['unknown_cost_calls'], 10)
        self.assertIsNone(report['conditions']['fixed_memory']['complete_totals']
                          ['interventions'])
        self.assertEqual(report['conditions']['fixed_memory']['unknown_attempts']
                         ['interventions'], 10)
        self.assertIsNone(report['conditions']['fixed_memory']['total_cost_by_currency'])

        missing = dict(attempts)
        del missing[self.rows[0]['attempt_id']]
        missing_operational = dict(operational)
        missing_operational.pop(self.rows[0]['attempt_id'], None)
        report = report_operational_metrics(self.protocol, self.rows, missing,
                                            missing_operational)
        condition = self.rows[0]['condition']
        self.assertEqual(report['conditions'][condition]['attempts'], 9)
        self.assertEqual(report['attempts'][0]['status'], 'missing')

    def test_rejects_foreign_episode_and_invalid_counts(self):
        row = self.rows[0]
        attempts = {row['attempt_id']: self.episode(row, True)}
        with self.assertRaisesRegex(ProtocolError, 'identity mismatch'):
            report_operational_metrics(self.protocol, self.rows, attempts,
                {row['attempt_id']: {'episode_id': 'other'}})
        with self.assertRaisesRegex(ProtocolError, 'invalid interventions'):
            report_operational_metrics(self.protocol, self.rows, attempts,
                {row['attempt_id']: {'episode_id': row['attempt_id'],
                                     'interventions': -1}})

    def test_missing_timing_does_not_become_zero(self):
        row = self.rows[0]
        outcome = replace(self.episode(row, True), rollout_seconds=None,
                          cumulative_wait_seconds=None)
        report = report_operational_metrics(
            self.protocol, self.rows, {row['attempt_id']: outcome}, {})
        group = report['conditions'][row['condition']]
        self.assertEqual(group['attempts'], 1)
        self.assertEqual(group['unknown_attempts']['elapsed_seconds'], 1)
        self.assertIsNone(group['complete_totals']['elapsed_seconds'])
        self.assertIsNone(group['complete_totals']['waiting_seconds'])
        self.assertEqual(group['complete_totals']['executed_actions'], 1)


if __name__ == '__main__':
    unittest.main()
