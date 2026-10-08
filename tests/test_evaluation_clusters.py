"""Public paired-state analysis contract with complete replay episodes."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_clusters import (assemble_paired_state_clusters,
                                 report_paired_cluster_intervals)
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_schedule import build_evaluation_schedule
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration, member


class PairedClusterTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fields = declaration()
        fields['seeds'] = {'pairing': [17, 18, 19], 'scheduling': 701,
                           'bootstrap': 991}
        fields['allocation'] = {f'task_{task}': {'pairs': 3}
                                for task in range(10)}
        fields['splits']['held_out'].extend(member(task, 2)
                                            for task in range(10))
        self.protocol = EvaluationProtocol.freeze(
            Path(temporary.name) / 'protocol.json', **fields)
        self.rows = build_evaluation_schedule(self.protocol)

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

    def test_repeated_trials_are_one_state_with_all_arms_preserved(self):
        attempts = {row['attempt_id']: self.episode(
            row, row['condition'] == 'baseline')
                    for row in self.rows}
        report = assemble_paired_state_clusters(self.protocol, self.rows,
                                                attempts)
        self.assertEqual(report['independent_starting_states'], 20)
        self.assertEqual(report['diagnostics'], [])
        task_zero = [cluster for cluster in report['clusters']
                     if cluster['task_id'] == 0]
        self.assertEqual([len(cluster['repetitions']) for cluster in task_zero],
                         [2, 1])
        self.assertEqual([cluster['paired_repetitions'] for cluster in task_zero],
                         [2, 1])
        for cluster in task_zero:
            for repetition in cluster['repetitions']:
                self.assertEqual(set(repetition['conditions']),
                                 {'baseline', 'no_memory', 'fixed_memory'})
                self.assertEqual({arm['status'] for arm in
                                  repetition['conditions'].values()},
                                 {'success', 'post_start_failure'})

    def test_missing_arm_diagnosed_without_dropping_its_state(self):
        target = next(row for row in self.rows if row['task_id'] == 0 and
                      row['repetition'] == 2 and row['condition'] == 'fixed_memory')
        attempts = {row['attempt_id']: self.episode(row, False)
                    for row in self.rows if row != target}
        report = assemble_paired_state_clusters(self.protocol, self.rows,
                                                attempts)
        self.assertEqual(report['independent_starting_states'], 20)
        cluster = next(cluster for cluster in report['clusters']
                       if cluster['task_id'] == 0 and cluster['state_id'] == 1)
        self.assertEqual(cluster['paired_repetitions'], 1)
        self.assertEqual(cluster['repetitions'][1]['conditions']['fixed_memory']
                         ['status'], 'missing')
        self.assertIn('task 0 state 1 repetition 2: fixed_memory: missing',
                      report['diagnostics'])

    def test_conflicting_state_identity_rejected_by_sealed_schedule(self):
        changed = [dict(row) for row in self.rows]
        changed[0]['selected_sha256'] = '0' * 64
        with self.assertRaisesRegex(ProtocolError, 'sealed declaration'):
            assemble_paired_state_clusters(self.protocol, changed, {})
        changed = [dict(row) for row in self.rows]
        changed[0]['environment_seed'] += 1
        with self.assertRaisesRegex(ProtocolError, 'sealed declaration'):
            assemble_paired_state_clusters(self.protocol, changed, {})

    def test_bootstrap_keeps_repetitions_in_their_state_cluster(self):
        # Task 0 has two repetitions of its successful state and one of its
        # unsuccessful state. Every other task has zero difference.
        attempts = {}
        for row in self.rows:
            success = (row['task_id'] == 0 and row['state_id'] == 1 and
                       row['condition'] == 'fixed_memory')
            attempts[row['attempt_id']] = self.episode(row, success)
        report = report_paired_cluster_intervals(self.protocol, self.rows,
                                                 attempts)
        contrast = 'fixed_memory_minus_baseline'
        self.assertAlmostEqual(report['macro_differences_pp'][contrast], 20 / 3)
        self.assertEqual(report['intervals_pp'][contrast],
                         {'lower': 0.0, 'upper': 10.0})
        self.assertEqual(report['intervals_pp']['no_memory_minus_baseline'],
                         {'lower': 0.0, 'upper': 0.0})
        self.assertEqual(report['intervals_pp']['fixed_memory_minus_no_memory'],
                         report['intervals_pp'][contrast])
        self.assertEqual(report['analysis_seed'], 991)
        self.assertEqual(report['resamples'], 10_000)
        self.assertEqual(report['independent_starting_states'], 20)
        self.assertEqual(report['status'], 'estimated')
        self.assertEqual(report['limitations'], [])
        self.assertEqual(report_paired_cluster_intervals(self.protocol, self.rows,
                                                        attempts), report)

    def test_bootstrap_rejects_incomplete_pairs_and_wrong_sealed_settings(self):
        attempts = {row['attempt_id']: self.episode(row, False)
                    for row in self.rows}
        missing = dict(attempts)
        del missing[self.rows[0]['attempt_id']]
        report = report_paired_cluster_intervals(self.protocol, self.rows, missing)
        self.assertEqual(report['status'], 'inconclusive')
        first = self.rows[0]
        self.assertIn(f"task {first['task_id']} state {first['state_id']} "
                      f"repetition {first['repetition']}: "
                      f"{first['condition']}: missing", report['limitations'])
        self.assertTrue(all(value is None for value in report['intervals_pp'].values()))

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fields = declaration()
        fields['seeds']['bootstrap'] = -1
        wrong = EvaluationProtocol.freeze(Path(temporary.name) / 'wrong.json',
                                          **fields)
        with self.assertRaisesRegex(ProtocolError, 'bootstrap seed'):
            report_paired_cluster_intervals(wrong, [], {})

    def test_all_success_has_degenerate_primary_interval(self):
        attempts = {row['attempt_id']: self.episode(row, True)
                    for row in self.rows}
        report = report_paired_cluster_intervals(self.protocol, self.rows,
                                                 attempts)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['macro_differences_pp']
                         ['fixed_memory_minus_baseline'], 0)
        self.assertEqual(report['intervals_pp']
                         ['fixed_memory_minus_baseline'],
                         {'lower': 0.0, 'upper': 0.0})
        self.assertIn('degenerate zero-width bootstrap interval',
                      report['limitations'][0])

    def test_sparse_and_missing_task_clusters_are_inconclusive(self):
        attempts = {row['attempt_id']: self.episode(row, False)
                    for row in self.rows}
        sparse = {identity: outcome for identity, outcome in attempts.items()
                  if not any(row['attempt_id'] == identity and
                             row['task_id'] == 1 and row['state_id'] == 2
                             for row in self.rows)}
        report = report_paired_cluster_intervals(self.protocol, self.rows,
                                                 sparse)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertTrue(any('task 1: only 1 independent paired' in reason
                            for reason in report['limitations']))
        missing = {row['attempt_id']: attempts[row['attempt_id']]
                   for row in self.rows if row['task_id'] != 2}
        report = report_paired_cluster_intervals(self.protocol, self.rows,
                                                 missing)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertIn('task 2: no paired initial-state clusters',
                      report['limitations'])
        self.assertIsNone(report['macro_differences_pp']
                          ['fixed_memory_minus_baseline'])


if __name__ == '__main__':
    unittest.main()
