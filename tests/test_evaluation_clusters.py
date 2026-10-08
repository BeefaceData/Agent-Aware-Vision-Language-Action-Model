"""Public paired-state analysis contract with complete replay episodes."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_clusters import assemble_paired_state_clusters
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
        fields['splits']['held_out'].append(member(0, 2))
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
        self.assertEqual(report['independent_starting_states'], 11)
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
        self.assertEqual(report['independent_starting_states'], 11)
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


if __name__ == '__main__':
    unittest.main()
