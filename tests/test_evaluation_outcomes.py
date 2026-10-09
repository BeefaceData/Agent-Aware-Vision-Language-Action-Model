"""Public schedule scoring with complete replay and mixed attempt outcomes."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, PreStartFailure, run_episode
from evaluation_outcomes import (report_macro_improvement, report_task_outcomes,
                                 score_evaluation_outcomes)
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

    def test_optional_horizon_outcome_cannot_enter_primary_rate(self):
        row = self.rows[0]
        optional = replace(self.episode(success=True, terminated=True),
                           evaluation_scope='optional_uncapped_action_horizon')
        report = score_evaluation_outcomes(self.protocol, self.rows,
                                           {row['attempt_id']: optional})
        self.assertEqual(report['dispositions'][0]['status'], 'unscoreable')
        self.assertIn('optional ablation', report['dispositions'][0]['reason'])
        self.assertIsNone(report['success_rate'])

    def test_task_report_keeps_regression_and_missing_tasks_visible(self):
        # Public complete-episode replay supplies the outcomes; unique IDs
        # identify the retained evidence for each scheduled attempt.
        attempts = {}
        for row in self.rows:
            task, condition = row['task_id'], row['condition']
            if task == 3 or (task == 2 and condition == 'fixed_memory'):
                continue
            success = ((task == 0 and condition != 'fixed_memory') or
                       (task == 1 and condition == 'fixed_memory'))
            attempts[row['attempt_id']] = replace(
                self.episode(success=success, terminated=True),
                episode_id=row['attempt_id'])

        report = report_task_outcomes(self.protocol, self.rows, attempts)
        self.assertEqual(report['protocol_id'], self.protocol.reference['protocol_id'])
        self.assertEqual(len(report['tasks']), 10)
        self.assertEqual(
            [[task['conditions'][condition]['successes']
              for condition in ('baseline', 'no_memory', 'fixed_memory')]
             for task in report['tasks']],
            [[1, 1, 0], [0, 0, 1]] + [[0, 0, 0]] * 8)
        self.assertEqual(
            [[task['conditions'][condition]['missing_evidence']
              for condition in ('baseline', 'no_memory', 'fixed_memory')]
             for task in report['tasks']],
            [[0, 0, 0]] * 2 + [[0, 0, 1], [1, 1, 1]] +
            [[0, 0, 0]] * 6)
        harmed = report['tasks'][0]
        self.assertEqual(harmed['conditions']['baseline']['successes'], 1)
        self.assertEqual(harmed['conditions']['fixed_memory']['attempts'], 1)
        self.assertEqual(harmed['differences_pp']['fixed_memory_minus_baseline'], -100)
        self.assertEqual(report['tasks'][1]['differences_pp']
                         ['fixed_memory_minus_baseline'], 100)
        incomplete = report['tasks'][2]
        self.assertEqual(incomplete['conditions']['fixed_memory']
                         ['missing_evidence'], 1)
        self.assertIsNone(incomplete['differences_pp']
                          ['fixed_memory_minus_baseline'])
        self.assertEqual(report['tasks'][3]['conditions']['baseline']
                         ['attempts'], 0)
        self.assertIn('task 3: no valid starts', report['warnings'])
        self.assertIn('task 2 fixed_memory: 1 missing, 0 unscoreable',
                      report['warnings'])

    def test_task_report_shows_pre_start_exclusions_and_unscoreable_evidence(self):
        baseline = next(row for row in self.rows
                        if row['task_id'] == 0 and row['condition'] == 'baseline')
        fixed = next(row for row in self.rows
                     if row['task_id'] == 0 and row['condition'] == 'fixed_memory')
        pre_start = replace(self.interruption('infrastructure_failure'),
                            last_observation=None,
                            pre_start_failure=PreStartFailure(
                                'environment_reset', 17, True, False))
        incomplete = replace(self.episode(success=True, terminated=True),
                             artifact_status='incomplete')
        report = report_task_outcomes(self.protocol, self.rows, {
            baseline['attempt_id']: pre_start,
            fixed['attempt_id']: incomplete,
        })
        task = report['tasks'][0]
        self.assertEqual(task['conditions']['baseline']['pre_start_exclusions'], 1)
        self.assertEqual(task['conditions']['fixed_memory']['unscoreable'], 1)
        self.assertIsNone(task['differences_pp']['fixed_memory_minus_baseline'])
        self.assertIn('task 0 fixed_memory: 0 missing, 1 unscoreable',
                      report['warnings'])

    def test_macro_average_weights_tasks_not_pooled_attempts(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fields = declaration()
        fields['seeds'] = {'pairing': [17, 18], 'scheduling': 701,
                           'bootstrap': 991}
        fields['allocation'] = {f'task_{task}': {'pairs': 2}
                                for task in range(10)}
        protocol = EvaluationProtocol.freeze(Path(temporary.name) / 'protocol.json',
                                             **fields)
        rows = build_evaluation_schedule(protocol)
        attempts = {}
        for row in rows:
            task, condition = row['task_id'], row['condition']
            if task == 0 and condition == 'fixed_memory' and row['repetition'] == 1:
                # A declared pre-start exclusion makes valid-start counts unequal.
                interruption = replace(
                    self.interruption('infrastructure_failure'),
                    last_observation=None,
                    pre_start_failure=PreStartFailure('environment_reset',
                                                      row['environment_seed'],
                                                      True, False))
                evidence = interruption
            else:
                success = (condition == 'fixed_memory' and task in (0, 1)
                           and row['repetition'] == 0)
                evidence = self.episode(success=success, terminated=True)
            attempts[row['attempt_id']] = replace(evidence,
                                                  episode_id=row['attempt_id'])

        report = report_macro_improvement(protocol, rows, attempts)
        self.assertEqual(report['macro_differences_pp']
                         ['fixed_memory_minus_baseline'], 15.0)
        self.assertEqual(report['macro_differences_pp']
                         ['no_memory_minus_baseline'], 0.0)
        self.assertEqual(report['macro_differences_pp']
                         ['fixed_memory_minus_no_memory'], 15.0)
        self.assertEqual(report['pooled_counts']['fixed_memory']['successes'], 2)
        self.assertEqual(report['pooled_counts']['fixed_memory']['attempts'], 19)
        self.assertEqual(report['pooled_counts']['baseline']['attempts'], 20)
        self.assertNotEqual(15.0, 100 * 2 / 19)

        del attempts[next(row['attempt_id'] for row in rows
                          if row['task_id'] == 9 and
                          row['condition'] == 'fixed_memory')]
        incomplete = report_macro_improvement(protocol, rows, attempts)
        self.assertIsNone(incomplete['macro_differences_pp']
                          ['fixed_memory_minus_baseline'])
        self.assertIsNone(incomplete['macro_differences_pp']
                          ['fixed_memory_minus_no_memory'])
        self.assertEqual(incomplete['macro_differences_pp']
                         ['no_memory_minus_baseline'], 0.0)
        self.assertEqual(incomplete['pooled_counts']['fixed_memory']
                         ['missing_evidence'], 1)

        attempts = {identity: evidence for identity, evidence in attempts.items()
                    if identity not in {row['attempt_id'] for row in rows
                                        if row['task_id'] == 9}}
        missing_task = report_macro_improvement(protocol, rows, attempts)
        self.assertTrue(all(value is None for value in
                            missing_task['macro_differences_pp'].values()))
        self.assertIn('task 9: no valid starts', missing_task['warnings'])


if __name__ == '__main__':
    unittest.main()
