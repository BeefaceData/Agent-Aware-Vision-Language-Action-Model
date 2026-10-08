"""Public schedule audit and complete synthetic episode contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_schedule import (audit_evaluation_schedule,
                                 build_evaluation_schedule)
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration, member


class EvaluationScheduleTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'protocol.json'

    def protocol(self, *, pairs=4):
        fields = declaration()
        fields['splits']['held_out'] = [member(task, state)
                                         for task in range(10) for state in (1, 2)]
        fields['seeds'] = {'pairing': list(range(17, 17 + pairs)),
                           'scheduling': 701, 'bootstrap': 991}
        fields['allocation'] = {f'task_{task}': {'pairs': pairs}
                                for task in range(10)}
        return EvaluationProtocol.freeze(self.path, **fields)

    def test_reproducible_schedule_audits_task_balance_pairing_and_order(self):
        protocol = self.protocol(pairs=6)
        first = build_evaluation_schedule(protocol)
        self.assertEqual(first, build_evaluation_schedule(protocol))
        report = audit_evaluation_schedule(protocol, first)
        self.assertEqual(report['attempts'], 180)
        self.assertEqual(set(report['attempts_per_task'].values()), {18})
        self.assertEqual(len({row['attempt_id'] for row in first}), 180)
        self.assertEqual({row['state_id'] for row in first}, {1, 2})
        for task in range(10):
            for position in range(3):
                conditions = [first[index + position]['condition']
                              for index in range(0, len(first), 3)
                              if first[index]['task_id'] == task]
                self.assertEqual(set(conditions),
                                 {'baseline', 'no_memory', 'fixed_memory'})
        for position in range(3):
            counts = {condition: sum(first[index + position]['condition'] == condition
                                     for index in range(0, len(first), 3))
                      for condition in ('baseline', 'no_memory', 'fixed_memory')}
            self.assertEqual(set(counts.values()), {20})

    def test_audit_rejects_modified_pair_and_duplicate_attempt(self):
        protocol = self.protocol()
        rows = build_evaluation_schedule(protocol)
        changed = [dict(row) for row in rows]
        changed[1]['environment_seed'] += 1
        with self.assertRaises(ProtocolError):
            audit_evaluation_schedule(protocol, changed)
        changed = [dict(row) for row in rows]
        changed[1]['attempt_id'] = changed[0]['attempt_id']
        with self.assertRaisesRegex(ProtocolError, 'duplicate attempt'):
            audit_evaluation_schedule(protocol, changed)
        changed = [dict(row) for row in rows]
        first, second = changed[:3], changed[3:6]
        changed[:6] = second + first
        for sequence, row in enumerate(changed):
            row['sequence'] = sequence
        with self.assertRaisesRegex(ProtocolError, 'scheduling seed'):
            audit_evaluation_schedule(protocol, changed)

    def test_schedule_seed_changes_order_without_changing_pairs(self):
        first = build_evaluation_schedule(self.protocol())
        self.path.unlink()
        fields = declaration()
        fields['splits']['held_out'] = [member(task, state)
                                         for task in range(10) for state in (1, 2)]
        fields['seeds'] = {'pairing': list(range(17, 21)),
                           'scheduling': 702, 'bootstrap': 991}
        fields['allocation'] = {f'task_{task}': {'pairs': 4}
                                for task in range(10)}
        second = build_evaluation_schedule(EvaluationProtocol.freeze(self.path, **fields))
        self.assertNotEqual([(row['task_id'], row['condition']) for row in first],
                            [(row['task_id'], row['condition']) for row in second])
        self.assertEqual({(row['task_id'], row['state_id'], row['environment_seed'],
                           row['condition']) for row in first},
                         {(row['task_id'], row['state_id'], row['environment_seed'],
                           row['condition']) for row in second})

    def test_rejects_incomplete_or_unequal_allocation(self):
        protocol = self.protocol()
        self.path.unlink()
        fields = declaration()
        fields['seeds']['scheduling'] = 701
        with self.assertRaisesRegex(ProtocolError, 'allocation'):
            build_evaluation_schedule(EvaluationProtocol.freeze(self.path, **fields))

    def test_scheduled_pair_runs_three_complete_replay_episodes(self):
        protocol = self.protocol(pairs=1)
        pair = build_evaluation_schedule(protocol)[:3]
        self.assertEqual({row['environment_seed'] for row in pair}, {17})
        outcomes = []
        for row in pair:
            seed = row['environment_seed']
            policy = ReplayPolicy((('visible', 'reach'),))
            environment = ReplayEnvironment(seed, 'visible', (
                ('reach', ReplayStep('held', 1.0, True, True, False)),))
            outcome = run_episode(EpisodeConfig(seed=seed, max_steps=2),
                                  policy, environment, ReplayRecorder())
            outcomes.append((outcome.success, outcome.steps, environment.actions))
        self.assertEqual(outcomes, [(True, 1, ['reach'])] * 3)


if __name__ == '__main__':
    unittest.main()
