"""Public campaign gate and complete replay through the gated entry point."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from evaluation_preflight import (preflight_evaluation_campaign,
                                  start_evaluation_campaign)
from evaluation_protocol import EvaluationProtocol, ProtocolError
from evaluation_schedule import build_evaluation_schedule
from intervention_memory import InterventionMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceRecorder
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from tests.test_evaluation_protocol import declaration, member


class CampaignPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fields = declaration()
        fields['seeds']['scheduling'] = 701
        fields['allocation'] = {f'task_{task}': {'pairs': 1}
                                for task in range(10)}
        fields['splits']['held_out'] = [member(task, 1) for task in range(10)]
        self.snapshot_path = self.root / 'snapshot.json'
        self.snapshot = MemorySnapshot.freeze(
            self.snapshot_path, InterventionMemory(self.root / 'memory'), [],
            metadata={'population': 'synthetic-development'},
            retrieval={'ranking_policy': 'exact-context-failure-v1',
                       'summary_policy': 'whole-historical-summary-v1',
                       'max_entries': 0, 'max_summary_bytes': 0,
                       'max_context_bytes': 2}).reference
        for condition in fields['conditions']:
            fields['conditions'][condition].update(
                model_id='frozen-policy-revision',
                configuration_id=f'{condition}-sealed-config')
        fields['conditions']['fixed_memory']['memory'] = self.snapshot
        fields['horizons'].update(supervisor_calls=4, wall_clock_seconds=120)
        self.protocol = EvaluationProtocol.freeze(self.root / 'protocol.json', **fields)
        self.rows = build_evaluation_schedule(self.protocol)
        self.gates = {
            'resolved_conditions': {
                name: {'model_id': 'frozen-policy-revision',
                       'configuration_id': f'{name}-sealed-config'}
                for name in fields['conditions']},
            'observed_states': deepcopy(fields['splits']['development'] +
                                        fields['splits']['held_out']),
            'snapshot_path': self.snapshot_path,
            'runtime_caps': {'primary_actions': 5, 'supervisor_calls': 4,
                             'wall_clock_seconds': 120},
            'resource_records': {
                name: {'protocol_id': self.protocol.reference['protocol_id'],
                       'approved': True, 'reference': f'approval-{name}',
                       'max_seconds': 3600, 'max_model_calls': 120}
                for name in ('compute', 'api', 'data_use')},
        }

    def report(self, gates):
        return preflight_evaluation_campaign(self.protocol, self.rows, **gates)

    def test_complete_gate_runs_one_full_replay_episode(self):
        self.assertEqual(self.report(self.gates)['status'], 'ready')
        called = []

        def replay(row, attempt):
            called.append(row['attempt_id'])
            config = EpisodeConfig(seed=row['environment_seed'], max_steps=2)
            recorder = TraceRecorder(attempt / 'replay', config, ReplayRecorder())
            outcome = run_episode(
                config, ReplayPolicy((('visible', 'reach'),)),
                ReplayEnvironment(row['environment_seed'], 'visible', (
                    ('reach', ReplayStep('held', 1.0, True, True, False)),)),
                recorder)
            recorder.seal(outcome)
            return outcome

        outcomes = start_evaluation_campaign(
            self.protocol, self.rows, self.root / 'attempts', replay,
            **self.gates, max_new=1)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]['success'])

    def test_each_missing_or_mismatched_category_blocks_before_execution(self):
        mutations = {
            'model': lambda g: g['resolved_conditions']['baseline'].pop('model_id'),
            'configuration': lambda g: g['resolved_conditions']['no_memory'].update(
                configuration_id='different'),
            'dataset': lambda g: g['observed_states'].pop(),
            'snapshot': lambda g: g.update(snapshot_path=None),
            'call cap': lambda g: g['runtime_caps'].pop('supervisor_calls'),
            'time cap': lambda g: g['runtime_caps'].pop('wall_clock_seconds'),
            'action cap': lambda g: g['runtime_caps'].pop('primary_actions'),
            'compute': lambda g: g['resource_records'].pop('compute'),
            'api': lambda g: g['resource_records'].pop('api'),
            'data use': lambda g: g['resource_records'].pop('data_use'),
            'compute allowance': lambda g: g['resource_records']['compute'].update(
                max_seconds=1),
            'api allowance': lambda g: g['resource_records']['api'].update(
                max_model_calls=1),
        }
        for category, mutate in mutations.items():
            with self.subTest(category=category):
                gates = deepcopy(self.gates)
                mutate(gates)
                report = self.report(gates)
                self.assertEqual(report['status'], 'blocked')
                self.assertTrue(report['reasons'])
                destination = self.root / 'blocked-attempts'
                with self.assertRaisesRegex(ProtocolError, 'campaign preflight blocked'):
                    start_evaluation_campaign(
                        self.protocol, self.rows, destination,
                        lambda *_: self.fail('preflight must precede execution'),
                        **gates)
                self.assertFalse(destination.exists())

    def test_changed_snapshot_file_blocks_before_execution(self):
        self.snapshot_path.write_text('{}', encoding='utf-8')
        report = self.report(self.gates)
        self.assertEqual(report['status'], 'blocked')
        self.assertTrue(any('snapshot' in reason for reason in report['reasons']))
        with self.assertRaisesRegex(ProtocolError, 'snapshot'):
            start_evaluation_campaign(
                self.protocol, self.rows, self.root / 'attempts',
                lambda *_: self.fail('must not run'), **self.gates)
        self.assertFalse((self.root / 'attempts').exists())


if __name__ == '__main__':
    unittest.main()
