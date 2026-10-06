"""Public memory provenance contracts over complete synthetic episode replay."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from action_capabilities import libero_native_capabilities
from attempt_identity import AttemptIdentityRecorder
from episode_harness import EpisodeConfig, RecoverySequence, run_episode
from intervention_memory import InterventionMemory
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_identity import FrozenSupervisorManifest
from recovery_monitor import check_recovery
import test_intervention_limits as fixtures
import test_single_action_adjustment as adjustments


def pinned(path):
    return path, sha256(path.read_bytes()).hexdigest()


class InterventionMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.make_episode()

    def make_episode(self, name='source', *, success=False, historical_check=None):
        self.source = self.root / name
        self.source.mkdir()
        fixture = fixtures.InterventionLimitTests()
        fixture.setUp()
        config, policy, environment, select, _ = fixture.episode([fixtures.TOOL, fixtures.TOOL])
        if success:
            step = environment.step

            def succeed(action):
                return replace(step(action), success=True)

            environment.step = succeed
        supervisor = FrozenSupervisorManifest.freeze(self.source / 'supervisor.json',
            {'model': 'synthetic-frozen-supervisor', 'prompt_sha256': 'a' * 64})
        identity = AttemptIdentityRecorder(self.source, config, lambda: {
            'task': {'suite': 'synthetic', 'task_id': 0, 'instruction': 'place object'},
            'policy_assets': {'revision': 'synthetic-frozen-policy'},
            'settings': {'action_capabilities': asdict(libero_native_capabilities()),
                         'mode': 'synthetic-recovery'}, **supervisor.reference}, ReplayRecorder())
        contexts = []

        def retain_context(proposal):
            resolution = select(proposal)
            context = {'episode_id': proposal.observation.episode_id,
                'proposal_id': proposal.proposal_id,
                'observation_sequence': proposal.observation.sequence,
                'diagnosis': {'category': 'suspected_missed_grasp',
                    'summary': 'Synthetic missed grasp assessment',
                    'evidence': [{'observation_sequence': proposal.observation.sequence,
                                 'source': 'main', 'description': 'Synthetic open gripper'}]},
                'request': resolution.recovery.request}
            path = self.source / f'context-{len(contexts)}.json'
            with path.open('x') as stream:
                json.dump(context, stream)
            contexts.append(path)
            return resolution

        class HistoricalTrace(TraceRecorder):
            # Retain the formerly supported local assessment at termination,
            # independently of the evaluator's success result.
            def record_step(self, step, source, action, result, ingestion):
                recovery = deepcopy(result.action_record.recovery)
                plan = RecoverySequence(**recovery['sequence'])
                assessment = fixture.recovery.observe(plan, 0, result.observation)
                if historical_check == 'aborted':
                    assessment = replace(assessment, gripper_open_confirmed=False)
                recovery['check'] = check_recovery(plan, 0, result.observation, assessment, 10.)
                result = replace(result, action_record=replace(result.action_record, recovery=recovery))
                super().record_step(step, source, action, result, ingestion)

        recorder = HistoricalTrace if historical_check else TraceRecorder
        trace = recorder(self.source / 'trace', config, identity)
        self.outcome = run_episode(config, policy, environment, trace,
            action_selector=retain_context, recovery_observer=fixture.recovery.observe,
            clock=lambda: 10.)
        trace.seal(self.outcome)
        self.contexts = contexts
        self.arguments = dict(trace=pinned(trace.directory / 'manifest.json'),
            attempt=pinned(identity.path), supervisor=pinned(supervisor.path),
            context=pinned(contexts[0]))
        self.store = InterventionMemory(self.root / 'memory')

    def read(self, reference):
        return self.store.read(reference['record_id'], expected_sha256=reference['sha256'])

    def test_complete_record_round_trip_and_failed_episode_replay(self):
        reference = self.store.append(**self.arguments)
        record = self.read(reference)
        self.assertEqual(record['task']['instruction'], 'place object')
        self.assertEqual(record['models']['policy']['revision'], 'synthetic-frozen-policy')
        self.assertEqual(record['robot_capabilities']['control_frequency_hz'], 20)
        self.assertEqual(record['diagnosis']['category'], 'suspected_missed_grasp')
        self.assertEqual(record['request']['tool_name'], fixtures.TOOL)
        self.assertEqual(len(record['execution']), 2)
        self.assertTrue(all(row['action_record']['executed_action'] is not None
                            for row in record['execution']))
        self.assertFalse(record['episode_outcome']['success'])
        self.assertEqual(record['local_outcome'], {'status': 'completed', 'reason': None})
        replayed = load_recorded_replay(self.source / 'trace').run()
        self.assertEqual((replayed.steps, replayed.success), (4, False))
        record['diagnosis']['category'] = 'invented'
        self.assertEqual(self.read(reference)['diagnosis']['category'], 'suspected_missed_grasp')

    def test_failed_local_recovery_is_not_promoted_by_terminal_success(self):
        self.make_episode('failed-local', success=True, historical_check='aborted')
        record = self.read(self.store.append(**self.arguments))
        self.assertEqual(record['local_outcome'],
                         {'status': 'aborted', 'reason': 'gripper_open_confirmed'})
        self.assertTrue(record['episode_outcome']['success'])
        self.assertTrue(load_recorded_replay(self.source / 'trace').run().success)

    def test_terminal_cancellation_remains_aborted_despite_task_success(self):
        self.make_episode('cancelled', success=True)
        record = self.read(self.store.append(**self.arguments))
        self.assertEqual(record['local_outcome'],
                         {'status': 'aborted', 'reason': 'episode_terminated'})
        self.assertTrue(record['episode_outcome']['success'])
        self.assertTrue(load_recorded_replay(self.source / 'trace').run().success)

    def test_unfinished_local_check_stays_unknown_despite_task_success(self):
        self.make_episode('unknown', success=True, historical_check='continuing')
        record = self.read(self.store.append(**self.arguments))
        self.assertEqual(record['local_outcome'],
                         {'status': 'unknown', 'reason': 'no_terminal_local_check'})
        self.assertTrue(record['episode_outcome']['success'])
        self.assertTrue(load_recorded_replay(self.source / 'trace').run().success)

    def test_local_outcome_tampering_rejected_even_with_new_record_digest(self):
        reference = self.store.append(**self.arguments)
        record = self.read(reference)
        record['local_outcome']['status'] = 'aborted'
        path = self.store.directory / (reference['record_id'] + '.json')
        path.write_text(json.dumps(record))
        with self.assertRaises(TraceError):
            self.store.read(reference['record_id'], expected_sha256=pinned(path)[1])

    def test_legacy_record_reads_without_rewriting_immutable_bytes(self):
        reference = self.store.append(**self.arguments)
        record = self.read(reference)
        record['version'] = 1
        del record['local_outcome']
        path = self.store.directory / (reference['record_id'] + '.json')
        path.write_text(json.dumps(record))
        original = path.read_bytes()
        loaded = self.store.read(reference['record_id'], expected_sha256=pinned(path)[1])
        self.assertEqual(loaded, record)
        self.assertEqual(path.read_bytes(), original)

    def test_append_preserves_prior_bytes_and_duplicate_is_rejected(self):
        first = self.store.append(**self.arguments)
        path = self.store.directory / (first['record_id'] + '.json')
        original = path.read_bytes()
        second = self.store.append(**(self.arguments | {'context': pinned(self.contexts[1])}))
        self.assertNotEqual(first['record_id'], second['record_id'])
        with self.assertRaises(FileExistsError):
            self.store.append(**self.arguments)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(self.read(first)['execution']), 2)
        self.assertEqual(len(self.read(second)['execution']), 2)

    def test_missing_or_changed_pinned_provenance_never_creates_store(self):
        for name in self.arguments:
            with self.subTest(name=name), self.assertRaises(TraceError):
                self.store.append(**(self.arguments | {name: (self.arguments[name][0], '0' * 64)}))
        self.contexts[0].unlink()
        with self.assertRaises(TraceError):
            self.store.append(**self.arguments)
        self.assertFalse(self.store.directory.exists())

    def test_missing_and_foreign_context_are_rejected_even_with_new_digest(self):
        path = self.contexts[0]
        original = json.loads(path.read_text())
        damaged = [dict(original, episode_id='foreign'), dict(original, diagnosis=None),
                   dict(original, observation_sequence=1), dict(original, request={}),
                   {key: value for key, value in original.items() if key != 'diagnosis'},
                   dict(original, outcome={'success': True})]
        bad_request = original['request'] | {'parameters': [['retreat_m', 0.03]]}
        damaged.append(dict(original, request=bad_request))
        future = {'category': 'stall', 'summary': 'future', 'evidence': [
            {'observation_sequence': 3, 'source': 'main', 'description': 'future'}]}
        damaged.append(dict(original, diagnosis=future))
        for value in damaged:
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(TraceError):
                self.store.append(**(self.arguments | {'context': pinned(path)}))
        self.assertFalse(self.store.directory.exists())

    def test_missing_identity_and_foreign_config_are_rejected(self):
        path = self.arguments['attempt'][0]
        original = json.loads(path.read_text())
        for key in ('task', 'policy_assets', 'settings', 'supervisor_sha256'):
            value = dict(original)
            del value[key]
            path.write_text(json.dumps(value))
            with self.subTest(key=key), self.assertRaises(TraceError):
                self.store.append(**(self.arguments | {'attempt': pinned(path)}))
        value = original | {'episode_config': original['episode_config'] | {'seed': 99}}
        path.write_text(json.dumps(value))
        with self.assertRaises(TraceError):
            self.store.append(**(self.arguments | {'attempt': pinned(path)}))

    def test_reads_reject_changed_records_and_changed_or_missing_evidence(self):
        reference = self.store.append(**self.arguments)
        path = self.store.directory / (reference['record_id'] + '.json')
        original = path.read_bytes()
        value = json.loads(original)
        value['episode_outcome']['success'] = True
        path.write_text(json.dumps(value))
        with self.assertRaises(TraceError):
            self.read(reference)
        with self.assertRaises(TraceError):
            self.store.read(reference['record_id'], expected_sha256=pinned(path)[1])
        path.write_bytes(original)
        decisions = self.source / 'trace/decisions.jsonl'
        decisions.write_bytes(decisions.read_bytes() + b'\n')
        with self.assertRaises(TraceError):
            self.read(reference)
        decisions.unlink()
        with self.assertRaises(TraceError):
            self.read(reference)

    def test_common_parent_can_move_and_record_paths_cannot_escape_by_id(self):
        reference = self.store.append(**self.arguments)
        moved = self.root / 'moved'
        moved.mkdir()
        shutil.move(self.source, moved / 'source')
        shutil.move(self.store.directory, moved / 'memory')
        self.store = InterventionMemory(moved / 'memory')
        self.assertFalse(self.read(reference)['episode_outcome']['success'])
        with self.assertRaises(TraceError):
            self.store.read('../source/context-0', expected_sha256=reference['sha256'])

    def test_single_adjustment_retains_request_and_actual_action(self):
        fixture = adjustments.SingleActionAdjustmentTests()
        fixture.setUp()
        source = self.root / 'adjustment'
        source.mkdir()
        config = EpisodeConfig(17, 1)
        supervisor = FrozenSupervisorManifest.freeze(source / 'supervisor.json',
                                                     {'model': 'synthetic'})
        identity = AttemptIdentityRecorder(source, config, lambda: {
            'task': {'instruction': 'place'}, 'policy_assets': {'revision': 'synthetic'},
            'settings': {'action_capabilities': asdict(libero_native_capabilities())},
            **supervisor.reference}, ReplayRecorder())
        initial = {'task': 'place'}
        adjusted = [0.1 + 0.01 / 0.05, 0.2 - 0.02 / 0.05,
                    0.3 + 0.005 / 0.05, 0.4, 0.5, 0.6, -1]
        environment = ReplayEnvironment(17, initial, [
            (adjusted, ReplayStep({'task': 'place'}, 1, True, True, False))])
        context = source / 'context.json'

        def select(proposal):
            request = fixture.request(proposal)
            context.write_text(json.dumps({
                'episode_id': proposal.observation.episode_id,
                'proposal_id': proposal.proposal_id, 'observation_sequence': 0,
                'diagnosis': {'category': 'unknown', 'summary': 'Synthetic host probe',
                              'evidence': []}, 'request': request}))
            return fixture.executor.resolve(proposal, request)

        trace = TraceRecorder(source / 'trace', config, identity)
        outcome = run_episode(config, ReplayPolicy([(initial, fixture.action)]),
                              environment, trace, action_selector=select)
        trace.seal(outcome)
        reference = self.store.append(trace=pinned(trace.directory / 'manifest.json'),
            attempt=pinned(identity.path), supervisor=pinned(supervisor.path), context=pinned(context))
        record = self.read(reference)
        self.assertEqual(record['request']['residual']['translation_x'], 0.01)
        self.assertEqual(record['execution'][0]['action_record']['executed_action'], adjusted)
        self.assertTrue(record['episode_outcome']['success'])
        self.assertEqual(record['local_outcome'],
                         {'status': 'unknown', 'reason': 'no_local_assessment'})
        self.assertTrue(load_recorded_replay(trace.directory).run().success)


if __name__ == '__main__':
    unittest.main()
