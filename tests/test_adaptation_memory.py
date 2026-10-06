"""Public adaptation boundaries with complete sealed synthetic episodes."""

from dataclasses import asdict
import json
import unittest

from adaptation_memory import AdaptationMemory
from attempt_identity import AttemptIdentityRecorder
from episode_harness import ActionResolution, EpisodeConfig, run_episode
from fixed_memory import FixedMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_identity import FrozenSupervisorManifest
import test_memory_compatibility as fixtures
import test_decision_memory as decision_fixtures
from test_intervention_memory import pinned
from test_memory_snapshot import METADATA, RETRIEVAL
from test_supervisor_vlm import proposal


class AdaptationMemoryTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain
    episode = decision_fixtures.DecisionMemoryTests.episode

    def query(self, proposal):
        return dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
                    progress_context=fixtures.PROGRESS, failure_category='unknown')

    def adaptive(self, refs=()):
        snapshot = MemorySnapshot.freeze(self.root / 'start.json', self.store, refs,
            metadata=METADATA, retrieval=RETRIEVAL)
        return AdaptationMemory(self.root / 'start.json',
            expected_reference=snapshot.reference, query=self.query)

    def attempt(self, memory, name, success=False):
        source = self.root / name
        source.mkdir()
        config = EpisodeConfig(17, 2)
        supervisor = FrozenSupervisorManifest.freeze(source / 'supervisor.json',
                                                     {'model': 'synthetic'})
        identity = AttemptIdentityRecorder(source, config, lambda: {
            'task': fixtures.TASK, 'policy_assets': {'revision': 'synthetic'},
            'settings': {'action_capabilities': asdict(self.single)},
            **supervisor.reference}, ReplayRecorder())
        action = [0.] * len(self.single.components)
        correction = [0.1] * len(action)
        policy = ReplayPolicy([('start', action), ('middle', action)])
        environment = ReplayEnvironment(17, 'start', [
            (correction, ReplayStep('middle', 0, False, False, False)),
            (action, ReplayStep('done', 0, success, True, False))])
        policy.action_capabilities = environment.action_capabilities = self.single
        trace = TraceRecorder(source / 'trace', config, identity)
        contexts = []
        before = memory.reference

        def select(item):
            contexts.append(memory.prepare(item))
            # Actual ongoing execution has no sealed outcome to admit.
            with self.assertRaises(TraceError):
                memory.complete_episode(trace=(trace.directory / 'manifest.json', '0' * 64),
                    references=[], snapshot_path=self.root / 'premature.json')
            self.assertEqual(memory.reference, before)
            if item.observation.sequence:
                return ActionResolution('pass')
            component = self.single.components[0]
            context = {'episode_id': item.observation.episode_id,
                'proposal_id': item.proposal_id, 'observation_sequence': 0,
                'progress_context': fixtures.PROGRESS,
                'diagnosis': {'category': 'unknown', 'summary': 'synthetic', 'evidence': []},
                'request': {'episode_id': item.observation.episode_id,
                    'proposal_id': item.proposal_id, 'observation_sequence': 0,
                    'decision_id': 'synthetic', 'kind': 'adjustment', 'scope': 'single_action',
                    'target': component.arm, 'frame': component.frame,
                    'units': {component.name: component.unit},
                    'residual': {component.name: 0.1}}}
            (source / 'context.json').write_text(json.dumps(context))
            return ActionResolution('override', correction)

        outcome = run_episode(config, policy, environment, trace, action_selector=select)
        self.assertEqual((outcome.steps, outcome.success), (2, success))
        trace.seal(outcome)
        replayed = load_recorded_replay(trace.directory).run()
        self.assertEqual((replayed.steps, replayed.success), (2, success))
        ref = self.store.append(trace=pinned(trace.directory / 'manifest.json'),
            attempt=pinned(identity.path), supervisor=pinned(supervisor.path),
            context=pinned(source / 'context.json'))
        return ref, outcome, contexts, pinned(trace.directory / 'manifest.json')

    def test_failed_and_successful_outcomes_enter_only_the_next_episode_with_lineage(self):
        development = self.retain('development')
        memory = self.adaptive([development])
        starting = memory.reference
        expected = {development['record_id']}
        for index, success in enumerate((False, True)):
            before = memory.reference
            ref, outcome, contexts, trace = self.attempt(memory, str(index), success)
            for context in contexts:
                self.assertEqual({r['record_id'] for r in context['selected']}, expected)
                self.assertEqual(context['snapshot'], before)
            # Even an independently written, fully verified record is not yet visible.
            pending = memory.prepare(proposal(episode=outcome.episode_id))
            self.assertEqual({r['record_id'] for r in pending['selected']}, expected)
            path = self.root / f'after-{index}.json'
            lineage = memory.complete_episode(trace=trace, references=[ref], snapshot_path=path)
            self.assertEqual(lineage['before'], before)
            self.assertEqual(lineage['after'], memory.reference)
            self.assertEqual(lineage['starting_snapshot'], starting)
            self.assertEqual(lineage['episode_id'], outcome.episode_id)
            document = MemorySnapshot(path, expected_reference=memory.reference).read()
            self.assertEqual(document['metadata']['adaptation'],
                             {k: v for k, v in lineage.items() if k != 'after'})
            self.assertEqual(document['retrieval'], RETRIEVAL)
            expected.add(ref['record_id'])
        # Restart from an explicitly retained current pin preserves admitted membership.
        memory = AdaptationMemory(path, expected_reference=memory.reference, query=self.query)
        self.assertEqual({r['record_id'] for r in memory.prepare(proposal())['selected']}, expected)

    def test_sealed_provider_replay_uses_before_and_after_snapshots(self):
        memory = self.adaptive()
        for index, abstain in enumerate((True, False)):
            ref, _, _, trace = self.attempt(memory, f'admit-{index}', success=not abstain)
            memory.complete_episode(trace=trace, references=[ref],
                                    snapshot_path=self.root / f'admitted-{index}.json')
            before = memory.reference
            directory, rows, sent = self.episode(memory, f'provider-{index}', abstain)
            self.assertTrue(rows)
            self.assertEqual(len(rows), len(sent))
            for row, payload in zip(rows, sent):
                key = 'supervisor_abstention' if abstain else 'supervisor_pass'
                context = row['action_record'][key]['memory_context']
                self.assertEqual(context['snapshot'], before)
                self.assertIn(ref, context['references'])
                self.assertEqual(context['context_json'], payload['messages'][0]['content'][1]['text'])
            lineage = memory.complete_episode(trace=pinned(directory / 'manifest.json'), references=[],
                snapshot_path=self.root / f'empty-{index}.json')
            self.assertEqual(lineage['admitted'], [])
            self.assertNotEqual(lineage['before'], lineage['after'])

    def test_foreign_unverified_and_corrupt_evidence_cannot_advance_membership(self):
        foreign = self.retain('foreign')
        memory = self.adaptive()
        ref, outcome, _, trace = self.attempt(memory, 'active')
        original = memory.prepare(proposal(episode=outcome.episode_id))
        cases = [(trace, [foreign]),
                 (pinned(self.root / 'foreign/trace/manifest.json'), [foreign]),
                 ((trace[0], '0' * 64), [ref]),
                 (trace, [dict(ref, sha256='0' * 64)]),
                 (trace, [ref, ref])]
        for source, refs in cases:
            with self.subTest(source=source, refs=refs), self.assertRaises(TraceError):
                memory.complete_episode(trace=source, references=refs,
                                        snapshot_path=self.root / 'rejected.json')
            self.assertEqual(memory.prepare(proposal(episode=outcome.episode_id)), original)
            self.assertFalse((self.root / 'rejected.json').exists())
        (self.root / 'active/context.json').write_text('{}')
        with self.assertRaises(TraceError):
            memory.complete_episode(trace=trace, references=[ref], snapshot_path=self.root / 'bad.json')
        self.assertEqual(memory.reference, original['snapshot'])

    def test_episode_switch_self_retrieval_and_failed_snapshot_write_are_rejected(self):
        ref = self.retain('source')
        memory = self.adaptive([ref])
        record = self.store.read(ref['record_id'], expected_sha256=ref['sha256'])
        with self.assertRaisesRegex(TraceError, 'own experience'):
            memory.begin_episode(record['episode_id'])
        ref, outcome, _, trace = self.attempt(memory, 'new')
        original = memory.prepare(proposal(episode=outcome.episode_id))
        with self.assertRaisesRegex(TraceError, 'active adaptation episode'):
            memory.prepare(proposal(episode='foreign'))
        with self.assertRaises(TraceError):
            memory.complete_episode(trace=trace, references=[ref], snapshot_path=self.root / 'start.json')
        self.assertEqual(memory.prepare(proposal(episode=outcome.episode_id)), original)
        result = memory.complete_episode(trace=trace, references=[ref], snapshot_path=self.root / 'valid.json')
        pin = memory.reference
        result['after'].clear()
        memory.reference.clear()
        self.assertEqual(memory.reference, pin)
        with self.assertRaisesRegex(TraceError, 'conflicting completed episode'):
            memory.complete_episode(trace=trace, references=[], snapshot_path=self.root / 'again.json')

    def test_repeated_outcomes_after_restart_leave_membership_and_exposure_unchanged(self):
        memory = self.adaptive()
        attempts = []
        for index, success in enumerate((False, True)):
            ref, outcome, _, trace = self.attempt(memory, f'completed-{index}', success)
            path = self.root / f'completed-{index}.json'
            memory.complete_episode(trace=trace, references=[ref], snapshot_path=path)
            attempts.append((ref, outcome, trace))
            pin = memory.reference
            retry = memory.complete_episode(trace=trace, references=[ref], snapshot_path=path)
            self.assertEqual(retry['before'], retry['after'])
            self.assertEqual(memory.reference, pin)
        memory = AdaptationMemory(path, expected_reference=pin, query=self.query)
        original = path.read_bytes()
        expected = FixedMemory(path, expected_reference=pin, query=self.query).prepare(proposal())
        for ref, outcome, trace in attempts:
            retry = memory.complete_episode(trace=trace, references=[ref],
                snapshot_path=self.root / 'unused.json')
            self.assertTrue(retry['replayed'])
            self.assertEqual(retry['after'], pin)
            with self.assertRaisesRegex(TraceError, 'own experience'):
                memory.begin_episode(outcome.episode_id)
            with self.assertRaisesRegex(TraceError, 'conflicting completed episode'):
                memory.complete_episode(trace=trace, references=[],
                    snapshot_path=self.root / 'unused.json')
        document = MemorySnapshot(path, expected_reference=pin).read()
        self.assertEqual(len(document['records']), 2)
        self.assertEqual(len(document['metadata']['completed_episodes']), 2)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / 'unused.json').exists())
        self.assertEqual(memory.prepare(proposal()), expected)

    def test_empty_exposure_retry_is_idempotent_and_cannot_be_reopened(self):
        memory = self.adaptive()
        directory, _, _ = self.episode(memory, 'empty', True)
        trace = pinned(directory / 'manifest.json')
        path = self.root / 'empty.json'
        memory.complete_episode(trace=trace, references=[], snapshot_path=path)
        pin = memory.reference
        memory = AdaptationMemory(path, expected_reference=pin, query=self.query)
        self.assertTrue(memory.complete_episode(trace=trace, references=[],
            snapshot_path=path)['replayed'])
        episode_id = load_recorded_replay(directory).evidence()['source_episode_id']
        with self.assertRaisesRegex(TraceError, 'repeat exposure'):
            memory.begin_episode(episode_id)
        document = MemorySnapshot(path, expected_reference=pin).read()
        self.assertEqual(document['records'], [])
        self.assertEqual(len(document['metadata']['completed_episodes']), 1)
        # A retry must still verify the completed trace, even with no members.
        trace[0].write_bytes(trace[0].read_bytes() + b' ')
        with self.assertRaisesRegex(TraceError, 'pinned sealed trace'):
            memory.complete_episode(trace=trace, references=[], snapshot_path=path)
        self.assertEqual(memory.reference, pin)

    def test_retry_cannot_complete_or_replace_another_active_episode(self):
        memory = self.adaptive()
        ref, _, _, trace = self.attempt(memory, 'first')
        memory.complete_episode(trace=trace, references=[ref],
            snapshot_path=self.root / 'first.json')
        before = memory.prepare(proposal(episode='active'))
        with self.assertRaisesRegex(TraceError, 'active episode'):
            memory.complete_episode(trace=trace, references=[ref],
                snapshot_path=self.root / 'retry.json')
        self.assertEqual(memory.prepare(proposal(episode='active')), before)


if __name__ == '__main__':
    unittest.main()
