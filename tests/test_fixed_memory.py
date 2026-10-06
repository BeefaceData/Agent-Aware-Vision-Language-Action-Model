"""Public read-only evaluation contracts over sealed synthetic episodes."""

from dataclasses import asdict
import json
import unittest

from fixed_memory import FixedMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError
import test_memory_compatibility as fixtures
import test_decision_memory as decision_fixtures
from test_intervention_memory import pinned
from test_memory_snapshot import METADATA, RETRIEVAL


class FixedMemoryTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain
    episode = decision_fixtures.DecisionMemoryTests.episode

    def query(self, proposal):
        return dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
                    progress_context=fixtures.PROGRESS, failure_category='stall')

    def fixed(self, refs, **settings):
        snapshot = MemorySnapshot.freeze(self.root / 'snapshot.json', self.store, refs,
            metadata=METADATA, retrieval=RETRIEVAL | settings)
        return FixedMemory(self.root / 'snapshot.json',
                           expected_reference=snapshot.reference, query=self.query)

    def test_reject_writes_and_keep_membership_across_complete_episodes_and_restart(self):
        ref = self.retain('development', success=True)
        memory = self.fixed([ref])
        original = memory.prepare(None)
        pin = memory.reference
        self.assertEqual([row['record_id'] for row in original['selected']], [ref['record_id']])
        self.assertEqual(original['references'], [ref])
        for index, abstain in enumerate((False, True)):
            # A separate recorder may retain new outcomes, even in the same store.
            self.retain('evaluation' + str(index), success=not abstain)
            source = self.root / ('evaluation' + str(index))
            with self.assertRaisesRegex(TraceError, 'read-only'):
                memory.append(trace=pinned(source / 'trace/manifest.json'),
                    attempt=pinned(source / 'attempt.json'),
                    supervisor=pinned(source / 'supervisor.json'),
                    context=pinned(source / 'context.json'))
            _, rows, sent = self.episode(memory, 'run' + str(index), abstain)
            self.assertTrue(rows)
            self.assertEqual(len(rows), len(sent))
            for row, payload in zip(rows, sent):
                key = 'supervisor_abstention' if abstain else 'supervisor_pass'
                evidence = row['action_record'][key]['memory_context']
                self.assertEqual(evidence, json.loads(json.dumps(original)))
                self.assertEqual(payload['messages'][0]['content'][1]['text'],
                                 original['context_json'])
            self.assertEqual(memory.reference, pin)
            memory = FixedMemory(self.root / 'snapshot.json',
                                 expected_reference=pin, query=self.query)
            self.assertEqual(memory.prepare(None), original)

    def test_detached_results_and_pins_cannot_change_retrieval(self):
        memory = self.fixed([self.retain('source')], max_entries=1)
        original = memory.prepare(None)
        result = memory.prepare(None)
        result['references'].clear()
        result['selected'].clear()
        result['settings']['max_entries'] = 0
        memory.reference['snapshot_id'] = '0' * 64
        self.assertEqual(memory.prepare(None), original)

    def test_query_cannot_override_snapshot_membership_or_settings(self):
        memory = self.fixed([self.retain('source')])
        for key, value in [('max_entries', 0), ('references', []),
                           ('snapshot', None), ('ranking_policy', 'foreign')]:
            query = self.query(None) | {key: value}
            changed = FixedMemory(self.root / 'snapshot.json',
                expected_reference=memory.reference, query=lambda _: query)
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed.prepare(None)

    def test_empty_snapshot_stays_empty_after_new_records(self):
        memory = self.fixed([])
        original = memory.prepare(None)
        self.retain('later')
        self.assertEqual(memory.prepare(None), original)
        self.assertEqual(original['context_json'], '[]')
        self.assertEqual(original['retrieval'], 'enabled')

    def test_manifest_and_source_drift_fail_even_with_zero_budget(self):
        memory = self.fixed([self.retain('source')], max_entries=0)
        for path in (self.root / 'snapshot.json', self.root / 'source/context.json'):
            raw = path.read_bytes()
            for missing in (False, True):
                if missing:
                    path.unlink()
                else:
                    path.write_bytes(raw + b' ')
                with self.subTest(path=path, missing=missing), self.assertRaises(TraceError):
                    memory.prepare(None)
                path.write_bytes(raw)


if __name__ == '__main__':
    unittest.main()
