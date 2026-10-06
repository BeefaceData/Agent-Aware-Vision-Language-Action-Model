"""Snapshot contracts using public storage and complete sealed episode fixtures."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import shutil
import unittest

from intervention_memory import InterventionMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError, load_recorded_replay
import test_memory_compatibility as fixtures


RETRIEVAL = {'ranking_policy': 'exact-context-failure-v1',
    'summary_policy': 'whole-historical-summary-v1', 'max_entries': 4,
    'max_summary_bytes': 1024, 'max_context_bytes': 4096}
METADATA = {'population': 'synthetic-development', 'selection_revision': 'fixture-v1'}


class MemorySnapshotTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain

    def freeze(self, refs, name='snapshot.json', **changes):
        return MemorySnapshot.freeze(self.root / name, self.store, refs,
            **({'metadata': METADATA, 'retrieval': RETRIEVAL} | changes))

    def test_round_trip_retrieval_and_complete_replay(self):
        refs = [self.retain('failed', category='stall'),
                self.retain('successful', success=True)]
        snapshot = self.freeze(refs)
        loaded = MemorySnapshot(self.root / 'snapshot.json', expected_reference=snapshot.reference)
        document = loaded.read()
        self.assertEqual(document['records'], sorted(refs, key=lambda ref: ref['record_id']))
        self.assertEqual(document['metadata'], METADATA)
        self.assertEqual(document['retrieval'], RETRIEVAL)
        self.assertEqual(loaded.reference, snapshot.reference)
        query = dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
            progress_context=fixtures.PROGRESS, failure_category='stall',
            **{key: value for key, value in document['retrieval'].items() if key.startswith('max_')})
        reopened = InterventionMemory(self.root / document['store'])
        self.assertEqual(reopened.retrieve_context(document['records'], **query),
                         self.store.retrieve_context(refs, **query))
        for name, success in [('failed', False), ('successful', True)]:
            outcome = load_recorded_replay(self.root / name / 'trace').run()
            self.assertEqual((outcome.steps, outcome.success), (2, success))

    def test_deterministic_identity_and_membership_metadata_configuration_binding(self):
        refs = [self.retain('one'), self.retain('two')]
        original = self.freeze(refs)
        reordered = self.freeze(reversed(refs), 'reordered.json')
        self.assertEqual(original.reference, reordered.reference)
        for name, members, changes in [
            ('membership', refs[:1], {}),
            ('metadata', refs, {'metadata': METADATA | {'selection_revision': 'fixture-v2'}}),
            ('retrieval', refs, {'retrieval': RETRIEVAL | {'max_entries': 1}})]:
            changed = self.freeze(members, name + '.json', **changes)
            self.assertNotEqual(original.reference['snapshot_id'], changed.reference['snapshot_id'])
        before = original.read()
        self.retain('later')
        self.assertEqual(original.read(), before)

    def test_exclusive_creation_detached_reads_and_portability(self):
        ref = self.retain('source')
        snapshot = self.freeze([ref])
        with self.assertRaises(FileExistsError):
            self.freeze([])
        document = snapshot.read()
        document['records'].clear()
        document['retrieval']['max_entries'] = 0
        pin = snapshot.reference
        pin['sha256'] = '0' * 64
        self.assertEqual(snapshot.read()['records'], [ref])
        destination = self.root / 'relocated'
        destination.mkdir()
        for name in ('snapshot.json', 'memory', 'source'):
            source = self.root / name
            shutil.move(str(source), destination / name)
        relocated = MemorySnapshot(destination / 'snapshot.json', expected_reference=snapshot.reference)
        self.assertEqual(relocated.read()['records'], [ref])

    def test_manifest_changes_and_wrong_external_pins_fail(self):
        snapshot = self.freeze([self.retain('source')])
        path = self.root / 'snapshot.json'
        raw = path.read_bytes()
        for field in ('records', 'metadata', 'retrieval', 'store', 'snapshot_id'):
            with self.subTest(field=field):
                document = json.loads(raw)
                document[field] = [] if field == 'records' else 'changed'
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(TraceError, 'manifest digest mismatch'):
                    snapshot.read()
                path.write_bytes(raw)
        for field in ('snapshot_id', 'sha256'):
            with self.assertRaises(TraceError):
                MemorySnapshot(path, expected_reference=snapshot.reference | {field: '0' * 64})
        # A new file hash cannot bless changed metadata under the old identity.
        document = json.loads(raw)
        document['metadata']['selection_revision'] = 'tampered'
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(TraceError, 'identity mismatch'):
            MemorySnapshot(path, expected_reference=snapshot.reference | {
                'sha256': sha256(path.read_bytes()).hexdigest()})

    def test_missing_or_changed_records_and_transitive_evidence_fail(self):
        ref = self.retain('source')
        snapshot = self.freeze([ref])
        files = [self.store.directory / (ref['record_id'] + '.json'),
                 self.root / 'source' / 'context.json',
                 self.root / 'source' / 'trace' / 'manifest.json']
        # Include a manifest-pinned trace payload, not just direct record refs.
        trace = self.root / 'source' / 'trace'
        manifest = json.loads((trace / 'manifest.json').read_bytes())
        files.extend(trace / name for name in manifest['files'])
        for path in files:
            raw = path.read_bytes()
            for missing in (False, True):
                with self.subTest(path=path.name, missing=missing):
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(raw + b' ')
                    with self.assertRaisesRegex(TraceError, ref['record_id']):
                        snapshot.read()
                    path.write_bytes(raw)
        path = self.root / 'snapshot.json'
        path.unlink()
        with self.assertRaises(TraceError):
            snapshot.read()

    def test_freeze_rejects_corrupt_evidence_and_duplicate_or_malformed_members(self):
        ref = self.retain('source')
        for references in ([ref, ref], [ref | {'sha256': '0' * 64}],
                           [{'record_id': '../source'}], [ref | {'extra': 1}]):
            with self.subTest(references=references), self.assertRaises(TraceError):
                self.freeze(references)
            self.assertFalse((self.root / 'snapshot.json').exists())
        (self.root / 'source' / 'context.json').unlink()
        with self.assertRaises(TraceError):
            self.freeze([ref])
        self.assertFalse((self.root / 'snapshot.json').exists())

    def test_empty_snapshot_and_invalid_configuration(self):
        snapshot = self.freeze([])
        self.assertEqual(snapshot.read()['records'], [])
        for changes in ({'metadata': {}}, {'metadata': {'value': float('nan')}},
                        {'retrieval': {}},
                        {'retrieval': RETRIEVAL | {'ranking_policy': 'unknown'}},
                        {'retrieval': RETRIEVAL | {'max_entries': True}},
                        {'retrieval': RETRIEVAL | {'max_summary_bytes': -1}},
                        {'retrieval': RETRIEVAL | {'max_context_bytes': 1}}):
            with self.subTest(changes=changes), self.assertRaises(TraceError):
                self.freeze([], 'invalid.json', **deepcopy(changes))
            self.assertFalse((self.root / 'invalid.json').exists())


if __name__ == '__main__':
    unittest.main()
