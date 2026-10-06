"""Development admission through public APIs and complete sealed replay."""

from copy import deepcopy
from dataclasses import replace
import unittest

from development_memory import freeze_development_snapshot
from fixed_memory import FixedMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError
import test_memory_compatibility as fixtures
import test_fixed_memory as fixed_fixtures
from test_memory_snapshot import RETRIEVAL


class DevelopmentMemoryTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain
    episode = fixed_fixtures.FixedMemoryTests.episode
    query = fixed_fixtures.FixedMemoryTests.query

    def declarations(self, refs):
        records = [self.store.read(r['record_id'], expected_sha256=r['sha256']) for r in refs]
        first = records[0]
        return ({'revision': 'reviewed-synthetic-split-v1', 'episodes': {
                    r['episode_id']: 'development' for r in records}},
                first['models'] | {'settings': first['configuration']['settings']})

    def freeze(self, refs, membership, compatibility, name='snapshot.json'):
        return freeze_development_snapshot(self.root / name, self.store, refs,
            membership=membership, compatibility=compatibility, retrieval=RETRIEVAL)

    def test_mixed_population_report_and_fixed_complete_episodes(self):
        refs = [self.retain('failed'), self.retain('success', success=True),
                self.retain('held'), self.retain('unknown'),
                self.retain('incompatible', replace(self.single, control_frequency_hz=10.)),
                self.retain('corrupt')]
        membership, compatible = self.declarations(refs)
        for index, split in ((2, 'held_out'), (3, 'unknown')):
            record = self.store.read(refs[index]['record_id'], expected_sha256=refs[index]['sha256'])
            membership['episodes'][record['episode_id']] = split
        (self.root / 'corrupt/context.json').unlink()
        snapshot = self.freeze(refs, membership, compatible)
        document = snapshot.read()
        self.assertEqual(document['records'], sorted(refs[:2], key=lambda r: r['record_id']))
        report = document['metadata']['development_selection']
        self.assertEqual({r['record_id']: r['reasons'] for r in report['excluded']}, {
            refs[2]['record_id']: ['held_out'], refs[3]['record_id']: ['unknown_origin'],
            refs[4]['record_id']: ['settings_mismatch'],
            refs[5]['record_id']: ['unverified_evidence']})
        self.assertEqual(len(report['included']), 2)
        memory = FixedMemory(self.root / 'snapshot.json',
            expected_reference=snapshot.reference, query=self.query)
        for index, abstain in enumerate((False, True)):
            _, rows, sent = self.episode(memory, 'evaluation' + str(index), abstain)
            self.assertTrue(rows)
            self.assertEqual(len(rows), len(sent))
            for row in rows:
                key = 'supervisor_abstention' if abstain else 'supervisor_pass'
                context = row['action_record'][key]['memory_context']
                self.assertEqual(context['references'], document['records'])
        reopened = MemorySnapshot(self.root / 'snapshot.json', expected_reference=snapshot.reference)
        self.assertEqual(reopened.read(), document)

    def test_model_mismatches_and_missing_origin_are_excluded(self):
        ref = self.retain('source')
        membership, compatible = self.declarations([ref])
        for key in ('policy', 'supervisor', 'settings'):
            changed = deepcopy(compatible)
            changed[key]['revision'] = 'different'
            snapshot = self.freeze([ref], membership, changed, key + '.json')
            self.assertEqual(snapshot.read()['records'], [])
            self.assertEqual(snapshot.read()['metadata']['development_selection']['excluded'],
                             [dict(ref, reasons=[key + '_mismatch'])])
        membership['episodes'].clear()
        snapshot = self.freeze([ref], membership, compatible)
        self.assertEqual(snapshot.read()['metadata']['development_selection']['excluded'],
                         [dict(ref, reasons=['unknown_origin'])])

    def test_selection_identity_binds_declarations_and_is_order_independent(self):
        refs = [self.retain('one'), self.retain('two')]
        membership, compatible = self.declarations(refs)
        snapshot = self.freeze(refs, membership, compatible)
        self.assertEqual(snapshot.reference,
            self.freeze(reversed(refs), membership, compatible, 'reversed.json').reference)
        changed = self.freeze(refs, membership | {'revision': 'review-v2'}, compatible, 'new.json')
        self.assertNotEqual(snapshot.reference, changed.reference)
        membership['episodes'].clear()
        compatible.clear()
        self.assertEqual(len(snapshot.read()['records']), 2)
        with self.assertRaises(FileExistsError):
            self.freeze(refs, *self.declarations(refs))

    def test_malformed_declarations_fail_before_publication(self):
        ref = self.retain('source')
        membership, compatible = self.declarations([ref])
        for refs, members, policy in (
            ([ref, ref], membership, compatible),
            ([ref | {'sha256': 'bad'}], membership, compatible),
            ([ref], {}, compatible), ([ref], membership, {}),
            ([ref], membership | {'episodes': {'episode': 'evaluation'}}, compatible),
            ([ref], membership | {'revision': ''}, compatible)):
            with self.subTest(members=members), self.assertRaises(TraceError):
                self.freeze(refs, members, policy)
            self.assertFalse((self.root / 'snapshot.json').exists())


if __name__ == '__main__':
    unittest.main()
