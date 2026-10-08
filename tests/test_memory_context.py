"""Public context budgets over verified complete synthetic episode replays."""

from dataclasses import asdict
import json
import unittest

from intervention_memory import InterventionMemory
from recorded_replay import TraceError
import test_memory_compatibility as fixtures
import test_intervention_memory as recovery_fixtures


class MemoryContextTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain

    def query(self, refs, **changes):
        settings = dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
            compatibility=fixtures.COMPATIBILITY,
            progress_context=fixtures.PROGRESS, failure_category='stall',
            max_entries=10, max_summary_bytes=10000, max_context_bytes=100000)
        return self.store.retrieve_context(refs, **(settings | changes))

    def test_exact_context_and_summary_boundaries_preserve_whole_evidence(self):
        ref = self.retain('unknown', truncated=True)
        result = self.query([ref])
        size = result['context_bytes']
        self.assertEqual(size, len(result['context_json'].encode('utf-8')))
        exact = self.query([ref], max_context_bytes=size, max_summary_bytes=size - 2)
        self.assertEqual(json.loads(exact['context_json']), exact['selected'])
        summary = exact['selected'][0]
        self.assertEqual({key: summary[key] for key in ref}, ref)
        self.assertEqual(summary['local_outcome']['status'], 'unknown')
        self.assertEqual(summary['task_outcome']['status'], 'withheld')
        self.assertIn('task_outcome_unverified', summary['evidence_limitations'])
        self.assertIn('causal_benefit_unverified', summary['evidence_limitations'])
        for change, reason in ((dict(max_context_bytes=size - 1), 'context_size_limit'),
                               (dict(max_summary_bytes=size - 3), 'summary_size_limit')):
            bounded = self.query([ref], **change)
            self.assertEqual(bounded['context_json'], '[]')
            self.assertEqual(bounded['omitted'], [dict(ref, reason=reason)])

    def test_entry_caps_commas_duplicates_order_and_reopening(self):
        refs = [self.retain('failed', category='stall'),
                self.retain('success', category='stall', success=True),
                self.retain('unknown', truncated=True)]
        full = self.query(refs)
        self.assertEqual(full['context_bytes'], len(full['context_json'].encode()))
        limited = self.query(refs, max_entries=2)
        self.assertEqual(len(limited['selected']), 2)
        self.assertEqual(limited['selected'], full['selected'][:2])
        self.assertEqual(limited['omitted'][0]['reason'], 'entry_limit')
        self.store = InterventionMemory(self.root / 'memory')
        self.assertEqual(self.query(reversed(refs + refs), max_entries=2), limited)
        exact = self.query(refs, max_context_bytes=limited['context_bytes'])
        self.assertEqual(exact['selected'], limited['selected'])

    def test_oversized_candidate_does_not_prevent_smaller_later_summary(self):
        large = self.retain('large', category='unknown', truncated=True)
        small = self.retain('small', category='stall')
        small_size = self.query([small])['context_bytes'] - 2
        result = self.query([small, large], failure_category='unknown',
                            max_summary_bytes=small_size)
        self.assertEqual([s['record_id'] for s in result['selected']], [small['record_id']])
        self.assertEqual(result['omitted'], [dict(large, reason='summary_size_limit')])

    def test_empty_and_zero_allowances_and_invalid_settings(self):
        self.assertEqual(self.query([])['context_json'], '[]')
        ref = self.retain('source')
        for settings in (dict(max_entries=0), dict(max_summary_bytes=0),
                         dict(max_context_bytes=2)):
            result = self.query([ref], **settings)
            self.assertEqual(result['selected'], [])
            self.assertEqual(result['context_bytes'], 2)
        for key in ('max_entries', 'max_summary_bytes', 'max_context_bytes'):
            for value in (-1, True, 1.5, None, '10'):
                with self.subTest(key=key, value=value), self.assertRaises(TraceError):
                    self.query([], **{key: value})
        with self.assertRaises(TraceError):
            self.query([], max_context_bytes=1)

    def test_caps_never_bypass_verification_and_raw_evidence_is_not_context(self):
        ref = self.retain('source')
        summary = self.query([ref])['selected'][0]
        self.assertTrue(set(summary).isdisjoint({'execution', 'provenance', 'models',
            'configuration', 'episode_outcome', 'request', 'diagnosis'}))
        summary['evidence_limitations'].clear()
        self.assertIn('causal_benefit_unverified',
                      self.query([ref])['selected'][0]['evidence_limitations'])
        with self.assertRaises(TraceError):
            self.query([ref, dict(ref, sha256='0' * 64)], max_entries=0)
        foreign = self.retain('foreign', progress={'stage': 'other'})
        self.assertEqual(self.query([foreign])['excluded'][0]['record_id'], foreign['record_id'])
        (self.root / 'foreign/context.json').unlink()
        with self.assertRaises(TraceError):
            self.query([ref, foreign], max_entries=0)

    def test_recovery_summary_preserves_abort_reason_from_sealed_replay(self):
        fixture = recovery_fixtures.InterventionMemoryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        replay = recovery_fixtures.load_recorded_replay(fixture.source / 'trace').run()
        self.assertEqual(replay.stop_reason, fixture.outcome.stop_reason)
        path = fixture.contexts[0]
        context = json.loads(path.read_text())
        context['progress_context'] = fixtures.PROGRESS
        path.write_text(json.dumps(context))
        ref = self.store.append(**(fixture.arguments | {
            'context': recovery_fixtures.pinned(path)}))
        record = self.store.read(ref['record_id'], expected_sha256=ref['sha256'])
        result = self.query([ref], robot_capabilities=record['robot_capabilities'],
            compatibility=record['models'] | {'settings': record['configuration']['settings']})
        summary = result['selected'][0]
        self.assertEqual(summary['intervention_kind'], 'recovery')
        self.assertEqual(summary['local_outcome'], record['local_outcome'])
        self.assertEqual(summary['evidence_limitations'], record['evidence_limitations'])


if __name__ == '__main__':
    unittest.main()
