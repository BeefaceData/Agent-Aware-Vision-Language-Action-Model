"""Public deterministic ranking contracts over pinned complete replay episodes."""

from dataclasses import asdict
from itertools import permutations
import unittest

from intervention_memory import InterventionMemory
from recorded_replay import TraceError
import test_memory_compatibility as fixtures


class MemoryRankingTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain

    def query(self, references, **changes):
        query = dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
                     progress_context=fixtures.PROGRESS, failure_category='stall')
        return self.store.rank_candidates(references, **(query | changes))

    def test_failure_match_precedes_success_with_other_diagnosis(self):
        success = self.retain('success', success=True, category='suspected_missed_grasp')
        failed = self.retain('failed', category='stall')
        uncertain = self.retain('uncertain', category='stall', truncated=True)
        expected = sorted([failed['record_id'], uncertain['record_id']]) + [success['record_id']]
        for references in permutations([success, failed, uncertain]):
            result = self.query(references)
            self.assertEqual([r['record_id'] for r in result['candidates']], expected)
        self.assertEqual({r['task_outcome']['status'] for r in result['candidates']},
                         {'success', 'failure', 'unknown'})
        self.assertTrue(all('causal_benefit_unverified' in r['evidence_limitations']
                            for r in result['candidates']))
        self.assertEqual(result['ranking'], {
            'policy': 'exact-context-failure-v1', 'failure_category': 'stall',
            'progress_context': fixtures.PROGRESS,
            'compatibility': 'exact-task-progress-control',
            'order': ['failure_category_match_desc', 'record_id_asc'],
            'outcome_preference': 'none'})

    def test_ties_ignore_outcomes_input_order_and_duplicate_references(self):
        refs = [self.retain('failed', category='stall'),
                self.retain('success', category='stall', success=True),
                self.retain('uncertain', category='stall', truncated=True)]
        expected = sorted(ref['record_id'] for ref in refs)
        first = self.query(refs)
        self.assertEqual([r['record_id'] for r in first['candidates']], expected)
        self.store = InterventionMemory(self.root / 'memory')
        self.assertEqual(self.query(list(reversed(refs)) + refs), first)
        first['candidates'][0]['diagnosis']['category'] = 'invented'
        first['ranking']['progress_context']['stage'] = 'invented'
        self.assertEqual(self.query(refs)['ranking']['progress_context'], fixtures.PROGRESS)
        self.assertTrue(all(r['diagnosis']['category'] == 'stall'
                            for r in self.query(refs)['candidates']))

    def test_unknown_is_not_a_wildcard_and_different_failure_queries_reorder(self):
        unknown = self.retain('unknown')
        stall = self.retain('stall', category='stall')
        refs = [unknown, stall]
        self.assertEqual(self.query(refs)['candidates'][0]['record_id'], stall['record_id'])
        result = self.query(refs, failure_category='unknown')
        self.assertEqual(result['candidates'][0]['record_id'], unknown['record_id'])

    def test_exact_progress_and_task_gates_apply_before_failure_ranking(self):
        grasp = self.retain('grasp')
        release_progress = fixtures.PROGRESS | {'stage': 'release'}
        release = self.retain('release', progress=release_progress, category='stall')
        foreign = self.retain('foreign', task=fixtures.TASK | {'instruction': 'fold towel'},
                              category='stall')
        missing = self.retain('missing', progress=None, category='stall')
        paired = self.retain('paired', fixtures.two_arm(), category='stall')
        refs = [release, foreign, grasp, missing, paired]
        result = self.query(refs)
        self.assertEqual([r['record_id'] for r in result['candidates']], [grasp['record_id']])
        self.assertEqual(result, self.query(reversed(refs)))
        reasons = {r['record_id']: r['reasons'] for r in result['excluded']}
        self.assertEqual(reasons[release['record_id']], ['progress_context_mismatch'])
        self.assertEqual(reasons[missing['record_id']], ['progress_context_missing'])
        self.assertEqual(reasons[foreign['record_id']], ['task_mismatch'])
        self.assertIn('component_count_mismatch', reasons[paired['record_id']])
        result = self.query(refs, progress_context=release_progress)
        self.assertEqual([r['record_id'] for r in result['candidates']], [release['record_id']])

    def test_invalid_queries_and_corrupt_excluded_or_duplicate_evidence_fail_closed(self):
        for value in (None, '', 'made_up', True, [], {}):
            with self.subTest(value=value), self.assertRaises(TraceError):
                self.query([], failure_category=value)
        with self.assertRaises(TraceError):
            self.query([], progress_context={})
        self.assertEqual(self.query([])['candidates'], [])
        ref = self.retain('retained')
        with self.assertRaises(TraceError):
            self.query([ref, dict(ref, sha256='0' * 64)])
        foreign = self.retain('foreign', progress={'stage': 'foreign'})
        (self.root / 'foreign/context.json').unlink()
        with self.assertRaises(TraceError):
            self.query([ref, foreign])


if __name__ == '__main__':
    unittest.main()
