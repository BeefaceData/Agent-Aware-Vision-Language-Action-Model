"""Public protocol freeze, loading, and evidence integrity contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation_protocol import EvaluationProtocol, ProtocolError


def member(task, state, digest=None):
    return {'suite': 'libero_10', 'task_id': task, 'state_id': state,
            'selected_sha256': digest or f'{task * 10 + state:064x}',
            'digest_encoding': 'little-endian-float64-vector-v1'}


def declaration():
    return {
        'conditions': {'baseline': {'supervisor': False},
                       'no_memory': {'supervisor': True, 'memory': 'empty'},
                       'fixed_memory': {'supervisor': True, 'memory': 'snapshot-pin'}},
        'splits': {'development': [member(0, 0)],
                   'held_out': [member(task, 1) for task in range(10)]},
        'seeds': {'pairing': [17], 'bootstrap': 701},
        'allocation': {'task_0': {'pairs': 1, 'order': ['baseline', 'no_memory',
                                                      'fixed_memory']}},
        'horizons': {'primary_actions': 5, 'recovery_actions_consume_budget': True},
        'analysis': {'estimator': 'equal-task-macro',
                     'interval': 'stratified-paired-cluster-bootstrap-95'},
        'outcomes': {'success': 'evaluator-terminal-success',
                     'post_start_system_failure': 'primary-denominator',
                     'pre_start_exclusion': 'declared-replacement-only'},
    }


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'protocol.json'

    def test_freeze_and_round_trip_with_external_reference(self):
        fields = declaration()
        frozen = EvaluationProtocol.freeze(self.path, **fields)
        reference = frozen.reference
        fields['conditions']['baseline']['supervisor'] = True
        loaded = EvaluationProtocol(self.path, expected_reference=reference)
        self.assertEqual(loaded.read()['conditions']['baseline']['supervisor'], False)
        self.assertEqual(loaded.read()['protocol_id'], reference['protocol_id'])
        self.assertEqual(loaded.reference, reference)
        with self.assertRaises(FileExistsError):
            EvaluationProtocol.freeze(self.path, **declaration())

    def test_changed_content_has_new_identity_and_cannot_replace_freeze(self):
        first = EvaluationProtocol.freeze(self.path, **declaration())
        second_fields = declaration()
        second_fields['horizons']['primary_actions'] = 6
        other = self.path.with_name('other.json')
        second = EvaluationProtocol.freeze(other, **second_fields)
        self.assertNotEqual(first.reference['protocol_id'], second.reference['protocol_id'])
        self.assertNotEqual(first.reference['sha256'], second.reference['sha256'])
        with self.assertRaises(ProtocolError):
            EvaluationProtocol(other, expected_reference=first.reference)

    def test_tampered_manifest_fails_against_external_reference(self):
        frozen = EvaluationProtocol.freeze(self.path, **declaration())
        altered = json.loads(self.path.read_text())
        altered['outcomes']['success'] = 'human-guess'
        self.path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ProtocolError, 'digest mismatch'):
            frozen.read()

    def test_missing_fields_and_bad_values_rejected_before_write(self):
        for field in declaration():
            with self.subTest(field=field):
                fields = declaration()
                fields[field] = {}
                with self.assertRaisesRegex(ProtocolError, field):
                    EvaluationProtocol.freeze(self.path, **fields)
                self.assertFalse(self.path.exists())
        fields = declaration()
        fields['seeds']['bootstrap'] = float('nan')
        with self.assertRaises(ProtocolError):
            EvaluationProtocol.freeze(self.path, **fields)
        self.assertFalse(self.path.exists())

    def test_missing_or_invalid_external_reference_rejected(self):
        frozen = EvaluationProtocol.freeze(self.path, **declaration())
        with self.assertRaises(ProtocolError):
            EvaluationProtocol(self.path, expected_reference={})
        self.path.unlink()
        with self.assertRaises(ProtocolError):
            frozen.read()

    def test_rejects_overlap_by_state_id_even_with_different_digest(self):
        fields = declaration()
        fields['splits']['held_out'][0] = member(0, 0, 'f' * 64)
        with self.assertRaisesRegex(ProtocolError, 'overlapping'):
            EvaluationProtocol.freeze(self.path, **fields)
        self.assertFalse(self.path.exists())

    def test_rejects_overlap_by_state_digest_even_with_different_id(self):
        fields = declaration()
        fields['splits']['held_out'][0] = member(0, 7, member(0, 0)['selected_sha256'])
        with self.assertRaisesRegex(ProtocolError, 'overlapping'):
            EvaluationProtocol.freeze(self.path, **fields)

    def test_rejects_duplicates_inside_each_split(self):
        for split in ('development', 'held_out'):
            fields = declaration()
            fields['splits'][split].append(dict(fields['splits'][split][0]))
            with self.subTest(split=split), self.assertRaisesRegex(ProtocolError, 'duplicate'):
                EvaluationProtocol.freeze(self.path, **fields)

    def test_rejects_absent_task_coverage_and_seed_only_membership(self):
        fields = declaration()
        fields['splits']['held_out'].pop()
        with self.assertRaisesRegex(ProtocolError, 'missing LIBERO-10 tasks: \\[9\\]'):
            EvaluationProtocol.freeze(self.path, **fields)
        fields = declaration()
        fields['splits']['development'] = [{'seed': 17}]
        with self.assertRaisesRegex(ProtocolError, 'task/state identity'):
            EvaluationProtocol.freeze(self.path, **fields)

    def test_disjoint_tasks_may_reuse_state_numbers_and_seeds(self):
        fields = declaration()
        fields['splits']['development'] = [member(9, 0)]
        fields['splits']['held_out'][0] = member(0, 0)
        frozen = EvaluationProtocol.freeze(self.path, **fields)
        self.assertEqual(frozen.read()['splits'], fields['splits'])


if __name__ == '__main__':
    unittest.main()
