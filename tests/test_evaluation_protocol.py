"""Public protocol freeze, loading, and evidence integrity contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation_protocol import EvaluationProtocol, ProtocolError


def declaration():
    return {
        'conditions': {'baseline': {'supervisor': False},
                       'no_memory': {'supervisor': True, 'memory': 'empty'},
                       'fixed_memory': {'supervisor': True, 'memory': 'snapshot-pin'}},
        'splits': {'development': ['state-0'], 'held_out': ['state-1']},
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


if __name__ == '__main__':
    unittest.main()
