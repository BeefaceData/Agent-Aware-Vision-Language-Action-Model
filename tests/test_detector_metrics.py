"""Controlled public metric fixtures; synthetic labels prove arithmetic only."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from development_annotations import load_annotation
from detector_metrics import detector_report


TRACE = Path(__file__).parent / 'fixtures/recorded_episode'
SAMPLE = Path(__file__).parent / 'fixtures/development_annotation.json'


def inputs():
    review = load_annotation(TRACE, SAMPLE)
    seed = review['events'][0]
    review['events'] = [
        dict(seed, id='miss', family='missed_grasp', uncertain=False, onset=[1, 2]),
        dict(seed, id='stall', family='stall', uncertain=False, onset=[2, 2]),
        dict(seed, id='productive', family='progress', example='productive',
             uncertain=False, onset=[0, 0]),
        dict(seed, id='uncertain', onset=None),
    ]
    evaluation = {
        'version': 1, 'source_episode_id': review['source_episode_id'],
        'trace_sha256': review['trace_sha256'], 'configuration_id': 'synthetic-v1',
        'provenance': 'Controlled metric fixture; not live detector output',
        'max_delay_sequences': 0,
        'events': [
            {'id': 'early', 'kind': 'detection', 'family': 'missed_grasp', 'sequence': 0},
            {'id': 'match', 'kind': 'detection', 'family': 'missed_grasp', 'sequence': 1},
            {'id': 'duplicate', 'kind': 'detection', 'family': 'missed_grasp', 'sequence': 2},
            {'id': 'wrong-family', 'kind': 'detection', 'family': 'lost_grasp', 'sequence': 2},
            {'id': 'unknown', 'kind': 'unknown', 'family': None, 'sequence': 0},
            {'id': 'abstain', 'kind': 'abstention', 'family': None, 'sequence': 0},
            {'id': 'false', 'kind': 'intervention', 'family': 'stall', 'sequence': 0},
            {'id': 'outside', 'kind': 'intervention', 'family': 'stall', 'sequence': 2},
        ],
        'productive_intervals': [{'annotation_id': 'productive', 'start': 0, 'end': 0,
                                  'reviewer': 'synthetic author', 'notes': 'fixture duration'}],
    }
    return review, evaluation


class DetectorMetricsTests(unittest.TestCase):
    def test_counts_delays_unmatched_and_productive_interventions(self):
        review, evaluation = inputs()
        before = deepcopy((review, evaluation))
        report = detector_report(TRACE, review, evaluation)
        missed = report['families']['missed_grasp']
        self.assertEqual(missed['precision'], 1 / 3)
        self.assertEqual(missed['recall'], 1)
        self.assertEqual(missed['matches'], [{'detection_id': 'match', 'annotation_id': 'miss',
                                              'delay_sequences': [-1, 0]}])
        self.assertEqual(missed['unmatched_detections'], ['early', 'duplicate'])
        self.assertEqual(report['families']['lost_grasp']['precision'], 0)
        self.assertIsNone(report['families']['lost_grasp']['recall'])
        stall = report['families']['stall']
        self.assertIsNone(stall['precision'])
        self.assertEqual(stall['recall'], 0)
        self.assertEqual(stall['unmatched_annotations'], ['stall'])
        self.assertEqual(stall['false_interventions'], ['false'])
        self.assertEqual(stall['interventions_outside_productive_coverage'], ['outside'])
        self.assertEqual(report['unknown_count'], 1)
        self.assertEqual(report['abstention_count'], 1)
        self.assertEqual(report['uncertain_annotations'], ['uncertain'])
        self.assertEqual(report['false_intervention_count'], 1)
        self.assertEqual((review, evaluation), before)

    def test_tolerance_and_inclusive_delay_boundary(self):
        review, evaluation = inputs()
        review['events'] = [dict(review['events'][0], onset=[0, 0])]
        evaluation['productive_intervals'] = []
        evaluation['events'] = [{'id': 'late', 'kind': 'detection',
                                 'family': 'missed_grasp', 'sequence': 2}]
        for tolerance, matched in [(0, 0), (1, 0), (2, 1)]:
            evaluation['max_delay_sequences'] = tolerance
            result = detector_report(TRACE, review, evaluation)['families']['missed_grasp']
            self.assertEqual(result['matched_events'], matched)
            if matched:
                self.assertEqual(result['matches'][0]['delay_sequences'], [2, 2])

    def test_one_to_one_overlapping_labels_and_order_independence(self):
        review, evaluation = inputs()
        review['events'] = [dict(review['events'][0], id='first', onset=[0, 1]),
                            dict(review['events'][0], id='second', onset=[1, 2])]
        evaluation['productive_intervals'] = []
        evaluation['events'] = [event for event in evaluation['events']
                                if event['id'] in ('match', 'duplicate')]
        report = detector_report(TRACE, review, evaluation)
        self.assertEqual([item['annotation_id'] for item in report['families']['missed_grasp']['matches']],
                         ['first', 'second'])
        evaluation['events'].reverse()
        review['events'].reverse()
        self.assertEqual(report['families'], detector_report(TRACE, review, evaluation)['families'])

    def test_absent_outputs_and_coverage_are_explicit(self):
        review, evaluation = inputs()
        evaluation['events'] = []
        evaluation['productive_intervals'] = []
        report = detector_report(TRACE, review, evaluation)
        self.assertEqual(report['productive_sequences'], 0)
        self.assertIsNone(report['families']['missed_grasp']['precision'])
        self.assertEqual(report['families']['missed_grasp']['recall'], 0)

    def test_invalid_identity_events_and_conflicting_coverage(self):
        review, evaluation = inputs()
        for patch in ({'source_episode_id': 'foreign'}, {'trace_sha256': 'bad'},
                      {'max_delay_sequences': True}, {'max_delay_sequences': -1},
                      {'configuration_id': ''}, {'version': True}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                detector_report(TRACE, review, dict(evaluation, **patch))
        for patch in ({'sequence': 3}, {'sequence': True}, {'family': 'invented'},
                      {'kind': 'proposal'}, {'id': ''}):
            changed = deepcopy(evaluation)
            changed['events'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                detector_report(TRACE, review, changed)
        changed = deepcopy(evaluation)
        changed['events'].append(changed['events'][0])
        with self.assertRaises(ValueError):
            detector_report(TRACE, review, changed)
        for patch in ({'end': 2}, {'annotation_id': 'uncertain'}, {'reviewer': ''},
                      {'start': True}, {'end': 99}):
            changed = deepcopy(evaluation)
            changed['productive_intervals'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                detector_report(TRACE, review, changed)

    def test_cli_and_source_artifact_preservation(self):
        original = {path.name: path.read_bytes() for path in TRACE.iterdir() if path.is_file()}
        review, evaluation = inputs()
        with TemporaryDirectory() as tmp:
            annotation = Path(tmp) / 'review.json'
            outputs = Path(tmp) / 'outputs.json'
            annotation.write_text(json.dumps(review), encoding='utf-8')
            outputs.write_text(json.dumps(evaluation), encoding='utf-8')
            result = subprocess.run([sys.executable, 'detector_metrics.py', str(TRACE),
                                     str(annotation), str(outputs)], check=True,
                                    capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), detector_report(TRACE, review, evaluation))
            outputs.write_text('{"version": 1, "version": 1}', encoding='utf-8')
            result = subprocess.run([sys.executable, 'detector_metrics.py', str(TRACE),
                                     str(annotation), str(outputs)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual(original, {path.name: path.read_bytes() for path in TRACE.iterdir() if path.is_file()})


if __name__ == '__main__':
    unittest.main()
