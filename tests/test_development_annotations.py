"""Public offline annotation workflow and provenance checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from development_annotations import (
    AnnotationError, annotation_template, load_annotation, save_annotation,
    validate_annotation,
)


TRACE = Path(__file__).parent / 'fixtures' / 'recorded_episode'
SAMPLE = Path(__file__).parent / 'fixtures' / 'development_annotation.json'


class DevelopmentAnnotationTests(unittest.TestCase):
    def test_sample_roundtrip_preserves_review_and_trace(self):
        original = {p.name: p.read_bytes() for p in TRACE.iterdir() if p.is_file()}
        annotation = load_annotation(TRACE, SAMPLE)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            save_annotation(TRACE, path, annotation)
            self.assertEqual(load_annotation(TRACE, path), annotation)
            with self.assertRaises(FileExistsError):
                save_annotation(TRACE, path, annotation)
        self.assertEqual(original, {p.name: p.read_bytes() for p in TRACE.iterdir()
                                    if p.is_file()})

    def test_all_example_types_and_unknown_onsets(self):
        annotation = load_annotation(TRACE, SAMPLE)
        for example, family in [('failed', 'missed_grasp'), ('successful', 'progress'),
                                ('paused', 'pause'), ('occluded', 'occlusion'),
                                ('productive', 'progress')]:
            with self.subTest(example=example):
                review = deepcopy(annotation)
                review['events'][0].update(example=example, family=family)
                self.assertEqual(validate_annotation(TRACE, review), review)
        annotation['events'][0].update(onset=None, uncertain=True, family='unknown')
        self.assertEqual(validate_annotation(TRACE, annotation), annotation)

    def test_invalid_reviews_are_rejected(self):
        annotation = load_annotation(TRACE, SAMPLE)
        for patch in ({'split': 'held_out'}, {'status': 'draft'}, {'reviewer': ''},
                      {'source_episode_id': 'foreign'}, {'trace_sha256': '0' * 64},
                      {'reviewed_at': '2026-10-03'}, {'events': []}, {'version': True}):
            with self.subTest(patch=patch), self.assertRaises(AnnotationError):
                validate_annotation(TRACE, dict(annotation, **patch))
        for patch in ({'onset': [2, 1]}, {'onset': [-1, 1]}, {'onset': [0, 99]},
                      {'onset': [True, 1]}, {'onset': None, 'uncertain': False},
                      {'family': 'unknown', 'uncertain': False}, {'uncertain': 'yes'},
                      {'evidence': []}, {'evidence': [{'sequence': 99, 'description': 'x'}]},
                      {'family': 'invented'}, {'notes': ''}):
            review = deepcopy(annotation)
            review['events'][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(AnnotationError):
                validate_annotation(TRACE, review)
        annotation['events'].append(deepcopy(annotation['events'][0]))
        with self.assertRaises(AnnotationError):
            validate_annotation(TRACE, annotation)

    def test_template_cli_and_validation_cli(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'draft.json'
            command = [sys.executable, 'development_annotations.py']
            subprocess.run(command + ['template', str(TRACE), str(path), '--reviewer',
                                      'fixture author', '--provenance', 'synthetic'],
                           check=True, capture_output=True)
            draft = json.loads(path.read_text())
            self.assertEqual(draft['status'], 'draft')
            result = subprocess.run(command + ['validate', str(TRACE), str(path)],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            result = subprocess.run(command + ['validate', str(TRACE), str(SAMPLE)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['status'], 'validated')

    def test_duplicate_fields_and_changed_trace_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            path.write_text('{"version": 1, "version": 1}')
            with self.assertRaises(AnnotationError):
                load_annotation(TRACE, path)
            bundle = Path(directory) / 'bundle'
            bundle.mkdir()
            for source in TRACE.iterdir():
                if source.is_file():
                    (bundle / source.name).write_bytes(source.read_bytes())
            manifest = bundle / 'manifest.json'
            value = json.loads(manifest.read_text())
            value['config']['seed'] += 1
            manifest.write_text(json.dumps(value))
            with self.assertRaises(AnnotationError):
                load_annotation(bundle, SAMPLE)

    def test_template_is_not_a_review(self):
        draft = annotation_template(TRACE, reviewer='reviewer', provenance='synthetic')
        with self.assertRaises(AnnotationError):
            validate_annotation(TRACE, draft)


if __name__ == '__main__':
    unittest.main()
