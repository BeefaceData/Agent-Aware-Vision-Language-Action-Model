"""Public export behavior against a complete sealed synthetic episode."""

from hashlib import sha256
import json
import subprocess
import sys
import unittest

from artifact_bundle import seal_artifact_bundle, verify_artifact_bundle
from recorded_replay import TraceError
from shareable_report import export_shareable_report
import test_artifact_bundle


class ShareableReportTests(unittest.TestCase):
    def setUp(self):
        test_artifact_bundle.ArtifactBundleTests.setUp(self)
        # A private raw document and credential can also occur inside inventoried
        # environment metadata; neither the default report nor seal is copied.
        (self.root / 'environment.json').write_text(json.dumps({
            'private_document': 'PRIVATE-DOCUMENT-SENTINEL',
            'credential': 'SECRET-CREDENTIAL-SENTINEL'}))
        (self.root / 'private-client.txt').write_text('UNLISTED-PRIVATE-SENTINEL')
        (self.root / 'bundle.json').unlink()
        seal_artifact_bundle(self.root)
        self.destination = self.root.with_name('export')

    def manifest(self):
        return json.loads((self.destination / 'export.json').read_text())

    def test_default_cli_excludes_raw_data_and_preserves_omission_provenance(self):
        subprocess.run([sys.executable, 'shareable_report.py', str(self.root),
                        str(self.destination)], check=True, capture_output=True)
        self.assertEqual({p.name for p in self.destination.iterdir()}, {'report.html', 'export.json'})
        output = ''.join(p.read_text() for p in self.destination.iterdir())
        for secret in ('PRIVATE-DOCUMENT-SENTINEL', 'SECRET-CREDENTIAL-SENTINEL',
                       'UNLISTED-PRIVATE-SENTINEL', 'synthetic-frozen-revision'):
            self.assertNotIn(secret, output)
        manifest = self.manifest()
        source = json.loads((self.root / 'bundle.json').read_text())
        self.assertEqual(manifest['source_bundle_sha256'],
                         sha256((self.root / 'bundle.json').read_bytes()).hexdigest())
        self.assertEqual({e['source_artifact']: e['sha256'] for e in manifest['artifacts']},
                         source['files'])
        self.assertTrue(all(not e['included'] and e['export_path'] is None
                            for e in manifest['artifacts']))
        self.assertEqual(manifest['summary'], {'success': True, 'steps': 1})
        self.assertTrue(verify_artifact_bundle(self.root).run().success)

    def test_private_selection_needs_permission_and_credentials_stay_excluded(self):
        export_shareable_report(self.root, self.destination,
            selected=['episode.mp4', 'environment.json'],
            classifications={'episode.mp4': 'private', 'environment.json': 'credentials'},
            permit_private=['environment.json'])
        entries = {e['source_artifact']: e for e in self.manifest()['artifacts']}
        self.assertEqual(entries['episode.mp4']['reason'], 'private permission required')
        self.assertEqual(entries['environment.json']['reason'], 'credentials excluded')
        self.assertFalse(any(e['included'] for e in entries.values()))

    def test_permitted_private_and_selected_public_are_copied_exactly(self):
        export_shareable_report(self.root, self.destination,
            selected=['episode.mp4', 'policy-assets.json'], permit_private=['episode.mp4'],
            classifications={'episode.mp4': 'private', 'policy-assets.json': 'public'})
        entries = [e for e in self.manifest()['artifacts'] if e['included']]
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual((self.destination / entry['export_path']).read_bytes(),
                             (self.root / entry['source_artifact']).read_bytes())
            self.assertIn(entry['export_path'], (self.destination / 'report.html').read_text())

    def test_bad_selection_and_permissions_fail_without_creating_output(self):
        for options in ({'selected': ['../private-client.txt']},
                        {'selected': ['private-client.txt']},
                        {'permit_private': ['episode.mp4']},
                        {'classifications': {'episode.mp4': 'unknown'}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                export_shareable_report(self.root, self.destination, **options)
            self.assertFalse(self.destination.exists())

    def test_corrupt_source_fails_even_when_nothing_selected(self):
        (self.root / 'episode.mp4').write_bytes(b'changed')
        with self.assertRaises(TraceError):
            export_shareable_report(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_destination_is_preserved(self):
        self.destination.mkdir()
        marker = self.destination / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            export_shareable_report(self.root, self.destination)
        self.assertEqual(marker.read_text(), 'keep')
