"""Offline runtime inspection contracts; no inference or simulator startup."""

import hashlib
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baseline_environment import capture_environment
from run_smolvla_episode import main


class BaselineEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'environment.json'
        source = self.root / 'controller.py'
        source.write_bytes(b'fixture controller source')
        self.controller = SimpleNamespace(
            version='1.4.0', files=['robosuite/controllers/osc.py'],
            locate_file=lambda name: source)
        self.torch = SimpleNamespace(
            version=SimpleNamespace(cuda='12.6'),
            cuda=SimpleNamespace(is_available=lambda: True,
                                 current_device=lambda: 1,
                                 get_device_name=lambda index: 'fixture GPU'))
        patches = [
            patch('baseline_environment.metadata.distributions', return_value=[
                SimpleNamespace(metadata={'Name': 'torch', 'Home-page': 'secret'},
                                version='9.0-fixture'),
                SimpleNamespace(metadata={'Name': 'lerobot'}, version='0.4.3')]),
            patch('baseline_environment.metadata.version', return_value='0.4.3'),
            patch('baseline_environment.metadata.distribution', return_value=self.controller),
            patch.dict('sys.modules', {'torch': self.torch}),
            patch.dict('os.environ', {'MUJOCO_GL': 'osmesa', 'HF_TOKEN': 'secret-token'}),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_round_trip_runtime_identity_separates_history_and_omits_secrets(self):
        record = capture_environment(self.path, 'cuda')
        self.assertEqual(json.loads(self.path.read_text()), record)
        self.assertEqual(record['preflight']['status'], 'passed')
        self.assertEqual(record['device']['resolved'], 'cuda:1')
        self.assertEqual(record['render_backend'], 'osmesa')
        self.assertTrue(record['python']['version'])
        self.assertEqual(record['historical_environment']['torch'], '2.7.1')
        self.assertIn({'name': 'torch', 'version': '9.0-fixture'}, record['dependencies'])
        self.assertEqual(record['controller']['source_sha256'], {
            'robosuite/controllers/osc.py': hashlib.sha256(b'fixture controller source').hexdigest()})
        self.assertNotIn('secret', self.path.read_text())
        self.assertNotIn(str(self.root), self.path.read_text())

    def test_cpu_does_not_query_gpu(self):
        self.torch.cuda = None
        self.assertEqual(capture_environment(self.path, 'cpu')['device']['resolved'], 'cpu')

    def test_unsupported_version_retains_actionable_failure_before_torch(self):
        with patch('baseline_environment.metadata.version', return_value='0.5.0'):
            with self.assertRaisesRegex(RuntimeError, 'lerobot==0.4.3'):
                capture_environment(self.path, 'cuda')
        record = json.loads(self.path.read_text())
        self.assertEqual(record['preflight']['status'], 'failed')
        self.assertIn('0.5.0', record['preflight']['diagnostic'])
        self.assertIsNone(record['device']['resolved'])

    def test_missing_controller_sources_fails_explicitly(self):
        self.controller.files = None
        with self.assertRaisesRegex(RuntimeError, 'controller files'):
            capture_environment(self.path, 'cpu')
        self.assertEqual(json.loads(self.path.read_text())['controller']['status'], 'unavailable')

    def test_missing_lerobot_has_install_guidance(self):
        with patch('baseline_environment.metadata.version', side_effect=PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, 'lerobot==0.4.3'):
                capture_environment(self.path, 'cpu')
        self.assertIn('install', json.loads(self.path.read_text())['preflight']['diagnostic'])

    def test_unexpected_failure_does_not_export_exception_secrets(self):
        with patch('baseline_environment.metadata.distribution',
                   side_effect=OSError('private/path/secret-token')):
            with self.assertRaises(OSError):
                capture_environment(self.path, 'cpu')
        record = json.loads(self.path.read_text())
        self.assertEqual(record['preflight']['error_type'], 'OSError')
        self.assertNotIn('secret', self.path.read_text())

    def test_cuda_unavailable_retains_unresolved_device(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(RuntimeError, '--device cpu'):
            capture_environment(self.path, 'cuda')
        record = json.loads(self.path.read_text())
        self.assertIsNone(record['device']['resolved'])
        self.assertEqual(record['preflight']['status'], 'failed')

    def test_existing_evidence_is_never_overwritten(self):
        capture_environment(self.path, 'cpu')
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            capture_environment(self.path, 'cuda')
        self.assertEqual(self.path.read_bytes(), original)

    def test_runner_persists_failure_before_policy_or_simulator_import(self):
        output = self.root / 'run'
        with patch('sys.argv', ['runner', '--output-dir', str(output)]), \
                patch('baseline_environment.metadata.version', return_value='unsupported'), \
                patch('run_smolvla_episode.resolve_policy_assets') as assets:
            with self.assertRaisesRegex(RuntimeError, 'lerobot==0.4.3'):
                main()
        assets.assert_not_called()
        self.assertEqual(json.loads((output / 'environment.json').read_text())[
            'preflight']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
