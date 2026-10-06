"""Offline asset and adapter contracts; these are not model-quality results."""

import hashlib
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from episode_harness import run_episode
from pinned_policy import (ASSET_LOCK, BACKBONE_REPOSITORY, POLICY_REPOSITORY,
                           POLICY_REVISION, resolve_policy_assets)
from policy_adapter import ResetOnResumePolicyAdapter
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay
from run_smolvla_episode import parse_args


class PinnedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = json.loads(ASSET_LOCK.read_text(encoding='utf-8'))
        self.directories = {}
        # Small synthetic bytes replace large weights, with explicit fixture-only
        # digests. Repository/revision identities and all filenames stay intact.
        for name, spec in self.lock['assets'].items():
            directory = self.root / name / 'snapshots' / spec['revision']
            directory.mkdir(parents=True)
            self.directories[spec['repo_id']] = directory
            for filename, entry in spec['files'].items():
                data = f'fixture {name} {filename}'.encode()
                (directory / filename).write_bytes(data)
                entry['size'] = len(data)
                if entry['algorithm'] == 'sha256':
                    entry['digest'] = hashlib.sha256(data).hexdigest()
                else:
                    entry['digest'] = hashlib.sha1(
                        f'blob {len(data)}\0'.encode() + data).hexdigest()
        lock_path = self.root / 'fixture-lock.json'
        lock_path.write_text(json.dumps(self.lock), encoding='utf-8')
        self.lock_patch = patch('pinned_policy.ASSET_LOCK', lock_path)
        self.lock_patch.start()
        self.addCleanup(self.lock_patch.stop)
        self.downloads = []

    def download(self, *, repo_id, revision, allow_patterns):
        self.downloads.append((repo_id, revision, allow_patterns))
        return str(self.directories[repo_id])

    def resolve(self):
        return resolve_policy_assets(snapshot_download=self.download)

    def test_records_verified_identity_and_pins_all_downloads(self):
        assets = self.resolve()
        identity = assets.identity()
        self.assertEqual(identity['verification'], 'file-digests-verified')
        self.assertEqual(identity['assets'], self.lock['assets'])
        self.assertEqual(identity['assets']['policy']['revision'], POLICY_REVISION)
        for repo, revision, filenames in self.downloads:
            spec = next(x for x in self.lock['assets'].values() if x['repo_id'] == repo)
            self.assertEqual(revision, spec['revision'])
            self.assertEqual(set(filenames), set(spec['files']))
        self.assertEqual(len(self.downloads), 2)

    def test_unpinned_repository_or_snapshot_cannot_masquerade_as_baseline(self):
        for repository in ('other/model', str(self.root), POLICY_REPOSITORY + '@main'):
            with self.subTest(repository=repository), self.assertRaises(ValueError):
                resolve_policy_assets(repository, snapshot_download=self.download)
        self.assertEqual(self.downloads, [])
        with self.assertRaisesRegex(ValueError, 'pinned snapshot'):
            resolve_policy_assets(snapshot_download=lambda **kw: self.root)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--policy', 'other/model'])

    def test_unavailable_assets_do_not_fall_back(self):
        download = Mock(side_effect=FileNotFoundError('pinned revision unavailable'))
        with self.assertRaisesRegex(FileNotFoundError, 'unavailable'):
            resolve_policy_assets(snapshot_download=download)
        self.assertEqual(download.call_count, 1)

    def test_missing_and_corrupt_assets_are_rejected_before_loading(self):
        for repo, directory in self.directories.items():
            for filename in next(s['files'] for s in self.lock['assets'].values()
                                 if s['repo_id'] == repo):
                with self.subTest(repo=repo, file=filename):
                    path = directory / filename
                    original = path.read_bytes()
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, 'missing'):
                        self.resolve()
                    path.write_bytes(b'x' * len(original))
                    with self.assertRaisesRegex(ValueError, 'digest-mismatched'):
                        self.resolve()
                    path.write_bytes(original)

    def test_mutation_after_resolution_cannot_reach_model_loader(self):
        assets = self.resolve()
        (assets.policy_directory / 'config.json').write_text('changed')
        config_loader = Mock()
        with self.assertRaises(ValueError):
            assets.load('cpu', config_loader=config_loader)
        config_loader.assert_not_called()

    def test_unexpected_root_asset_cannot_override_verified_files(self):
        (self.directories[BACKBONE_REPOSITORY] / 'chat_template.jinja').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'unexpected files'):
            self.resolve()

    def test_complete_episode_uses_loaded_frozen_policy_and_seals_replay(self):
        assets = self.resolve()
        fixture = successful_replay()
        config = SimpleNamespace(vlm_model_name=BACKBONE_REPOSITORY)
        model = SimpleNamespace(config=config, reset=fixture.policy.reset,
                                select_action=fixture.policy.act)
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        model.requires_grad_ = Mock()
        config_loader = Mock(return_value=config)
        policy_loader = Mock(return_value=model)
        processors = Mock(return_value=(lambda x: x, lambda x: x))
        policy, pre, post = assets.load('cpu', config_loader=config_loader,
                                       policy_loader=policy_loader,
                                       processor_factory=processors)
        config_loader.assert_called_once_with(str(assets.policy_directory), local_files_only=True)
        policy_loader.assert_called_once_with(str(assets.policy_directory), config=config,
                                               local_files_only=True, strict=True)
        self.assertEqual(config.vlm_model_name, str(assets.backbone_directory))
        self.assertEqual(processors.call_args.args[1], str(assets.policy_directory))
        self.assertEqual(processors.call_args.kwargs['preprocessor_overrides'], {
            'device_processor': {'device': 'cpu'},
            'tokenizer_processor': {'tokenizer_name': str(assets.backbone_directory)}})
        model.requires_grad_.assert_called_once_with(False)
        adapter = ResetOnResumePolicyAdapter(policy.reset,
            lambda packet: post(policy.select_action(pre(packet))))
        recorder = TraceRecorder(self.root / 'episode', fixture.config, fixture.recorder)
        outcome = run_episode(fixture.config, adapter, fixture.environment, recorder)
        recorder.seal(outcome)
        replay = load_recorded_replay(self.root / 'episode')
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(assets.identity()['assets'], self.lock['assets'])
        replay_outcome = replay.run()
        self.assertTrue(replay_outcome.success)
        self.assertEqual(replay_outcome.steps, 2)

    def test_incompatible_model_load_failure_is_not_silenced(self):
        assets = self.resolve()
        processors = Mock()
        with self.assertRaisesRegex(RuntimeError, 'missing weight'):
            assets.load('cpu', config_loader=Mock(return_value=SimpleNamespace(
                vlm_model_name=BACKBONE_REPOSITORY)),
                policy_loader=Mock(side_effect=RuntimeError('missing weight')),
                processor_factory=processors)
        processors.assert_not_called()


if __name__ == '__main__':
    unittest.main()
