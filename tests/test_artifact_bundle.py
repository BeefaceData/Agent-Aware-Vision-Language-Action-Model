"""Public final-bundle verification with a complete synthetic camera episode."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from artifact_bundle import seal_artifact_bundle, verify_artifact_bundle
from attempt_identity import AttemptIdentityRecorder
from camera_evidence import CameraEvidenceRecorder
from episode_harness import EpisodeConfig, run_episode
from recorded_replay import TraceError, TraceRecorder
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class ArtifactBundleTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'episode'
        self.root.mkdir()
        root = self.root

        class Writer:
            def __init__(self, path):
                self.stream = path.open('wb')

            def append_data(self, frame):
                self.stream.write(json.dumps(frame).encode() + b'\n')

            def close(self):
                self.stream.close()

        cameras = CameraEvidenceRecorder(root / 'episode.mp4', root / 'episode_wrist.mp4',
            root / 'frames.jsonl', Writer, lambda frame: frame)

        class Recorder(ReplayRecorder):
            def begin(self, packet):
                super().begin(packet)
                cameras.record(packet)

            def record_step(self, step, source, action, result, ingestion):
                super().record_step(step, source, action, result, ingestion)
                cameras.record(result.observation)

            def finish(self):
                cameras.close()
                (root / 'steps.jsonl').write_text('{}\n')
                return cameras.artifacts

        initial = {'pixels': {'image': [[[0]]], 'image2': [[[1]]]}}
        terminal = {'pixels': {'image': [[[2]]]}}
        config = EpisodeConfig(17, 2)
        assets = {'revision': 'synthetic-frozen-revision'}
        identity = AttemptIdentityRecorder(root, config,
            lambda: {'policy_assets': assets, 'settings': {'device': 'synthetic'}}, Recorder())
        trace = TraceRecorder(root / 'replay', config, identity)
        outcome = run_episode(config, ReplayPolicy(((initial, [0.1]),)),
            ReplayEnvironment(17, initial, (([0.1], ReplayStep(terminal, 1., True, True, False)),)), trace)
        trace.seal(outcome)
        (root / 'environment.json').write_text('{}')
        (root / 'policy-assets.json').write_text(json.dumps(assets))
        (root / 'result.json').write_text(json.dumps({
            'status': 'completed', 'artifact_status': 'completed',
            'episode_id': outcome.episode_id, 'policy_assets': assets, **identity.reference}))
        seal_artifact_bundle(root)

    def test_intact_moved_bundle_verifies_and_replays_via_cli(self):
        moved = self.root.with_name('moved')
        shutil.move(self.root, moved)
        outcome = verify_artifact_bundle(moved).run()
        self.assertEqual((outcome.success, outcome.steps), (True, 1))
        result = subprocess.run([sys.executable, 'artifact_bundle.py', str(moved), '--replay'],
                                capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)['verified'])

    def test_changed_trace_is_rejected(self):
        with (self.root / 'replay/decisions.jsonl').open('ab') as stream:
            stream.write(b'\n')
        with self.assertRaisesRegex(TraceError, 'checksum'):
            verify_artifact_bundle(self.root)

    def test_missing_camera_artifact_is_rejected(self):
        (self.root / 'episode_wrist.mp4').unlink()
        with self.assertRaises(TraceError):
            verify_artifact_bundle(self.root)

    def test_removed_frame_index_entry_is_rejected(self):
        path = self.root / 'frames.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
        with self.assertRaises(TraceError):
            verify_artifact_bundle(self.root)
        (self.root / 'bundle.json').unlink()
        with self.assertRaisesRegex(TraceError, 'frame index'):
            seal_artifact_bundle(self.root)

    def test_unsupported_version_and_incomplete_inventory_are_rejected(self):
        path = self.root / 'bundle.json'
        original = json.loads(path.read_text())
        for version in (2, True):
            path.write_text(json.dumps({**original, 'version': version}))
            with self.assertRaisesRegex(TraceError, 'version'):
                verify_artifact_bundle(self.root)
        del original['files']['attempt.json']
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(TraceError, 'inventory'):
            verify_artifact_bundle(self.root)

    def test_model_identity_and_existing_seal_cannot_be_replaced(self):
        with self.assertRaises(FileExistsError):
            seal_artifact_bundle(self.root)
        (self.root / 'policy-assets.json').write_text('{"revision":"changed"}')
        with self.assertRaises(TraceError):
            verify_artifact_bundle(self.root)
        with self.assertRaisesRegex(TraceError, 'model identity'):
            seal_artifact_bundle(self.root)


if __name__ == '__main__':
    unittest.main()
