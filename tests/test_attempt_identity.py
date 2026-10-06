"""Synthetic attempt provenance, collision protection and complete replay."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from attempt_identity import AttemptIdentityRecorder, reserve_attempt
from episode_harness import run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay
from run_smolvla_episode import main


class AttemptIdentityTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def record(self, name, state='state-a', revision='revision-a'):
        fixture = successful_replay()
        out = reserve_attempt(self.root / name)
        evidence = {
            'task': {'suite': 'synthetic', 'task_id': 0, 'instruction': 'place item'},
            'initial_state': {'applied_sha256': state, 'status': 'verified'},
            'seeds': {'global': 17, 'environment_reset': 17},
            'policy_assets': {'revision': revision},
            'settings': {'device': 'cpu', 'action_horizon': 5},
        }

        def identity():
            self.assertEqual(fixture.environment.actions, [])
            return evidence

        recorder = AttemptIdentityRecorder(out, fixture.config, identity, fixture.recorder)
        trace = TraceRecorder(out / 'replay', fixture.config, recorder)
        act = fixture.policy.act

        def checked_act(packet):
            # Disk evidence is readable before even the first policy proposal.
            row = json.loads((out / 'attempt.json').read_text())
            self.assertEqual(row['episode_id'], packet.episode_id)
            self.assertEqual(row['initial_state'], evidence['initial_state'])
            self.assertEqual(row['episode_config']['seed'], 17)
            self.assertEqual(row['settings'], evidence['settings'])
            return act(packet)

        with patch.object(fixture.policy, 'act', side_effect=checked_act):
            outcome = run_episode(fixture.config, fixture.policy, fixture.environment, trace)
        trace.seal(outcome)
        raw = (out / outcome.artifacts['attempt_manifest']).read_bytes()
        self.assertEqual(outcome.artifacts['attempt_sha256'], sha256(raw).hexdigest())
        self.assertEqual(recorder.reference, dict(outcome.artifacts))
        replayed = load_recorded_replay(out / 'replay').run()
        self.assertEqual((replayed.success, replayed.steps), (True, 2))
        return json.loads(raw)

    def test_pre_action_identity_propagates_to_completed_outcome(self):
        row = self.record('attempt')
        self.assertEqual(row['task']['instruction'], 'place item')
        self.assertEqual(row['policy_assets']['revision'], 'revision-a')

    def test_state_and_revision_changes_are_distinguishable_beyond_episode_id(self):
        rows = [self.record('first'), self.record('state', state='state-b'),
                self.record('model', revision='revision-b')]
        for row in rows:
            row.pop('episode_id')
        self.assertNotEqual(rows[0], rows[1])
        self.assertNotEqual(rows[0], rows[2])

    def test_runner_rejects_existing_destination_before_loading(self):
        out = reserve_attempt(self.root / 'attempt')
        sentinel = out / 'result.json'
        sentinel.write_bytes(b'previous evidence')
        with patch('sys.argv', ['runner', '--output-dir', str(out)]), \
                patch('run_smolvla_episode.capture_environment') as capture:
            with self.assertRaises(FileExistsError):
                main()
        capture.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b'previous evidence')
        self.assertEqual(list(out.iterdir()), [sentinel])

    def test_manifest_collision_blocks_all_policy_actions(self):
        out = reserve_attempt(self.root / 'attempt')
        path = out / 'attempt.json'
        path.write_bytes(b'existing manifest')
        fixture = successful_replay()
        recorder = AttemptIdentityRecorder(out, fixture.config, lambda: {}, fixture.recorder)
        with self.assertRaises(FileExistsError):
            run_episode(fixture.config, fixture.policy, fixture.environment, recorder)
        self.assertEqual(fixture.policy.observations, [])
        self.assertEqual(fixture.environment.actions, [])
        self.assertEqual(path.read_bytes(), b'existing manifest')

    def test_failed_policy_retains_pre_action_identity(self):
        out = reserve_attempt(self.root / 'attempt')
        fixture = successful_replay()
        recorder = AttemptIdentityRecorder(out, fixture.config,
            lambda: {'policy_assets': {'revision': 'fixture'}}, fixture.recorder)
        with patch.object(fixture.policy, 'act', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                run_episode(fixture.config, fixture.policy, fixture.environment, recorder)
        self.assertEqual(fixture.environment.actions, [])
        raw = (out / 'attempt.json').read_bytes()
        self.assertEqual(recorder.reference['attempt_sha256'], sha256(raw).hexdigest())
        self.assertEqual(json.loads(raw)['policy_assets']['revision'], 'fixture')


if __name__ == '__main__':
    unittest.main()
