"""Frozen selection checks and an offline complete-episode provenance replay."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from attempt_identity import AttemptIdentityRecorder
from episode_harness import EpisodeConfig, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_identity import FrozenSupervisorManifest
from supervisor_provider import BoundedSupervisorProvider
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings, supervisor_identity
from test_supervisor_vlm import PNG, pass_message, proposal, raw


class SupervisorIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = VlmSettings('synthetic-version-1')
        self.manifest = FrozenSupervisorManifest.freeze(
            self.root / 'supervisor.json', supervisor_identity(self.settings))

    def test_records_exact_prompt_settings_and_version_limitation(self):
        requests = []

        def send(payload, deadline, cancellation):
            requests.append(payload)
            return pass_message(payload, deadline, cancellation)

        adapter = ChronologicalVlmAdapter(self.settings, lambda frame: PNG, send,
                                          frozen_manifest=self.manifest)
        adapter(proposal(), monotonic() + 1, Event())
        row = json.loads(self.manifest.path.read_text())
        self.assertEqual(row['provider_api_version'], '2023-06-01')
        self.assertEqual(row['model'], requests[0]['model'])
        self.assertEqual(row['prompt_template_sha256'],
                         sha256(requests[0]['system'].encode()).hexdigest())
        self.assertEqual(row['generation_settings']['max_tokens'], requests[0]['max_tokens'])
        self.assertEqual(row['generation_settings']['temperature'], 'provider_default')
        self.assertFalse(row['immutable_model_version'])
        self.assertIn('may drift', row['version_limitation'])

    def test_changed_selection_rejected_before_construction_sends(self):
        for settings in (replace(self.settings, model='other'),
                         replace(self.settings, max_tokens=99),
                         replace(self.settings, max_observations=2),
                         replace(self.settings, immutable_model_version=True,
                                 version_limitation='')):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'selection'):
                ChronologicalVlmAdapter(settings, lambda frame: PNG,
                                        frozen_manifest=self.manifest)
        for symbol in ('SUPERVISOR_PROMPT', 'PROVIDER_API_VERSION'):
            with patch('supervisor_vlm.' + symbol, 'changed'):
                with self.assertRaisesRegex(ValueError, 'selection'):
                    ChronologicalVlmAdapter(self.settings, lambda frame: PNG,
                                            frozen_manifest=self.manifest)

    def test_runtime_drift_cannot_reach_transport(self):
        sent = []
        adapter = ChronologicalVlmAdapter(self.settings, lambda frame: PNG,
                                          lambda *args: sent.append(args),
                                          frozen_manifest=self.manifest)
        adapter.settings = replace(self.settings, model='changed')
        with self.assertRaisesRegex(ValueError, 'selection'):
            adapter(proposal(), monotonic() + 1, Event())
        self.assertEqual(sent, [])

    def test_retained_file_cannot_be_overwritten_or_replaced(self):
        original = self.manifest.path.read_bytes()
        with self.assertRaises(FileExistsError):
            FrozenSupervisorManifest.freeze(self.manifest.path, {'model': 'changed'})
        self.assertEqual(self.manifest.path.read_bytes(), original)
        self.manifest.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.manifest.verify(supervisor_identity(self.settings))
        with self.assertRaisesRegex(ValueError, 'digest'):
            FrozenSupervisorManifest(self.manifest.path, self.manifest.reference['supervisor_sha256'])

    def test_reload_portable_reference_and_missing_evidence(self):
        moved = self.root / 'moved.json'
        self.manifest.path.rename(moved)
        loaded = FrozenSupervisorManifest(moved, self.manifest.reference['supervisor_sha256'])
        self.assertEqual(loaded.verify(supervisor_identity(self.settings))['supervisor_manifest'],
                         'moved.json')
        moved.unlink()
        with self.assertRaises(FileNotFoundError):
            loaded.verify(supervisor_identity(self.settings))

    def test_unverified_version_requires_explicit_limitation(self):
        for kwargs in ({'version_limitation': ''}, {'version_limitation': None},
                       {'immutable_model_version': 1}):
            with self.assertRaises(ValueError):
                VlmSettings('model', **kwargs)

    def test_complete_episode_records_identity_before_control_and_replays(self):
        observations = [raw(i) for i in range(3)]
        config = EpisodeConfig(17, 3)
        policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
        environment = ReplayEnvironment(17, observations[0], [
            ([0], ReplayStep(observations[1], 0, False, False, False)),
            ([1], ReplayStep(observations[2], 1, True, True, False))])
        recorder = AttemptIdentityRecorder(self.root, config,
            lambda: self.manifest.verify(supervisor_identity(self.settings)), ReplayRecorder())
        trace = TraceRecorder(self.root / 'trace', config, recorder)

        def send(payload, deadline, cancellation):
            attempt = json.loads((self.root / 'attempt.json').read_text())
            self.assertEqual(attempt['supervisor_sha256'], self.manifest.reference['supervisor_sha256'])
            return pass_message(payload, deadline, cancellation)

        adapter = ChronologicalVlmAdapter(self.settings, lambda frame: PNG, send,
                                          frozen_manifest=self.manifest)
        outcome = run_episode(config, policy, environment, trace,
                              supervisor_decider=BoundedSupervisorProvider(adapter, 1))
        trace.seal(outcome)
        self.assertEqual(environment.actions, [[0], [1]])
        self.assertTrue(load_recorded_replay(trace.directory).run().success)
        attempt_raw = (self.root / 'attempt.json').read_bytes()
        self.assertEqual(outcome.artifacts['attempt_sha256'], sha256(attempt_raw).hexdigest())


if __name__ == '__main__':
    unittest.main()
