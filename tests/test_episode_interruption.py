"""Public replay interruptions retain evidence despite recorder teardown faults."""
import json
import unittest
from dataclasses import asdict

from episode_harness import EpisodeConfig, run_episode
from replay_adapters import ReplayRecorder, successful_replay
from run_smolvla_episode import evidence_json


class EpisodeInterruptionTests(unittest.TestCase):
    def test_partial_replay_survives_error_and_teardown_failures(self):
        for error in (TimeoutError('transport'), KeyboardInterrupt('cancelled')):
            for stage in ('policy', 'supervisor', 'execution', 'recording', 'callback'):
                with self.subTest(error=type(error).__name__, stage=stage):
                    fixture = successful_replay()

                    class Policy:
                        def reset(self):
                            fixture.policy.reset()

                        def act(self, packet):
                            if stage == 'policy' and packet.sequence == 1:
                                raise error
                            return fixture.policy.act(packet)

                    class Environment:
                        def reset(self, seed, episode_id):
                            return fixture.environment.reset(seed, episode_id)

                        def step(self, action):
                            if stage == 'execution' and fixture.environment.actions:
                                raise error
                            return fixture.environment.step(action)

                    class Recorder(ReplayRecorder):
                        def record_step(self, step, source, action, result, ingestion):
                            if stage == 'recording' and step == 2:
                                raise error
                            super().record_step(step, source, action, result, ingestion)

                        def record_failure(self, *args):
                            raise OSError('failure log unavailable')

                        def finish(self):
                            raise OSError('video close failed')

                    def supervisor(packet, action):
                        if stage == 'supervisor' and packet.sequence == 1:
                            raise error

                    def progress(step, result):
                        if stage == 'callback' and step == 2:
                            raise error

                    with self.assertRaises(type(error)) as caught:
                        run_episode(fixture.config, Policy(), Environment(), Recorder(),
                                    progress, supervisor)
                    self.assertIs(caught.exception, error)
                    evidence = caught.exception.episode_interruption
                    count = 2 if stage in ('recording', 'callback') else 1
                    self.assertEqual(evidence.steps, count)
                    self.assertEqual(evidence.last_observation.sequence, count)
                    self.assertEqual(len(evidence.acknowledged_actions), count)
                    self.assertEqual(evidence.error_type, type(error).__name__)
                    self.assertEqual(evidence.stop_reason,
                                     'interrupted' if isinstance(error, KeyboardInterrupt) else 'timeout')
                    self.assertIn('OSError: video close failed', evidence.artifact_diagnostics)
                    if stage in ('supervisor', 'execution'):
                        self.assertIn('OSError: failure log unavailable', evidence.artifact_diagnostics)
                    for record in evidence.acknowledged_actions:
                        self.assertIsNotNone(record.execution_acknowledgement)
                    encoded = json.loads(json.dumps(asdict(evidence), default=evidence_json))
                    self.assertEqual(encoded['last_observation']['sequence'], count)
                    self.assertEqual(encoded['steps'], count)

    def test_reset_failure_has_no_fabricated_observation_or_action(self):
        fixture = successful_replay()

        class Policy:
            def reset(self):
                raise RuntimeError('reset failed')

        with self.assertRaises(RuntimeError) as caught:
            run_episode(fixture.config, Policy(), fixture.environment, fixture.recorder)
        partial = caught.exception.episode_interruption
        self.assertEqual(partial.steps, 0)
        self.assertIsNone(partial.last_observation)
        self.assertEqual(partial.acknowledged_actions, ())
        self.assertFalse(fixture.recorder.finalized)

    def test_cleanup_success_does_not_turn_cancellation_into_completion(self):
        fixture = successful_replay()

        def cancel(step, result):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt) as caught:
            run_episode(fixture.config, fixture.policy, fixture.environment,
                        fixture.recorder, cancel)
        self.assertTrue(fixture.recorder.finalized)
        partial = caught.exception.episode_interruption
        self.assertEqual(partial.stop_reason, 'interrupted')
        self.assertEqual(partial.steps, 1)
        self.assertEqual(partial.artifact_diagnostics, ())


if __name__ == '__main__':
    unittest.main()
