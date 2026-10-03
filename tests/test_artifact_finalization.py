"""Complete replay outcomes survive evidence sink failures."""

import json
import tempfile
import unittest
from pathlib import Path

from artifact_finalization import ArtifactCloseError, ArtifactResources
from camera_evidence import CameraEvidenceRecorder
from episode_harness import EpisodeConfig, run_episode
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class ArtifactFinalizationTests(unittest.TestCase):
    def test_interrupted_close_remains_incomplete_after_cleanup(self):
        resources = ArtifactResources()
        closed = []
        resources.add('other sink', lambda: closed.append('closed'))

        def interrupt():
            raise KeyboardInterrupt('close interrupted')

        resources.add('interrupted sink', interrupt)
        with self.assertRaises(KeyboardInterrupt):
            resources.close()
        self.assertEqual(closed, ['closed'])
        with self.assertRaisesRegex(ArtifactCloseError, 'close interrupted'):
            resources.close()

    def test_task_outcomes_and_all_close_failures_remain_independent(self):
        for stop in ('success', 'terminated', 'truncated', 'step_limit'):
            for failures in ((), ('log',), ('main',), ('log', 'main', 'wrist')):
                with self.subTest(stop=stop, failures=failures), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    closed = []

                    class Writer:
                        def __init__(self, name):
                            self.name = name

                        def append_data(self, frame):
                            pass

                        def close(self):
                            closed.append(self.name)
                            if self.name in failures:
                                raise OSError(f'{self.name} close failed')

                    cameras = CameraEvidenceRecorder(
                        root / 'main.mp4', root / 'wrist.mp4', root / 'frames.jsonl',
                        lambda path: Writer(path.stem), lambda frame: frame)
                    resources = ArtifactResources()
                    resources.add('cameras', cameras.close)
                    resources.add('steps log', Writer('log').close)

                    class Recorder(ReplayRecorder):
                        def begin(self, packet):
                            super().begin(packet)
                            cameras.record(packet)

                        def record_step(self, step, source, action, result, ingestion):
                            super().record_step(step, source, action, result, ingestion)
                            cameras.record(result.observation)

                        def finish(self):
                            resources.close()
                            return cameras.artifacts

                    initial = {'pixels': {'image': [0], 'image2': [1]}}
                    terminal = {'pixels': {'image': [2], 'image2': [3]}}
                    recorder = Recorder()
                    outcome = run_episode(
                        EpisodeConfig(17, 1), ReplayPolicy(((initial, 'move'),)),
                        ReplayEnvironment(17, initial, (('move', ReplayStep(
                            terminal, 1.0, stop == 'success', stop == 'terminated',
                            stop == 'truncated')),)), recorder)

                    self.assertEqual(outcome.success, stop == 'success')
                    self.assertEqual(outcome.stop_reason, stop)
                    self.assertEqual(outcome.steps, 1)
                    self.assertEqual(outcome.sum_rewards, 1.0)
                    self.assertEqual(len(recorder.steps), 1)
                    rows = [json.loads(line) for line in
                            (root / 'frames.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows), 4)
                    self.assertEqual(rows[-1]['observation_sequence'], 1)
                    self.assertCountEqual(closed, ['log', 'main', 'wrist'])
                    if failures:
                        self.assertEqual(outcome.artifact_status, 'incomplete')
                        self.assertEqual(outcome.artifacts, {})
                        for name in failures:
                            self.assertIn(f'{name} close failed', outcome.artifact_diagnostics[0])
                        with self.assertRaises(ArtifactCloseError):
                            resources.close()
                        if 'main' in failures:
                            with self.assertRaises(ArtifactCloseError):
                                cameras.close()
                            with self.assertRaises(RuntimeError):
                                _ = cameras.artifacts
                    else:
                        self.assertEqual(outcome.artifact_status, 'completed')
                        self.assertEqual(outcome.artifact_diagnostics, ())
                        self.assertEqual(len(outcome.artifacts), 3)
                        resources.close()
                    self.assertCountEqual(closed, ['log', 'main', 'wrist'])

    def test_recorder_finish_error_retains_terminal_identity(self):
        class Recorder(ReplayRecorder):
            def finish(self):
                raise OSError('steps log close failed')

        outcome = run_episode(
            EpisodeConfig(17, 2), ReplayPolicy((('initial', 'move'),)),
            ReplayEnvironment(17, 'initial', (('move', ReplayStep(
                'terminal', 1.0, True, True, False)),)), Recorder())
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.terminal_observation.sequence, 1)
        self.assertEqual(outcome.artifact_status, 'incomplete')
        self.assertEqual(outcome.artifact_diagnostics, ('OSError: steps log close failed',))


if __name__ == '__main__':
    unittest.main()
