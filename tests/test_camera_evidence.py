"""Camera association and gaps through the public episode replay interface."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from camera_evidence import CameraEvidenceRecorder
from episode_harness import (EpisodeConfig, ViewCapture,
                             frame_references_for_observation, run_episode)
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


CAPTURE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def views(main, wrist=None):
    pixels = {'image': [main]}
    if wrist is not None:
        pixels['image2'] = [wrist]
    return {'pixels': pixels}


def captures(sequence, main_time, wrist_time):
    return {
        'main': ViewCapture(sequence, CAPTURE_START + timedelta(seconds=main_time),
                            main_time),
        'wrist': ViewCapture(sequence, CAPTURE_START + timedelta(seconds=wrist_time),
                             wrist_time),
    }


class CameraEvidenceReplayTests(unittest.TestCase):
    def test_episode_artifacts_index_each_camera_without_filling_gap(self):
        class Writer:
            def __init__(self):
                self.frames = []

            def append_data(self, frame):
                self.frames.append(frame)

            def close(self):
                pass

        initial = views('main 0', 'wrist 0')
        terminal = views('main 1')
        environment = ReplayEnvironment(
            17, initial, (('finish', ReplayStep(
                terminal, 1.0, True, True, False)),), clock=lambda: 10.0)
        writers = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def make_writer(path):
                writer = Writer()
                writers[path.name] = writer
                return writer

            camera_recorder = CameraEvidenceRecorder(
                root / 'main.mp4', root / 'wrist.mp4', root / 'frames.jsonl',
                make_writer, lambda pixels: pixels[0])

            class Recorder(ReplayRecorder):
                def begin(self, packet):
                    super().begin(packet)
                    camera_recorder.record(packet)

                def record_step(self, step, source, action, result, ingestion):
                    super().record_step(step, source, action, result, ingestion)
                    if ingestion.accepted:
                        camera_recorder.record(result.observation)

                def finish(self):
                    super().finish()
                    camera_recorder.close()
                    return camera_recorder.artifacts

            outcome = run_episode(EpisodeConfig(17, 2),
                                  ReplayPolicy(((initial, 'finish'),)),
                                  environment, Recorder(), clock=lambda: 10.0)
            rows = [json.loads(line) for line in
                    (root / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]

        self.assertEqual(writers['main.mp4'].frames, ['main 0', 'main 1'])
        self.assertEqual(writers['wrist.mp4'].frames, ['wrist 0'])
        self.assertEqual([(row['observation_sequence'], row['camera'],
                           row['frame_index']) for row in rows],
                         [(0, 'main', 0), (0, 'wrist', 0),
                          (1, 'main', 1), (1, 'wrist', None)])
        self.assertEqual(rows[-1]['availability'], 'missing')
        self.assertIsNone(rows[-1]['video_path'])
        self.assertIsNone(rows[-1]['captured_at'])
        self.assertEqual(set(outcome.artifacts),
                         {'frames_path', 'video_path', 'wrist_video_path'})

    def test_capture_skew_and_missing_limit_cannot_be_called_verified(self):
        observation = views('main', 'wrist')
        camera_captures = captures(0, 1.0, 1.06)
        refs = frame_references_for_observation(
            observation, 0, CAPTURE_START + timedelta(seconds=2),
            2.0, camera_captures, max_skew_seconds=0.05)
        self.assertEqual([ref.synchronization for ref in refs],
                         ['unsynchronized', 'unsynchronized'])
        with self.assertRaises(ValueError):
            frame_references_for_observation(
                observation, 0, CAPTURE_START, 2.0, camera_captures)

    def test_complete_episode_keeps_paired_views_with_each_observation(self):
        initial = views('main 0', 'wrist 0')
        after_reach = views('main 1', 'wrist 1')
        terminal = views('main 2', 'wrist 2')
        policy = ReplayPolicy(((initial, 'reach'), (after_reach, 'place')))
        environment = ReplayEnvironment(
            17, initial, (
                ('reach', ReplayStep(after_reach, 0.0, False, False, False,
                                     captures(1, 0.96, 0.98))),
                ('place', ReplayStep(terminal, 1.0, True, True, False,
                                     captures(2, 1.96, 1.98))),
            ), clock=lambda: 10.0, initial_camera_captures=captures(0, -0.04, -0.02),
            max_camera_skew_seconds=0.05)
        recorder = ReplayRecorder()

        outcome = run_episode(EpisodeConfig(17, 3), policy, environment,
                              recorder, clock=lambda: 10.0)

        self.assertTrue(outcome.success)
        self.assertTrue(recorder.finalized)
        self.assertEqual([packet.sequence for packet in recorder.observations],
                         [0, 1, 2])
        for sequence, packet in enumerate(recorder.observations):
            self.assertEqual({ref.camera for ref in packet.frame_references},
                             {'main', 'wrist'})
            for ref in packet.frame_references:
                self.assertEqual(ref.observation_sequence, sequence)
                self.assertEqual(ref.synchronization, 'verified')
                self.assertEqual(ref.availability, 'available')
                self.assertEqual(ref.time_basis, 'camera_capture')
                self.assertEqual(ref.image_key,
                                 'pixels.image' if ref.camera == 'main'
                                 else 'pixels.image2')
                self.assertEqual(packet.observation['pixels'][
                    ref.image_key.split('.')[1]][0],
                    f'{ref.camera} {sequence}')
                self.assertEqual(ref.captured_at,
                                 CAPTURE_START + timedelta(
                                     seconds=ref.captured_monotonic))
        self.assertEqual([packet.sequence for packet in policy.observations], [0, 1])

    def test_missing_and_stale_wrist_are_explicit_without_replacement(self):
        initial = views('main 0', 'wrist 0')
        stale_wrist = views('main 1', 'wrist 0')
        terminal = views('main 2')
        stale_captures = {
            'main': ViewCapture(1, CAPTURE_START + timedelta(seconds=1), 1.0),
            'wrist': ViewCapture(0, CAPTURE_START, 0.0),
        }
        policy = ReplayPolicy(((initial, 'reach'), (stale_wrist, 'place')))
        environment = ReplayEnvironment(
            17, initial, (
                ('reach', ReplayStep(stale_wrist, 0.0, False, False, False,
                                     stale_captures)),
                ('place', ReplayStep(terminal, 1.0, True, True, False)),
            ), clock=lambda: 10.0, max_camera_skew_seconds=0.05)
        recorder = ReplayRecorder()

        outcome = run_episode(EpisodeConfig(17, 3), policy, environment,
                              recorder, clock=lambda: 10.0)

        self.assertTrue(outcome.success)
        stale = recorder.observations[1]
        refs = {ref.camera: ref for ref in stale.frame_references}
        self.assertEqual(refs['main'].synchronization, 'unsynchronized')
        self.assertEqual(refs['wrist'].synchronization, 'unsynchronized')
        self.assertEqual(refs['wrist'].observation_sequence, 0)
        self.assertEqual(stale.observation['pixels']['image2'][0], 'wrist 0')
        terminal_refs = {ref.camera: ref for ref in recorder.observations[2].frame_references}
        self.assertEqual(terminal_refs['wrist'].availability, 'missing')
        self.assertEqual(terminal_refs['wrist'].synchronization, 'unavailable')
        self.assertIsNone(terminal_refs['wrist'].captured_at)
        self.assertIsNone(terminal_refs['wrist'].captured_monotonic)
        self.assertNotIn('image2', recorder.observations[2].observation['pixels'])
        self.assertEqual(terminal_refs['main'].time_basis, 'observation_return')
        self.assertEqual(terminal_refs['main'].synchronization, 'unpaired')


if __name__ == '__main__':
    unittest.main()
