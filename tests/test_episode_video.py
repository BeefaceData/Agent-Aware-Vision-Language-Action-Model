import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from development.demos.episode_video import EpisodeVideo, RecordingEnvironment
from harness.environment.local import LocalEnvironmentClient
from harness.environment.base import Observation
from harness.environment.base import CAMERAS


def observation(sequence):
    return Observation(sequence, {name: np.zeros((240, 320, 3), dtype=np.uint8) for name in CAMERAS}, (0.,)*14)


def observation_with_supervisor_camera(sequence):
    images = {name: np.zeros((240, 320, 3), dtype=np.uint8) for name in CAMERAS}
    images["perception_camera"] = np.zeros((240, 320, 3), dtype=np.uint8)
    return Observation(sequence, images, (0.,)*14)


class VideoTests(unittest.TestCase):
    def test_supervisor_recording_uses_head_and_fixed_scene_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = EpisodeVideo(tmp)
            try:
                video.append(observation_with_supervisor_camera(0), 'reset', 0, 0)
            finally:
                video.close()
            reader = __import__('imageio.v2', fromlist=['get_reader']).get_reader(
                Path(tmp)/'episode.mp4')
            try:
                self.assertEqual(reader.get_data(0).shape, (272, 960, 3))
            finally:
                reader.close()
            self.assertEqual(video.metadata()['cameras'],
                             ['head_camera', 'perception_camera'])

    def test_real_encoding_and_frame_index(self):
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as tmp:
            video = EpisodeVideo(tmp)
            try:
                video.append(observation(0), 'reset', 0, 0)
                video.append(observation(1), 'vla_act', 1, 0)
            finally:
                video.close()
            video.close()
            reader = imageio.get_reader(Path(tmp)/'episode.mp4')
            try:
                self.assertEqual(reader.count_frames(), 2)
                self.assertEqual(reader.get_data(0).shape, (272, 960, 3))
            finally:
                reader.close()
            rows = [json.loads(x) for x in (Path(tmp)/'video_frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['observation_sequence'] for r in rows], [0, 1])
            self.assertAlmostEqual(rows[1]['playback_seconds'], 1/30)
            self.assertEqual(video.metadata()['fps'], 30)
            self.assertFalse(video.metadata()['real_time'])

    def test_recording_does_not_add_actions_and_observations_do_not_duplicate_frames(self):
        video = Mock()
        env = RecordingEnvironment(Mock(), video)
        current = [observation(0)]
        def native_qpos(*args, **kwargs):
            current[0] = observation(current[0].sequence+1)
        with patch.object(LocalEnvironmentClient, 'reset', return_value=current[0]) as reset, \
             patch.object(LocalEnvironmentClient, 'observe', side_effect=lambda: current[0]), \
             patch.object(LocalEnvironmentClient, 'apply_native_qpos', side_effect=native_qpos) as native_step, \
             patch.object(LocalEnvironmentClient, 'step', return_value=observation(28)) as step:
            env.reset(3)
            for _ in range(27):
                env.apply_native_qpos([0.] * 14, controlled_arm='left',
                                      controlled_velocity=[0.] * 6)
            self.assertEqual(video.append.call_count, 6)  # reset + ticks 5, 10, 15, 20, 25
            env.observe(); env.observe()
            self.assertEqual(video.append.call_count, 7)  # endpoint, once
            env.step([0.]*14)
            self.assertEqual(video.append.call_count, 8)
            self.assertEqual(native_step.call_count, 27)
            reset.assert_called_once_with(3)
            step.assert_called_once()

    def test_writer_failure_propagates(self):
        video = Mock()
        video.append.side_effect = OSError('disk full')
        env = RecordingEnvironment(Mock(), video)
        with patch.object(LocalEnvironmentClient, 'step', return_value=observation(1)) as step:
            with self.assertRaises(OSError): env.step([0.]*14)
            step.assert_called_once()
