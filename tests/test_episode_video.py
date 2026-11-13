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


class VideoTests(unittest.TestCase):
    def test_four_camera_video_contains_perception_panel(self):
        import imageio.v2 as imageio
        current = observation(0)
        current.images["perception_camera"] = np.full((240, 320, 3), 255, dtype=np.uint8)
        cameras = (*CAMERAS, "perception_camera")
        with tempfile.TemporaryDirectory() as tmp:
            video = EpisodeVideo(tmp, cameras=cameras)
            try:
                video.append(current, "baseline", 0, 0)
            finally:
                video.close()
            reader = imageio.get_reader(Path(tmp) / "episode.mp4")
            try:
                frame = reader.get_data(0)
            finally:
                reader.close()
            self.assertEqual(frame.shape, (512, 640, 3))
            self.assertGreater(frame[250:400, 330:600].mean(), 240)

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
            self.assertEqual(rows[1]['playback_seconds'], .1)
            self.assertFalse(video.metadata()['real_time'])

    def test_recording_does_not_add_actions_and_observations_do_not_duplicate_frames(self):
        video = Mock()
        env = RecordingEnvironment(Mock(), video)
        current = [observation(0)]
        def tick(*args, **kwargs):
            current[0] = observation(current[0].sequence+1)
        with patch.object(LocalEnvironmentClient, 'reset', return_value=current[0]) as reset, \
             patch.object(LocalEnvironmentClient, 'observe', side_effect=lambda: current[0]), \
             patch.object(LocalEnvironmentClient, 'tick', side_effect=tick) as native_tick, \
             patch.object(LocalEnvironmentClient, 'step', return_value=observation(28)) as step:
            env.reset(3)
            for _ in range(27): env.tick('left', [], [], 1., {})
            self.assertEqual(video.append.call_count, 2)  # reset + tick 25
            env.observe(); env.observe()
            self.assertEqual(video.append.call_count, 3)  # endpoint, once
            env.step([0.]*14)
            self.assertEqual(video.append.call_count, 4)
            self.assertEqual(native_tick.call_count, 27)
            reset.assert_called_once_with(3)
            step.assert_called_once()

    def test_writer_failure_propagates(self):
        video = Mock()
        video.append.side_effect = OSError('disk full')
        env = RecordingEnvironment(Mock(), video)
        with patch.object(LocalEnvironmentClient, 'step', return_value=observation(1)) as step:
            with self.assertRaises(OSError): env.step([0.]*14)
            step.assert_called_once()
