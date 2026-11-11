"""Opt-in sampled RGB replay; recording never advances simulator physics.

One frame per policy target and every 25 analytic ticks, encoded at 30 fps.
This is a diagnostic sampling convention, not a real-time motion recording.
"""

from pathlib import Path

import numpy as np

from harness.jsonl import append_jsonl
from harness.environment.local import LocalEnvironmentClient
from harness.environment.base import CAMERAS

VIDEO_FPS = 30


class EpisodeVideo:
    def __init__(self, output):
        import imageio.v2 as imageio
        self.output = Path(output)
        self.frames = 0
        self.closed = False
        self.cameras = None
        self.writer = imageio.get_writer(self.output / "episode.mp4", fps=VIDEO_FPS,
            codec="libx264", output_params=["-g", str(VIDEO_FPS),
                                             "-movflags", "+frag_keyframe+empty_moov"])

    def append(self, observation, kind, policy_actions, analytic_ticks):
        from PIL import Image, ImageDraw
        if self.closed:
            raise RuntimeError("video_closed")
        cameras = (("head_camera", "perception_camera")
                   if "perception_camera" in observation.images else CAMERAS)
        if self.cameras is None:
            self.cameras = cameras
        elif self.cameras != cameras:
            raise ValueError("recording_camera_set_changed")
        picture = Image.new("RGB", (960, 272))
        draw = ImageDraw.Draw(picture)
        panel_width = 960 // len(cameras)
        for i, camera in enumerate(cameras):
            label = "Supervisor camera" if camera == "perception_camera" else camera
            picture.paste(Image.fromarray(observation.images[camera]).resize((panel_width, 240)),
                          (i*panel_width, 0))
            draw.text((i*panel_width+4, 242), label, fill="white")
        draw.text((4, 257), f"SAMPLED {VIDEO_FPS} fps | {kind} | policy {policy_actions} | analytic ticks {analytic_ticks} | obs {observation.sequence}", fill="white")
        self.writer.append_data(np.asarray(picture))
        append_jsonl(self.output / "video_frames.jsonl", dict(frame=self.frames,
            playback_seconds=self.frames/VIDEO_FPS, observation_sequence=observation.sequence,
            kind=kind, policy_actions=policy_actions, analytic_ticks=analytic_ticks))
        self.frames += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.writer.close()

    def metadata(self):
        return dict(path="episode.mp4", frame_index="video_frames.jsonl", frames=self.frames,
                    fps=VIDEO_FPS, timing="sampled: one frame/policy target; one/25 analytic ticks; boundary frames",
                    real_time=False, closed=self.closed,
                    cameras=list(self.cameras or CAMERAS))


class RecordingEnvironment(LocalEnvironmentClient):
    def __init__(self, simulator, video, **options):
        super().__init__(simulator, **options)
        self.video = video
        self.policy_actions = self.analytic_ticks = 0
        self.policy_kind = "vla_act"
        self._recorded_sequence = None

    def _record(self, observation, kind):
        self.video.append(observation, kind, self.policy_actions, self.analytic_ticks)
        self._recorded_sequence = observation.sequence

    def reset(self, seed):
        observation = super().reset(seed)
        self.policy_actions = self.analytic_ticks = 0
        self._record(observation, "reset")
        return observation

    def step(self, action):
        observation = super().step(action)
        return self._record_policy_step(observation)

    def step_after_success(self, action):
        observation = super().step_after_success(action)
        return self._record_policy_step(observation)

    def _record_policy_step(self, observation):
        self.policy_actions += 1
        self._record(observation, self.policy_kind)
        return observation

    def tick(self, *args, **kwargs):
        super().tick(*args, **kwargs)
        self.analytic_ticks += 1
        if self.analytic_ticks % 25 == 0:
            self._record(super().observe(), "analytic")

    def observe(self):
        observation = super().observe()
        # Preserve the endpoint of a short analytic skill without duplicating
        # frames during repeated predicate/planner observation reads.
        if observation.sequence != self._recorded_sequence:
            self._record(observation, "boundary")
        return observation
