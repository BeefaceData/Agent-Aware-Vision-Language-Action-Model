"""Opt-in sampled RGB replay; recording never advances simulator physics.

One frame per policy target and every 25 analytic ticks, played at 10 fps.
This is a diagnostic sampling convention, not a real-time motion recording.
"""

from pathlib import Path

import numpy as np

from harness.assessment_log import append_jsonl
from harness.environment.local import LocalEnvironmentClient
from harness.trace import CAMERAS


class EpisodeVideo:
    def __init__(self, output):
        import imageio.v2 as imageio
        self.output = Path(output)
        self.frames = 0
        self.closed = False
        self.writer = imageio.get_writer(self.output / "episode.mp4", fps=10,
            codec="libx264", output_params=["-g", "10", "-movflags", "+frag_keyframe+empty_moov"])

    def append(self, observation, kind, policy_actions, analytic_ticks):
        from PIL import Image, ImageDraw
        if self.closed:
            raise RuntimeError("video_closed")
        picture = Image.new("RGB", (960, 272))
        draw = ImageDraw.Draw(picture)
        for i, camera in enumerate(CAMERAS):
            picture.paste(Image.fromarray(observation.images[camera]).resize((320, 240)), (i*320, 0))
            draw.text((i*320+4, 242), camera, fill="white")
        draw.text((4, 257), f"SAMPLED 10 fps | {kind} | policy {policy_actions} | analytic ticks {analytic_ticks} | obs {observation.sequence}", fill="white")
        self.writer.append_data(np.asarray(picture))
        append_jsonl(self.output / "video_frames.jsonl", dict(frame=self.frames,
            playback_seconds=self.frames/10, observation_sequence=observation.sequence,
            kind=kind, policy_actions=policy_actions, analytic_ticks=analytic_ticks))
        self.frames += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.writer.close()

    def metadata(self):
        return dict(path="episode.mp4", frame_index="video_frames.jsonl", frames=self.frames,
                    fps=10, timing="sampled: one frame/policy target; one/25 analytic ticks; boundary frames",
                    real_time=False, closed=self.closed)


class RecordingEnvironment(LocalEnvironmentClient):
    def __init__(self, simulator, video):
        super().__init__(simulator)
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
