"""Single-environment LIBERO vector adapter with explicit terminal evidence.

Supports Gymnasium final_observation/final_obs and final_info envelopes.
Terminal observations in these envelopes are unbatched; normal returns are
batched. Reset observations from same-step autoreset are never ingested.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from time import monotonic

from episode_harness import ObservationPacket, StepResult, frame_references_for_observation


class LiberoEnvironmentAdapter:
    def __init__(self, environment, clock=monotonic):
        if environment.num_envs != 1:
            raise ValueError('exactly one environment is required')
        self.environment = environment
        self.clock = clock
        self.ended = True

    def _packet(self, observation):
        captured = self.clock()
        wall = datetime.now(timezone.utc)
        observation = deepcopy(observation)
        return ObservationPacket(
            self.episode_id, self.sequence, wall, observation, captured,
            frame_references_for_observation(observation, self.sequence, wall, captured))

    def reset(self, seed, episode_id):
        self.episode_id = episode_id
        self.sequence = 0
        observation, _ = self.environment.reset(seed=[seed])
        self.ended = False
        return self._packet(observation)

    def step(self, action):
        import numpy as np

        if self.ended:
            raise RuntimeError('reset is required before another episode step')
        observation, reward, terminated, truncated, info = self.environment.step(action)
        done = bool(terminated[0]) or bool(truncated[0])
        self.ended = done
        success_info = info
        keys = [key for key in ('final_observation', 'final_obs') if key in info]
        mode = getattr(self.environment, 'metadata', {}).get('autoreset_mode')
        same_step = getattr(mode, 'value', mode) == 'SameStep'
        if keys or (done and (same_step or 'final_info' in info)):
            if not done or len(keys) != 1:
                raise ValueError('terminal evidence missing or inconsistent with termination')
            key = keys[0]
            if not bool(info.get('_' + key, [True])[0]):
                raise ValueError('terminal observation is unavailable')
            terminal = info[key][0]
            if not isinstance(terminal, Mapping):
                raise ValueError('terminal observation must be a mapping')

            def batch(value):
                if value is None:
                    return None
                if isinstance(value, Mapping):
                    return {name: batch(item) for name, item in value.items()}
                return np.expand_dims(value, 0)

            observation = batch(terminal)
            # Top-level info belongs to the reset episode in same-step mode.
            success_info = {}
            if 'final_info' in info:
                if not bool(info.get('_final_info', [True])[0]):
                    raise ValueError('terminal evaluator information is unavailable')
                final_info = info['final_info']
                success_info = (final_info if isinstance(final_info, Mapping)
                                else final_info[0])
                if not isinstance(success_info, Mapping):
                    raise ValueError('terminal evaluator information must be a mapping')
        self.sequence += 1
        success = bool(np.asarray(success_info.get('is_success', False)).reshape(-1)[0])
        self.ended = done or success
        return StepResult(self._packet(observation), float(reward[0]), success,
                          bool(terminated[0]), bool(truncated[0]))
