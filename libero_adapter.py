"""Single-environment LIBERO vector adapter with explicit terminal evidence.

Supports Gymnasium final_observation/final_obs and final_info envelopes.
Terminal observations in these envelopes are unbatched; normal returns are
batched. Reset observations from same-step autoreset are never ingested.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from operator import index
from time import monotonic

from episode_harness import ObservationPacket, StepResult, frame_references_for_observation


@dataclass(frozen=True)
class ActionHorizon:
    """Environment readback and the optional explicitly requested action limit."""

    effective: int
    requested_override: int | None
    source: str


def read_action_horizon(environment, requested_override=None) -> ActionHorizon:
    """Read the configured single-vector horizon; refuse an ignored override.

    Call after environment creation and before reset or execution. Never coerce
    a fractional or missing limit into a usable budget.
    """
    def positive_integer(value):
        try:
            result = index(value)
        except TypeError as exc:
            raise ValueError('action horizon must be a positive integer') from exc
        if isinstance(value, bool) or result <= 0:
            raise ValueError('action horizon must be a positive integer')
        return result

    if environment.num_envs != 1:
        raise ValueError('exactly one environment is required')
    requested = (None if requested_override is None
                 else positive_integer(requested_override))
    limits = environment.call('_max_episode_steps')
    if len(limits) != 1:
        raise ValueError('expected exactly one environment horizon')
    effective = positive_integer(limits[0])
    if requested is not None and effective != requested:
        raise ValueError(f'environment horizon {effective} differs from override {requested}')
    return ActionHorizon(effective, requested,
                         'environment_default' if requested is None else 'explicit_override')


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
        # Invalidate the previous attempt even if reset or packet capture fails.
        self.ended = True
        self.episode_id = episode_id
        self.sequence = 0
        observation, _ = self.environment.reset(seed=[seed])
        packet = self._packet(observation)
        self.ended = False
        return packet

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
