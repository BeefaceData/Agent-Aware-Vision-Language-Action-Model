"""Bounded repetition evidence requesting inspection, never a failure verdict."""

from collections import deque
from dataclasses import dataclass
from math import isfinite

from episode_harness import supervisor_observation


@dataclass(frozen=True)
class RepetitionSettings:
    window: int = 4
    action_tolerance: float = 0.0
    position_tolerance: float = 0.0

    def __post_init__(self):
        if type(self.window) is not int or self.window < 2:
            raise ValueError('window must be an integer >= 2')
        for value in (self.action_tolerance, self.position_tolerance):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError('tolerances must be finite and nonnegative')


@dataclass(frozen=True)
class RepetitionSignal:
    episode_id: str
    proposal_ids: tuple[str, ...]
    observation_sequences: tuple[int, ...]
    actions: tuple[tuple[float, ...], ...]
    positions: tuple[tuple[float, ...], ...]
    settings: RepetitionSettings
    reason: str = 'repeated_proposals_with_low_observed_position_change'


def _vector(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if (not isinstance(value, (tuple, list)) or not value or
            any(type(v) not in (int, float) or not isfinite(v) for v in value)):
        return None
    return tuple(value)


def _span(vectors):
    return max(max(axis) - min(axis) for axis in zip(*vectors))


class RepeatedProposalTrigger:
    """Callable assessment trigger; inspect ``signal`` immediately after a call.

    A window of consecutive proposals must have per-coordinate action and
    observed position spans <= the configured tolerances. Emits once until the
    condition clears. Missing/invalid vectors and sequence gaps break evidence;
    new episode identities reset it. Position is deployable eef.pos, falling
    back to robot_state.position only when eef.pos is absent. Units and frame
    consistency belong to the adapter; configure thresholds for that contract.

    This proxy cannot establish task productivity (e.g. a stationary gripper
    may be working). It requests assessment only. Retain emitted signals using
    the caller's evidence sink; this object keeps only one bounded window and
    the current signal. Proposed actions are not execution acknowledgements.
    """

    def __init__(self, settings=RepetitionSettings()):
        if not isinstance(settings, RepetitionSettings):
            raise ValueError('RepetitionSettings required')
        self.settings = settings
        self.signal = None
        self._rows = deque(maxlen=settings.window)
        self._episode = None
        self._sequence = None
        self._active = False

    def __call__(self, proposal):
        packet = supervisor_observation(proposal.observation)
        self.signal = None
        if (packet.episode_id != self._episode or self._sequence is None or
                packet.sequence != self._sequence + 1):
            self._rows.clear()
            self._active = False
        self._episode, self._sequence = packet.episode_id, packet.sequence
        state = packet.observation.get('robot_state', {})
        position = _vector(state.get('eef', {}).get('pos', state.get('position')))
        action = _vector(proposal.action)
        if action is None or position is None:
            self._rows.clear()
            self._active = False
            return False
        if self._rows and (len(action) != len(self._rows[-1][2]) or
                           len(position) != len(self._rows[-1][3])):
            self._rows.clear()
            self._active = False
        self._rows.append((proposal.proposal_id, packet.sequence, action, position))
        actions = tuple(row[2] for row in self._rows)
        positions = tuple(row[3] for row in self._rows)
        matches = (len(self._rows) == self.settings.window and
                   _span(actions) <= self.settings.action_tolerance and
                   _span(positions) <= self.settings.position_tolerance)
        emit = matches and not self._active
        self._active = matches
        if emit:
            self.signal = RepetitionSignal(packet.episode_id,
                tuple(row[0] for row in self._rows),
                tuple(row[1] for row in self._rows), actions, positions, self.settings)
        return emit
