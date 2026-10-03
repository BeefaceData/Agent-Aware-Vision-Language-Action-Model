"""Bounded observed displacement requests assessment, never correction."""

from collections import deque
from dataclasses import dataclass
from math import dist, isfinite

from episode_harness import supervisor_observation


@dataclass(frozen=True)
class ProgressSettings:
    frame: str
    position_units: str
    window: int = 4
    min_interval_seconds: float = 0.1
    max_interval_seconds: float = 2.0
    displacement_tolerance: float = 0.001

    def __post_init__(self):
        if any(type(v) is not str or not v.strip()
               for v in (self.frame, self.position_units)):
            raise ValueError('explicit frame and position units required')
        if type(self.window) is not int or self.window < 2:
            raise ValueError('window must be an integer >= 2')
        for v in (self.min_interval_seconds, self.max_interval_seconds,
                  self.displacement_tolerance):
            if type(v) not in (int, float) or not isfinite(v) or v < 0:
                raise ValueError('intervals and tolerance must be finite and nonnegative')
        if not 0 < self.min_interval_seconds <= self.max_interval_seconds:
            raise ValueError('require 0 < minimum interval <= maximum interval')


@dataclass(frozen=True)
class ProgressSample:
    proposal_id: str
    observation_sequence: int
    captured_monotonic: float
    position: tuple[float, ...]


@dataclass(frozen=True)
class ProgressEvidence:
    episode_id: str
    samples: tuple[ProgressSample, ...]
    settings: ProgressSettings
    displacement: float | None
    interval_seconds: float | None
    status: str
    limitations: tuple[str, ...]


class EndEffectorProgressTrigger:
    """Maximum pairwise Euclidean displacement in a consecutive sample window.

    Requires explicit matching eef.frame/position_units and finite xyz eef.pos.
    Uses packet capture times, not assumed sensor exposure times. Unknown state,
    gaps, incompatible frames and non-increasing times break the window. A match
    emits once until cleared. Callers retain ``evidence`` after each call.
    """

    def __init__(self, settings):
        if not isinstance(settings, ProgressSettings):
            raise ValueError('ProgressSettings required')
        self.settings = settings
        self.evidence = None
        self._rows = deque(maxlen=settings.window)
        self._episode = None
        self._active = False

    def __call__(self, proposal):
        packet = supervisor_observation(proposal.observation)
        state = packet.observation.get('robot_state', {}).get('eef', {})
        position = state.get('pos')
        timestamp = packet.captured_monotonic
        valid = (isinstance(position, list) and len(position) == 3 and
                 all(type(v) in (int, float) and isfinite(v) for v in position) and
                 state.get('frame') == self.settings.frame and
                 state.get('position_units') == self.settings.position_units and
                 type(timestamp) in (int, float) and isfinite(timestamp))
        if (packet.episode_id != self._episode or not valid or
                (self._rows and (packet.sequence != self._rows[-1].observation_sequence + 1
                 or timestamp <= self._rows[-1].captured_monotonic))):
            self._rows.clear()
            self._active = False
        self._episode = packet.episode_id
        limitations = ['motion_alone_cannot_distinguish_stall_from_intentional_pause',
                       'interval_uses_packet_capture_not_verified_sensor_capture']
        if valid:
            self._rows.append(ProgressSample(proposal.proposal_id, packet.sequence,
                                             timestamp, tuple(position)))
            while self._rows and timestamp - self._rows[0].captured_monotonic > self.settings.max_interval_seconds:
                self._rows.popleft()
        else:
            limitations.append('position_frame_units_or_time_unavailable_or_invalid')
        rows = tuple(self._rows)
        interval = rows[-1].captured_monotonic - rows[0].captured_monotonic if rows else None
        displacement = None
        status = 'unknown'
        if len(rows) == self.settings.window and interval >= self.settings.min_interval_seconds:
            displacement = max(dist(a.position, b.position) for a in rows for b in rows)
            if isfinite(displacement):
                status = ('low_progress' if displacement <= self.settings.displacement_tolerance
                          else 'motion_observed')
            else:
                displacement = None
                limitations.append('displacement_not_finite')
        else:
            limitations.append('insufficient_contiguous_interval')
        matches = status == 'low_progress'
        emit = matches and not self._active
        self._active = matches
        self.evidence = ProgressEvidence(packet.episode_id, rows, self.settings,
                                        displacement, interval, status, tuple(limitations))
        return emit
