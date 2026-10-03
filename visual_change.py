"""Bounded same-view pixel change requests inspection, never diagnoses failure."""

from dataclasses import dataclass
from math import isfinite

from episode_harness import supervisor_observation


@dataclass(frozen=True)
class VisualChangeSettings:
    view: str = 'main'
    pixel_max: float = 255
    grid_size: int = 16
    low_change: float = .01
    high_change: float = .5
    min_interval_seconds: float = .1
    max_interval_seconds: float = 2

    def __post_init__(self):
        if self.view not in ('main', 'wrist'):
            raise ValueError('view must be main or wrist')
        if type(self.grid_size) is not int or not 2 <= self.grid_size <= 64:
            raise ValueError('grid_size must be an integer from 2 to 64')
        for value in (self.pixel_max, self.low_change, self.high_change,
                      self.min_interval_seconds, self.max_interval_seconds):
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError('finite numeric settings required')
        if not (self.pixel_max > 0 and 0 <= self.low_change < self.high_change <= 1
                and 0 < self.min_interval_seconds <= self.max_interval_seconds):
            raise ValueError('invalid scale, thresholds or interval bounds')


@dataclass(frozen=True)
class VisualSample:
    proposal_id: str
    observation_sequence: int
    captured_monotonic: float
    time_basis: str
    shape: tuple[int, int, int]
    values: tuple[float, ...]


@dataclass(frozen=True)
class VisualChangeEvidence:
    episode_id: str
    settings: VisualChangeSettings
    samples: tuple[VisualSample, ...]
    interval_seconds: float | None
    mean_absolute_change: float | None
    status: str
    limitations: tuple[str, ...]


def _sample(frame, settings):
    """Sample a deterministic endpoint-inclusive grid in HW or HWC pixels."""
    if not isinstance(frame, list) or not frame or not isinstance(frame[0], list) or not frame[0]:
        return None
    height, width = len(frame), len(frame[0])
    if any(not isinstance(row, list) or len(row) != width for row in frame):
        return None
    color = isinstance(frame[0][0], list)
    channels = len(frame[0][0]) if color else 1
    if channels not in (1, 3):
        return None
    values = []
    rows, cols = min(height, settings.grid_size), min(width, settings.grid_size)
    for y in range(rows):
        for x in range(cols):
            pixel = frame[y * (height - 1) // max(1, rows - 1)][x * (width - 1) // max(1, cols - 1)]
            parts = pixel if color else [pixel]
            if not isinstance(parts, list) or len(parts) != channels:
                return None
            for value in parts:
                if (type(value) not in (int, float) or not isfinite(value)
                        or not 0 <= value <= settings.pixel_max):
                    return None
                values.append(value / settings.pixel_max)
    return (height, width, channels), tuple(values)


class VisualChangeTrigger:
    """Compare consecutive same-view grid samples; emit once per extreme regime.

    Only two grids are retained. Missing/stale frames, sequence gaps, incompatible
    shapes or time bases and invalid intervals break comparison continuity.
    Low and high image change are inspection hints, never task outcome labels.
    """

    def __init__(self, settings=VisualChangeSettings()):
        if not isinstance(settings, VisualChangeSettings):
            raise ValueError('VisualChangeSettings required')
        self.settings = settings
        self.evidence = None
        self._previous = None
        self._episode = None
        self._active = None

    def __call__(self, proposal):
        packet = supervisor_observation(proposal.observation)
        refs = [r for r in packet.frame_references if r.camera == self.settings.view]
        ref = refs[0] if len(refs) == 1 else None
        sample = None
        limitations = ['pixel_change_does_not_establish_task_progress_or_failure',
                       'camera_motion_lighting_and_occlusion_are_not_disambiguated',
                       'grid_sampling_can_miss_local_changes']
        if (ref is not None and ref.availability == 'available'
                and ref.observation_sequence == packet.sequence
                and ref.synchronization in ('verified', 'co_observed', 'unpaired')
                and ref.time_basis in ('camera_capture', 'observation_return')
                and ref.captured_monotonic is not None):
            frame = packet.observation.get('pixels', {}).get(ref.image_key.split('.')[1])
            result = _sample(frame, self.settings)
            if result is not None:
                sample = VisualSample(proposal.proposal_id, packet.sequence,
                    ref.captured_monotonic, ref.time_basis, *result)
        if sample is None:
            limitations.append('view_or_capture_or_pixels_unavailable_or_invalid')
        elif sample.time_basis == 'observation_return':
            limitations.append('interval_uses_observation_return_not_verified_camera_capture')
        previous = self._previous if self._episode == packet.episode_id else None
        interval = None
        change = None
        status = 'unknown'
        samples = (sample,) if sample else ()
        if (previous and sample and sample.observation_sequence == previous.observation_sequence + 1
                and sample.shape == previous.shape and sample.time_basis == previous.time_basis):
            interval = sample.captured_monotonic - previous.captured_monotonic
            if self.settings.min_interval_seconds <= interval <= self.settings.max_interval_seconds:
                samples = (previous, sample)
                change = sum(abs(a - b) for a, b in zip(previous.values, sample.values)) / len(sample.values)
                status = ('low_change' if change <= self.settings.low_change else
                          'high_change' if change >= self.settings.high_change else 'intermediate_change')
        if status == 'unknown':
            limitations.append('insufficient_compatible_temporal_evidence')
        active = status if status in ('low_change', 'high_change') else None
        emit = active is not None and active != self._active
        self._active = active
        self._previous, self._episode = sample, packet.episode_id
        self.evidence = VisualChangeEvidence(packet.episode_id, self.settings, samples,
            interval, change, status, tuple(limitations))
        return emit
