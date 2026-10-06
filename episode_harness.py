"""Baseline episode orchestration with injectable policy, environment and recorder.

The caller owns the adapters and releases any resources they hold. Artifact
finalization failures are returned separately from the observed task outcome.
Environment step results contain evaluator outcomes, which are never sent to the
policy through this interface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import wraps
from math import isfinite
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Literal, Mapping, Protocol, TYPE_CHECKING
from uuid import uuid4

from temporal_diagnosis import valid_temporal_diagnosis
from intervention_budget import InterventionBudget
from correction_expiry import check_expiry
from episode_deadline import EpisodeDeadline, EpisodeDeadlineExceeded, workers_busy
from supervisor_retry import RECOVERABLE_ERRORS
from decision_memory import disabled_memory, validate_memory

if TYPE_CHECKING:
    from baseline_fallback import BaselineFallback
    from observation_window import ObservationWindow, WindowSettings


@dataclass(frozen=True)
class EpisodeConfig:
    """Trusted episode limits; None resolves the intervention cap to max_steps.

    Each recovery sequence and each single-action override consumes one
    intervention. Unlisted tools have zero attempts; zero disables a budget.
    Recovery cooldown counts accepted baseline actions after local completion;
    zero disables cooldown, and early correction requests stop the episode.

    max_steps is the shared action horizon: a baseline or adjusted command
    costs one action, and each recovery command costs one action. Recovery
    admission requires room for the entire sequence; unused commands after
    terminal outcomes or aborts are not counted as executed actions.

    max_supervisor_calls defaults to max_steps and reserves one request before
    each due single-attempt supervisor callback. Failed requests are not refunded.
    Exhaustion stops dispatch by default; baseline_fallback explicitly opts into
    current-input/health/native-action admission checks. See docs/supervisor-allowance.md.

    max_episode_seconds bounds active rollout time, including model waits and
    corrections, from the ready initial observation through termination. None
    preserves legacy uncapped operation. Capped adapters must support concurrent,
    bounded interruption; reset/setup and artifact finalization are outside it.

    Declare both correction validity limits for expiry-enforced operation.
    None/None preserves the legacy trusted-selector/replay interface; it is
    not an expiry-ready active configuration. No physical limits are inferred.

    correction_mode='recovery_only' is an optional diagnostic ablation. It
    rejects numerical overrides, including identity overrides, while preserving
    normal recovery validation and every configured runtime limit. A rejected
    adjustment stops unless an explicit BaselineFallback admits the proposal.
    'adjustment_only' instead rejects recovery sequences and retains normal
    single-action adjustment validation and the same runtime limits.
    """
    seed: int
    max_steps: int
    supervisor_interval_actions: int = 1
    max_interventions: int | None = None
    recovery_attempt_limits: tuple[tuple[str, int], ...] = (('reopen_and_retreat', 1),)
    recovery_cooldown_actions: int = 0
    correction_timeout_seconds: float | None = None
    correction_max_age_seconds: float | None = None
    max_supervisor_calls: int | None = None
    supervisor_exhaustion_policy: str = "stop"
    max_episode_seconds: float | None = None
    supervisor_max_retries: int = 0
    supervisor_retry_delay_seconds: float = 0.0
    supervisor_retry_errors: tuple[str, ...] = RECOVERABLE_ERRORS
    correction_mode: str = 'combined'

    def __post_init__(self):
        if self.correction_mode not in ('combined', 'recovery_only', 'adjustment_only'):
            raise ValueError('invalid correction mode')
        if type(self.supervisor_max_retries) is not int or self.supervisor_max_retries < 0:
            raise ValueError('supervisor retry count must be a nonnegative integer')
        delay = self.supervisor_retry_delay_seconds
        if type(delay) not in (int, float) or not 0 <= delay < float('inf'):
            raise ValueError('supervisor retry delay must be nonnegative and finite')
        errors = self.supervisor_retry_errors
        if (type(errors) not in (tuple, list) or
                any(type(e) is not str or e not in RECOVERABLE_ERRORS for e in errors) or
                len(set(errors)) != len(errors)):
            raise ValueError('invalid supervisor retry error classes')
        if self.supervisor_max_retries and (not errors or self.max_episode_seconds is None):
            raise ValueError('supervisor retries require eligible errors and an episode time cap')
        if (type(self.supervisor_interval_actions) is not int or
                self.supervisor_interval_actions <= 0):
            raise ValueError('supervisor_interval_actions must be a positive integer')
        if self.max_interventions is None:
            if type(self.max_steps) is not int or self.max_steps < 0:
                raise ValueError('max_steps must be a positive integer')
            object.__setattr__(self, 'max_interventions', self.max_steps)
        if type(self.max_interventions) is not int or self.max_interventions < 0:
            raise ValueError('max_interventions must be a nonnegative integer')
        if type(self.recovery_cooldown_actions) is not int or self.recovery_cooldown_actions < 0:
            raise ValueError('recovery_cooldown_actions must be a nonnegative integer')
        if self.max_supervisor_calls is None:
            object.__setattr__(self, 'max_supervisor_calls', self.max_steps)
        if type(self.max_supervisor_calls) is not int or self.max_supervisor_calls < 0:
            raise ValueError('max_supervisor_calls must be a nonnegative integer')
        if self.supervisor_exhaustion_policy not in ('stop', 'baseline_fallback'):
            raise ValueError('invalid supervisor exhaustion policy')
        if self.max_episode_seconds is not None and (
                type(self.max_episode_seconds) not in (int, float) or
                not isfinite(self.max_episode_seconds) or self.max_episode_seconds <= 0):
            raise ValueError('max_episode_seconds must be positive and finite')
        expiry_limits = (self.correction_timeout_seconds, self.correction_max_age_seconds)
        if expiry_limits != (None, None) and any(
                type(value) not in (int, float) or not isfinite(value) or value <= 0
                for value in expiry_limits):
            raise ValueError('correction expiry requires two positive finite limits')
        limits = self.recovery_attempt_limits
        if (type(limits) not in (tuple, list) or
                any(type(item) not in (tuple, list) or len(item) != 2 or
                    type(item[0]) is not str or not item[0].isidentifier() or
                    type(item[1]) is not int or item[1] < 0 for item in limits) or
                len({item[0] for item in limits}) != len(limits)):
            raise ValueError('recovery_attempt_limits require unique tool names and nonnegative integers')
        object.__setattr__(self, 'recovery_attempt_limits', tuple(tuple(item) for item in limits))


@dataclass(frozen=True)
class ObservationPacket:
    """One captured observation, identified within a single episode.

    ``captured_at`` is a timezone-aware source capture time for provenance.
    ``captured_monotonic`` is sampled at capture from the same monotonic clock
    as ``run_episode``; it alone is used for durations. Sequences start at zero.
    """

    episode_id: str
    sequence: int
    captured_at: datetime
    observation: Any
    captured_monotonic: float | None = None
    frame_references: tuple[FrameReference, ...] = ()
    robot_state_capture: RobotStateCapture | None = None
    state_fields: tuple[StateField, ...] = ()


@dataclass(frozen=True)
class StateField:
    """Deployable state for one arm, timed at observation return unless verified.

    A missing measurement has no value or timestamp. ``observation_return`` is
    not a claim of sensor capture time or synchronized robot and camera clocks.
    """

    arm: str
    name: str
    frame: str
    units: str
    availability: Literal['available', 'missing']
    value: tuple[float, ...] | None
    captured_at: datetime | None
    captured_monotonic: float | None
    time_basis: Literal['observation_return'] | None


@dataclass(frozen=True)
class ViewCapture:
    """A camera capture on the packet's monotonic clock, when known."""

    observation_sequence: int
    captured_at: datetime
    captured_monotonic: float


@dataclass(frozen=True)
class RobotStateCapture:
    """Measured robot-state capture on the packet's monotonic clock."""

    observation_sequence: int
    captured_at: datetime
    captured_monotonic: float


@dataclass(frozen=True)
class FrameReference:
    """Reference to one view in an observation's ``pixels`` mapping.

    A missing view has no capture time. ``observation_return`` timestamps mark
    when the paired simulator observation returned, not camera exposure time.
    ``co_observed`` therefore does not claim measured camera synchronization.
    """

    camera: Literal['main', 'wrist']
    observation_sequence: int
    image_key: str
    captured_at: datetime | None
    captured_monotonic: float | None
    availability: Literal['available', 'missing']
    synchronization: Literal['co_observed', 'verified', 'unsynchronized',
                             'unpaired', 'unavailable']
    time_basis: Literal['observation_return', 'camera_capture'] | None


def supervisor_observation(packet: ObservationPacket) -> ObservationPacket:
    """Copy only declared deployable fields across the supervision boundary.

    Raw environment mappings belong to the policy/recorder. Unknown fields,
    including nested metadata, are omitted rather than filtered by secret names.
    Adapters must map additional sensors to this contract before exposing them.
    Numeric payloads are rebuilt as plain values, removing array/object metadata.
    """
    def measurement(value):
        if type(value) in (int, float, bool):
            return value if isfinite(value) else None
        if isinstance(value, (list, tuple)):
            items = [measurement(item) for item in value]
            return items if items and all(item is not None for item in items) else None
        if hasattr(value, 'tolist'):
            try:
                return measurement(value.tolist())
            except (TypeError, ValueError, RuntimeError):
                return None
        return None

    raw = packet.observation if isinstance(packet.observation, Mapping) else {}
    allowed = {}
    if type(raw.get('task')) is str:
        allowed['task'] = raw['task']
    pixels = raw.get('pixels')
    if isinstance(pixels, Mapping):
        allowed['pixels'] = {key: measurement(pixels[key])
                             for key in ('image', 'image2') if key in pixels}
    state = raw.get('robot_state')
    if isinstance(state, Mapping):
        clean_state = {}
        for key in ('position', 'orientation', 'joint_positions', 'joint_velocities'):
            if key in state:
                clean_state[key] = measurement(state[key])
        for group, fields in (('eef', ('pos', 'quat')),
                              ('gripper', ('qpos', 'qvel', 'closed'))):
            if isinstance(state.get(group), Mapping):
                clean_state[group] = {key: measurement(state[group][key])
                                      for key in fields if key in state[group]}
                if group == 'eef':
                    for key in ('frame', 'position_units'):
                        if type(state[group].get(key)) is str:
                            clean_state[group][key] = state[group][key]
        allowed['robot_state'] = clean_state
    def valid_capture(sequence, wall, mono):
        return (type(sequence) is int and type(wall) is datetime and
                type(mono) in (int, float) and isfinite(mono))

    # Reject malformed metadata rather than forwarding nested payloads in typed slots.
    for ref in packet.frame_references:
        if (type(ref) is not FrameReference or
            ref.camera not in ('main', 'wrist') or
            ref.image_key != ('pixels.image' if ref.camera == 'main' else 'pixels.image2') or
            type(ref.observation_sequence) is not int or
            (ref.captured_at is not None and type(ref.captured_at) is not datetime) or
            (ref.captured_monotonic is not None and
             (type(ref.captured_monotonic) not in (int, float) or
              not isfinite(ref.captured_monotonic))) or
            ref.availability not in ('available', 'missing') or
            ref.synchronization not in ('co_observed', 'verified', 'unsynchronized',
                                        'unpaired', 'unavailable') or
            ref.time_basis not in (None, 'observation_return', 'camera_capture')):
            raise ValueError('invalid supervisor camera metadata')
    # Reconstruct typed metadata rather than copying arbitrary attached attributes.
    references = tuple(FrameReference(
        ref.camera, ref.observation_sequence, ref.image_key, ref.captured_at,
        ref.captured_monotonic, ref.availability, ref.synchronization, ref.time_basis)
        for ref in packet.frame_references)
    capture = packet.robot_state_capture
    if capture is not None and (
        type(capture) is not RobotStateCapture or not valid_capture(
            capture.observation_sequence, capture.captured_at, capture.captured_monotonic)
    ):
        raise ValueError('invalid supervisor robot-state metadata')
    state_capture = (RobotStateCapture(capture.observation_sequence,
                                      capture.captured_at, capture.captured_monotonic)
                     if type(capture) is RobotStateCapture else None)
    state_fields = []
    seen = set()
    for item in packet.state_fields:
        if (type(item) is not StateField or
                any(type(text) is not str or not text for text in
                    (item.arm, item.name, item.frame, item.units)) or
                (item.arm, item.name) in seen):
            raise ValueError('invalid supervisor state field')
        seen.add((item.arm, item.name))
        if item.availability == 'missing':
            valid = all(value is None for value in (
                item.value, item.captured_at, item.captured_monotonic, item.time_basis))
        else:
            valid = (item.availability == 'available' and
                     type(item.value) is tuple and bool(item.value) and
                     all(type(value) in (int, float) and isfinite(value)
                         for value in item.value) and
                     type(item.captured_at) is datetime and
                     item.captured_at.utcoffset() is not None and
                     type(item.captured_monotonic) in (int, float) and
                     isfinite(item.captured_monotonic) and
                     item.time_basis == 'observation_return')
        if not valid:
            raise ValueError('invalid supervisor state field')
        state_fields.append(StateField(
            item.arm, item.name, item.frame, item.units, item.availability,
            item.value, item.captured_at, item.captured_monotonic, item.time_basis))
    return ObservationPacket(packet.episode_id, packet.sequence, packet.captured_at,
                             allowed, packet.captured_monotonic, references,
                             state_capture, tuple(state_fields))


def frame_references_for_observation(
    observation: Mapping[str, Any], sequence: int,
    captured_at: datetime, captured_monotonic: float,
    camera_captures: Mapping[str, ViewCapture] | None = None,
    max_skew_seconds: float | None = None,
) -> tuple[FrameReference, FrameReference]:
    """Describe LIBERO's main/wrist views without filling a missing view.

    Explicit camera capture metadata verifies association and skew. When the
    environment only supplies a paired observation, both times are the shared
    observation-return timestamp and synchronization remains unverified.
    """
    if (type(sequence) is not int or sequence < 0 or
        not isinstance(captured_at, datetime) or captured_at.utcoffset() is None or
        isinstance(captured_monotonic, bool) or
        not isinstance(captured_monotonic, (int, float)) or
        not isfinite(captured_monotonic)):
        raise ValueError('valid observation identity and timestamps required')
    if camera_captures is not None and (
        isinstance(max_skew_seconds, bool) or
        not isinstance(max_skew_seconds, (int, float)) or
        not isfinite(max_skew_seconds) or max_skew_seconds < 0
    ):
        raise ValueError('explicit camera captures require a finite skew limit')
    if not isinstance(observation, Mapping):
        raise ValueError('observation must be a mapping')
    if camera_captures is not None and set(camera_captures) - {'main', 'wrist'}:
        raise ValueError('unknown camera capture')
    pixels = observation.get('pixels')
    if not isinstance(pixels, Mapping):
        pixels = {}
    available = {camera: pixels.get(key) is not None
                 for camera, key in (('main', 'image'), ('wrist', 'image2'))}
    if camera_captures is not None:
        for camera, capture in camera_captures.items():
            if not isinstance(capture, ViewCapture) or not available[camera]:
                raise ValueError('camera capture requires a present view')
            if (type(capture.observation_sequence) is not int or
                not isinstance(capture.captured_at, datetime) or
                capture.captured_at.utcoffset() is None or
                not isinstance(capture.captured_monotonic, (int, float)) or
                isinstance(capture.captured_monotonic, bool) or
                not isfinite(capture.captured_monotonic) or
                capture.captured_monotonic > captured_monotonic or
                capture.captured_at > captured_at):
                raise ValueError('invalid camera capture metadata')
    captures = camera_captures or {}
    pair_verified = (all(available.values()) and len(captures) == 2 and
                     all(captures[camera].observation_sequence == sequence
                         for camera in ('main', 'wrist')) and
                     abs(captures['main'].captured_monotonic -
                         captures['wrist'].captured_monotonic) <= max_skew_seconds and
                     abs((captures['main'].captured_at -
                          captures['wrist'].captured_at).total_seconds()) <=
                     max_skew_seconds)
    references = []
    for camera, key in (('main', 'image'), ('wrist', 'image2')):
        if not available[camera]:
            references.append(FrameReference(camera, sequence, f'pixels.{key}',
                                             None, None, 'missing', 'unavailable', None))
            continue
        capture = captures.get(camera)
        status = ('verified' if pair_verified else
                  'unsynchronized' if camera_captures is not None else
                  'co_observed' if all(available.values()) else 'unpaired')
        references.append(FrameReference(
            camera, capture.observation_sequence if capture else sequence,
            f'pixels.{key}', capture.captured_at if capture else captured_at,
            capture.captured_monotonic if capture else captured_monotonic,
            'available', status,
            'camera_capture' if capture else 'observation_return'))
    return tuple(references)


@dataclass(frozen=True)
class InputFreshness:
    """One required input's availability and age at a decision time.

    An ``observation_return`` age is only a lower bound on sensor age, so an
    input with that time basis cannot be called fresh. ``sensor_capture``
    measures age since capture. Status does not establish valid geometry.
    """

    name: Literal['main', 'wrist', 'robot_state']
    available: bool
    age_seconds: float | None
    max_age_seconds: float
    status: Literal['fresh', 'unverified', 'missing', 'stale', 'invalid_metadata']
    diagnostic: str
    captured_monotonic: float | None
    observation_sequence: int | None
    time_basis: Literal['sensor_capture', 'observation_return'] | None


def _camera_payload_present(value: Any) -> bool:
    """Check for data without testing pixel truthiness or importing array libraries."""
    if value is None or isinstance(value, Mapping):
        return False
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_camera_payload_present(v) for v in value)
    if isinstance(value, (int, float)):
        return isfinite(value)  # Zero-valued pixels are still measurements.
    try:
        shape = getattr(value, 'shape', None)
        if shape is not None:
            # len(array) alone misses zero elements on a later axis.
            return len(shape) > 0 and all(dimension > 0 for dimension in shape)
        return len(value) > 0
    except (TypeError, ValueError, RuntimeError):
        return False


def check_observation_freshness(
    packet: ObservationPacket, now_monotonic: float,
    max_age_seconds: Mapping[str, float],
) -> tuple[InputFreshness, ...]:
    """Assess configured required inputs without accepting absent data as fresh.

    Limits must name one or more of ``main``, ``wrist``, and ``robot_state``.
    A missing input has no age. An explicit capture from an older observation
    retains that older sequence and timestamp. Invalid or future metadata is
    reported as invalid, rather than being clamped to age zero.
    """
    names = ('main', 'wrist', 'robot_state')
    if not isinstance(packet, ObservationPacket) or (
        not isinstance(now_monotonic, (int, float)) or
        isinstance(now_monotonic, bool) or not isfinite(now_monotonic)
    ):
        raise ValueError('packet and finite decision time required')
    if (not isinstance(packet.captured_at, datetime) or
        packet.captured_at.utcoffset() is None or
        packet.captured_monotonic is None or
        not isinstance(packet.captured_monotonic, (int, float)) or
        isinstance(packet.captured_monotonic, bool) or
        not isfinite(packet.captured_monotonic) or
        packet.captured_monotonic > now_monotonic):
        raise ValueError('packet capture must precede the decision')
    if not isinstance(max_age_seconds, Mapping) or not max_age_seconds or (
        set(max_age_seconds) - set(names)
    ):
        raise ValueError('freshness limits require known input names')
    for limit in max_age_seconds.values():
        if (not isinstance(limit, (int, float)) or isinstance(limit, bool) or
            not isfinite(limit) or limit < 0):
            raise ValueError('freshness limits must be finite and nonnegative')

    observation = packet.observation if isinstance(packet.observation, Mapping) else {}
    pixels = observation.get('pixels')
    pixels = pixels if isinstance(pixels, Mapping) else {}
    refs = {ref.camera: ref for ref in packet.frame_references
            if isinstance(ref, FrameReference)}
    results = []
    for name in names:
        if name not in max_age_seconds:
            continue
        limit = max_age_seconds[name]
        if name == 'robot_state':
            state = observation.get('robot_state')
            def present(value: Any) -> bool:
                if value is None:
                    return False
                if isinstance(value, Mapping):
                    return bool(value) and all(present(v) for v in value.values())
                if isinstance(value, (list, tuple)):
                    return bool(value) and all(present(v) for v in value)
                if isinstance(value, bool):
                    return True
                if isinstance(value, (int, float)):
                    return isfinite(value)
                if hasattr(value, 'tolist'):
                    try:
                        return present(value.tolist())
                    except (TypeError, ValueError, RuntimeError):
                        return False
                return False
            available = isinstance(state, Mapping) and present(state)
            capture = packet.robot_state_capture
            captured = (capture.captured_monotonic if isinstance(capture, RobotStateCapture)
                        else packet.captured_monotonic if capture is None else None)
            sequence = (capture.observation_sequence if isinstance(capture, RobotStateCapture)
                        else packet.sequence if capture is None else None)
            basis = 'sensor_capture' if capture is not None else 'observation_return'
            metadata_valid = (capture is None or (
                isinstance(capture, RobotStateCapture) and
                type(capture.observation_sequence) is int and
                0 <= capture.observation_sequence <= packet.sequence and
                isinstance(capture.captured_at, datetime) and
                capture.captured_at.utcoffset() is not None and
                capture.captured_at <= packet.captured_at))
        else:
            key = 'image' if name == 'main' else 'image2'
            ref = refs.get(name)
            available = _camera_payload_present(pixels.get(key))
            captured = ref.captured_monotonic if ref is not None else None
            sequence = ref.observation_sequence if ref is not None else None
            basis = (('sensor_capture' if ref.time_basis == 'camera_capture'
                      else 'observation_return') if ref is not None and
                     ref.time_basis is not None else None)
            metadata_valid = (ref is not None and
                ref.image_key == f'pixels.{key}' and
                ref.availability == ('available' if available else 'missing') and
                type(ref.observation_sequence) is int and
                0 <= ref.observation_sequence <= packet.sequence and
                (not available or (
                    isinstance(ref.captured_at, datetime) and
                    ref.captured_at.utcoffset() is not None and
                    ref.captured_at <= packet.captured_at and
                    ref.time_basis in ('camera_capture', 'observation_return'))))
        if not available:
            status, diagnostic = 'missing', f'{name}: obtain a new observation with this input'
            age = None
            captured = None
            sequence = None
            basis = None
        elif (not metadata_valid or not isinstance(captured, (int, float)) or
              isinstance(captured, bool) or not isfinite(captured) or
              captured > packet.captured_monotonic):
            status, diagnostic = 'invalid_metadata', f'{name}: repair capture metadata before use'
            age = None
        else:
            age = now_monotonic - captured
            if age > limit:
                status = 'stale'
                diagnostic = (f'{name}: age {age:.3f}s exceeds {limit:.3f}s; '
                              'obtain a new capture')
            elif basis == 'observation_return':
                status = 'unverified'
                diagnostic = (f'{name}: sensor capture time unavailable; '
                              'obtain timestamped input before claiming freshness')
            else:
                status, diagnostic = 'fresh', f'{name}: within configured age limit'
        results.append(InputFreshness(name, available, age, limit, status,
                                      diagnostic, captured, sequence, basis))
    return tuple(results)


@dataclass(frozen=True)
class IngestionOutcome:
    accepted: bool
    code: str
    episode_id: str
    received_episode_id: str | None
    expected_sequence: int
    received_sequence: int | None


class ObservationRejected(ValueError):
    """An environment packet was rejected before it could drive a decision."""

    def __init__(self, outcome: IngestionOutcome):
        self.outcome = outcome
        super().__init__(f'observation rejected: {outcome.code}; '
                         f'expected episode {outcome.episode_id}, '
                         f'received {outcome.received_episode_id}; '
                         f'expected sequence {outcome.expected_sequence}, '
                         f'received {outcome.received_sequence}')


class ObservationIngestor:
    """Accept a contiguous, time-ordered packet stream for one episode."""

    def __init__(self, episode_id: str):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_id must be a nonempty string')
        self.episode_id = episode_id
        self._next_sequence = 0
        self._last_capture: datetime | None = None
        self._last_monotonic_capture: float | None = None

    def ingest(self, packet: ObservationPacket,
               received_at: float | None = None) -> IngestionOutcome:
        expected = self._next_sequence
        received = packet.sequence if isinstance(packet, ObservationPacket) else None
        received_episode = packet.episode_id if isinstance(packet, ObservationPacket) else None
        code = 'accepted'
        if not isinstance(packet, ObservationPacket):
            code = 'invalid_packet'
        elif packet.episode_id != self.episode_id:
            code = 'wrong_episode'
        elif type(packet.sequence) is not int or packet.sequence < 0:
            code = 'invalid_sequence'
        elif not isinstance(packet.captured_at, datetime) or (
            packet.captured_at.utcoffset() is None
        ):
            code = 'invalid_capture_time'
        elif (packet.captured_monotonic is None or
              isinstance(packet.captured_monotonic, bool) or
              not isinstance(packet.captured_monotonic, (int, float)) or
              not isfinite(packet.captured_monotonic)):
            code = 'invalid_monotonic_capture_time'
        elif received_at is not None and packet.captured_monotonic > received_at:
            code = 'future_monotonic_capture_time'
        elif packet.sequence < self._next_sequence:
            code = 'duplicate'
        elif packet.sequence > self._next_sequence:
            code = 'out_of_order'
        elif self._last_capture is not None and packet.captured_at < self._last_capture:
            code = 'out_of_order_capture_time'
        elif (self._last_monotonic_capture is not None and
              packet.captured_monotonic < self._last_monotonic_capture):
            code = 'out_of_order_monotonic_capture_time'
        if code == 'accepted':
            self._next_sequence += 1
            self._last_capture = packet.captured_at
            self._last_monotonic_capture = packet.captured_monotonic
        return IngestionOutcome(code == 'accepted', code, self.episode_id,
                                received_episode, expected, received)


@dataclass(frozen=True)
class StepTiming:
    """Seconds on one shared monotonic clock, from source capture to execution.

    Decision latency is request-to-response time for the observation-only
    supervisor and any injected action selector. Without either it is zero.
    The cumulative wait sums those latencies, independently of executed actions.
    """

    capture_at: float
    request_at: float
    response_at: float
    execution_started_at: float
    execution_finished_at: float
    cumulative_wait_seconds: float

    def __post_init__(self) -> None:
        times = (self.capture_at, self.request_at, self.response_at,
                 self.execution_started_at, self.execution_finished_at)
        if (any(not isfinite(value) for value in times) or
            any(earlier > later for earlier, later in zip(times, times[1:])) or
            not isfinite(self.cumulative_wait_seconds) or
            self.cumulative_wait_seconds < 0):
            raise ValueError('timing requires finite, ordered monotonic times')

    @property
    def observation_age_seconds(self) -> float:
        return self.request_at - self.capture_at

    @property
    def decision_latency_seconds(self) -> float:
        return self.response_at - self.request_at

    @property
    def execution_seconds(self) -> float:
        return self.execution_finished_at - self.execution_started_at


@dataclass(frozen=True)
class FailedStepTiming:
    """Timing for a failed request or environment call on the episode clock.

    ``request_finished_at`` marks either a response or an exception. A missing
    response and missing execution boundaries mean those events did not occur.
    ``execution_finished_at`` marks a call ending in error, not a completed
    simulator action.
    """

    capture_at: float
    request_at: float
    request_finished_at: float
    response_at: float | None
    execution_started_at: float | None
    execution_finished_at: float | None
    cumulative_wait_seconds: float

    def __post_init__(self) -> None:
        boundaries = (self.capture_at, self.request_at,
                      self.request_finished_at, self.execution_started_at,
                      self.execution_finished_at)
        present = tuple(value for value in boundaries if value is not None)
        if (any(not isfinite(value) for value in present) or
            any(earlier > later for earlier, later in zip(present, present[1:])) or
            (self.response_at is not None and
             self.response_at != self.request_finished_at) or
            (self.execution_started_at is None) !=
            (self.execution_finished_at is None) or
            (self.response_at is None and self.execution_started_at is not None) or
            not isfinite(self.cumulative_wait_seconds) or
            self.cumulative_wait_seconds < 0):
            raise ValueError('failure timing requires finite, ordered monotonic times')

    @property
    def observation_age_seconds(self) -> float:
        return self.request_at - self.capture_at

    @property
    def decision_latency_seconds(self) -> float:
        return self.request_finished_at - self.request_at


@dataclass(frozen=True)
class StepFailure:
    stage: str  # 'supervisor', 'selection', or 'execution'
    error_type: str
    timing: FailedStepTiming
    action_record: ActionRecord | None = None


@dataclass(frozen=True)
class ActionProposal:
    """Identity and detached native action for one policy decision."""

    proposal_id: str
    observation: ObservationPacket
    action: Any


@dataclass(frozen=True)
class SupervisorPass:
    """Explicit permission to execute exactly one current policy proposal."""

    episode_id: str
    observation_sequence: int
    proposal_id: str
    temporal_diagnosis: dict | None = None
    suppressed_correction: Literal['recovery', 'adjustment'] | None = None
    memory_context: dict = field(default_factory=disabled_memory)
    kind: Literal['pass'] = field(default='pass', init=False)


class SupervisorResponseError(ValueError):
    """Structured response rejected before execution; message is safe evidence."""


@dataclass(frozen=True)
class SupervisorAbstention:
    """Unknown diagnosis; continue the unchanged proposal while execution is healthy.

    Evidence availability is the supervisor's reported assessment, not a sensor
    verification or permission for a correction. Name each assessed input and
    mark unassessed availability explicitly as unknown.
    """

    episode_id: str
    observation_sequence: int
    proposal_id: str
    reason: str
    evidence_availability: dict[str, Literal['available', 'missing', 'stale', 'unknown']]
    temporal_diagnosis: dict | None = None
    memory_context: dict = field(default_factory=disabled_memory)
    kind: Literal['abstain'] = field(default='abstain', init=False)
    diagnosis: Literal['unknown'] = field(default='unknown', init=False)


def valid_abstention_details(reason, evidence_availability):
    """Validate inspectable uncertainty evidence at execution and trace loading."""
    return (type(reason) is str and bool(reason.strip()) and
            type(evidence_availability) is dict and bool(evidence_availability) and
            all(type(name) is str and bool(name.strip()) and
                type(status) is str and status in ('available', 'missing', 'stale', 'unknown')
                for name, status in evidence_availability.items()))


@dataclass(frozen=True)
class RecoverySequence:
    """Host-validated finite commands; no model code or environment ownership.

    The original suspended proposal remains in each action record. ``request``
    identifies its source; subsequent step IDs identify execution, not inference.
    Local physical completion is separate from exhausting these commands.
    """

    actions: tuple
    action_limit: int
    request: dict
    envelope_id: str
    controller_evidence_sha256: str
    monitor: dict

    def __post_init__(self):
        from recovery_monitor import validate_monitor
        validate_monitor(self.monitor)
        if (type(self.actions) not in (list, tuple) or not self.actions or
                type(self.action_limit) is not int or
                not 2 == len(self.actions) <= self.action_limit or
                any(action is None for action in self.actions) or
                type(self.request) is not dict or
                any(type(self.request.get(key)) is not str or not self.request[key]
                    for key in ('episode_id', 'proposal_id', 'decision_id', 'tool_name')) or
                type(self.request.get('observation_sequence')) is not int or
                self.request['observation_sequence'] < 0 or
                type(self.envelope_id) is not str or not self.envelope_id or
                type(self.controller_evidence_sha256) is not str or
                len(self.controller_evidence_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.controller_evidence_sha256)):
            raise ValueError('invalid bounded recovery sequence')
        object.__setattr__(self, 'actions', tuple(deepcopy(self.actions)))
        object.__setattr__(self, 'request', deepcopy(self.request))
        object.__setattr__(self, 'monitor', deepcopy(self.monitor))


@dataclass(frozen=True)
class ActionResolution:
    """Execution choice, optionally bound to its source proposal by the executor.

    Unbound choices are reserved for trusted host selectors and recorded replay;
    supervisor-derived overrides must retain their source identity until dispatch.
    """

    kind: Literal['pass', 'override', 'reject', 'recovery']
    action: Any = None
    reason: str | None = None
    recovery: RecoverySequence | None = None
    source_identity: tuple[str, int, str] | None = None
    memory_context: dict | None = None


@dataclass(frozen=True)
class ExecutionAcknowledgement:
    """The environment returned a step result for this proposal.

    This confirms the adapter call completed, not independent physical actuation.
    """

    proposal_id: str
    result_episode_id: str | None
    result_sequence: int | None


@dataclass(frozen=True)
class ActionRecord:
    """Snapshot of what was proposed, selected, and confirmed at the step seam.

    ``executed_action`` is present only after ``environment.step`` returns.
    ``selected_action`` retains the attempted command if the call fails. For a
    rejection both fields and the acknowledgement are absent.
    """

    proposal_id: str
    proposed_action: Any
    selected_action: Any | None
    executed_action: Any | None
    execution_acknowledgement: ExecutionAcknowledgement | None
    disposition: Literal['unmodified', 'overridden', 'rejected', 'unconfirmed']
    rejection_reason: str | None = None
    supervisor_pass: SupervisorPass | None = None
    supervisor_abstention: SupervisorAbstention | None = None
    recovery: dict | None = None
    intervention_budget: dict | None = None
    fallback: dict | None = None
    interruption: dict | None = None
    dispatch: dict | None = None
    correction_expiry: dict | None = None
    supervisor_call_budget: dict | None = None
    correction_memory: dict | None = None


@dataclass(frozen=True)
class StepResult:
    observation: ObservationPacket
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    timing: StepTiming | None = None
    action_record: ActionRecord | None = None


@dataclass(frozen=True)
class ObservationReference:
    episode_id: str
    sequence: int


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    success: bool
    steps: int
    stop_reason: str
    sum_rewards: float
    rollout_seconds: float
    artifacts: Mapping[str, str]
    cumulative_wait_seconds: float = 0.0
    step_timings: tuple[StepTiming, ...] = ()
    terminal_observation: ObservationReference | None = None
    artifact_status: Literal['incomplete', 'completed'] = 'incomplete'
    artifact_diagnostics: tuple[str, ...] = ()
    supervisor_call_budget: dict | None = None
    wall_clock_limit: dict | None = None
    episode_clock: dict | None = None
    task_status: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'task_status',
                           'success' if self.success else
                           'failure' if self.stop_reason in ('terminated', 'step_limit')
                           else 'unknown')


def exception_stop_reason(error: BaseException) -> str:
    """Classify the original failure independently of cleanup diagnostics."""
    if not isinstance(error, Exception):
        return 'interrupted'
    return 'timeout' if isinstance(error, TimeoutError) else 'infrastructure_failure'


@dataclass(frozen=True)
class PreStartFailure:
    """Reset/initial-observation failure before an episode can start.

    Completion flags acknowledge contract returns, not a verified simulator
    state. A failed reset must be retried through a new run_episode call.
    """

    stage: Literal['policy_reset', 'environment_reset', 'initial_observation']
    seed: int
    policy_reset_completed: bool
    environment_reset_completed: bool


@dataclass(frozen=True)
class EpisodeInterruption:
    """Partial evaluator evidence attached to the original raised exception.

    Acknowledgements confirm adapter returns, not independent physical actuation.
    This is never a completed episode, even when evidence cleanup succeeds.
    """

    episode_id: str
    stop_reason: str
    error_type: str
    steps: int
    sum_rewards: float
    last_observation: ObservationPacket | None
    acknowledged_actions: tuple[ActionRecord, ...]
    artifact_diagnostics: tuple[str, ...]
    task_status: str = 'unknown'
    artifact_status: str = 'incomplete'
    pre_start_failure: PreStartFailure | None = None
    failed_action: ActionRecord | None = None
    terminal_observation: ObservationReference | None = None
    terminal_reason: str | None = None


class PolicyAdapter(Protocol):
    """Reset must discard queued actions and all per-attempt policy state."""

    def reset(self) -> None: ...
    def act(self, observation: ObservationPacket) -> Any: ...


class ResumablePolicyAdapter(PolicyAdapter, Protocol):
    """Optional override capability; reset is never inferred as a substitute.

    Resume discards obsolete queued actions and synchronizes policy state with
    the accepted post-override packet. It must not execute an action. The next
    act receives that same observation. Raise if safe resumption is unavailable.
    """

    def resume(self, observation: ObservationPacket) -> None: ...


class EnvironmentAdapter(Protocol):
    """Reset replaces terminal state and returns sequence zero for the new ID.

    A failed reset must disable stepping until a successful reset. A returned
    StepResult acknowledges completion of the adapter step call.
    """

    def reset(self, seed: int, episode_id: str) -> ObservationPacket: ...
    def step(self, action: Any) -> StepResult: ...


class EpisodeRecorder(Protocol):
    """Records completed steps and failed attempts; failure actions are proposals."""

    def begin(self, observation: ObservationPacket) -> None: ...
    def record_step(self, step: int, source: ObservationPacket,
                    action: Any, result: StepResult,
                    ingestion: IngestionOutcome) -> None: ...
    def record_failure(self, step: int, source: ObservationPacket,
                       action: Any, failure: StepFailure) -> None: ...
    def finish(self) -> Mapping[str, str]: ...


class ExecutionBusy(RuntimeError):
    """An overlapping episode was rejected before touching shared adapters."""


_ownership_lock = Lock()
_active_adapters: set[int] = set()


def _exclusive_episode(run):
    @wraps(run)
    def owned(config, policy, environment, recorder, *args, **kwargs):
        # Identity, not equality/hash: adapters may be mutable or unhashable.
        # The call holds strong references until release, preventing ID reuse.
        identities = {id(policy), id(environment), id(recorder)}
        with _ownership_lock:
            if identities & _active_adapters or workers_busy(identities | {
                id(kwargs[key]) for key in ('supervisor', 'window_supervisor',
                    'supervisor_decider', 'action_selector', 'recovery_observer',
                    'assessment_trigger', 'on_step', 'baseline_fallback') if kwargs.get(key) is not None
            }):
                raise ExecutionBusy('episode adapter already has an execution owner')
            _active_adapters.update(identities)
        try:
            return run(config, policy, environment, recorder, *args, **kwargs)
        finally:
            with _ownership_lock:
                _active_adapters.difference_update(identities)
    return owned


@_exclusive_episode
def run_episode(
    config: EpisodeConfig,
    policy: PolicyAdapter,
    environment: EnvironmentAdapter,
    recorder: EpisodeRecorder,
    on_step: Callable[[int, StepResult], None] | None = None,
    supervisor: Callable[[ObservationPacket, Any], None] | None = None,
    clock: Callable[[], float] = monotonic,
    action_selector: Callable[[ActionProposal], ActionResolution] | None = None,
    window_supervisor: Callable[[ObservationWindow, Any], None] | None = None,
    window_settings: WindowSettings | None = None,
    supervisor_decider: Callable[[ActionProposal], SupervisorPass | SupervisorAbstention] | None = None,
    assessment_trigger: Callable[[ActionProposal], bool] | None = None,
    recovery_observer: Callable[[RecoverySequence, int, ObservationPacket], Any] | None = None,
    baseline_fallback: BaselineFallback | None = None,
    frozen_contract=None,
) -> EpisodeOutcome:
    """Run one attempt and return task outcome and artifact finalization status.

    The initial task field remains fixed through terminal observation. Optional
    ``frozen_contract`` verifies trusted host configuration readers before reset,
    inference and dispatch and after each step. Drift records a rejection and
    invokes the adapter's declared interruption behavior before raising.

    One call owns its policy, environment and recorder from before reset through
    finalization, including recovery and resumption. Concurrent or reentrant
    calls sharing any adapter raise ExecutionBusy immediately without reset,
    dispatch or recorder cleanup; they are never queued for later execution.
    Independent adapters can run concurrently. This is an in-process ownership
    boundary: callers must not bypass it with direct adapter calls or distinct
    wrappers/processes controlling the same underlying robot.

    ``policy`` provides reset/act(packet); ``environment`` provides
    reset(seed, episode_id)/step(action); ``recorder`` provides begin(packet),
    record_step(step, source_packet, action, result, ingestion),
    record_failure(step, source_packet, action, failure), and finish()
    -> artifact references. A finish exception returns incomplete artifacts and
    diagnostics without erasing task success, failure or terminal identity.
    Rejected result packets and failed calls are recorded
    before raising; failures do not produce a completed outcome.
    On failure, recorder finalization is attempted and the original exception is
    re-raised with ``episode_interruption`` containing detached partial evidence.
    Cleanup errors never replace that exception. The caller releases adapters. ``clock`` must be
    monotonic and share a timebase with each packet's ``captured_monotonic``.
    ``supervisor`` observes an allowlisted packet and a copy of the proposed action before
    execution; it has no execution authority here. ``action_selector`` is an
    injected, deterministic execution choice for replay or a validated executor.
    Its input and output actions are copied so neither it nor the environment can
    mutate the policy proposal or recorded evidence. Its rejection is recorded and
    ends the attempt without an environment call. Supervisor and selector wait is
    measured. Overrides must come from a separately validated executor.
    A recovery resolution supplies a finite host-validated RecoverySequence.
    ``recovery_observer`` assesses local conditions from deployable observations
    after each recovery command. Failed checks end the episode without resuming
    the policy; this declared stop path does not implement a physical stop.
    It pauses policy inference and callbacks, counts each command in max_steps,
    and resumes once from the final accepted nonterminal packet. The whole
    sequence must fit the remaining horizon before its first command executes.
    Per-action recovery evidence links the suspended original policy proposal;
    later execution IDs do not represent additional policy inference.
    Episode-owned counters charge an intervention before its first dispatch,
    with an additional per-tool charge for recovery. Continuation actions do
    not charge another attempt. Aborts/failures never refund a charge; exhausted
    requests are recorded and rejected without dispatch or further selection.
    Policy inference, recorder and ``on_step`` time are excluded from that wait.
    ``window_supervisor(window, proposed_action)`` is the temporal alternative to
    ``supervisor``. It receives bounded sanitized history and acknowledged prior
    actions, using ``window_settings`` (defaults to eight packets/eight actions).
    Both callbacks are observation-only and cannot be configured together.
    ``supervisor_decider`` instead receives a sanitized, detached ActionProposal
    and must return a SupervisorPass or SupervisorAbstention matching its episode,
    observation and proposal. Abstention records uncertainty and continues the
    unchanged baseline proposal only after baseline_fallback confirms fresh required
    inputs, healthy controller state and validity of the current native proposal.
    Without this host-owned guard, abstention refuses dispatch. Any execution or
    observation fault ends the attempt.
    It cannot be combined with other supervision or action selection callbacks.
    Accepted responses are retained in action evidence, including execution failures.
    Simulation scheduling is synchronous: one proposal waits for its decision
    before any environment step or next policy call. The environment must advance
    only on step(); this is not a physical-controller hold/stop implementation.
    Use a bounded decider to impose a deadline; callback failures retain waiting
    time and terminate the attempt without executing the pending proposal unless
    baseline_fallback is configured and admits that proposal. Explicit selector
    rejection can also use this guard; internal budget/cooldown rejections cannot.
    Interruptions and selector exceptions never trigger fallback. Active decision
    configurations require an adapter interruption_contract and interrupt(request)
    before reset. Rejected dispatch invokes that declared hold/stop operation;
    an unconfirmed acknowledgement ends with interruption_failed. No zero-action
    substitute is generated, and interruption does not count as a policy action.
    The decider is assessed before action 1 and every configured interval of
    acknowledged actions thereafter. An optional sanitized assessment_trigger
    adds assessments between these fixed boundaries; it never postpones them.
    Coincident periodic/event requests use one call and pending calls block the
    loop. Skipped assessments retain no supervisor response in action evidence.
    An accepted terminal result ends correction processing before recovery
    assessment or user callbacks. Remaining recovery commands are cancelled;
    no decision queue is drained and no policy resume occurs. Late provider
    work has no dispatch authority. A new attempt gets a new episode identity,
    so a cached executor resolution cannot be reused across that boundary.
    """
    from frozen_runtime import EpisodeInstruction, FrozenContractViolation, FrozenRuntimeContract
    if frozen_contract is not None:
        if type(frozen_contract) is not FrozenRuntimeContract:
            raise ValueError('FrozenRuntimeContract required')
        frozen_contract.verify()
    from action_capabilities import validate_action_pair
    validate_action_pair(policy, environment)
    from baseline_fallback import BaselineFallback, fallback_evidence
    if baseline_fallback is not None and type(baseline_fallback) is not BaselineFallback:
        raise ValueError('BaselineFallback required')
    if type(config.max_steps) is not int or config.max_steps <= 0:
        raise ValueError('max_steps must be a positive integer')
    if config.supervisor_max_retries and supervisor_decider is None:
        raise ValueError('supervisor retries require a supervisor decider')
    if supervisor is not None and window_supervisor is not None:
        raise ValueError('choose one supervisor callback')
    if assessment_trigger is not None and (
            supervisor_decider is None or not callable(assessment_trigger)):
        raise ValueError('assessment_trigger requires supervisor_decider and a callable')
    if config.supervisor_interval_actions != 1 and (
            supervisor is not None or window_supervisor is not None):
        raise ValueError('periodic assessment requires supervisor_decider')
    if window_settings is not None and window_supervisor is None:
        raise ValueError('window_settings requires window_supervisor')
    if supervisor_decider is not None and any(callback is not None for callback in
                                             (supervisor, window_supervisor, action_selector)):
        raise ValueError('supervisor_decider requires exclusive decision ownership')

    from environment_interruption import require_interruption, interrupt
    interruption_contract = (require_interruption(environment) if any(
        item is not None for item in (action_selector, supervisor_decider, baseline_fallback))
        or config.max_episode_seconds is not None
        or (config.max_supervisor_calls < config.max_steps and
            (supervisor is not None or window_supervisor is not None))
        else None)

    from observation_window import ObservationWindowBuilder, WindowAction, WindowSettings
    settings = window_settings if window_settings is not None else WindowSettings()
    if not isinstance(settings, WindowSettings):
        raise ValueError('WindowSettings required')
    history = None

    episode_id = uuid4().hex
    ingestor = ObservationIngestor(episode_id)
    last_observation = None
    acknowledged_actions = []
    recorder_started = False
    reward_sum = 0.0
    cumulative_wait = 0.0
    steps = 0
    interruption_diagnostics = []
    failed_action = None
    terminal_observation = None
    terminal_reason = None
    task_status = 'unknown'
    startup_stage = 'policy_reset'
    policy_reset_completed = False
    environment_reset_completed = False

    def record_failure(*args):
        try:
            recorder.record_failure(*args)
        except BaseException as recording_error:
            interruption_diagnostics.append(
                f'{type(recording_error).__name__}: {recording_error}')

    try:
        policy.reset()
        policy_reset_completed = True
        startup_stage = 'environment_reset'
        observation = environment.reset(config.seed, episode_id)
        environment_reset_completed = True
        startup_stage = 'initial_observation'
        initial_ingestion = ingestor.ingest(observation, clock())
        if not initial_ingestion.accepted:
            raise ObservationRejected(initial_ingestion)
        last_observation = deepcopy(observation)
        startup_stage = None
        recorder_started = True
        instruction_guard = EpisodeInstruction(observation)
        recorder.begin(observation)
        deadline = EpisodeDeadline(config.max_episode_seconds, clock,
            (policy, environment, recorder, supervisor, window_supervisor,
             supervisor_decider, action_selector, recovery_observer, assessment_trigger,
             on_step, baseline_fallback))
        rollout_start = deadline.started_at
        wall_clock_limit = None
        reward_sum = 0.0
        stop_reason = 'step_limit'
        success = False
        steps = 0
        cumulative_wait = 0.0
        step_timings: list[StepTiming] = []
        terminal_observation = None
        recovery_sequence = None
        recovery_index = 0
        budget = InterventionBudget(config.max_interventions, config.recovery_attempt_limits)
        recovery_budget = None
        recovery_validity = None
        cooldown_remaining = 0
        supervisor_calls = 0
        supervisor_exhausted = False

        for step in range(1, config.max_steps + 1):
            call_budget = None
            correction_memory = None
            deadline.check('policy')
            instruction_guard.verify(observation)
            if frozen_contract is not None:
                frozen_contract.verify()
            budget_record = recovery_budget if recovery_sequence is not None else None
            if recovery_sequence is not None and history is not None:
                history.append(observation)
            # A pending recovery owns execution. Do not infer, assess or select
            # another policy action until its last accepted nonterminal packet.
            action = (deadline.call('policy', policy.act, observation) if recovery_sequence is None
                      else proposed_action)
            proposed_action = deepcopy(action)
            proposal_id = f'{episode_id}:{step}'
            pass_response = None
            abstention_response = None
            fallback_cause = None
            fallback = None
            expiry = None
            call_budget = None
            allowance_refused = False
            continuing_recovery = recovery_sequence is not None
            proposal_for_supervisor = (deepcopy(proposed_action)
                                       if supervisor is not None or window_supervisor is not None
                                       else None)
            request_at = clock()
            if observation.captured_monotonic > request_at:
                raise ValueError('observation capture is in the future of the episode clock')
            if recovery_sequence is None and (supervisor is not None or window_supervisor is not None or
                supervisor_decider is not None):
                try:
                    assessment_due = (step - 1) % config.supervisor_interval_actions == 0
                    if assessment_trigger is not None:
                        triggered = deadline.call('assessment_trigger', assessment_trigger, ActionProposal(
                            proposal_id, supervisor_observation(observation),
                            deepcopy(proposed_action)))
                        if type(triggered) is not bool:
                            raise ValueError('assessment_trigger must return bool')
                        assessment_due = assessment_due or triggered
                    if window_supervisor is not None:
                        if history is None:
                            task = supervisor_observation(observation).observation.get('task')
                            history = ObservationWindowBuilder(episode_id, task, settings)
                        history.append(observation)
                    if assessment_due:
                        allowance_refused = supervisor_calls >= config.max_supervisor_calls
                        if allowance_refused:
                            supervisor_exhausted = True
                            fallback_cause = 'supervisor call allowance exhausted'
                        else:
                            # Reserve before invoking user/provider code. Failures, timeouts,
                            # malformed replies and cancellation never refund an attempt.
                            supervisor_calls += 1
                        call_budget = dict(limit=config.max_supervisor_calls,
                                           attempted=supervisor_calls,
                                           admitted=not allowance_refused,
                                           policy=config.supervisor_exhaustion_policy)
                    if allowance_refused:
                        pass
                    elif supervisor_decider is not None and assessment_due:
                        from supervisor_provider import ProviderRequestError
                        attempts = []
                        if config.supervisor_max_retries:
                            call_budget['retry_attempts'] = attempts
                        while True:
                            attempt = dict(started_at=clock(), finished_at=None,
                                           status='pending', error_class=None)
                            attempts.append(attempt)
                            try:
                                response = deadline.call('supervisor', supervisor_decider, ActionProposal(
                                    proposal_id, supervisor_observation(observation),
                                    deepcopy(proposed_action)))
                            except BaseException as exc:
                                attempt.update(finished_at=clock(), status='error')
                                if isinstance(exc, EpisodeDeadlineExceeded):
                                    attempt['status'] = 'timeout'
                                if isinstance(exc, ProviderRequestError):
                                    attempt.update(status=exc.result.status,
                                                   error_class=exc.result.error_class)
                                eligible = (isinstance(exc, ProviderRequestError) and
                                            exc.result.status == 'error' and
                                            exc.result.error_class in config.supervisor_retry_errors)
                                if not eligible or len(attempts) > config.supervisor_max_retries:
                                    raise
                                deadline.check('supervisor')
                                if supervisor_calls >= config.max_supervisor_calls:
                                    supervisor_exhausted = allowance_refused = True
                                    call_budget['admitted'] = False
                                    fallback_cause = 'supervisor call allowance exhausted'
                                    break
                                # The main thread waits; no abandoned worker can initiate a retry.
                                delay_end = clock() + config.supervisor_retry_delay_seconds
                                while clock() < delay_end:
                                    deadline.check('supervisor')
                                    sleep(min(.01, max(0, delay_end - clock())))
                                deadline.check('supervisor')
                                supervisor_calls += 1
                                call_budget['attempted'] = supervisor_calls
                            else:
                                attempt.update(finished_at=clock(), status='response')
                                break
                        if allowance_refused:
                            # Enter the existing guarded fallback/stop path below.
                            raise SupervisorResponseError('supervisor call allowance exhausted')
                        if (type(response) not in (SupervisorPass, SupervisorAbstention) or
                            type(response.episode_id) is not str or
                            type(response.proposal_id) is not str or
                            type(response.observation_sequence) is not int or
                            response.episode_id != episode_id or
                            response.observation_sequence != observation.sequence or
                            response.proposal_id != proposal_id):
                            raise SupervisorResponseError(
                                'supervisor response must reference the current proposal')
                        if not valid_temporal_diagnosis(response.temporal_diagnosis, response.kind):
                            raise SupervisorResponseError('invalid temporal diagnosis')
                        if (type(response) is SupervisorPass and
                                response.suppressed_correction not in (None, 'recovery', 'adjustment')):
                            raise SupervisorResponseError('invalid suppressed correction kind')
                        if type(response) is SupervisorAbstention:
                            if (response.kind != 'abstain' or response.diagnosis != 'unknown' or
                                not valid_abstention_details(response.reason,
                                                             response.evidence_availability)):
                                raise ValueError('invalid supervisor abstention evidence')
                            abstention_response = deepcopy(response)
                        else:
                            pass_response = deepcopy(response)
                    elif window_supervisor is not None:
                        deadline.call('supervisor', window_supervisor, history.snapshot(), proposal_for_supervisor)
                    elif supervisor is not None:
                        deadline.call('supervisor', supervisor, supervisor_observation(observation), proposal_for_supervisor)
                except EpisodeDeadlineExceeded:
                    raise
                except BaseException as exc:
                    if allowance_refused or (baseline_fallback is not None and isinstance(exc, Exception)):
                        # Retain a safe error category, never provider payloads.
                        if not allowance_refused:
                            fallback_cause = f'supervisor unavailable or rejected: {type(exc).__name__}'
                    else:
                        request_finished_at = clock()
                        cumulative_wait += request_finished_at - request_at
                        record_failure(
                            step, observation, deepcopy(proposed_action),
                            StepFailure('supervisor', type(exc).__name__, FailedStepTiming(
                                observation.captured_monotonic, request_at,
                                request_finished_at, None, None, None, cumulative_wait),
                                ActionRecord(proposal_id, deepcopy(proposed_action), None,
                                             None, None,
                                             'rejected' if isinstance(exc, SupervisorResponseError)
                                             else 'unconfirmed',
                                             str(exc) if isinstance(exc, SupervisorResponseError)
                                             else None,
                                             supervisor_call_budget=deepcopy(call_budget))))
                        raise
                response_at = clock()
            else:
                response_at = request_at
            cumulative_wait += response_at - request_at
            if recovery_sequence is not None:
                resolution = ActionResolution('recovery', recovery=recovery_sequence)
            elif action_selector is None or fallback_cause is not None:
                resolution = ActionResolution('pass')
            else:
                try:
                    resolution = deadline.call('selection', action_selector, ActionProposal(
                        proposal_id, observation, deepcopy(proposed_action)))
                    if isinstance(resolution, ActionResolution) and resolution.memory_context is not None:
                        validate_memory(resolution.memory_context)
                        correction_memory = deepcopy(resolution.memory_context)
                    if not isinstance(resolution, ActionResolution) or resolution.kind not in (
                        'pass', 'override', 'reject', 'recovery'
                    ) or (resolution.kind == 'reject' and not resolution.reason) or (
                        resolution.kind != 'reject' and resolution.reason is not None
                    ) or (resolution.kind != 'override' and resolution.action is not None):
                        raise ValueError('invalid action resolution')
                    if ((resolution.kind == 'recovery' and type(resolution.recovery) is not RecoverySequence) or
                            (resolution.kind != 'recovery' and resolution.recovery is not None)):
                        raise ValueError('invalid recovery resolution')
                    if config.correction_mode == 'recovery_only' and resolution.kind == 'override':
                        resolution = ActionResolution(
                            'reject', reason='adjustment disallowed in recovery_only mode')
                    if config.correction_mode == 'adjustment_only' and resolution.kind == 'recovery':
                        resolution = ActionResolution(
                            'reject', reason='recovery disallowed in adjustment_only mode')
                    if resolution.kind == 'reject' and baseline_fallback is not None:
                        fallback_cause = 'correction rejected: ' + resolution.reason
                        resolution = ActionResolution('pass')
                    if (resolution.source_identity is not None and
                            resolution.source_identity !=
                            (episode_id, observation.sequence, proposal_id)):
                        resolution = ActionResolution(
                            'reject', reason='resolution must reference the current proposal')
                    if resolution.kind in ('override', 'recovery') and not callable(
                        getattr(policy, 'resume', None)
                    ):
                        raise NotImplementedError('policy does not support override/resume')
                    if resolution.kind in ('override', 'recovery') and cooldown_remaining:
                        resolution = ActionResolution('reject', reason=
                            f'recovery cooldown: {cooldown_remaining} baseline actions remaining')
                    if resolution.kind == 'recovery':
                        plan = RecoverySequence(**asdict(resolution.recovery))
                        if not callable(recovery_observer):
                            raise ValueError('recovery requires a local observation assessor')
                        if (plan.request['episode_id'] != episode_id or
                                plan.request['proposal_id'] != proposal_id or
                                plan.request['observation_sequence'] != observation.sequence):
                            raise ValueError('recovery must reference the current proposal')
                        if len(plan.actions) > config.max_steps - step + 1:
                            resolution = ActionResolution('reject', reason='insufficient recovery action horizon')
                        else:
                            recovery_sequence, recovery_index = plan, 0
                except EpisodeDeadlineExceeded:
                    raise
                except BaseException as exc:
                    selection_finished_at = clock()
                    cumulative_wait += selection_finished_at - response_at
                    record_failure(
                        step, observation, deepcopy(proposed_action),
                        StepFailure('selection', type(exc).__name__, FailedStepTiming(
                            observation.captured_monotonic, request_at,
                            selection_finished_at, selection_finished_at,
                            None, None, cumulative_wait),
                            ActionRecord(proposal_id, deepcopy(proposed_action), None,
                                         None, None, 'unconfirmed')))
                    raise
                selection_finished_at = clock()
                cumulative_wait += selection_finished_at - response_at
                response_at = selection_finished_at
            if abstention_response is not None:
                fallback_cause = 'supervisor abstained: ' + abstention_response.reason
            if fallback_cause is not None:
                proposal = ActionProposal(proposal_id, deepcopy(observation),
                                          deepcopy(proposed_action))
                fallback = (deadline.call('fallback', baseline_fallback.assess, proposal, fallback_cause, clock)
                            if baseline_fallback is not None and not (allowance_refused and
                                config.supervisor_exhaustion_policy == 'stop') else
                            fallback_evidence(proposal, fallback_cause, clock()))
                checked_at = clock()
                cumulative_wait += checked_at - response_at
                response_at = checked_at
                if fallback['selected'] == 'refuse':
                    resolution = ActionResolution('reject', reason=fallback['reason'])
            # Prepare detached transport data before the final clock sample. No
            # extension callback or queue wait may sit between this gate and step.
            selected_action = deepcopy(recovery_sequence.actions[recovery_index]
                                       if recovery_sequence is not None else
                                       proposed_action if resolution.kind == 'pass' else resolution.action)
            recovery_evidence = (dict(sequence=asdict(recovery_sequence), action_index=recovery_index)
                                 if recovery_sequence is not None else None)
            transport_action = deepcopy(selected_action)
            instruction_guard.verify(observation)
            if frozen_contract is not None:
                frozen_contract.verify()
            deadline.check('dispatch')
            execution_started_at = clock()
            if resolution.kind in ('override', 'recovery'):
                if config.correction_timeout_seconds is not None:
                    if not continuing_recovery:
                        validity = (request_at, observation.captured_monotonic, proposal_id)
                        if resolution.kind == 'recovery':
                            recovery_validity = validity
                    else:
                        validity = recovery_validity
                    expiry = check_expiry(config, *validity, execution_started_at)
                    if not expiry['valid']:
                        resolution = ActionResolution('reject', reason=expiry['reason'])
                if resolution.kind != 'reject' and not continuing_recovery:
                    budget_record = budget.admit(resolution.kind,
                        recovery_sequence.request['tool_name'] if recovery_sequence is not None else None)
                    if not budget_record['admitted']:
                        resolution = ActionResolution('reject', reason=budget_record['reason'])
                    elif recovery_sequence is not None:
                        recovery_budget = budget_record
            if resolution.kind == 'reject':
                cumulative_wait += execution_started_at - response_at
                response_at = execution_started_at
                interruption = interrupt(environment, interruption_contract,
                    ActionProposal(proposal_id, observation, proposed_action),
                    resolution.reason, clock)
                recorder.record_failure(
                    step, observation, deepcopy(proposed_action),
                    StepFailure('selection', 'ProposalRejected', FailedStepTiming(
                        observation.captured_monotonic, request_at, response_at,
                        response_at, None, None, cumulative_wait),
                        ActionRecord(proposal_id, deepcopy(proposed_action), None,
                                     None, None, 'rejected', resolution.reason,
                                     intervention_budget=deepcopy(budget_record),
                                     supervisor_abstention=abstention_response,
                                     correction_memory=correction_memory,
                                     fallback=deepcopy(fallback),
                                     interruption=interruption, correction_expiry=expiry,
                                     supervisor_call_budget=deepcopy(call_budget))))
                stop_reason = ('proposal_rejected' if interruption['confirmed']
                               else 'interruption_failed')
                break
            disposition = 'unmodified' if resolution.kind == 'pass' else 'overridden'
            try:
                result = deadline.call('execution', environment.step, transport_action,
                                       accept_late_return=True)
                if not isinstance(result, StepResult):
                    from execution_failure import ExecutionFailure
                    raise ExecutionFailure('adapter returned no valid step acknowledgement')
            except BaseException as exc:
                execution_finished_at = clock()
                from execution_failure import ExecutionFailure
                # Baseline adapters may omit this capability. Never invent a stop.
                try:
                    failure_contract = interruption_contract or require_interruption(environment)
                except ValueError:
                    interruption = None
                    interruption_diagnostics.append('controller interruption unavailable')
                else:
                    try:
                        interruption = interrupt(environment, failure_contract,
                            ActionProposal(proposal_id, observation, proposed_action),
                            'execution acknowledgement unavailable', clock)
                    except BaseException as stop_error:
                        interruption = None
                        interruption_diagnostics.append(
                            'controller interruption raised ' + type(stop_error).__name__)
                if recovery_evidence is not None:
                    from recovery_monitor import aborted
                    recovery_evidence['check'] = aborted('controller_failure', recovery_index)
                failed_action = ActionRecord(
                    proposal_id, deepcopy(proposed_action), deepcopy(selected_action),
                    None, None, 'unconfirmed', supervisor_pass=pass_response,
                    correction_memory=correction_memory,
                    supervisor_abstention=abstention_response, recovery=recovery_evidence,
                    intervention_budget=deepcopy(budget_record), fallback=deepcopy(fallback),
                    interruption=interruption, correction_expiry=expiry,
                    supervisor_call_budget=deepcopy(call_budget),
                    dispatch=dict(attempted=True,
                                  sent=exc.sent if isinstance(exc, ExecutionFailure) else None,
                                  acknowledged=False, error_type=type(exc).__name__))
                record_failure(
                    step, observation, deepcopy(proposed_action),
                    StepFailure('execution', type(exc).__name__, FailedStepTiming(
                        observation.captured_monotonic, request_at, response_at,
                        response_at, execution_started_at, execution_finished_at,
                        cumulative_wait),
                        deepcopy(failed_action)))
                raise
            execution_finished_at = clock()
            timing = StepTiming(observation.captured_monotonic, request_at,
                                response_at, execution_started_at,
                                execution_finished_at, cumulative_wait)
            acknowledgement = ExecutionAcknowledgement(
                proposal_id,
                result.observation.episode_id if isinstance(result.observation,
                                                            ObservationPacket) else None,
                result.observation.sequence if isinstance(result.observation,
                                                          ObservationPacket) else None)
            result = replace(result, timing=timing, action_record=ActionRecord(
                proposal_id, deepcopy(proposed_action), deepcopy(selected_action),
                deepcopy(selected_action), acknowledgement, disposition,
                supervisor_pass=pass_response,
                correction_memory=correction_memory,
                supervisor_abstention=abstention_response,
                recovery=recovery_evidence, intervention_budget=deepcopy(budget_record),
                fallback=deepcopy(fallback), correction_expiry=expiry,
                supervisor_call_budget=deepcopy(call_budget)))
            acknowledged_actions.append(deepcopy(result.action_record))
            reward_sum += result.reward
            steps = step
            step_timings.append(timing)
            ingestion = ingestor.ingest(result.observation, execution_finished_at)
            if ingestion.accepted:
                # Capture evaluator evidence before any extension callback can
                # fail or mutate its copy of the terminal observation.
                last_observation = deepcopy(result.observation)
                if result.success:
                    task_status = 'success'
                elif result.terminated:
                    task_status = 'failure'
                elif not result.truncated and step == config.max_steps:
                    task_status = 'failure'
                if result.success or result.terminated or result.truncated:
                    terminal_observation = ObservationReference(
                        result.observation.episode_id, result.observation.sequence)
                    terminal_reason = ('success' if result.success else
                                       'terminated' if result.terminated else 'truncated')
            recovery_aborted = False
            expired_after_action = None
            if recovery_sequence is not None:
                from recovery_monitor import aborted, check_recovery
                if not ingestion.accepted:
                    check = aborted('stale_observation', recovery_index + 1)
                elif terminal_reason is not None:
                    check = aborted('episode_terminated', recovery_index + 1)
                    recovery_sequence = None
                else:
                    try:
                        assessment = deadline.call('recovery', recovery_observer, deepcopy(recovery_sequence),
                            recovery_index, supervisor_observation(result.observation))
                    except EpisodeDeadlineExceeded as exc:
                        expired_after_action = exc
                        assessment = None
                    except Exception:
                        assessment = None
                    check = check_recovery(recovery_sequence, recovery_index,
                                           result.observation, assessment, clock())
                if expired_after_action is not None:
                    check = aborted('episode_wall_clock', recovery_index + 1)
                recovery_evidence['check'] = check
                result = replace(result, action_record=replace(result.action_record,
                                 recovery=deepcopy(recovery_evidence)))
                acknowledged_actions[-1] = deepcopy(result.action_record)
                recovery_aborted = check['status'] == 'aborted'
            recorder.record_step(step, observation, deepcopy(selected_action), result,
                                 ingestion)
            if not ingestion.accepted:
                raise ObservationRejected(ingestion)
            if history is not None:
                history.record_action(WindowAction(
                    observation.sequence, proposal_id, proposed_action,
                    selected_action, execution_finished_at))
            success = result.success
            if expired_after_action is not None:
                raise expired_after_action
            if terminal_reason is None:
                deadline.check('after_execution')
            if on_step is not None:
                deadline.call('on_step', on_step, step, result)
            instruction_guard.verify(result.observation)
            if frozen_contract is not None:
                frozen_contract.verify()
            if terminal_reason is not None:
                stop_reason = terminal_reason
                break
            if recovery_aborted:
                stop_reason = 'recovery_aborted'
                break
            observation = result.observation
            recovery_finished = False
            if recovery_sequence is not None:
                recovery_index += 1
                recovery_finished = recovery_index == len(recovery_sequence.actions)
                if recovery_finished:
                    recovery_sequence = None
                    cooldown_remaining = config.recovery_cooldown_actions
            elif resolution.kind == 'pass':
                cooldown_remaining = max(0, cooldown_remaining - 1)
            if (resolution.kind == 'override' or recovery_finished) and step < config.max_steps:
                # Only accepted, nonterminal evidence can seed the next proposal.
                # Keep adapter mutation separate from recorded environment evidence.
                deadline.call('resume', policy.resume, deepcopy(observation))

    except BaseException as exc:
        if isinstance(exc, FrozenContractViolation) and last_observation is not None:
            stop = None
            try:
                contract = interruption_contract or require_interruption(environment)
                stop = interrupt(environment, contract,
                    ActionProposal(f'{episode_id}:{steps + 1}', last_observation, None),
                    str(exc), clock)
                interruption_diagnostics.append('frozen contract stop confirmed: ' + str(stop['confirmed']))
            except Exception as stop_error:
                interruption_diagnostics.append('frozen contract stop unavailable: ' + type(stop_error).__name__)
            now = clock()
            failed_action = ActionRecord(f'{episode_id}:{steps + 1}', None, None,
                                         None, None, 'rejected', str(exc), interruption=stop)
            record_failure(steps + 1, last_observation, None,
                StepFailure('selection', type(exc).__name__, FailedStepTiming(
                    last_observation.captured_monotonic, now, now, now, None, None, cumulative_wait),
                    failed_action))
        if (isinstance(exc, EpisodeDeadlineExceeded) and failed_action is None
                and terminal_reason is None):
            # No further policy, supervisor or recovery work can reach dispatch.
            # The last accepted observation binds the stop even if no proposal exists.
            stop = interrupt(environment, interruption_contract,
                ActionProposal(f'{episode_id}:{steps + 1}', last_observation, None),
                'episode wall-clock limit exhausted', clock)
            if exc.stage in ('supervisor', 'assessment_trigger'):
                cumulative_wait += max(0., clock() - request_at)
            elif exc.stage in ('selection', 'fallback'):
                cumulative_wait += max(0., clock() - response_at)
            pending = call_budget if steps < step else None
            wall_clock_limit = deadline.evidence(exc.stage, stop, deepcopy(pending))
            stop_reason = 'wall_clock_limit' if stop['confirmed'] else 'interruption_failed'
            success = False
            recovery_sequence = None
        else:
            if recorder_started:
                try:
                    recorder.finish()
                except BaseException as cleanup_error:
                    interruption_diagnostics.append(
                        f'{type(cleanup_error).__name__}: {cleanup_error}')
            exc.episode_interruption = EpisodeInterruption(
                episode_id, exception_stop_reason(exc),
                type(exc).__name__, steps, reward_sum, last_observation,
                tuple(acknowledged_actions), tuple(interruption_diagnostics), task_status,
                pre_start_failure=(PreStartFailure(
                    startup_stage, config.seed, policy_reset_completed,
                    environment_reset_completed) if startup_stage is not None else None),
                failed_action=deepcopy(failed_action),
                terminal_observation=terminal_observation, terminal_reason=terminal_reason)
            raise

    rollout_seconds = clock() - rollout_start
    artifacts = {}
    artifact_status = 'incomplete'
    diagnostics = ()
    try:
        artifacts = dict(recorder.finish())
        artifact_status = 'completed'
    except Exception as exc:
        # Task outcome is already known. Encoding failure must not erase it or
        # present partial paths as a successfully finalized evidence package.
        diagnostics = (f'{type(exc).__name__}: {exc}',)
    except BaseException as exc:
        exc.episode_interruption = EpisodeInterruption(
            episode_id, exception_stop_reason(exc), type(exc).__name__,
            steps, reward_sum, last_observation, tuple(acknowledged_actions),
            (f'{type(exc).__name__}: {exc}',),
            'success' if success else 'failure' if stop_reason in
            ('terminated', 'step_limit') else 'unknown',
            terminal_observation=terminal_observation, terminal_reason=terminal_reason)
        raise
    return EpisodeOutcome(episode_id, success, steps, stop_reason, reward_sum,
                          rollout_seconds, artifacts, cumulative_wait,
                          tuple(step_timings), terminal_observation,
                          artifact_status, diagnostics,
                          dict(limit=config.max_supervisor_calls, attempted=supervisor_calls,
                               exhausted=supervisor_exhausted,
                               policy=config.supervisor_exhaustion_policy), wall_clock_limit,
                          (dict(started_at=deadline.started_at, deadline=deadline.deadline)
                           if config.max_episode_seconds is not None else None))
