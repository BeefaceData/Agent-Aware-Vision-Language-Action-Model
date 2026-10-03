"""Baseline episode orchestration with injectable policy, environment and recorder.

The caller owns the adapters and releases any resources they hold. Artifact
finalization failures are returned separately from the observed task outcome.
Environment step results contain evaluator outcomes, which are never sent to the
policy through this interface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from math import isfinite
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from observation_window import ObservationWindow, WindowSettings


@dataclass(frozen=True)
class EpisodeConfig:
    seed: int
    max_steps: int


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
    return ObservationPacket(packet.episode_id, packet.sequence, packet.captured_at,
                             allowed, packet.captured_monotonic, references,
                             state_capture)


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
class ActionResolution:
    """A deterministic execution choice; rejection does not call the environment."""

    kind: Literal['pass', 'override', 'reject']
    action: Any = None
    reason: str | None = None


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


class PolicyAdapter(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: ObservationPacket) -> Any: ...


class EnvironmentAdapter(Protocol):
    """A returned StepResult acknowledges completion of the adapter step call."""

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
) -> EpisodeOutcome:
    """Run one attempt and return task outcome and artifact finalization status.

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
    Policy inference, recorder and ``on_step`` time are excluded from that wait.
    ``window_supervisor(window, proposed_action)`` is the temporal alternative to
    ``supervisor``. It receives bounded sanitized history and acknowledged prior
    actions, using ``window_settings`` (defaults to eight packets/eight actions).
    Both callbacks are observation-only and cannot be configured together.
    """
    if type(config.max_steps) is not int or config.max_steps <= 0:
        raise ValueError('max_steps must be a positive integer')
    if supervisor is not None and window_supervisor is not None:
        raise ValueError('choose one supervisor callback')
    if window_settings is not None and window_supervisor is None:
        raise ValueError('window_settings requires window_supervisor')

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
    steps = 0
    interruption_diagnostics = []
    task_status = 'unknown'

    def record_failure(*args):
        try:
            recorder.record_failure(*args)
        except BaseException as recording_error:
            interruption_diagnostics.append(
                f'{type(recording_error).__name__}: {recording_error}')

    try:
        policy.reset()
        observation = environment.reset(config.seed, episode_id)
        initial_ingestion = ingestor.ingest(observation, clock())
        if not initial_ingestion.accepted:
            raise ObservationRejected(initial_ingestion)
        last_observation = deepcopy(observation)
        recorder_started = True
        recorder.begin(observation)
        rollout_start = clock()
        reward_sum = 0.0
        stop_reason = 'step_limit'
        success = False
        steps = 0
        cumulative_wait = 0.0
        step_timings: list[StepTiming] = []
        terminal_observation = None

        for step in range(1, config.max_steps + 1):
            action = policy.act(observation)
            proposed_action = deepcopy(action)
            proposal_id = f'{episode_id}:{step}'
            proposal_for_supervisor = (deepcopy(proposed_action)
                                       if supervisor is not None or window_supervisor is not None
                                       else None)
            request_at = clock()
            if observation.captured_monotonic > request_at:
                raise ValueError('observation capture is in the future of the episode clock')
            if supervisor is not None or window_supervisor is not None:
                try:
                    if window_supervisor is not None:
                        if history is None:
                            task = supervisor_observation(observation).observation.get('task')
                            history = ObservationWindowBuilder(episode_id, task, settings)
                        history.append(observation)
                        window_supervisor(history.snapshot(), proposal_for_supervisor)
                    else:
                        supervisor(supervisor_observation(observation), proposal_for_supervisor)
                except BaseException as exc:
                    request_finished_at = clock()
                    cumulative_wait += request_finished_at - request_at
                    record_failure(
                        step, observation, deepcopy(proposed_action),
                        StepFailure('supervisor', type(exc).__name__, FailedStepTiming(
                            observation.captured_monotonic, request_at,
                            request_finished_at, None, None, None, cumulative_wait),
                            ActionRecord(proposal_id, deepcopy(proposed_action), None,
                                         None, None, 'unconfirmed')))
                    raise
                response_at = clock()
            else:
                response_at = request_at
            cumulative_wait += response_at - request_at
            if action_selector is None:
                resolution = ActionResolution('pass')
            else:
                try:
                    resolution = action_selector(ActionProposal(
                        proposal_id, observation, deepcopy(proposed_action)))
                    if not isinstance(resolution, ActionResolution) or resolution.kind not in (
                        'pass', 'override', 'reject'
                    ) or (resolution.kind == 'reject' and not resolution.reason) or (
                        resolution.kind != 'reject' and resolution.reason is not None
                    ) or (resolution.kind != 'override' and resolution.action is not None):
                        raise ValueError('invalid action resolution')
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
            if resolution.kind == 'reject':
                recorder.record_failure(
                    step, observation, deepcopy(proposed_action),
                    StepFailure('selection', 'ProposalRejected', FailedStepTiming(
                        observation.captured_monotonic, request_at, response_at,
                        response_at, None, None, cumulative_wait),
                        ActionRecord(proposal_id, deepcopy(proposed_action), None,
                                     None, None, 'rejected', resolution.reason)))
                stop_reason = 'proposal_rejected'
                break
            selected_action = deepcopy(proposed_action if resolution.kind == 'pass'
                                       else resolution.action)
            disposition = 'unmodified' if resolution.kind == 'pass' else 'overridden'
            execution_started_at = clock()
            try:
                result = environment.step(deepcopy(selected_action))
            except BaseException as exc:
                execution_finished_at = clock()
                record_failure(
                    step, observation, deepcopy(proposed_action),
                    StepFailure('execution', type(exc).__name__, FailedStepTiming(
                        observation.captured_monotonic, request_at, response_at,
                        response_at, execution_started_at, execution_finished_at,
                        cumulative_wait),
                        ActionRecord(proposal_id, deepcopy(proposed_action),
                                     deepcopy(selected_action), None, None,
                                     'unconfirmed')))
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
                deepcopy(selected_action), acknowledgement, disposition))
            acknowledged_actions.append(deepcopy(result.action_record))
            reward_sum += result.reward
            steps = step
            step_timings.append(timing)
            ingestion = ingestor.ingest(result.observation, execution_finished_at)
            if ingestion.accepted:
                last_observation = deepcopy(result.observation)
                if result.success:
                    task_status = 'success'
                elif result.terminated:
                    task_status = 'failure'
                elif not result.truncated and step == config.max_steps:
                    task_status = 'failure'
            recorder.record_step(step, observation, deepcopy(selected_action), result,
                                 ingestion)
            if not ingestion.accepted:
                raise ObservationRejected(ingestion)
            if history is not None:
                history.record_action(WindowAction(
                    observation.sequence, proposal_id, proposed_action,
                    selected_action, execution_finished_at))
            success = result.success
            if on_step is not None:
                on_step(step, result)
            if success or result.terminated or result.truncated:
                terminal_observation = ObservationReference(
                    result.observation.episode_id, result.observation.sequence)
                stop_reason = ('success' if success else
                               'terminated' if result.terminated else 'truncated')
                break
            observation = result.observation

    except BaseException as exc:
        if recorder_started:
            try:
                recorder.finish()
            except BaseException as cleanup_error:
                interruption_diagnostics.append(
                    f'{type(cleanup_error).__name__}: {cleanup_error}')
        exc.episode_interruption = EpisodeInterruption(
            episode_id, exception_stop_reason(exc),
            type(exc).__name__, steps, reward_sum, last_observation,
            tuple(acknowledged_actions), tuple(interruption_diagnostics), task_status)
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
            ('terminated', 'step_limit') else 'unknown')
        raise
    return EpisodeOutcome(episode_id, success, steps, stop_reason, reward_sum,
                          rollout_seconds, artifacts, cumulative_wait,
                          tuple(step_timings), terminal_observation,
                          artifact_status, diagnostics)
