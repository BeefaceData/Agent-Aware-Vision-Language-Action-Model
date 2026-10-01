"""Baseline episode orchestration with injectable policy, environment and recorder.

The caller owns the adapters and releases any resources they hold. The recorder
finalizes its artifacts on success; the caller handles cleanup on failure.
Environment step results contain evaluator outcomes, which are never sent to the
policy through this interface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol
from uuid import uuid4


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
) -> EpisodeOutcome:
    """Run one attempt and return its outcome after artifacts are finalized.

    ``policy`` provides reset/act(packet); ``environment`` provides
    reset(seed, episode_id)/step(action); ``recorder`` provides begin(packet),
    record_step(step, source_packet, action, result, ingestion),
    record_failure(step, source_packet, action, failure), and finish()
    -> artifact references. Rejected result packets and failed calls are recorded
    before raising; failures do not produce a completed outcome.
    The caller releases adapter resources on any exception. ``clock`` must be
    monotonic and share a timebase with each packet's ``captured_monotonic``.
    ``supervisor`` observes the packet and a copy of the proposed action before
    execution; it has no execution authority here. ``action_selector`` is an
    injected, deterministic execution choice for replay or a validated executor.
    Its input and output actions are copied so neither it nor the environment can
    mutate the policy proposal or recorded evidence. Its rejection is recorded and
    ends the attempt without an environment call. Supervisor and selector wait is
    measured. Overrides must come from a separately validated executor.
    Policy inference, recorder and ``on_step`` time are excluded from that wait.
    """
    if config.max_steps <= 0:
        raise ValueError('max_steps must be positive')

    episode_id = uuid4().hex
    ingestor = ObservationIngestor(episode_id)
    policy.reset()
    observation = environment.reset(config.seed, episode_id)
    initial_ingestion = ingestor.ingest(observation, clock())
    if not initial_ingestion.accepted:
        raise ObservationRejected(initial_ingestion)
    recorder.begin(observation)
    rollout_start = clock()
    reward_sum = 0.0
    stop_reason = 'step_limit'
    success = False
    steps = 0
    cumulative_wait = 0.0
    step_timings: list[StepTiming] = []

    for step in range(1, config.max_steps + 1):
        action = policy.act(observation)
        proposed_action = deepcopy(action)
        proposal_id = f'{episode_id}:{step}'
        proposal_for_supervisor = deepcopy(proposed_action) if supervisor is not None else None
        request_at = clock()
        if observation.captured_monotonic > request_at:
            raise ValueError('observation capture is in the future of the episode clock')
        if supervisor is not None:
            try:
                supervisor(observation, proposal_for_supervisor)
            except BaseException as exc:
                request_finished_at = clock()
                cumulative_wait += request_finished_at - request_at
                recorder.record_failure(
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
                recorder.record_failure(
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
            recorder.record_failure(
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
        step_timings.append(timing)
        ingestion = ingestor.ingest(result.observation, execution_finished_at)
        recorder.record_step(step, observation, deepcopy(selected_action), result,
                             ingestion)
        if not ingestion.accepted:
            raise ObservationRejected(ingestion)
        reward_sum += result.reward
        steps = step
        success = result.success
        if on_step is not None:
            on_step(step, result)
        if success or result.terminated or result.truncated:
            stop_reason = ('success' if success else
                           'terminated' if result.terminated else 'truncated')
            break
        observation = result.observation

    rollout_seconds = clock() - rollout_start
    artifacts = recorder.finish()
    return EpisodeOutcome(episode_id, success, steps, stop_reason, reward_sum,
                          rollout_seconds, artifacts, cumulative_wait,
                          tuple(step_timings))
