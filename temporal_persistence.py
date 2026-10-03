"""Caller-owned persistence gate over the shared temporal diagnosis contract."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json

from episode_harness import SupervisorAbstention
from supervisor_response import SupervisorResponseDecoder


@dataclass(frozen=True)
class PersistenceSettings:
    required_windows: int
    assessment_stride: int
    calibration_id: str

    def __post_init__(self):
        if (type(self.required_windows) is not int or
                not 2 <= self.required_windows <= 32 or
                type(self.assessment_stride) is not int or self.assessment_stride < 1 or
                type(self.calibration_id) is not str or
                not self.calibration_id.strip() or len(self.calibration_id) > 128):
            raise ValueError('declare 2..32 windows, positive stride and calibration identity')


@dataclass(frozen=True)
class PersistenceVerdict:
    episode_id: str
    proposal_id: str
    observation_sequence: int
    family: str | None
    supporting_windows: tuple[int, ...]
    eligible: bool
    reason: str
    settings: PersistenceSettings


class TemporalPersistenceGate:
    """Assess once per declared stride, using adapter-owned observation windows.

    Any adapter emitting the shared structured temporal diagnosis can participate.
    Rolling windows may overlap, but each must cite its new current observation.
    References are resolved before counting; their semantic truth is not verified.
    Call for quiet/unknown assessments too: skipping them cannot preserve a streak.
    Eligibility is only for consideration, never execution or detector readiness.
    """

    def __init__(self, settings):
        if type(settings) is not PersistenceSettings:
            raise ValueError('PersistenceSettings required')
        self.settings = settings
        self.verdict = None
        self._last_time = None

    def eligible_for(self, proposal):
        """A result cannot be reused for another proposal or observation."""
        v = self.verdict
        return bool(v and v.eligible and
                    (v.episode_id, v.proposal_id, v.observation_sequence) ==
                    (proposal.observation.episode_id, proposal.proposal_id,
                     proposal.observation.sequence))

    def consider(self, proposal, candidate):
        """Invoke a correction-candidate producer only for the current eligible proposal.

        The returned candidate still needs all independent correction checks.
        This method neither interprets nor executes it.
        """
        if not callable(candidate):
            raise ValueError('candidate producer must be callable')
        if not self.eligible_for(proposal):
            return None
        return candidate(deepcopy(proposal))

    def assess(self, response, proposal, *, window):
        """Return a traceable pass/abstention, retaining a bounded verdict.

        Missing or malformed input clears eligibility even if decoding raises.
        The harness retains the verdict in the diagnosis summary or abstention
        reason, including settings and supporting window endpoints.
        """
        previous, previous_time = self.verdict, self._last_time
        self.verdict = None
        self._last_time = None
        if window is None:
            raise ValueError('declared observation window required')
        decision = SupervisorResponseDecoder().decode(response, proposal, window=window)
        temporal = deepcopy(decision.temporal_diagnosis)
        packet = proposal.observation
        family = temporal['category'] if temporal else None
        failure = family in ('stall', 'suspected_missed_grasp', 'suspected_lost_grasp')
        current_evidence = temporal and any(
            item['observation_sequence'] == packet.sequence for item in temporal['evidence'])
        continuous = (previous is not None and previous.episode_id == packet.episode_id and
                      packet.sequence == previous.observation_sequence + self.settings.assessment_stride and
                      previous_time is not None and packet.captured_monotonic > previous_time)
        windows = ()
        if not failure or decision.kind == 'abstain':
            reason = 'no_consistent_failure'
        elif window.missing_intervals or not current_evidence:
            reason = 'incomplete_or_reused_evidence'
        else:
            if continuous and previous.family == family:
                windows = previous.supporting_windows
            windows = (*windows, packet.sequence)[-self.settings.required_windows:]
            reason = ('persistence_passed' if len(windows) >= self.settings.required_windows
                      else 'insufficient_persistent_windows')
        eligible = reason == 'persistence_passed'
        self.verdict = PersistenceVerdict(packet.episode_id, proposal.proposal_id,
            packet.sequence, family, windows, eligible, reason, self.settings)
        self._last_time = packet.captured_monotonic
        report = json.dumps(dict(persistence=reason, eligible=eligible, family=family,
            windows=windows, required_windows=self.settings.required_windows,
            assessment_stride=self.settings.assessment_stride,
            calibration_id=self.settings.calibration_id), separators=(',', ':'))
        if temporal is not None:
            temporal['summary'] = (report + ' ' + temporal['summary'])[:2000]
        if eligible or family == 'progress':
            return replace(decision, temporal_diagnosis=temporal)
        if temporal is not None:
            temporal['category'] = 'unknown'
        return SupervisorAbstention(packet.episode_id, packet.sequence, proposal.proposal_id,
            report, {source: 'unknown' for source in ('main', 'wrist', 'robot_state')},
            temporal_diagnosis=temporal)
