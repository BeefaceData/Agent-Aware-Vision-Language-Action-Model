"""Local recovery checks from host-assessed deployable evidence, not task success."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite


COMPLETION = ('gripper_open_confirmed', 'retreat_target_reached')
ABORT = ('clearance_unverified', 'stale_observation', 'controller_failure',
         'action_limit_reached')


def _nonnegative(value):
    try:
        return type(value) in (int, float) and isfinite(value) and value >= 0
    except OverflowError:
        return False


def validate_monitor(contract):
    if (type(contract) is not dict or set(contract) != {
            'completion_conditions', 'abort_conditions', 'required_observations',
            'max_age_seconds', 'abort_path'} or
            any(type(contract[key]) not in (list, tuple) or
                any(type(value) is not str for value in contract[key]) or
                len(set(contract[key])) != len(contract[key])
                for key in ('completion_conditions', 'abort_conditions', 'required_observations')) or
            tuple(contract['completion_conditions']) != COMPLETION or
            set(contract['abort_conditions']) != set(ABORT) or
            not contract['required_observations'] or
            set(contract['required_observations']) - {'main', 'wrist', 'robot_state'} or
            not _nonnegative(contract['max_age_seconds']) or
            contract['abort_path'] != 'stop_episode'):
        raise ValueError('unsupported recovery monitoring contract')


@dataclass(frozen=True)
class RecoveryAssessment:
    """Trusted local assessor result bound to the current packet and envelope.

    Every condition uses all declared required observations. None is unknown.
    This interface does not implement perception or certify an assessor.
    """

    episode_id: str
    observation_sequence: int
    envelope_id: str
    gripper_open_confirmed: bool | None
    retreat_target_reached: bool | None
    clearance_unverified: bool | None

    def __post_init__(self):
        if (any(type(v) is not str or not v for v in (self.episode_id, self.envelope_id)) or
                type(self.observation_sequence) is not int or self.observation_sequence < 0 or
                any(v is not None and type(v) is not bool for v in (
                    self.gripper_open_confirmed, self.retreat_target_reached,
                    self.clearance_unverified))):
            raise ValueError('invalid local recovery assessment')


def aborted(reason, executed_actions, *, assessment=None, checked_at=None, evidence=()):
    return dict(status='aborted', reason=reason, path='stop_episode',
                executed_actions=executed_actions, assessment=assessment,
                checked_at=checked_at, evidence=list(evidence))


def check_recovery(plan, action_index, packet, assessment, now):
    """Check after each acknowledged action; never use evaluator outcomes.

    No retries are generated. At the tool limit an unmet completion condition
    reports exhaustion; earlier failure aborts immediately with its condition.
    """
    from episode_harness import check_observation_freshness, supervisor_observation

    validate_monitor(plan.monitor)
    current = supervisor_observation(packet)
    count = action_index + 1
    evidence = [asdict(item) for item in check_observation_freshness(
        current, now, dict.fromkeys(plan.monitor['required_observations'],
                                   plan.monitor['max_age_seconds']))]
    data = asdict(assessment) if type(assessment) is RecoveryAssessment else None

    def abort(reason):
        return aborted(reason, count, assessment=data, checked_at=now, evidence=evidence)

    if any(item['status'] != 'fresh' or item['observation_sequence'] != current.sequence
           for item in evidence):
        return abort('stale_observation')
    if data is None:
        return abort('missing_assessment')
    if (assessment.episode_id != current.episode_id or
            assessment.observation_sequence != current.sequence or
            assessment.envelope_id != plan.envelope_id):
        return abort('assessment_identity_mismatch')
    if assessment.clearance_unverified is not False:
        return abort('clearance_unverified')
    required = COMPLETION[:action_index + 1]
    failed = next((name for name in required if getattr(assessment, name) is not True), None)
    if failed:
        return abort('action_limit_reached' if count >= plan.action_limit else failed)
    return dict(status='completed' if count == len(plan.actions) else 'continuing',
                reason=None, path='resume' if count == len(plan.actions) else 'continue',
                executed_actions=count, assessment=deepcopy(data), checked_at=now,
                evidence=evidence)
