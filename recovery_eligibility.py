"""Fail-closed scene eligibility for the predefined reopen-and-retreat tool."""

from dataclasses import asdict, dataclass
from math import isfinite

from correction_validation import CorrectionValidator
from episode_harness import (ActionProposal, check_observation_freshness,
                             supervisor_observation)
from observation_window import ObservationWindow
from recovery_registry import ResolvedRecovery
from supervisor_recovery import RecoveryEvidenceReference, SupervisorRecoveryRequest
from supervisor_response import SupervisorResponseDecoder


def _nonnegative(value):
    try:
        return type(value) in (int, float) and isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class RecoverySceneAssessment:
    """Host assessment from deployable observations, never supervisor JSON.

    The envelope identity names reviewed target/frame/direction and gripper
    sweep semantics. Clearance covers that entire sweep and retreat corridor,
    not a point distance. None means unknown. The host must retain the assessor
    and envelope provenance; this data structure is not a perception algorithm.
    """

    episode_id: str
    observation_sequence: int
    proposal_id: str
    envelope_id: str
    diagnosis: str
    possible_held_payload: bool | None
    gripper_sweep_clear: bool | None
    clear_retreat_m: float | None
    evidence: tuple[RecoveryEvidenceReference, ...]

    def __post_init__(self):
        if (type(self.evidence) not in (tuple, list) or not self.evidence or
                any(type(ref) is not RecoveryEvidenceReference for ref in self.evidence)):
            raise ValueError('scene evidence references required')
        object.__setattr__(self, 'evidence', tuple(self.evidence))


@dataclass(frozen=True)
class ReopenRetreatEligibility:
    """Executor-owned gate; success grants no execution or readiness authority.

    Construct the validator, envelope identity and sensor age limit from trusted
    host configuration. An envelope must bind the selected tool's arm, frame,
    retreat direction and opening sweep. There is no default robot envelope.
    """

    validator: CorrectionValidator
    envelope_id: str
    max_age_seconds: float

    def __post_init__(self):
        if (type(self.validator) is not CorrectionValidator or
                type(self.envelope_id) is not str or not self.envelope_id.strip() or
                not _nonnegative(self.max_age_seconds)):
            raise ValueError('invalid recovery eligibility configuration')

    def check(self, request: dict | SupervisorRecoveryRequest,
              proposal: ActionProposal, *, scene: RecoverySceneAssessment,
              now_monotonic: float, window: ObservationWindow | None = None
              ) -> ResolvedRecovery:
        """Revalidate request, resolve evidence, and require known scene clearance.

        Raises ValueError with an inspectable reason on ineligibility. Current
        required inputs must have sensor capture timestamps. Historical request
        references must resolve in the supplied bounded window; scene evidence
        must cite the current packet. Recheck immediately before any future
        execution; returned selections must never be cached as permission.
        """
        selection = self.validator.validate(request, proposal)
        if (type(selection) is not ResolvedRecovery or
                selection.tool.name != 'reopen_and_retreat' or
                set(dict(selection.request.parameters)) != {'retreat_m'}):
            raise ValueError('unsupported recovery eligibility contract')
        required = set(selection.tool.required_observations)
        if not required or required - {'main', 'wrist', 'robot_state'}:
            raise ValueError('unsupported required recovery observation')
        if (type(scene) is not RecoverySceneAssessment or
                type(scene.observation_sequence) is not int or
                scene.episode_id != proposal.observation.episode_id or
                scene.observation_sequence != proposal.observation.sequence or
                scene.proposal_id != proposal.proposal_id):
            raise ValueError('scene assessment must reference the current proposal')
        if scene.envelope_id != self.envelope_id:
            raise ValueError('scene envelope does not match reviewed local envelope')

        # Reuse the public diagnosis boundary to resolve references against the
        # exact permitted context, including source availability and identity.
        def resolve(references):
            response = dict(
                kind='pass', episode_id=selection.request.episode_id,
                observation_sequence=selection.request.observation_sequence,
                proposal_id=selection.request.proposal_id,
                temporal_diagnosis=dict(category='suspected_missed_grasp',
                    summary='Recovery eligibility evidence', evidence=[
                        {**asdict(ref), 'description': 'Host eligibility reference'}
                        for ref in references]))
            SupervisorResponseDecoder().decode(response, proposal, window=window)

        resolve(selection.request.evidence)
        resolve(scene.evidence)
        if (len(set(scene.evidence)) != len(scene.evidence) or
                any(ref.observation_sequence != proposal.observation.sequence
                    for ref in scene.evidence) or
                not required.issubset({ref.source for ref in scene.evidence})):
            raise ValueError('current declared scene evidence required')
        current = supervisor_observation(proposal.observation)
        scene_sources = required | {ref.source for ref in scene.evidence}
        freshness = check_observation_freshness(
            current, now_monotonic, {name: self.max_age_seconds for name in scene_sources})
        if any(item.status != 'fresh' or
               item.observation_sequence != current.sequence for item in freshness):
            raise ValueError('recovery observations are missing, stale or unverified')
        if scene.diagnosis != 'suspected_missed_grasp':
            raise ValueError('missed-grasp evidence required')
        if scene.possible_held_payload is not False:
            raise ValueError('possible or uncertain held payload')
        if scene.gripper_sweep_clear is not True:
            raise ValueError('gripper clearance is unknown or insufficient')
        if (not _nonnegative(scene.clear_retreat_m) or
                scene.clear_retreat_m < dict(selection.request.parameters)['retreat_m']):
            raise ValueError('retreat clearance is unknown or insufficient')
        return selection
