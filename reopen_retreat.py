"""Code-defined two-command reopen/retreat, for a reviewed single Panda envelope."""

from dataclasses import asdict, dataclass
from math import isclose, isfinite

from episode_harness import ActionResolution, RecoverySequence
from recovery_eligibility import ReopenRetreatEligibility
from translation_conversion import LiberoTranslationConverter


@dataclass(frozen=True)
class ReopenRetreatControl:
    """Trusted adapter semantics, never inferred from a supervisor response.

    The reviewed envelope binds target, world direction, gripper opening value,
    and the zero relative pose command. No gripper sign is assumed by default.
    Supplying this contract does not establish physical safety or readiness.
    """

    envelope_id: str
    target: str
    direction_world: tuple[float, float, float]
    gripper_open_native: float

    def __post_init__(self):
        try:
            valid = (type(self.direction_world) in (tuple, list) and
                     len(self.direction_world) == 3 and
                     all(type(v) in (float, int) and isfinite(v) for v in self.direction_world) and
                     isclose(sum(v * v for v in self.direction_world), 1., rel_tol=0, abs_tol=1e-12) and
                     type(self.gripper_open_native) in (float, int) and
                     isfinite(self.gripper_open_native) and -1 <= self.gripper_open_native <= 1)
        except OverflowError:
            valid = False
        if (not valid or any(type(v) is not str or not v.strip()
                             for v in (self.envelope_id, self.target))):
            raise ValueError('invalid reviewed reopen/retreat control contract')
        object.__setattr__(self, 'direction_world', tuple(self.direction_world))


class ReopenRetreatExecutor:
    """Select a bounded sequence; the harness remains the only environment caller.

    The host must enforce readiness, expiry and resource authorization before
    calling resolve. Eligibility is checked synchronously for each new request.
    Use one instance per execution session; consumed proposals cannot be retried.
    Commands request opening then a single relative retreat with the gripper
    open. Command acknowledgement is not proof of local physical completion.
    """

    def __init__(self, gate: ReopenRetreatEligibility,
                 converter: LiberoTranslationConverter, control: ReopenRetreatControl):
        if (type(gate) is not ReopenRetreatEligibility or
                type(converter) is not LiberoTranslationConverter or
                type(control) is not ReopenRetreatControl or
                gate.envelope_id != control.envelope_id):
            raise ValueError('matching reviewed eligibility and control contracts required')
        self._gate, self._converter, self._control = gate, converter, control
        self._consumed = set()

    def resolve(self, proposal, request, *, scene, now_monotonic, window=None):
        """Return rejection or two validated commands, with detached provenance."""
        try:
            selection = self._gate.check(request, proposal, scene=scene,
                                         now_monotonic=now_monotonic, window=window)
            identity = (selection.request.episode_id, selection.request.proposal_id)
            if identity in self._consumed:
                raise ValueError('recovery already consumed for this proposal')
            if selection.tool.action_limit < 2:
                raise ValueError('reopen/retreat requires two actions within the tool limit')
            distance = dict(selection.request.parameters)['retreat_m']
            if distance < 0:
                raise ValueError('retreat distance must be nonnegative')
            names = ('translation_x', 'translation_y', 'translation_z')
            adjustment = dict(kind='adjustment', scope='single_action',
                episode_id=selection.request.episode_id,
                observation_sequence=selection.request.observation_sequence,
                proposal_id=selection.request.proposal_id,
                decision_id=selection.request.decision_id, target=self._control.target,
                frame='world', units=dict.fromkeys(names, 'm'),
                residual=dict(zip(names, (distance * d for d in self._control.direction_world))))
            converted = self._converter.convert(adjustment, proposal)
            opening = (0.,) * 6 + (self._control.gripper_open_native,)
            retreat = converted.native_residual[:3] + (0.,) * 3 + (self._control.gripper_open_native,)
            for action in (opening, retreat):
                if any(not isfinite(v) or not -1 <= v <= 1 for v in action):
                    raise ValueError('recovery command outside native bounds')
            plan = RecoverySequence((opening, retreat), selection.tool.action_limit,
                asdict(selection.request), self._control.envelope_id, converted.evidence_sha256)
        except ValueError as exc:
            return ActionResolution('reject', reason=str(exc))
        self._consumed.add(identity)
        return ActionResolution('recovery', recovery=plan)
