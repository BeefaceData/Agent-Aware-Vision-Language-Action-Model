"""Verified LIBERO translation conversion; returns data, never executes actions."""

from dataclasses import dataclass, field
import hashlib
import json
from math import isclose, isfinite

from correction_validation import CorrectionValidator
from episode_harness import ActionProposal
from recovery_registry import RecoveryRegistry
from supervisor_adjustment import (AdjustmentComponent, AdjustmentRequestDecoder,
                                   SupervisorAdjustmentRequest)


_COMPONENTS = ('translation_x', 'translation_y', 'translation_z')


def _controller_contract():
    # Deliberately one verified mapping, not a general frame-transform engine.
    return dict(name='OSC_POSE', robot='OnTheGroundPanda',
                reference_frame='world', translation_unit='m',
                input_min=[-1.0] * 6, input_max=[1.0] * 6,
                output_min=[-0.05] * 3 + [-0.5] * 3,
                output_max=[0.05] * 3 + [0.5] * 3,
                metres_per_native_translation_unit=[0.05] * 3,
                native_translation_indices=[0, 1, 2],
                native_action_dimension=7, use_delta=True, impedance_mode='fixed')


def _numbers(values, length):
    try:
        return (type(values) in (list, tuple) and len(values) == length and
                all(type(v) in (int, float) and isfinite(v) for v in values))
    except OverflowError:
        return False


def _check_controller(controller):
    if type(controller) is not dict:
        raise ValueError('missing controller contract')
    for key, expected in _controller_contract().items():
        actual = controller.get(key)
        if (actual != expected or type(actual) is not type(expected) or
                (type(expected) is list and not _numbers(actual, len(expected)))):
            raise ValueError('unsupported controller contract: ' + key)


def _verify(report):
    if type(report) is not dict or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('unsupported controller evidence')
    _check_controller(report.get('controller'))
    verification = report.get('verification')
    if (type(verification) is not dict or verification.get('passed') is not True or
            verification.get('components') != list(_COMPONENTS)):
        raise ValueError('controller evidence did not pass')
    probes = verification.get('probes')
    if type(probes) is not list or len(probes) != 6:
        raise ValueError('six signed axis probes required')
    covered = set()
    for probe in probes:
        if type(probe) is not dict:
            raise ValueError('invalid controller probe')
        native = probe.get('native_pose_command')
        delta = probe.get('goal_delta_world_m')
        recovered = probe.get('recovered_native_translation')
        if not (_numbers(native, 6) and _numbers(delta, 3) and _numbers(recovered, 3)):
            raise ValueError('invalid controller probe vectors')
        axes = [i for i, value in enumerate(native) if value != 0]
        if len(axes) != 1 or axes[0] >= 3 or abs(native[axes[0]]) != 0.2:
            raise ValueError('unsupported controller probe direction')
        covered.add((axes[0], native[axes[0]]))
        if any(not isclose(delta[i], native[i] * 0.05, rel_tol=0, abs_tol=1e-10) or
               not isclose(recovered[i], native[i], rel_tol=0, abs_tol=1e-9)
               for i in range(3)):
            raise ValueError('controller probe contradicts conversion')
    if len(covered) != 6:
        raise ValueError('missing signed axis coverage')


@dataclass(frozen=True)
class NativeTranslationResidual:
    """Proposal-bound native offset with its physical request and evidence ID."""

    request: SupervisorAdjustmentRequest
    native_residual: tuple[float, ...]
    evidence_sha256: str


@dataclass(frozen=True, init=False)
class LiberoTranslationConverter:
    """Host-owned converter for the verified single Panda world-frame mapping.

    Evidence bytes and their expected SHA-256 come from retained host artifacts.
    Runtime controller settings must be read by the host from the actual adapter,
    including position_limits=None; never accept these inputs from a supervisor.
    Construct again after controller configuration or installation changes.
    """

    evidence_sha256: str
    _validator: CorrectionValidator = field(repr=False)

    def __init__(self, evidence: bytes, *, expected_sha256: str,
                 runtime_controller: dict, target: str, maximum_metres: float):
        if type(evidence) is not bytes or hashlib.sha256(evidence).hexdigest() != expected_sha256:
            raise ValueError('controller evidence hash mismatch')
        try:
            report = json.loads(evidence)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('invalid controller evidence JSON') from exc
        _verify(report)
        _check_controller(runtime_controller)
        if 'position_limits' not in runtime_controller or runtime_controller['position_limits'] is not None:
            raise ValueError('unverified position clipping geometry')
        if not _numbers([maximum_metres], 1) or not 0 < maximum_metres <= 0.05:
            raise ValueError('translation bound must be in (0, 0.05] metres')
        validator = CorrectionValidator(RecoveryRegistry(()), adjustment=AdjustmentRequestDecoder(
            target, 'world', tuple(AdjustmentComponent(name, 'm', -maximum_metres, maximum_metres)
                                   for name in _COMPONENTS)))
        object.__setattr__(self, 'evidence_sha256', expected_sha256)
        object.__setattr__(self, '_validator', validator)

    def convert(self, request: dict | SupervisorAdjustmentRequest,
                proposal: ActionProposal) -> NativeTranslationResidual:
        """Revalidate request limits and divide metre offsets by 0.05 m/unit.

        Zeros in rotation/gripper slots are residuals, not a hold command.
        The output is not a composed or approved action. Final-action bounds,
        freshness, scene eligibility and operational gates remain mandatory.
        """
        checked = self._validator.validate(request, proposal)
        if not _numbers(proposal.action, 7):
            raise ValueError('expected a finite seven-component native proposal')
        offsets = tuple(value / 0.05 for _, value in checked.residual)
        return NativeTranslationResidual(checked, offsets + (0.0,) * 4,
                                         self.evidence_sha256)
