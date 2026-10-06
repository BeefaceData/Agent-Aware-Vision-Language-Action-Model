"""One-proposal translation composition at the harness action-selector seam."""

from math import isfinite

from episode_harness import ActionProposal, ActionResolution
from supervisor_adjustment import SupervisorAdjustmentRequest
from translation_conversion import LiberoTranslationConverter


def _validate_native_action(action):
    """Check the complete command for the converter's verified Panda mapping."""
    if type(action) not in (list, tuple) or len(action) != 7:
        raise ValueError('adjusted action must have seven native components')
    try:
        finite = all(type(value) in (int, float) and isfinite(value)
                     for value in action)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('adjusted action must contain finite numeric values')
    if any(not -1 <= value <= 1 for value in action):
        raise ValueError('adjusted action outside native action bounds')


class SingleActionAdjustment:
    """Compose host-approved requests for synchronous, single-owner execution.

    Use one instance for an execution session. The caller must enforce scene,
    expiry, readiness and budget gates before supplying a request. This class
    does not call a model or environment and is not a live authorization gate.
    Returned overrides are consumed even if subsequent execution fails: retrying
    an uncertain command requires a new observation and proposal.
    """

    def __init__(self, converter: LiberoTranslationConverter):
        if type(converter) is not LiberoTranslationConverter:
            raise ValueError('verified LIBERO translation converter required')
        self._converter = converter
        self._consumed = set()

    def resolve(self, proposal: ActionProposal,
                request: dict | SupervisorAdjustmentRequest | None) -> ActionResolution:
        """Return a single override, explicit rejection, or unchanged pass.

        None means no adjustment on this proposal; residuals are never queued.
        Invalid requests reject rather than silently falling back to baseline.
        """
        if request is None:
            return ActionResolution('pass')
        try:
            converted = self._converter.convert(request, proposal)
            identity = (converted.request.episode_id, converted.request.proposal_id)
            if identity in self._consumed:
                raise ValueError('adjustment already consumed for this proposal')
            # The verified mapping fixes translation to slots 0/1/2. Preserve
            # rotation and gripper exactly, including their numeric types.
            action = list(proposal.action)
            if any(not -1 <= value <= 1 for value in action):
                raise ValueError('original proposal outside native action bounds')
            for index in range(3):
                action[index] += converted.native_residual[index]
            _validate_native_action(action)
        except ValueError as exc:
            return ActionResolution('reject', reason=str(exc))
        self._consumed.add(identity)
        return ActionResolution('override', action)
