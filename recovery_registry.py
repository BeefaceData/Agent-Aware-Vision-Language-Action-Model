"""Static, trusted recovery declarations and executor-side request resolution."""

from dataclasses import dataclass

from episode_harness import ActionProposal
from supervisor_recovery import (RecoveryParameter, RecoveryRequestDecoder,
                                 SupervisorRecoveryRequest)


def _names(values, label):
    if isinstance(values, str):
        raise ValueError(f'{label} must be a sequence of names')
    names = tuple(values)
    if (not names or any(type(name) is not str or not name.isidentifier() or
                         len(name) > 128 for name in names) or
            len(set(names)) != len(names)):
        raise ValueError(f'invalid {label}')
    return names


@dataclass(frozen=True)
class RecoveryTool:
    """Host-owned contract; condition names refer to executor checks, not code.

    Configuration must come from reviewed application code/configuration, never
    supervisor JSON. Capability names must describe semantics (including arm,
    frame and units where relevant), not just a robot/model name. No numerical
    limits here establish calibration or permission to operate a robot.
    """

    name: str
    parameter_bounds: tuple[RecoveryParameter, ...]
    required_capabilities: tuple[str, ...]
    required_observations: tuple[str, ...]
    action_limit: int
    completion_conditions: tuple[str, ...]
    abort_conditions: tuple[str, ...]

    def __post_init__(self):
        decoder = RecoveryRequestDecoder(self.name, self.parameter_bounds)
        object.__setattr__(self, 'parameter_bounds', decoder.parameter_bounds)
        for field in ('required_capabilities', 'required_observations',
                      'completion_conditions', 'abort_conditions'):
            object.__setattr__(self, field, _names(getattr(self, field), field))
        if type(self.action_limit) is not int or self.action_limit <= 0:
            raise ValueError('recovery action limit must be a positive integer')


@dataclass(frozen=True)
class ResolvedRecovery:
    """Inspectable selection for an executor; grants no execution authority."""

    tool: RecoveryTool
    request: SupervisorRecoveryRequest


@dataclass(frozen=True)
class RecoveryRegistry:
    """An immutable allowlist built before processing any supervisor request.

    An empty registry disables all recovery tools. There is no registration,
    code-loading or implementation override API. Actual executor dispatch must
    be predefined by the host, with readiness and scene checks before execution.
    """

    tools: tuple[RecoveryTool, ...]

    def __post_init__(self):
        tools = tuple(self.tools)
        if (any(type(tool) is not RecoveryTool for tool in tools) or
                len({tool.name for tool in tools}) != len(tools)):
            raise ValueError('invalid or duplicate recovery tool declaration')
        object.__setattr__(self, 'tools', tools)

    def resolve(self, response: dict, proposal: ActionProposal, *,
                capabilities: tuple[str, ...]) -> ResolvedRecovery:
        """Decode a wire request and check trusted adapter capability support.

        Capabilities come from the adapter contract, never the response. This
        checks declared compatibility only: observation presence/freshness,
        diagnosis, clearance and execution-time bounds remain executor gates.
        Evidence references retain the existing decoder's structural semantics.
        """
        if type(response) is not dict or type(response.get('tool_name')) is not str:
            raise ValueError('invalid recovery tool selection')
        tool = next((tool for tool in self.tools
                     if tool.name == response['tool_name']), None)
        if tool is None:
            raise ValueError('unknown recovery tool')
        request = RecoveryRequestDecoder(tool.name, tool.parameter_bounds).decode(
            response, proposal)
        # No capabilities is valid adapter input, but incompatible with any tool.
        if isinstance(capabilities, str):
            raise ValueError('capabilities must be a sequence of names')
        capabilities = tuple(capabilities)
        if capabilities:
            capabilities = _names(capabilities, 'adapter capabilities')
        if not set(tool.required_capabilities).issubset(capabilities):
            raise ValueError('incompatible robot/tool capabilities')
        return ResolvedRecovery(tool, request)
