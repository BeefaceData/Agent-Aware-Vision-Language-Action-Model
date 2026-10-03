# Static recovery registry

`recovery_registry.RecoveryRegistry` is the public executor-side selection
boundary for #65. Construct it once from trusted host declarations before
processing supervisor responses. `tools` exposes immutable `RecoveryTool`
contracts, each with a name, inclusive parameter bounds, required adapter
capabilities, required observations, positive action limit, and named
completion/abort conditions. Empty registries disable recovery; duplicate names
and incomplete declarations fail at construction.

```python
from recovery_registry import RecoveryRegistry, RecoveryTool
from supervisor_recovery import RecoveryParameter

# Synthetic contract example only; these are not reviewed robot limits.
tool = RecoveryTool(
    name='reopen_and_retreat',
    parameter_bounds=(RecoveryParameter('retreat_m', 0, 0.03),),
    required_capabilities=('single_arm_translation_metres', 'gripper_open'),
    required_observations=('main', 'robot_state'),
    action_limit=3,
    completion_conditions=('gripper_open_confirmed', 'retreat_target_reached'),
    abort_conditions=('clearance_unverified', 'stale_observation',
                      'controller_failure', 'action_limit_reached'),
)
registry = RecoveryRegistry((tool,))
# response: recorded recovery JSON; proposal: current ActionProposal;
# adapter_capabilities: trusted declarations from the robot adapter.
selection = registry.resolve(response, proposal, capabilities=adapter_capabilities)
```

The response uses the existing `RecoveryRequestDecoder` wire schema. Selection
returns an immutable `ResolvedRecovery` with the exact declared tool and decoded
request. Unknown tools, implementation/code/definition fields, altered limits,
undeclared parameters, invalid numerical values and stale proposal identities
are rejected. Capability matching requires every declared capability; extra
adapter capabilities are allowed. Names must distinguish relevant control
semantics, including units, frames and arm coordination. Merely naming a robot
does not establish compatibility, and bimanual support cannot be inferred from
a single-arm capability.

This registry resolves declarations only. It accepts no executable callbacks,
imports no provider-selected modules, and has no runtime registration API.
Completion/abort identifiers describe checks the host executor must implement;
they are never evaluated as code. Host declarations and adapter capabilities
must not be constructed from provider payloads. No default physical tool or
calibration is enabled.

Observation requirements are declared, not proof of current sensing or clearance.
Evidence references retain the decoder's structural validation. Scene eligibility
is #71, recovery execution is #72, completion/abort enforcement is #73 and active
readiness is #94. Resolution neither invokes a controller nor changes
`run_episode`; passing a recovery request to its current decision callback still
fails closed. Consequently this change requires no new execution episode path.

Run the public executor fixture with:

```powershell
python -m unittest discover -s tests -p test_recovery_registry.py -v
```

It resolves an eligible synthetic contract, rejects incompatible robot/tool
pairs and executable overrides, verifies per-tool bounds and immutable
declarations, and establishes interface behavior only.
