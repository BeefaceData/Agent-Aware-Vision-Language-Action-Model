# Frozen v1 runtime contract

`run_episode` freezes the first observation's `task` field through the terminal
observation. A changed or newly introduced/removed instruction raises
`FrozenContractViolation` before the next inference or dispatch. Legacy replay
without a task field remains supported. The LIBERO runner also verifies the task
inserted into the policy batch against the declared instruction.

Use `FrozenRuntimeContract` to capture trusted, JSON-compatible readers before
starting a supervised episode:

```python
contract = FrozenRuntimeContract(
    policy=policy_identity,
    supervisor=lambda: supervisor_identity(vlm.settings),
    tools=lambda: asdict(registry),
    controller=lambda: asdict(environment.action_capabilities),
)
# Persist contract.identity in the attempt manifest before control.
outcome = run_episode(config, policy, environment, recorder,
                      supervisor_decider=provider, frozen_contract=contract)
```

The host supplies these bounded readers. They must report current declared
state, including any configured numeric limits; do not construct them from
supervisor responses or memory. Checks run before reset, inference and dispatch,
and after each acknowledged step, including terminal steps. Changes stop further
execution, retain the acknowledged action count and rejected-action evidence,
and request the adapter's declared hold/stop. Unavailable or failed interruption
is recorded; no zero action or successful physical stop is invented.

The baseline runner retains its runtime declaration in `attempt.json` and checks
the declared policy assets, backbone revision, device, instruction and native
controller capabilities. Asset bytes are verified when loading the pinned
policy; per-action checks compare declarations rather than rehashing model
weights. `ChronologicalVlmAdapter` also rejects model/settings/prompt/API-version
drift for its lifetime, even without an on-disk supervisor manifest. Use the
retained supervisor manifest for cross-process selection verification.

Supervisor responses have a closed schema: instruction rewrites, model asset
changes, prompt changes, tool installation and controller-limit edits are not
operations. Memory and evidence text are data, never executable configuration.
The registry and correction validators remain host-owned. The supervisor gets
detached observations, so editing its task/memory fields cannot edit policy input.

This is a runtime contract for trusted adapters, not a Python sandbox. It does
not detect arbitrary in-process weight edits, prove remote-provider weight
immutability, or authorize live trials. A reader omitted from the contract is
not verified. The public tests exercise rejected mutation requests, declaration
drift, instruction changes, detached content and a complete sealed replay:

```powershell
python -m unittest discover -s tests -p test_frozen_runtime.py -v
```
