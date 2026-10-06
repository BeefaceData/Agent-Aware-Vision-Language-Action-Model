# One-action translation adjustments

`single_action_adjustment.SingleActionAdjustment` implements #68 through the
existing harness `action_selector` interface. Construct it with the verified,
host-configured `LiberoTranslationConverter` described in
[translation conversion](translation-conversion.md). Keep one instance for the
synchronous execution session and call `resolve(current_proposal, request)`.

```python
executor = SingleActionAdjustment(converter)

def select(proposal):
    # Host code supplies a request only after its execution gates have passed.
    # Return None when no adjustment is requested for this proposal.
    request = host_approved_request_for(proposal)
    return executor.resolve(proposal, request)

outcome = run_episode(config, policy, environment, recorder,
                      action_selector=select)
```

The converter revalidates both wire and decoded requests against the current
episode, observation and proposal identities, host residual limits, target,
frame and units. Composition adds only the three verified native translation
offsets. Rotation and gripper values remain unchanged. Both original and final
commands must fit the native [-1, 1] range; rejection never clips a command or
silently substitutes baseline execution.

After composition, the complete command is explicitly revalidated as seven
finite numeric components (booleans are invalid), each inside the inclusive
native [-1, 1] bounds. These bounds apply to the verified Panda mapping only.
An invalid command returns `reject` with a reason and no override action.
The harness records that reason as `rejection_reason`, leaves execution absent,
and ends the episode with `proposal_rejected` without calling the environment.
This implements #69; rejection does not consume an action or attempt clipping.

A successful resolution consumes that episode/proposal identity before dispatch.
Further adjustments for the same identity are rejected, even under a new decision
ID. This is conservative after a failed dispatch: an uncertain command is not
retried. This in-memory protection assumes one synchronous execution owner; it is
not persistent or distributed deduplication. A new proposal requires a new bound
request. `None` returns a pass and carries no previous residual forward.
Successful overrides retain their source episode, observation and proposal in
`ActionResolution.source_identity`. The harness rejects a cached resolution
returned for a later proposal, including after recovery. Hosts must preserve
this binding when forwarding an executor result.

The harness dispatches the returned override once, counts it toward the episode
action horizon, records the original proposal separately from the selected and
acknowledged execution, and resumes a compatible policy from the fresh result
observation before requesting another action. Existing recorder sealing and replay
preserve those distinctions. The policy must support the harness resume contract.

This component is an execution building block, not a complete live supervisor.
The host must enforce scene eligibility, expiry, readiness and operational budgets
before supplying requests. No live runner is enabled by this change. The current
identity check does not establish a wall-clock expiry bound. The caller should
retain request and controller-evidence provenance alongside its run configuration;
the existing action record retains proposal/execution evidence, not a new request
schema.

Run `python -m unittest discover -s tests -p test_single_action_adjustment.py -v`.
Eight public tests cover composition, unchanged nontranslation components, inclusive
bounds, malformed commands across all seven components, signed overflow on all
translation axes, invalid and stale requests, duplicate consumption, and complete sealed
successful/rejected replays. Synthetic controller evidence establishes software
contract behavior only, not task improvement or physical correction safety.
