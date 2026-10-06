# Baseline fallback admission

Issue #79 adds `baseline_fallback.BaselineFallback` to `run_episode`. An
abstention can execute the unchanged policy proposal only after all three
host-owned prerequisites pass: fresh required inputs for the current observation,
an affirmative controller health check, and validation of the current native
proposal. Supervisor-reported availability cannot satisfy these checks.

```python
guard = BaselineFallback(
    max_age_seconds=reviewed_baseline_sensor_limits,
    controller_healthy=adapter_controller_health_check,
    proposal_valid=adapter_native_proposal_validator,
)
outcome = run_episode(config, policy, environment, recorder,
                      supervisor_decider=decide, baseline_fallback=guard)
```

The adapter health callback takes no arguments and must query the current
controller/transport state. The proposal validator receives a detached
`ActionProposal` with its episode, observation and proposal identity. It must
validate the adapter's native action shape, finite numeric values, bounds and
current execution constraints. Each callback must return the boolean `True`;
`False`, unknown values, truthy non-booleans and exceptions refuse fallback.
Never wire these callbacks to model claims or unconditional live defaults.

Required sensor names are `main`, `wrist` and/or `robot_state`, each with an
explicit maximum age in seconds. They describe baseline execution requirements,
which can differ from supervisor assessment requirements. The harness evaluates
capture age after both host callbacks, using its own clock. Inputs must carry
measured sensor timestamps and the current observation sequence. Missing, stale,
unverified return-time-only and invalid metadata cannot grant admission. Limits
and callbacks must come from reviewed adapter configuration; this change supplies
no live robot configuration or additional sensor timestamp claims.

With a guard configured, unavailable or malformed supervision and explicit
selector rejection can also request baseline fallback. Supervisor error records
retain the exception category rather than potentially private provider text.
Internal intervention-budget, recovery-cooldown and horizon refusals still stop
execution. Selector exceptions and process interruptions do not request fallback.
Without a guard, abstention refuses dispatch, while supervisor exceptions retain
their existing raise-and-finalize behavior and selector rejection stops normally.

`ActionRecord.fallback` retains the cause, current identity, check time, required
input reports, controller/proposal check results, selected `baseline` or `refuse`,
and reason. Baseline admission executes exactly the original proposal, consumes
one shared action-horizon slot and no intervention allowance, and does not resume
or reset the policy. Refusal leaves selection, execution and acknowledgement
absent and ends with `proposal_rejected`. Admission is not an acknowledgement:
a subsequent controller exception still records no executed action.

Sealed replay recomputes sensor freshness and admission from retained evidence,
checks the selected fallback against dispatch, and reproduces complete successful
and refused episodes. It does not re-query a live controller or prove authenticity
of an author's health attestations. Historical bundles lacking the new field
remain readable without changing their annotation fingerprints.

Run `python -m unittest discover -s tests -p test_baseline_fallback.py -v`.
Fixtures cover healthy and refused fallback, per-proposal checks, delayed checks,
interruptions and resealed evidence tampering. These are synthetic software
contracts. Physical adapter hold/stop remains issue #80; refusal here stops
harness dispatch and does not claim to stop physical motion.
