# Environment hold and stop

Active `run_episode` configurations (`action_selector`, `supervisor_decider`,
or `baseline_fallback`) require the environment to expose an
`InterruptionContract` as `interruption_contract` and a callable
`interrupt(request)`. Validation happens before policy/environment reset.
Baseline and observation-only execution do not require this capability.

The adapter chooses `hold` or `stop` and documents its contract identity and
acknowledgement condition. `InterruptionRequest` carries the episode, proposal,
observation sequence, declared operation, and refusal reason. An adapter must
return exactly `True` only after its documented condition holds. An absent,
ambiguous or false acknowledgement, or a transport exception, is unconfirmed.
Physical transports must bound their own synchronous interruption calls and
document their deadlines; the harness cannot make a blocking driver safe.

When fallback is unsafe, or action selection rejects dispatch, the harness
invokes this operation once before writing the rejection and finalizing the
episode. It never substitutes a zero action. A confirmed interruption preserves
`proposal_rejected`; an unconfirmed operation returns `interruption_failed`.
Both retain an unknown task outcome rather than claiming task failure/success.
No further policy inference, action dispatch or automatic resume occurs.
Hold also ends this attempt: a new attempt requires reset under the adapter's
contract. Interruption does not add an executed policy action or task reward.

Action evidence retains the contract, request, request/completion times,
confirmation and exception category, separately from the absent selected and
executed action. Exception messages are omitted. Sealed replay validates this
evidence and reproduces the episode disposition without a live controller.
Historical traces without the interruption field remain readable; they do not
establish that a controller was stopped.

`ReplayEnvironment` stops scripted stepping until reset.
`LiberoEnvironmentAdapter` disables adapter stepping until reset and relies on
its synchronous simulation advancing only through `step`. Neither issues a
zero command, nor claims a physical stop. The client adapter must implement its
own documented controller operation (#158); LIBERO's contract is not suitable
for a continuously running robot. Execution transport failures use the path below;
recovery abort and operational wall-clock stop integration have their own
runtime contracts and are not established by this refusal path.

Run the public contract fixtures with:

```powershell
python -m unittest discover -s tests -p test_environment_interruption.py -v
```

These synthetic tests verify hold/stop requests after one acknowledged action,
unconfirmed acknowledgements and transport errors, preflight rejection, LIBERO
adapter dispatch disabling, and successful replay or rejection of tampered
sealed evidence. They establish software behavior, not physical safety or task
performance.

## Failed execution acknowledgements

When `environment.step` raises or returns no `StepResult`, the harness invokes
the declared interruption once, before recording the failure. No retry, fallback,
further dispatch or resume follows. The original exception is re-raised with
`episode_interruption.failed_action`; this detached evidence survives recorder
failure. A failed interruption never replaces the original execution error.
Legacy baseline adapters without an interruption contract retain an explicit
`controller interruption unavailable` diagnostic; no stop is claimed.

Adapters may raise `execution_failure.ExecutionFailure(sent=True/False/None)`.
True means the adapter established transmission, False means it established no
transmission, and None means unknown. Ordinary exceptions and missing results
leave transmission unknown. This receipt is transport evidence, not physical
execution evidence. The failure record retains proposed and selected actions,
dispatch attempt, transmission certainty, absent execution acknowledgement, and
the separate interruption acknowledgement. Private exception messages are omitted.

Only returned step results count toward acknowledged action totals and rewards.
A failed second call after one acknowledged action therefore retains one action,
the prior reward, and unknown task status, even if transmission was confirmed.
The physical action count may be greater and is unknown; a confirmed stop does
not retroactively confirm the command. As before, malformed/stale observation
packets in a returned StepResult follow the observation-validation contract.

Controller-fault episodes retain partial trace files without a completion seal.
They cannot be loaded as successful sealed replay. The deterministic whole-attempt
replay fixtures exercise startup, one acknowledged action, a controller fault,
interruption and finalization through public interfaces:

```powershell
python -m unittest discover -s tests -p test_execution_failure.py -v
```

These fixtures establish failure accounting only, not physical stopping behavior.
