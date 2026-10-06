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
for a continuously running robot. Execution transport failure handling is #81;
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
