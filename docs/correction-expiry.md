# Correction request expiry

Issue #83 adds two trusted `EpisodeConfig` settings:
`correction_timeout_seconds` and `correction_max_age_seconds`. Declare both as
positive finite seconds on the same monotonic clock as observation capture and
`run_episode(clock=...)`. These are host settings, never supervisor fields.
Neither value is a measured robot-safe default. Both absent preserves historical
trusted-selector fixtures and replay; that mode does not establish expiry readiness
and must not be selected for active deployment. Readiness integration remains #94.

For each new correction the harness establishes a deadline from `request_at +
correction_timeout_seconds`, before supervisor and selector work. It checks again
after selector/queue delays and detached command preparation, immediately before
dispatch. Both adjustments and recovery commands require `now < deadline` and
`now - original_capture <= correction_max_age_seconds`. Equality at the request
deadline rejects; equality at the maximum observation age is permitted. Wall-clock
calendar timestamps and provider-reported arrival times cannot extend validity.

Every recovery command retains the original request/capture/deadline. Fresh
intermediate observations do not renew that request. The recovery monitor still
checks its separately declared sensor requirements; this gate does not replace
per-camera/state freshness, scene eligibility, final-action bounds or readiness.

Expiry refuses the correction, invokes the adapter's declared interruption and
ends the attempt. It does not implicitly fall back, resume the policy, or issue a
zero command. A correction rejected before its first command consumes no
intervention attempt; a partially executed recovery retains its charged attempt
and acknowledged action count. This checks dispatch to the adapter, not eventual
physical actuation: an adapter with its own transport queue still needs a bounded
transport/controller contract.

`ActionRecord.correction_expiry` retains source proposal, original monotonic
request/capture times, deadline, age limit, check time, age and verdict. Sealed
replay checks these against configuration, execution timing, continuation identity
and rejection/interruption evidence before playback. Historical annotation
fingerprints remain unchanged. Replay advances a synthetic clock through recorded
validity checks, requiring no model or robot.

Run the public boundary and complete-episode replay checks:

```powershell
python -m unittest discover -s tests -p test_correction_expiry.py -v
```

Synthetic evidence verifies software contracts only; no live trials or physical
stopping guarantees are established.

## Episode operational cap

Issue #85 adds `EpisodeConfig(max_episode_seconds=...)`. Use a positive finite
host-declared duration on the observation/harness monotonic clock. `None` keeps
historical uncapped operation and does not establish operational readiness.
The allowance begins after reset, initial observation validation and recorder
startup. It includes policy inference, supervisor waits, selectors, fallback
checks, command execution, local recovery assessment, callbacks and resumption.
Setup and artifact finalization are outside the active rollout allowance.

The episode deadline is independent of correction validity and action count.
Equality expires it. A new episode establishes a new deadline; fresh observations,
recovery commands and new requests never extend the existing one. Once expired,
the harness invalidates pending results, dispatches no further commands and calls
the adapter's declared hold/stop operation. It does not select baseline fallback.
`wall_clock_limit` reports a confirmed interruption; `interruption_failed` retains
an unconfirmed stop. Neither is task success. A terminal evaluator result already
accepted by the harness remains terminal evidence.

Capped execution bounds callback waiting with daemon workers. Python cannot kill
an uncooperative callback or cancel a remote request: late results are discarded,
and affected adapters/callbacks remain busy until their worker exits. Adapters
must support interruption concurrently with an outstanding step and provide their
own bounded stop transport. The cap is a software dispatch/wait limit, not a
physical stopping-time guarantee. Recorder writes and finalization must also be
bounded by the embedding application; they are not asynchronous controller work.

An environment step still outstanding at expiry has unknown execution status.
The harness requests interruption and raises `EpisodeDeadlineExceeded`, retaining
unconfirmed dispatch and partial outcome evidence through `episode_interruption`.
It does not retry, count the command as acknowledged, or seal a complete trace.
An acknowledged action retains its count even if subsequent recovery assessment
expires; remaining recovery commands are discarded.

`EpisodeOutcome.episode_clock` retains the configured rollout start and deadline.
`wall_clock_limit` retains expiry stage/time, interruption acknowledgement and any
supervisor call reserved before expiry but not attached to an executed action.
Sealed replay validates these against configuration, dispatch times, observations,
call accounting and outcomes, including expiry before the first action. Historical
bundles retain their original configuration and annotation fingerprints.

```powershell
python -m unittest discover -s tests -p test_episode_deadline.py -v
```
