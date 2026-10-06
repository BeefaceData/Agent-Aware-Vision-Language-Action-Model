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
