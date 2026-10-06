# Per-episode supervisor call allowance

`EpisodeConfig(max_supervisor_calls=12, supervisor_exhaustion_policy="stop", ... )`
reserves at most twelve supervisor requests during one `run_episode` attempt.
The default allowance is the episode action horizon; zero disables requests.
Only nonnegative integers are accepted. A new episode starts a new allowance.

Periodic and event-triggered assessments use the same allowance. Coincident
requests cost one reservation; skipped assessments and pending recovery commands
cost none. Reserve before invoking the callback: transport errors, timeouts,
invalid responses and cancellation do not refund the request. A busy provider
also consumes a reservation conservatively, even though it sends no new request.
The existing `BoundedSupervisorProvider` and `ChronologicalVlmAdapter` make at
most one provider attempt per callback. The harness can explicitly retry a
decider using the policy below. Custom callbacks
must obey that same single-attempt contract; hidden transport retries or model
calls from an assessment trigger are unsupported. `ModelCallJournal` retains
actual transport accounting separately, including late completion and unknown
usage/cost. Request reservations can therefore exceed actual provider calls.

On the next due assessment after the allowance is used, no callback is invoked.
`stop` (the default) refuses dispatch and invokes the adapter's declared hold/stop,
even if a healthy baseline fallback was supplied. `baseline_fallback` instead
runs the existing current-observation freshness, controller-health and native
proposal checks. Missing or failed checks refuse dispatch and invoke hold/stop.
A skipped assessment continues to follow the configured assessment cadence.
Reaching the numeric limit alone does not stop an otherwise terminal episode.

Decision records retain `supervisor_call_budget` admission and cumulative count;
the returned outcome and sealed manifest retain the limit, attempted reservations,
policy and whether a request was refused for exhaustion. Complete-episode replay
validates those counts, refusal policy and fallback evidence before executing its
recorded actions, and returns the retained accounting. Historical bundles without
these fields remain readable without adding fields to annotation fingerprints.
Exceptional, unsealed attempts retain reservation evidence on recorded supervisor
failures; they are not complete replay evidence.

Observation-only callbacks share the cap. They require an interruption contract
when the allowance is smaller than the action horizon, since refusal can then
occur. Existing observation-only callers whose allowance covers the full horizon
retain their adapter contract. Bounded active deciders always require hold/stop.

Run `python -m unittest discover -s tests -p test_supervisor_allowance.py -v`.
Fixtures cover mixed scheduling, errors, invalid responses, timeout/busy accounting,
zero allowance, stop versus guarded fallback, observation callbacks, per-episode
reset and resealed accounting tampering. They establish software behavior only;
no paid calls, live benchmark or physical stopping performance is established.

## Explicit supervisor retries

Retries are disabled by default. Declare, for example,
`supervisor_max_retries=1`, `supervisor_retry_delay_seconds=0.25`,
`supervisor_retry_errors=('rate_limited', 'unavailable')`, together with
`max_episode_seconds` and `max_supervisor_calls` in `EpisodeConfig`.
The retry count is additional attempts per due assessment. Retry-enabled
episodes require a decider and a finite episode time cap. No delay or retry
extends the episode deadline or changes the proposal identity or instruction.

Only a completed transport failure explicitly classified with
`RecoverableProviderError` can qualify. The supplied HTTPS transport classifies
HTTP 429 as `rate_limited` and 503 as `unavailable`. Other HTTP failures,
arbitrary exceptions, malformed/invalid control responses, provider timeouts,
cancellation and busy refusals do not qualify. The bounded provider still makes
one attempt; it carries the safe class to the harness without raw error text.
Custom transports must not hide retries or classify decoding failures as
recoverable. The VLM adapter reuses the same observation history for an identical
proposal after a recoverable transport error; each transport attempt is journaled.

Every retry reserves another call before invocation. Delays and all attempts
consume the original episode time allowance. A depleted call allowance invokes
the declared stop or guarded baseline fallback policy. Retry-count exhaustion
and noneligible failures use the existing supervisor-error path: guarded fallback
when configured, otherwise an exceptional attempt with recorded failure evidence.
Episode-time exhaustion always interrupts; it cannot fall back or dispatch a late
response. Python workers and remote requests may outlive caller waiting, so actual
usage/cost remains in the model-call journal and may be unknown until completion.

Decision budgets retain `retry_attempts` with start/end times, status and safe
error class. Deadline outcomes retain the pending decision's attempts, including
expiry during the retry delay. Sealed replay checks eligibility, delay, count and
episode/call limits without contacting a provider. Run
`python -m unittest discover -s tests -p test_supervisor_retry.py -v`.
