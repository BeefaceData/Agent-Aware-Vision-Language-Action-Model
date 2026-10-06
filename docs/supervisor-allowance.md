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
most one provider attempt per callback, with no automatic retry. Custom callbacks
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
