# Bounded reopen-and-retreat execution

`ReopenRetreatExecutor.resolve` selects the first code-defined recovery for
issue #72. It rechecks the #71 scene gate, revalidates the request's tool and
parameter bounds, and uses `LiberoTranslationConverter` for the verified Panda
world-frame translation mapping. It returns data to the harness, never calls
the environment and never accepts generated robot code.

The trusted host supplies `ReopenRetreatControl`: a reviewed envelope identity,
target, unit retreat direction in the world frame, and native gripper-open
command. The envelope must cover that gripper sweep, zero relative pose command
and retreat corridor. There is no default gripper sign or assumed safe direction.
Construct the converter using retained controller evidence and actual runtime
settings as described in [translation conversion](translation-conversion.md).
The controller evidence verifies translation goal scaling, not gripper semantics
or achieved displacement. Host scene assessments and the control contract must
come from permitted, reviewed evidence; the supervisor cannot supply them.

After validation the sequence contains exactly two replacement commands:

1. Zero relative translation/rotation with the declared gripper-open command.
2. One bounded relative translation along the reviewed direction, with zero
   relative rotation and the gripper-open command retained.

The distance is the request's `retreat_m`; the converter enforces its own metre
limits and divides by the verified 0.05 metres/native unit. All seven final
components must be within [-1, 1]. The sequence requires at least two actions
in the registry's tool allowance. A proposal is consumed on selection; uncertain
execution cannot be retried with that identity.

Pass the result through `run_episode(..., action_selector=select,
recovery_observer=observe)`. The selector
calls `executor.resolve(current, request, scene=assessment,
now_monotonic=clock(), window=history)` only after the host's applicable
readiness, expiry and resource gates. Return `ActionResolution('pass')` when no
recovery is requested. Use one executor instance per execution session.

The harness checks proposal identity and resume support before execution. Both
commands must fit the remaining shared episode horizon; otherwise it records a
rejection before opening. During recovery it pauses policy inference and
supervision/selection callbacks, while retaining observations in temporal
history. Every command passes through the normal step, ingestion and recording
path. Accepted terminal results, invalid observations and execution failures
stop pending commands. After confirmed local completion on the final accepted nonterminal observation, if
there is remaining horizon, the harness calls policy `resume` once before fresh
inference. No environment reset is used for recovery.

Action records preserve the original suspended policy proposal and distinguish
each executed command. Their `recovery` metadata contains the full sequence,
source request, tool limit, envelope, controller evidence digest and zero-based
action index. Continuation step IDs identify execution slots, not additional
policy inference. A failed environment call retains its selected action without
inventing execution acknowledgement. Recorded replay validates sequence order,
origin, bounds on action count and command consistency before replaying it.

Run the synthetic public checks with:

```powershell
python -m unittest discover -s tests -p test_reopen_retreat.py -v
```

The full-episode fixture demonstrates two recorded recovery commands followed
by fresh-state policy resume and a scripted successful outcome. Other fixtures
cover rejection, terminal interruption, exact horizon, stale/invalid evidence,
controller failure and corrupted provenance. These are execution-contract
checks, not measured recovery efficacy or robot safety evidence. Exhausting the
sequence is not confirmation that the gripper opened or the retreat target was
reached. Local completion/abort monitoring is described below;
active-correction readiness remains #94. This module enables no live run by
default and establishes no simulation or physical improvement claim.

## Local completion and abort checks

The host supplies `observe(plan, action_index, packet)`, returning a
`recovery_monitor.RecoveryAssessment` bound to the current episode, observation
sequence and reviewed envelope. The packet contains only deployable fields;
the callback never receives evaluator success, reward or privileged state.
The host assessor must derive its booleans from the declared required sensor
inputs using reviewed local geometry and gripper semantics. `None` means
unknown. This interface supplies no perception algorithm or default physical
thresholds and does not establish active readiness.

The plan retains the tool's declared conditions, required observations and the
eligibility gate's sensor-age bound. Unsupported condition contracts reject
before execution. A missing observer also prevents execution. After every
acknowledged command the harness checks current sensor captures, clearance and
gripper-open confirmation. After retreat it additionally requires confirmation
that the retreat target was reached; a native goal request is insufficient.
It makes no additional corrective attempts. Unmet conditions at the tool's
action limit report `action_limit_reached`; earlier failures name the unmet
condition. Missing or failing assessors and stale, missing or uncertain evidence
abort, even if all commands were acknowledged.

Each action's `recovery.check` retains the assessment, sensor freshness evidence,
check time, local status, reason, acknowledged action count and selected path.
The declared abort path is `stop_episode`: no further command or policy resume
occurs and the outcome is `recovery_aborted`. This is a discrete harness stop,
not a physical controller hold/stop command (tracked separately in #80).
Controller exceptions and rejected packets retain partial evidence and use the
existing interruption path. Accepted terminal evaluator outcomes still end the
episode immediately and remain separate from local recovery status: a locally
completed recovery is not task success, nor does an abort erase observed task
success. Successful nonterminal recovery uses the existing fresh-state resume.

Recovery also retires the pre-recovery proposal for execution (#75). Proposal
identities include the episode and a monotonically advancing action index; they
are never reused when policy inference resumes. Both correction executors retain
the source episode, observation sequence and proposal identity in their returned
resolution. The harness checks this binding immediately before selection, so
even a cached, previously validated override cannot act on the recovered scene.
Late raw requests are revalidated against the new proposal and rejected too.
Rejections retain a reason and execute no additional action.

Before asking for a new proposal, the harness passes the accepted final recovery
observation to policy `resume`. The adapter must discard obsolete queued actions.
`ResetOnResumePolicyAdapter` implements this for policies whose reset clears all
cached actions, and rejects subsequent input from before that resume observation.
An arbitrary native action has no independently verifiable capture identity, so
custom policy adapters remain responsible for their queue invalidation contract.
Unbound `ActionResolution` values are a trusted host/replay interface, not a wire
format for supervisor replies; do not strip executor source bindings.

Synthetic complete-episode tests deliver late recovery and adjustment requests,
cached recovery plans and cached adjustment overrides after recovery. Each stops
without a third command and reproduces the rejection from a sealed trace. A
queued-policy fixture separately proves successful fresh inference and replay.

Sealed replay recomputes local checks from retained assessments and sensor
packets, rejects inconsistent check outcomes, and reproduces early aborts.
Legacy recovery traces without checks cannot establish monitored completion and
are rejected. The synthetic tests cover missing/stale observations, failed
opening/retreat, exhaustion, clearance uncertainty, assessor failures, and both
successful and unsuccessful complete-episode replays.

## Per-episode intervention and recovery attempt limits

Set trusted limits before calling the harness, for example:

```python
config = EpisodeConfig(
    seed=17, max_steps=100, max_interventions=3,
    recovery_attempt_limits=(("reopen_and_retreat", 2),),
)
```

`max_interventions` counts recovery sequences and single-action adjustments
together. When omitted (`None`), it resolves to `max_steps` and the concrete
integer is retained in new trace manifests. The default per-tool allowance is
one `reopen_and_retreat` attempt. Unlisted tools have zero allowance. Counts
must be nonnegative integers; zero disables the corresponding intervention or
tool. These are execution limits, not reviewed physical readiness settings.

The harness owns counters for the entire episode, independent of executor
instances, policy resume, and tool state. Once identity, monitor and horizon
checks pass, it charges the attempt immediately before dispatch. A two-command
recovery consumes one intervention and one tool attempt, plus two actions from
the shared horizon. A completed recovery never refunds either count. An abort
or controller exception retains the charged attempt and ends the episode under
the existing stop contract; there is no retry or counter reset through abort.
Only a new `run_episode` call, with a new episode identity and environment reset,
starts new counters.

An exhausted request records `proposal_rejected`, its specific limit reason,
and an `intervention_budget` snapshot containing the requested kind/tool,
admission outcome and cumulative counts. It dispatches no command and makes no
further selection call. Pass-through policy actions do not consume intervention
counts and remain possible when no further correction is requested. Each
admitted recovery command retains the same attempt snapshot, including partial
abort/failure evidence.

Sealed replay recomputes admissions and rejects inconsistent counts, exhaustion
reasons or execution beyond declared limits. Older annotation fingerprints retain
their historical configuration fields. Run the dedicated public tests with
`python -m unittest discover -s tests -p test_intervention_limits.py -v`.
The persistent-failure fixture completes local recovery without task success,
then requests another recovery until the exact configured limit rejects it.
Separate fixtures verify immediate abort, failed dispatch, independent tool
counts, adjustments sharing the cap, new-episode reset and corrupted accounting.
All evidence is synthetic and establishes execution behavior only.

## Matched action horizon

`EpisodeConfig.max_steps` is the same environment-action allowance for baseline
and supervised attempts. An adjusted action replaces the policy proposal and
costs one action. A two-command recovery costs two actions, even though it is
one intervention. Policy proposals, local checks and resume calls do not add
environment actions. A terminal result or abort counts only commands already
executed; the remaining recovery commands are never dispatched.

The entire recovery must fit before its first command is dispatched. With one
action remaining, a two-command recovery is rejected without consuming an
action or intervention attempt. If recovery finishes on the last allowed
action, the episode ends with `step_limit` unless that action reports success,
termination, truncation or a recovery abort. No resume or new policy inference
occurs after exhaustion. Local recovery completion alone is not task success.

The public contract checks compare baseline, numerical adjustments and recovery
under the same horizon, including consecutive recoveries, insufficient remaining
budget, terminal interruption and sealed complete-episode replay:

```powershell
python -m unittest discover -s tests -p test_corrective_action_horizon.py -v
```

These synthetic checks establish accounting behavior, not measured task gains.

## Recovery cooldown

`EpisodeConfig.recovery_cooldown_actions` declares a nonnegative integer count
of acknowledged, unmodified baseline actions required after each successfully
completed recovery sequence. For example, `recovery_cooldown_actions=2` permits
the next recovery or numerical adjustment only after two baseline actions and
their accepted observations. Recovery commands themselves do not count. The
cooldown applies across all tools and adjustments, even if the executor is
replaced. It starts again after every completed recovery; a new episode starts
with no outstanding cooldown. Aborted recoveries already terminate the episode.

The default `0` explicitly disables cooldown for compatibility with existing
configurations; active configurations must select a reviewed count. This is an
action-based contract: elapsed time, inference calls, and waiting cannot expire
it. It does not establish physical safety or observed task progress.

The declared policy rejects an otherwise valid correction during cooldown and
ends the episode with `proposal_rejected`, recording the number of baseline
actions still required. It dispatches no pending action, adds no hidden wait or
retry, and consumes no intervention attempt. An unchanged baseline proposal can
continue during cooldown under the ordinary execution/observation checks.

The effective configuration is retained in the sealed trace manifest. Replay
checks admitted interventions and cooldown rejections against acknowledged
actions; historical trace fingerprints retain their original configuration.
Run the synthetic public boundary, reset and complete-episode replay fixtures:

```powershell
python -m unittest discover -s tests -p test_recovery_cooldown.py -v
```
