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
