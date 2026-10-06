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

Pass the result through `run_episode(..., action_selector=select)`. The selector
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
stop pending commands. After the final accepted nonterminal observation, if
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
reached. Observation-based local completion/abort monitoring remains #73;
active-correction readiness remains #94. This module enables no live run by
default and establishes no simulation or physical improvement claim.
