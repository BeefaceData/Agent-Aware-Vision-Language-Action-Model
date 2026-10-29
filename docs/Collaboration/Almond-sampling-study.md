# Feedback sampling: offline study and runtime uncertainty fix

Almond · 29 October 2026 · Beeface base `43ceae36e454c4aed9ebdd92eaf65699c265170c`

## What changes

The Anthropic planner already requests one bounded VLA chunk when public
perception is unavailable. Final runtime-field normalization could overwrite
that request with the phase's 64/128-action initial window. The fix preserves
the uncertainty limit and its feedback-horizon metadata, including a remaining
budget shorter than one chunk. Clear-evidence behavior is unchanged.

`harness/sampling.py` adds a provider-independent inspection scheduler.
`development/sampling/replay.py` compares schedules on recorded trajectories.
The new scheduler is **not connected to the live orchestrator**. No runtime
preset, frozen-policy weight, action interface, or dependency changes.

## Evidence and authorship

We collected ten serial RoboTwin runs on a Colab L4: environment seeds 0–7,
plus repeatability runs for 0 and 1. The eight distinct seeds contain 6,870
executed actions. Seeds 0–3 are development; seeds 4–7 are held out. Policy RNG
seed is `1000 + environment seed`; execution chunks contain 16 actions.

The collection script recorded simulator-rendered camera views and action
indices. **The AI assistant reviewed extracted frames offline and authored the
annotations.** No independent human annotation or inter-reviewer validation
was performed. SAM3 did not produce these labels. No external planner,
detector, or semantic-predicate calls occurred during collection or replay;
this does not mean that no AI was used in the offline analysis.

Review used 32/64-action overview intervals, 8-action refinements, and a
2-action refinement around seed 4's brief drop. Transition intervals and
unknown windows preserve ambiguity. This is not exhaustive frame-by-frame
ground truth. Native simulator success is evaluation-only. Handoff means
table-mediated release and right pickup; the last visual milestone is rack
approach, not independently verified released hanging.

Boniface's 36 videos informed the earlier qualitative investigation. They do
not supply the action-aligned quantitative cohort below: their recording
configuration and frame-to-action relationship were unavailable.

## Method and result

The scheduler receives scene evidence only after requesting an inspection.
Phase clocks start at observed transitions, not evaluator-known transition
times. Unknown evidence or a reported retry restores the 16-action interval.
Per-action hard guards remain separate; stale evidence at a due inspection
stops the scheduler. Replay is alarm-only: an alarm does not alter the saved
trajectory or execute recovery.

Development selection minimizes inspection count among admissible candidates,
with delay and simplicity tie-breaks. The gate rejects additional event misses,
missed persistent visible failures, later failure detection than the 16-action
reference, and uncertainty rechecks longer than 16 actions. The 64-action and
phase-preset schedules are comparisons, not adoption candidates. The design
was frozen before held-out review.

| Schedule | Development inspections | Held-out inspections | Gate: dev / test |
| --- | ---: | ---: | --- |
| 16-action reference | 173 | 263 | Pass / Pass |
| 32 actions, 16 on uncertainty/retry | 122 | 210 | Pass / Pass |
| 64 actions, 16 on uncertainty/retry | 94 | 179 | Fail / Fail |
| Phase preset | 123 | 211 | Pass / Pass |
| Supported phase-aware 16–32 | 169 | 259 | Pass / Pass |
| Phase-aware plus gripper-command trigger | 184 | 274 | Fail / Pass |
| 16 actions plus gripper-command trigger | 188 | 278 | Fail / Pass |

The selected 32/16 rule reduces held-out inspection opportunities by **20.15%**
(263 to 210). This is conditional on the AI-authored annotations. Both schedules
detect persistent visible failures on seeds 6 and 7 at the same actions, with
onset-to-detection intervals of 0–8 and 8–16 actions. **Both miss seed 4's brief
drop at actions 468–470**, before the object leaves the camera views. Passing
the comparison gate does not mean detecting every failure.

Only four held-out episodes were tested. No real early-acquisition failure or
retry episode occurred; synthetic tests cover those cases. A 32-action interval
can have a larger worst-case detection delay than 16. A gripper command is not
proof of grasp, and extra trigger inspections can move later deadlines. The
phase-aware candidate had limited supported phase-age coverage.

Frozen weights do not imply deterministic actions: three unseeded calls on the
same input differed, while resetting the policy seed reproduced a chunk.
The repeated episodes matched actions and camera hashes in this environment.
This does not establish a universal action distribution or cross-platform
determinism.

## Using and checking the implementation

The selected configuration must be explicit; the experimental module's default
configuration is the phase-aware candidate, not the selected study rule:

```python
from harness.sampling import Action, Config, Evidence, InspectionScheduler

scheduler = InspectionScheduler(Config(
    mode="uniform", base_gap=16, max_gap=32, cheap_triggers=False,
))
decision = scheduler.tick(0, 0, hard_failure=False)
assert decision.action == Action.INSPECT
scheduler.observe(0, 0, Evidence(phase="grasp", quality="clear"))
assert scheduler.next_due == 32
```

Call `tick` for each executed-action index, retaining independent execution
guards. Only call `observe` after `INSPECT`, using the matching action and
observation sequence. Do not supply simulator-only labels to a live inspector.

```bash
python -m unittest tests.test_sampling tests.test_sampling_replay tests.test_anthropic_planner -q
python -m unittest discover -s tests -q
python -m development.sampling.replay --help
```

With the separately retained study bundle under ignored `data/sampling/`,
the replay commands are:

```bash
python -m development.sampling.replay --annotations data/sampling/annotations-development.json --data data/sampling/data-v2 --protocol data/sampling/PROTOCOL.md --out data/sampling/replayed-development
python -m development.sampling.replay --annotations data/sampling/annotations-test.json --data data/sampling/data-v2 --protocol data/sampling/PROTOCOL.md --frozen data/sampling/replayed-development/frozen-design.json --out data/sampling/replayed-test
```

Each run directory supplies `trace.jsonl` and `guards.jsonl`. Source, protocol,
configuration and split checks reject changed frozen evaluations. Preserve
the original frozen record when reproducing historical results. The raw bundle
is retained locally, not checked into this repository; unit tests use synthetic
fixtures and require neither that bundle nor a GPU/API key.

## Handoff for live integration and Gabriel's analysis

Inspection counts are not actual API, token, latency, or sensor-render savings.
Policy conditioning and repeated sensor reads continue. Across the eight
distinct seeds, policy calls by annotated phase were 56 grasp, 111 handoff,
250 hang, and 15 ambiguous-boundary calls. Phase names describe observed
progress, not confirmed success. Planner/detector/predicate counts are zero.

After credentials are available, connect the scheduler to the public inspector
and compare 16 versus 32/16 in bounded serial runs with the same policy, seeds,
chunk size, and hard guards. First independently validate the annotations and
inspect early failures. Record actual planner/perception requests, tokens,
encoding/network/inference durations, unique observations versus repeat reads,
missed/late failures, and task outcomes. Gabriel can then attribute cost and
latency to measured events. Changing the live trajectory requires a new
closed-loop evaluation; replay cannot predict recovery success.
