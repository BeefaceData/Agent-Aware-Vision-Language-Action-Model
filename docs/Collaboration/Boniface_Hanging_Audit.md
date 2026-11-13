# Hanging implementation: steps 1–4

Reviewed 29 October 2026. Working folder: normal `RoboTwin-Modal` checkout.
Branch: `hang/stage-evidence`; starting HEAD:
`10fd7f0cb78c27dd268380625df9e11418d475a1`.
Locally fetched `origin/main`: `ca49175cd8150726f1429fca491423d4fdbc3e5b`.
`git merge-base --is-ancestor origin/main HEAD` passed. No network fetch was
performed for this audit, so this does not assert GitHub has no newer changes.

## 1. Baseline and PR scope

Before this change, 85 hanging-focused tests passed. After adding the composed
scenario, 91 pass. No live simulator, VLA, SAM or Claude call ran during these
steps. Existing source edits were preserved; no branch switch, commit or push.

This checkout has unrelated uncommitted setup, video, baseline-runner and SAM
work. A plain `git add .` would mix scopes. Review these hanging files separately:

- Existing hanging change: `harness/perception/hang_targets.py` and its regression
  tests in `tests/test_hang_targets.py`.
- Existing offline scoring: `development/hanging/evaluation.py`, its package
  initializer, and `tests/test_hang_evaluation.py`.
- New composed CPU scenario: `tests/hanging_scenario.py` and
  `tests/test_hanging_scenario.py`.
- Relevant reviewed hanging contracts/audit documentation.

General baseline capture, camera/video layout, SAM service/cache and setup edits
are not automatically part of the hanging PR. No files were removed or moved to
separate those changes; scope separation remains a delivery requirement.

## 2. Hanging-entry contract

A recorded trial is eligible for hanging evaluation only when all of the
following have evidence:

| Requirement | Evidence needed | Missing/contradictory evidence |
|---|---|---|
| Upstream sequence completed | Recorded left acquisition, table handoff and right pickup, or validated replay provenance | Unknown/invalid entry; do not fabricate history |
| Right arm currently holds mug | Current public visual/robot evidence supporting an actual grasp; commanded closure alone is insufficient | Unknown/invalid entry |
| Measured robot state current | Measured joints and TCP pose for the command observation | No analytic move |
| Rack reference valid | Public same-frame RGB-D grounding cached in this episode, with matching camera contract and static-rack assumption | Reacquire or reject |
| Mug functional frame current | Mug-handle geometry grounded in the current command observation | Reacquire or reject |
| Execution budget available | Bounded skill/tick/retry budgets | Typed budget failure |

`HangTrial.entry` stores a reviewed tri-state assessment, sequence, source,
artifact reference and explanation. It is not an automatic grasp/history detector.
Its explanation must identify evidence for the entry requirements above. Failed
entry is `invalid_entry`; absent/ambiguous entry is `entry_unknown`, not a hanging
failure. Artifact references are not yet byte-attested by the evaluator.

The synthetic scenario assumes a valid right-held mug and fixed grasp. It does
not execute acquisition/handoff, establish actual grasp support or restore a
checkpoint. All tests are explicitly synthetic. Oracle truth, if used in later
offline scoring, must remain in a separate `development_oracle` condition.

## 3. Stage outcomes and evidence

| Stage | Passing evidence | What completion alone cannot establish |
|---|---|---|
| Approach | Measured endpoint within target tolerances after bounded rack-relative motion | Peg contact or mug support |
| Seat alignment | Measured endpoint matches the freshly composed seat target | Physical contact/support |
| Contact | Independent public contact/support assessment, or explicitly separate development label | Not inferable from `rack_seat_aligned` alone |
| Release | Actual relinquishment assessed separately from a completed opening command | Open gripper does not prove supported hanging |
| Retreat | Measured withdrawal and declared clearance with no observed dislodgement | Continued support after the hand leaves |
| Sustained hang | Released mug independently observed supported after retreat for the declared window | Native task checker alone is insufficient |

Runtime endpoint defaults are 15 mm and 0.10 rad; these are existing controller
tolerances, not experimentally validated hanging tolerances. Offline initial
support criteria are at least three samples spanning 1.0 simulated second with
no interval over 0.5 seconds. These provisional choices must be calibrated before
live comparative evaluation. Unknown evidence stays unknown. Report native
benchmark success separately from visual hanging verification.

## 4. Existing coverage and new composed scenario

| Area | Coverage already present | Added here / remaining gap |
|---|---|---|
| Rack/mug target composition | Current mug frame, measured grasp/TCP transform, cached rack frame | Scenario recomposes targets during a synthetic rigid translation |
| Freshness and calibration | Stale mug reports, camera changes, depth discontinuity, explicit cache clear | Scenario checks new observation sequence at each target preparation |
| Target validation | Both targets validated before either is published | No additional target algorithm change |
| Motion | Two measured starts, waypoint bounds, endpoint settling, invalid plans, tracking errors | Composed sequence checks two entry poses and failed seating feedback |
| Release gate | Completed seat move, wrong arm, drift, explicit authorization | Composed scenario confirms alignment permits opening without proving contact; drift blocks opening |
| Recovery tests | Approach/opening order, release support loss, withdrawal ordering | Some cases exercise development/legacy recovery; they do not prove production public-perception behavior |
| Runtime grounding feedback | Allow-listed preparation errors returned to planner | New scenario covers missing/stale grounding, depth discontinuity and out-of-reach errors before any motion |
| Stage scoring | Failed/unknown entry, unsupported release, support loss and observation gaps | No automatic visual support detector or continuous contact proof |
| VLA upstream behavior | Existing orchestrator/policy tests | This scenario makes zero VLA calls and changes no upstream trajectory |

The scenario uses the real `SkillExecutor`, `SkillOrchestrator`,
`RackSeatReleaseGate`, `AnchorRegistry`, cached rack RGB-D frame and pose-composition
function. It reuses `AnalyticEnvironment`; its fake planner schedules four bounded
right-arm commands, then an explicit end-of-scenario abort. Successful execution
therefore ends with `scenario_complete`, not `task_success`.

Synthetic contact and mug-release predicates remain unknown. The scenario's
fresh synthetic mug frame assumes rigid translation with the current TCP and no
rotation; it does not replace real perception tests or validate dynamic slip.
The sampled artifact references (`synthetic/<sequence>.npz`) are test labels,
not files claiming a recorded observation. No production planner is introduced.

Run from the normal repository folder:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_hanging_scenario -v
.\.venv\Scripts\python.exe -m unittest tests.test_hanging_scenario tests.test_hang_evaluation tests.test_analytic_skills tests.test_hang_targets tests.test_rack_seat_gate tests.test_rack_recovery tests.test_skill_orchestrator -q
```

## Next stage — not completed by this audit

### Step 11 follow-up: recorded entry and replay inspection

Inspected the four existing `data/runs/` folders locally, without simulator,
policy or perception-service calls. No valid right-arm hanging entry was found.

| Recording | Available evidence | Entry assessment |
|---|---|---|
| `baseline_camera_20261029_195201` | Seed 3; 64 requested/confirmed policy actions, eight chunks, 65 RGB-D/TCP observations, video, extra perception camera | Reviewed all 65 head-camera frames: mug remains at the table; no demonstrated handoff/right-held rack approach |
| `baseline_seed3_20261029T173841Z_093800` | Seed 3; 64 requested/confirmed actions, eight chunks, 65 RGB-D/TCP observations, video | Existing contact sheet at 0/16/32/48/64 shows table acquisition activity; no validated hanging entry |
| `modal_smoke_01` | Two observations, 724 analytic ticks, development-oracle log, zero policy actions | Acquisition unconfirmed; not an eligible natural hanging entry |
| `T4_full_20261029_121546` | Reset RGB-D observation and one video frame; zero policy actions | Planner request/validation failed; rack initialization reports BadRequestError |

Both baseline logs have contiguous requested indices 1–64 and matching confirmed
indices. Their manifests specify a 64-action limit; neither is the proposed
1,200-step recording. This checks journal pairing, not physical execution or
replay determinism. Saved command state is not proof of measured joint tracking.
NPZ files also contain left/right TCP poses, RGB-D, intrinsics and extrinsics.

The camera run's cached SAM results cover sequences 0/16/32/48/64. Mug/table
detections and occasional rack detections are available; they do not establish
right-hand grasp, peg/handle frames, contact or support. No oracle labels were
used to certify a public entry.

Replay tools exist, but no `data/checkpoints` bank or `data/entries` folder is
present. `development/checkpoints/replay.py` expects a validated checkpoint bank,
hashed assets, complete RecordedFrame prefix and matching local provenance. It
resets and replays actions on a reserved simulator, then compares command state
and final RGB. It does not restore full physics state. Baseline journals are not
direct checkpoint-bank inputs and lack the required provenance/asset manifest.
`extract.py` extracts earlier boundaries from an existing bank checkpoint; it
does not import these baseline folders. `acquisition_entry.py` requires a
successful natural acquisition episode, which these summaries do not provide.
Offline perception replay loads saved observations/masks only; it cannot restore
a robot state or establish a held mug.

Blocker: no recorded eligible entry and no validated replay restoration. Further
hanging evaluation requires a longer unchanged upstream rollout or a supplied
validated entry recording, followed by public entry review and serial simulator
replay validation. No upstream trajectory change is justified by this inspection.
Only this audit documentation changed in step 11; a contact sheet was generated
in the system temporary folder for inspection. No live run, API, commit or push.

### Step 10 follow-up: failure matrix audit

The following matrix distinguishes synthetic public-interface tests (fabricated
RGB-D, reports and measured state; no real scene validation) from privileged
development-oracle recovery checks. Neither constitutes a physical hanging trial.

| Failure / boundary | Existing or extended test modules | Evidence and limit |
|---|---|---|
| Missing/stale rack or mug grounding, wrong camera, future cache | `test_hang_target_preparation`, `test_hang_targets` | Synthetic public inputs; structured feedback before motion |
| Weak confidence, depth discontinuity, changed calibration | `test_hang_target_preparation`, `test_hang_targets` | Added three actual preparation/runtime rejection cases; zero ticks |
| Occluded static rack, moving camera, explicit cache clear | `test_hang_targets` | Synthetic public geometry; no moved-rack detection or automatic episode identity check |
| Unreachable target, invalid plan, bounded waypoints | `test_hang_target_preparation`, `test_analytic_skills` | Synthetic public state; no collision-free real-scene guarantee |
| Measured starts, tracking error, settling, cancellation, budgets | `test_analytic_skills`, `test_hang_motion` | Fake robot measurements/controller execution |
| Failed/uncertain seating, wrong-arm or stale/oracle authorization | `test_rack_seat_gate`, `test_hanging_release`, `test_hanging_scenario` | Synthetic public gate contract; oracle anchor rejected |
| Alignment unavailable/lost during opening | `test_hanging_release` | Synthetic measured TCP; checked at monitor boundaries |
| Observed support unknown/lost during opening | `test_analytic_skills` | Injected perception-provider values; does not validate a support detector |
| Grasp slip, simulator contact support, legacy withdrawal prerequisites | `test_rack_recovery` | Development-oracle recovery only; not production public evidence |
| Failed release followed by retreat, failed retreat, dislodgement | `test_hanging_scenario`, `test_hanging_retreat`, `test_hang_evaluation` | Scripted sequencing and annotated support; no automatic public dislodgement detector |
| Native success during opening, bounded grounding retries | `test_hanging_runtime` | Synthetic checker and planner; native result distinct from stage failure |
| Mixed public/oracle scoring and ambiguous support | `test_hang_evaluation` | Separate source modes enforced; annotations trusted, artifacts not attested |
| Upstream handoff/interface compatibility | `test_handoff`, `test_skill_orchestrator` | CPU regression coverage; no new policy inference or trajectory evidence |

Only missing integration coverage was added: weak confidence, discontinuous mug
depth and changed intrinsics now run through the actual preparer and runtime
feedback boundary. All pass without production changes; no new defect was
reproduced. The combined CPU suite passes 203 tests. `git diff --check` passes
with the existing evidence-document line-ending warning. No live runs, APIs,
commits or pushes occurred; unrelated edits and grasp/handoff behavior remain.

Remaining validation gaps: actual hanging entry/replay, public grasp-slip and
physical support detection, moved-rack detection, collision/retreat clearance,
and sustained support after release. Synthetic success or development-oracle
checks must not be reported as closing these gaps. Full compatible Linux-suite
validation also remains outstanding. Next inspect existing recordings for a
valid hanging entry before requesting any new live run.

### Step 9 follow-up: hanging runtime integration

Four CPU tests in `tests/test_hanging_runtime.py` validate assembly's shared
perception/anchor/release-gate wiring, fresh anchors on successive observations,
and the initial grounding attempt plus two bounded reselections without motion.
Native success during a correctly authorized opening allows full opening, then
terminates the benchmark episode; it does not prove retreat or sustained hanging.

A regression reproduced native success masking a later
`rack_seat_alignment_lost` failure during guarded hanging release. The terminal
outcome now preserves that failed guarded-release result and marks phase FAILED,
while the separate `task_success` field retains the native checker evidence.
The change is restricted to rack-seat-authorized RELEASE; observed-support
opening, upstream grasp/handoff and the frozen VLA interface are unchanged.

Validation: 202 tests pass including runtime assembly, hanging integration,
release/retreat, geometry, analytic motion, handoff and Anthropic planner CPU
tests. `git diff --check` passes apart from the existing line-ending warning.
No live runs, external APIs, commits or pushes. Automatic post-success retreat
was not introduced; physical withdrawal and support still require separate
validation rather than extending a terminal episode implicitly.

### Step 8 follow-up: hanging retreat

Four new tests in `tests/test_hanging_retreat.py` validate withdrawal from the
post-release measured pose/joints (including changes since seating), keeping the
hand open and preserving the inactive arm, workspace rejection before planning,
and planning rejection before any new writes. A completed release is retained
as such when retreat planning fails. Separate scoring tests confirm that completed
retreat and even native success cannot override unknown or explicitly lost support.
All new tests pass against the existing controller; no production code change
was justified in this step.

Existing composed scenario tests already stop sequencing after failed seating
or release. Existing legacy/development recovery tests explicitly require
authorized full opening before `withdraw()`. The public runtime instead has a
generic MOVE_TO selected by a planner: it has no independent dedicated retreat
authorization gate. These tests do not establish that every planner will choose
the correct sequence or that an arbitrary retreat vector is collision-free.
The 5 cm vertical offset is synthetic test geometry, not a validated rack-relative
live withdrawal prescription. Live clearance, contact and dislodgement remain
unverified, and native success may terminate the runtime before retreat.

No new global MOVE_TO restriction was added: that would also affect legitimate
recovery/transport and risk changing grasp/handoff behavior. Runtime integration
and actual recorded hanging evidence must address those remaining boundaries.

### Step 7 follow-up: guarded hanging release

Four new test methods cover public-anchor/result provenance, uncertain/terminal
seating, wrong-arm opening, and alignment loss/unavailable measured pose during
the opening ramp. A regression reproduced that the gate accepted stale/future
anchor sequences, oracle anchors, and mismatched skill/controlled-arm results.
It now requires a perception-derived anchor from the move's starting observation
and a completed MOVE_TO result for the commanded arm, in addition to its existing
uncertainty, terminal and endpoint checks. Invalid records clear prior approval.

Opening itself required no code change: wrong-arm release is rejected with zero
ticks, and lost/unavailable alignment stops further opening at the existing
25-tick monitor boundary. This polling interval does not provide continuous
support verification between checks. Geometric seat authorization still does
not establish physical contact/support. Existing observed-support gripper tests
and upstream scenario tests remain passing; no grasp/handoff trajectory changed.

Validation: 107 focused tests pass, `git diff --check` passes with only the existing
line-ending warning. No live runs, external APIs, commits or pushes.

### Step 6 follow-up: measured-start motion

Existing tests already cover measured starts, bounded translational/rotational
waypoints, tracking on both arms and settling with persistent endpoint errors.
Four new tests in `tests/test_hang_motion.py` reproduce a settling gap for both
right-arm MOVE_TO and ROTATE_WRIST, and exercise cancellation after planning,
cancellation mid-motion, and the remaining episode tick budget. Previously a
single final in-tolerance sample could report completion after the settling
window despite failing the existing three-stable-sample requirement. Completion
now also requires that stability condition; the existing `target_not_reached`
result and endpoint measurements are retained on failure. No trajectory,
tolerance, upstream grasp/handoff or release authorization logic was changed.

Validation: 103 focused tests passed, `git diff --check` passed (existing CRLF
warning only). No live runs, external calls, commits or pushes. Cancellation and
budget tests passed before the fix; only settling completion needed code changes.

### Step 5 follow-up: actual target preparation

The real `PerceptionAnchorPreparer`/RGB-D path is now exercised through the
orchestrator in `tests/test_hang_target_preparation.py`. Three new test methods
cover eight rejection scenarios plus current measured-TCP changes across
observations and rejection of stale bound anchors. Degenerate keypoint axes,
missing depth and mug-camera mismatches previously became generic controller
errors; these specific grounding failures now return bounded planner feedback.
Rack references from a later observation are rejected before any target is used.
Missing rack/stale mug and an actual out-of-reach rack composition also stop
before motion. Unknown errors remain errors rather than being broadly swallowed.

Validation: 99 tests pass across preparation, target geometry, scenarios,
evaluation, analytic skills, release gate, recovery and orchestrator. No live
run, external API call, commit or push. Existing unrelated edits were preserved.
Same-observation anchor reuse and episode-static rack assumptions remain; sequence
checks cannot detect a moved rack or distinguish episodes with reused sequences.
Cache construction/clear per episode remains the caller's lifecycle obligation.

Continue geometry/execution failure coverage only where the audit shows a gap.
Then obtain a validated hanging-entry replay and independently assess actual
release, support and retreat. The current static-rack cache cannot detect a rack
moving while retaining the same camera calibration. A passing synthetic sequence
does not show safe or repeatable physical hanging. Full Linux checks and live
evidence are still needed before claiming a measured hanging improvement.
