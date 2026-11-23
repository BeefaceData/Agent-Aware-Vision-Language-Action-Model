# Hanging-phase implementation checklist

Owner: Boniface. Updated: 29 October 2026.

> **Status, 13 November 2026.** The checklist below reflects 29 October. Since
> then: guarded Entry 660 replay recovery (native-checker success, support
> unverified), the public supervised hanging runtime, close perception cameras,
> two-frame geometry agreement, and a supervisor that hands control back to the
> frozen policy instead of ending a run. A policy-only seed-5 run reached native
> success at action 521 on its own. Results, numbers and next steps are in
> `Boniface_Close_Camera_Evaluation.md`; the paired five-seed evaluation (seeds
> 15-19) is produced by `development/hanging/seed_evaluation.py`.

Goal: improve repeatable rack hanging while keeping the Neotix policy frozen,
the existing 14-DoF action interface, and the acquisition/table-handoff trajectory.
This checklist plans the implementation; unchecked items are not claimed complete.

## 0. Starting position

- [x] Windows client environment and Modal authentication work.
- [x] Frozen policy deployed; model loading and history reset succeeded.
- [x] Modal simulator image and asset download completed.
- [x] T4 spike passed rendering, physics and CuRobo checks.
- [x] Scripted smoke episode completed with expected `step_budget`: 724 analytic
  ticks, zero policy actions, no reported exception or timeout.
- [x] SAM 3 deployed.
- [x] SAM 3 gated model access and startup/inference test pass on T4. Test input
  is a uniform gray image; zero detections do not validate real-scene perception.
- [x] Existing five hanging-related test modules passed: 73 tests.
- [x] Offline stage-evidence evaluator and 10 regression tests implemented on
  `hang/stage-evidence` in `data/worktrees/hang-stage-evidence`; 85 focused tests pass
  including two additional target/cache regression cases.
- [x] Real-scene SAM 3 detections recorded for mug, rack and table. Full episode
  stopped before motion; a minimal API probe confirmed Anthropic usage limits.
- [ ] Actual policy action inference validated in an episode.
- [ ] A recorded episode reaches the hanging stage.

## 1. Prepare an isolated development branch — can do now

- [x] Create a clean worktree/clone and branch `hang/<short-topic>`; preserve the
  current checkout's uncommitted setup edits. Do not reset, stash or force-push it.
- [x] Carry the LF renderer-patch fix into the test checkout deliberately; record
  its diff along with the image-builder override `2025.06` and simulator GPU T4.
- [x] Record starting commit, config hashes, deployed-worker identity and seed.
  Record unresolved remote model/dependency pinning as a reproducibility limit.
- [x] Re-run the focused tests from the new checkout before changing behavior.

Use that checkout's Python environment for:

```powershell
python -m unittest tests.test_analytic_skills tests.test_hang_targets tests.test_rack_seat_gate tests.test_rack_recovery tests.test_skill_orchestrator -q
```

Done when: a clean baseline and its test results are recorded independently of
setup edits. No simulator is needed for this step.

## 2. Define outcomes and the evidence contract — can do now

- [ ] Specify hanging entry: acquisition/table handoff finished, right gripper
  holds the mug, and the current pose/evidence are available. Keep unknown
  acquisition or grasp status explicit rather than assuming entry succeeded.
- [ ] Define separate approach, seat/contact, release, retreat and sustained-hang
  results. Each result includes status, observation sequence, evidence source,
  measurements, and typed failure reason where applicable.
- [ ] Preserve unknown when evidence is unavailable or ambiguous.
- [ ] Distinguish geometric `rack_seat_aligned` from physical contact/support.
  Keep native benchmark success separate from public visual verification.
- [ ] Choose and record a post-release observation window and stability criterion
  before collecting evaluation runs. Treat tolerances as hypotheses to validate.
- [ ] Define what retreat clearance and loss of grasp/support mean using available
  sensors; do not claim force control or measured forces without an interface.

Done when: the report can represent a successful approach followed by a failed
release, without converting either into hanging success.

## 3. Build the offline hanging test matrix — can do now

Extend existing fixtures and tests; add missing cases rather than duplicating
coverage. Synthetic RGB-D/keypoints do not establish real perception quality.

| Case | Expected behavior | Main coverage |
|---|---|---|
| Valid rack frame, fresh mug frame, measured TCP | Bind rack-relative approach/seat targets | `test_hang_targets.py` |
| Missing/stale mug anchor, wrong camera or arm | Reject before motion | `test_hang_targets.py` |
| Missing rack cache or changed camera intrinsics | Reject grounding | `test_hang_targets.py` |
| Discontinuous/invalid depth or weak grounding | Return unavailable/failure; no assumed target | Perception and hang-target tests |
| Occluded episode-static rack with valid cached frame | Preserve world reference under documented static-rack assumption | `test_hang_targets.py` |
| Rack moved or cache reused across episodes | Invalidate/reacquire; do not silently reuse | Cache lifecycle tests |
| Out-of-reach pose or planning failure | Typed failure before unsafe execution | Analytic and target tests |
| Tracking error, cancellation or tick exhaustion | Stop with measured diagnostic | `test_analytic_skills.py` |
| Failed/uncertain seat move or wrong-arm release | No release authorization | `test_rack_seat_gate.py` |
| Drift or lost support during opening | Stop further opening with explicit reason | Analytic/recovery tests |
| Gripper open but mug not supported | Release may succeed; hanging must not | Outcome/report tests |
| Mug dislodged during retreat | Retreat/hanging failure recorded | Outcome/report tests |
| VLA handoff before hanging | No invented arm lock or changed action shape | Orchestrator/compatibility tests |

- [ ] For the moved-rack case, document how motion can be detected or declare the
  static-rack limitation. The current cache does not visually reverify rack pose
  on every lookup; same-camera checks alone cannot prove the rack stayed still.
- [ ] Check that preparation/geometry errors become structured planner results
  at the runtime boundary, rather than unhandled exceptions or success.
- [ ] Keep oracle contact checks in `development/` and offline scoring; audit
  planner contexts and public traces for simulator-only pose/contact leakage.

Done when: each failure case has a focused assertion and no unsupported success.

## 4. Validate grounded targets and bounded execution — can do now

Review existing implementations before introducing replacements:

- [ ] `hang_targets.py`, `pose_grounding.py`, `anchors.py`: compose rack-relative
  geometry with the current mug-handle frame and measured grasp/TCP transform.
- [ ] Keep rack frames episode-local. Freshness of the mug frame and bound anchors
  must match the command observation; cached rack validity has a separate contract.
- [ ] `preparation.py`, `assembly.py`: prepare the required target before each
  analytic command; do not fabricate world coordinates in planner responses.
- [ ] `analytic.py`, `move_to.py`, `rotate_wrist.py`: begin from measured current
  arm state, use bounded waypoints and existing motion/tracking guards.
- [ ] `gripper.py`, `release.py`, `rack_seat_gate.py`: authorize opening explicitly
  and retain alignment/support-loss checks through the opening ramp.
- [ ] `orchestrator.py`: return fresh observations and typed failures for replanning;
  bound retry counts and total execution budgets.
- [ ] Confirm the frozen VLA still controls most of the trajectory, with bounded
  analytic motion reserved for recovery or final contact behavior.

Done when: test evidence supports each change, without modifying upstream
acquisition or handoff behavior. No hard-coded starts, teleports or retraining.

## 5. Obtain a reproducible hanging entry — live data required

- [ ] After SAM 3 approval, pass its startup check and validate public perception
  on a real captured RGB-D frame. Verify Anthropic model access before a long run.
- [ ] Run one bounded, recorded Modal episode using the unchanged upstream
  acquisition/handoff route and a unique ignored `data/runs/` path.
- [ ] Confirm actual `VLA_ACT` calls and nonzero policy actions.
- [ ] Label the observation/action boundary at which the right arm holds the mug
  and hanging begins. Save complete action history and source provenance.
- [ ] Check whether existing action replay can reproduce that boundary. Replay
  from reset; do not claim a file restores full simulator or remote model state.
- [ ] Check the replayed state/evidence against the declared entry conditions;
  reject invalid entry attempts rather than scoring them as hanging failures.
- [ ] Adapt a development-only replay runner for Modal if needed. The current
  episode launcher does not automatically run arbitrary checkpoint tools.

Done when: at least one validated near-rack entry can be reproduced and inspected.
If the full trajectory never reaches it, record that upstream blocker separately.
Synthetic tests can continue, but cannot replace this validation.

## 6. Measure one baseline hanging attempt — one live run at a time

- [ ] Freeze seed, entry/replay identity, config, policy worker and evidence mode.
- [ ] Record untouched baseline approach, seating, release and retreat behavior.
- [ ] Save video, RGB-D, command/measurement logs, grounding metadata, native
  checker result and independent stage assessments.
- [ ] Identify the first failed stage using the evidence; retain uncertainty.
- [ ] Select one bounded intervention based on that failure. Do not adjust grasp,
  handoff, perception and motion simultaneously.

Done when: a report explains where the baseline failed and why the proposed
change addresses the observed failure. A positive checker alone is insufficient.

## 7. Compare recovery against baseline

- [ ] Predeclare development and held-out evaluation seeds/entries and budgets.
- [ ] Compare frozen VLA baseline, monitoring-only supervision, and supervision
  plus grounded analytic hanging recovery where each condition is executable.
- [ ] Keep entry conditions comparable. Report hanging success conditioned on
  valid entry separately from end-to-end success across all attempts.
- [ ] Report counts/denominators and uncertainty, plus stage failure reasons,
  invalid entries, interventions, elapsed time and planner/policy call counts.
- [ ] Repeat with varied valid entries/seeds; do not claim repeatability from a
  single success or treat repeated deterministic replay as diverse evidence.
- [ ] Keep development tuning separate from held-out results. Build a separate
  hanging evaluation report: the existing frozen acquisition evaluator is not
  automatically a public-perception hanging benchmark.

Done when: improvements and remaining risks are supported by comparable trials.

## 8. Review and deliver

- [ ] Focused tests pass; run the full suite in a compatible Linux environment
  and document any unresolved failures. Windows-only failures must be identified.
- [ ] Run `git diff --check` and review only intended hanging/test changes.
- [ ] Update this checklist with evidence paths and measured results.
- [ ] Commit implementation, tests and reviewed documentation only; exclude raw
  RGB-D, videos, checkpoints, model weights, credentials and generated run reports.
- [ ] PR includes problem, behavioral change, seed/entry, public evidence mode,
  validation, artifact locations, results and limitations. Push/create the PR
  only as part of the requested delivery workflow.

## Literature to consult during implementation

- [ReKep](https://arxiv.org/abs/2409.01652): relational keypoint constraints for
  rack/handle geometry; informs steps 2 and 4.
- [REFLECT](https://arxiv.org/abs/2306.15724): structured failure summaries;
  informs steps 2, 3 and 6.
- [Manipulation in contact survey](https://doi.org/10.1016/j.robot.2022.104224):
  contact/constraint reasoning; informs seating and release evaluation.
- [Inner Monologue](https://arxiv.org/abs/2207.05608): environmental feedback for
  replanning; informs bounded execution and monitoring.

## First implementation milestone

Steps 1–4 of the hanging-only implementation sequence are now documented in
`docs/Collaboration/Boniface_Hanging_Audit.md`: baseline/scope audit, entry contract,
stage evidence contract, and composed CPU hanging scenario. Six new tests pass
in `tests/test_hanging_scenario.py`; 91 hanging-focused tests pass in total. No
live run or upstream trajectory change was made. This is not completion of the
entire broader checklist above; real entry evidence and support validation remain.

Current working location: the normal `RoboTwin-Modal` folder on branch
`hang/stage-evidence`. The earlier worktree is retained detached as a backup;
do not use it for further implementation. All 108 focused tests pass here.
The short baseline completed 64 policy actions in eight chunks without Claude
or execution errors. Artifacts are in
`data/runs/baseline_seed3_20261029T173841Z_093800/`. No hanging success is claimed.

Complete steps 1–3: clean branch, explicit stage/evidence contract, and missing
synthetic failure cases. The offline evaluator, tests, and
`docs/Collaboration/Boniface_Hanging_Evidence.md` now live in this normal checkout.
Baseline identity/config hashes are saved locally in
`data/diagnostics/hang-stage-baseline.json`. Stage labels and provisional support
criteria are implemented; visual support detection, geometry/cache coverage audit
and live validation remain unfinished. Offline work can continue while Anthropic
usage is blocked; SAM 3 access is now working.
Do not start a live run solely to complete this planning checklist.
