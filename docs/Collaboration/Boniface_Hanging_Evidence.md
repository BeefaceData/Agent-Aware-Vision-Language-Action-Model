# Offline hanging evidence contract

Branch: `hang/stage-evidence`. Baseline: `43ceae36e454c4aed9ebdd92eaf65699c265170c`.
Implementation: `development/hanging/evaluation.py`; CPU tests:
`tests/test_hang_evaluation.py`.

## Scope and input requirements

This evaluator scores explicit annotations from saved observations. It does not
detect contact, generate robot actions, alter release gates, restore physics,
or send evidence to a planner. Public annotations must be based on public
observations. Oracle scoring is a separately labeled development condition.

Input JSON must match `HangTrial`. Each assessment contains `value` (true, false
or null), observation `sequence`, `source`, an `artifact` reference, and `reason`.
Artifact references are provenance labels: this initial evaluator does not load
or hash their contents or verify an annotator's claim. Support samples additionally
include monotonically increasing simulated `elapsed_seconds`. Use simulator time,
not RPC latency, playback time or the number of frames in a sampled video.

| Field | Annotation meaning |
|---|---|
| entry | Prior acquisition/table handoff complete; right gripper currently holds mug; current evidence available |
| approach | Measured TCP reached the grounded approach target within declared motion tolerances |
| seat_alignment | Measured TCP/mug alignment with the grounded peg-seat geometry |
| contact | Independently assessed rack contact, not inferred from alignment |
| release | Gripper opened and relinquished the mug, not merely commanded open |
| retreat | Hand withdrew with the declared clearance and without dislodging the mug |
| support_samples | Mug independently observed supported by rack after release and retreat |
| native_success | Separate native benchmark checker result; never used to promote hanging success |

Stage evidence is ordered; absent evidence remains unknown. Missing or failed entry
is reported separately from hanging failure. An independent support assessment may
verify hanging even if direct contact detection remains unknown, for example when
peg contact is occluded but the released mug's sustained support is observable.
Report all stages alongside the final outcome; a final result does not erase a
failed approach or alignment assessment.

## Provisional observation criteria

The initial criteria require three or more explicitly positive support samples,
covering at least 1.0 simulated second, with no gap exceeding 0.5 seconds, after
release and retreat. Any observed support loss is failure; unknown evidence,
missing samples or temporal coverage gaps are unknown. Criteria are serialized
in the report and may be configured before a trial. These numbers are experimental
choices, not calibrated physical guarantees. Pose stability, camera coverage,
retreat clearance and visual support annotation still need a documented protocol
and real data. This is not continuous-contact proof between observations.

## Run offline

From this worktree, use an installed Python environment containing planner deps:

```powershell
python -m unittest tests.test_hang_evaluation -v
python -m development.hanging.evaluation PATH_TO_LABELED_EVIDENCE.json --output data/reports/hang_01.json
```

The output file must be new. Reports and raw observations stay under ignored
`data/`. Tests provide a synthetic input example in `trial_data()`; it is not a
recorded successful robot episode. No GPU or API calls are needed.

## Record a Claude-free baseline on Modal

From this worktree:

```powershell
$env:MODAL_IMAGE_BUILDER_VERSION = '2025.06'
$env:SIM_GPU = 'T4'
$env:MODAL_ENVIRONMENT = 'main'
python -m modal run modal_apps/simulator.py::baseline --seed 3 --max-actions 64 --chunk-size 8 --time-limit-minutes 15
```

Use an environment containing the client dependencies (the main checkout's
`.venv` is reusable). After the short integration run works, use `--max-actions
900 --time-limit-minutes 30`. Run serially. Outputs are under this worktree's
`data/runs/` and on `robotwin-runs`; the CLI prints both paths. There are no Claude,
SAM 3, oracle control, arm locks or recovery commands in this rollout. It stops
at native success or its action/time budget and does not verify hanging stability.
RGB-D and video are captured each acknowledged action; replay is diagnostic and
not real-time. Action requests and confirmations remain distinct on lost replies.

Seven CPU baseline tests pass and Modal CLI help loads. Cloud execution remains
to be validated; soft time limits cannot interrupt a blocking RPC or reset.

Update: the three-camera baseline ran on T4: 64 actions, eight chunks, no Claude
calls, outcome `action_limit`, no reported error. Optional `--perception-camera
default` now records the existing fixed RGB-D camera and four-panel video while
policy packing excludes the extra image. This optional integration has CPU tests
but has not yet run on a live scene.

For post-run segmentation, use `python -m development.replays.sam3_cache --frames
RUN_FOLDER --camera perception_camera --every 16 --max-frames 5`. This invokes the
deployed SAM 3 service on saved images; only subsequent cached replay is fully
offline. Cache creation does not alter the baseline trajectory and does not
establish contact/support. Frame selection avoids querying every view of every
saved action. No Claude calls are involved.

## Validation and remaining work

- 85 focused tests passed: 10 new offline evaluation tests, two new target/cache
  regression cases, plus 73 existing hanging tests.
- A regression first reproduced partial anchor publication when approach was
  reachable but seating was not. Target preparation now validates both poses
  before binding either anchor. An explicit cache-clear regression also passes.
- Full Windows suite: 464 tests, one failure and five errors, all matching the
  previous environment limitations (path separators, symlink privilege,
  `os.killpg`, and three missing-PyTorch checks). No new failures. Log is in the
  main checkout at `data/hang-stage-full-tests.log`; Linux verification remains.
- CLI example was exercised with synthetic evidence and produced
  `data/hanging/synthetic_report.json` inside this worktree. Its `hang_verified`
  outcome is synthetic and is not evidence of a successful robot hanging trial.
- The isolated worktree deliberately includes the LF renderer patch and
  `.gitattributes` setup fix; no acquisition/handoff trajectory changes.
- Simulator configuration used during prior setup: seed 3, T4, image builder
  `2025.06`; deployed workers `robotwin-policy` and `robotwin-perception-sam3`.
  Remote model weights and dependency bytes are not attested by these names.
- Anthropic usage currently blocks full episodes. Earlier real-scene SAM 3
  detections succeeded, but no policy actions or hanging entry were recorded.
- Cache assumes an episode-static rack; unchanged camera intrinsics cannot
  establish that the rack has not moved. Clear cache on new episode and require
  reacquisition if motion is suspected; automatic moved-rack detection is absent.
- Next: inspect preparation-error handling, add missing geometry/cache lifecycle
  regression cases, obtain a validated right-gripper entry, annotate its evidence,
  and calibrate support/clearance criteria before comparative live trials.
