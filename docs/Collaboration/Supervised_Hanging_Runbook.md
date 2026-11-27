# Supervised hanging: contributor runbook

This is the current workflow for full `hanging_mug` episodes after rebasing a
contributor branch onto `main`. It uses the frozen VLA for acquisition and
handoff, public RGB-D perception at planner feedback boundaries, immediate
analytic hanging takeover after a confirmed right-arm lift, and an optional
development-oracle fallback for evidence that public perception cannot yet
provide.

The important latency rule is: **one structured visual assessment per planner
boundary, reused by every predicate and by the planner**. Do not call Opus once
per predicate or once per policy action.

## 1. Rebase and verify locally

Work on a feature branch, never directly on `main`:

```bash
git fetch origin
git rebase origin/main
conda run -n robotwin python -m unittest discover -s tests
```

Resolve conflicts by preserving the current boundary cadence and the single
`PublicRGBDHangingProvider` instance. In particular, do not restore a planner
path that uploads the same observation images independently of the perception
provider.

## 2. Modal account setup

Select the intended Modal profile and confirm its secrets:

```bash
conda run -n robotwin modal profile current
conda run -n robotwin modal secret list
```

The account needs:

- `anthropic` with `ANTHROPIC_API_KEY`;
- `huggingface` with `HF_TOKEN` when deploying SAM 3.

Never place either value in the repository or a run command.

Deploy the persistent applications after a rebase or source change:

```bash
conda run -n robotwin modal deploy modal_apps/policy.py
conda run -n robotwin modal deploy modal_apps/sam3.py
conda run -n robotwin modal deploy modal_apps/simulator.py
```

All three applications default to T4. The supervised launcher below calls the
deployed `robotwin-simulator/run_supervised_hanging` function; it does not
create a new ephemeral Modal application.

## 3. Launch one hybrid-development episode

Use a unique output directory:

```bash
conda run -n robotwin python -m modal_apps.deployed_supervised_hanging \
  --model claude-opus-5 \
  --seed 3 \
  --max-actions 900 \
  --chunk-size 16 \
  --time-limit-minutes 75 \
  --max-turns 80 \
  --complementary-view \
  --perception-view close \
  --development-oracle-fallback \
  --output data/validation/NAME
```

`--development-oracle-fallback` means:

1. public perception is always asked first at a feedback boundary;
2. a public `true` or `false` result is retained;
3. the simulator oracle is queried only when the public result is `unknown` or
   unavailable;
4. hanging motion uses public geometry first, then development-oracle targets
   only when public geometry cannot be validated;
5. RoboTwin native success may be the final simulator-only judge.

Remove the flag for a public-only experiment. Public-only runs are currently
expected to expose perception gaps and should not be presented as the validated
baseline.

## 4. Resume a detached call

The launcher writes `<output>.modal-call.json` before waiting. If the terminal
disconnects, do not launch a duplicate episode. Resume it with:

```bash
conda run -n robotwin python -m modal_apps.deployed_supervised_hanging \
  --resume-state data/validation/NAME.modal-call.json
```

The remote episode continues independently of the local terminal. The launcher
downloads the completed artifacts from the `robotwin-runs` volume.

## 5. Current Opus cadence

With the standard chunk size of 16 actions:

| Task period | First feedback | Later feedback |
|---|---:|---:|
| GRASP | after 64 policy actions | every 2 chunks (32 actions) |
| HANDOFF | after 128 policy actions | every 2 chunks (32 actions) |
| HANG | after 32 policy actions | hanging takeover is immediate once right lift is confirmed |

At each boundary:

1. `PublicRGBDHangingProvider.observe()` sends the available labeled camera
   images once and caches the structured `VisibleHangingReport` for that
   observation sequence.
2. All public predicate checks reuse that report. Fourteen predicate checks do
   **not** mean fourteen Opus calls.
3. The planner receives the structured public assessment with
   `include_observation_images=False`; it does not upload the same scene again.
4. Per-action grasp-failure monitoring uses local evidence. In the hybrid
   development mode, the development oracle owns this monitor; remote visual
   perception is not polled inside every VLA chunk.

Do not reduce the feedback interval globally merely to satisfy a temporal
perception predicate. If a milestone needs multiple close observations, add a
bounded milestone-specific evidence burst triggered by cheap local evidence.

## 6. Sequential cross-seed validation

Run one Modal episode at a time because the frozen-policy service maintains one
history:

```bash
for seed in 3 5 7 9; do
  output="data/validation/hybrid_seed${seed}_YYYYMMDD"
  conda run -n robotwin python -m modal_apps.deployed_supervised_hanging \
    --model claude-opus-5 \
    --seed "${seed}" \
    --max-actions 900 \
    --chunk-size 16 \
    --time-limit-minutes 75 \
    --max-turns 80 \
    --complementary-view \
    --perception-view close \
    --development-oracle-fallback \
    --output "${output}"
  echo "seed ${seed} exit status $?"
done
```

Do not run these episodes in parallel and do not reuse an existing output path.

## 7. What to inspect

Each output contains:

- `report.json`: terminal outcome, Opus usage, evidence routing and recovery
  summaries;
- `supervisor_events.jsonl`: boundary assessments, predicate sources, VLA
  chunks, analytic stages and fallback reasons;
- `episode.mp4`: smooth head-camera review video;
- `observations/*.npz`: saved RGB-D boundary observations;
- `video_frames.jsonl`: recording provenance.

Key report fields:

```text
summary.evidence_resolution.mode
summary.evidence_resolution.resolution_counts
summary.evidence_resolution.oracle_fallbacks_by_predicate
summary.hanging_recovery.information_source
summary.hanging_recovery.task_success_confirmation
claude_usage.by_tool.report_public_hanging
claude_usage.by_tool.vla_act
```

An `oracle_fallback_total` is a count of predicate resolutions, not API calls.
Use `perception_requests` and
`claude_usage.by_tool.report_public_hanging.requests` to count visual Opus
requests.

## 8. Before opening a PR

```bash
conda run -n robotwin python -m unittest discover -s tests
git diff --check
git status --short
```

The PR description should separate:

- public evidence that controlled a decision;
- development-oracle fallback evidence;
- native simulator success;
- visual observations made manually from the video.

Do not describe hybrid-development success as public-perception-only success.
