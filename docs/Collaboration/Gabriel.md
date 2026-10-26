# Gabriel — latency analysis

## Goal

Find where wall-clock time is spent in a supervised episode and recommend changes supported by event-level measurements. Separate simulator/reset and action round trips, VLA inference, image encoding/network transfer, Opus planner requests, perception, predicate evaluation, analytic ticks, and artifact/video writing. Do not infer a bottleneck from episode duration alone.

## Where to work

- `harness/vla/modal_client.py`: observation packing, policy request, and response timing.
- `harness/skills/vla_act.py`: VLA chunk inference and per-action execution events.
- `harness/runtime/orchestrator.py`: planner selection and full turn timing.
- `harness/perception/vision_request.py`, perception providers, and `run_skills.py`: image/perception requests, preparation, initialization, and output setup.
- `harness/environment/local.py` and `environment/robotwin.py`: local simulator observation/action timing.
- `harness/skills/executor.py` and `skills/motion.py`: skill elapsed time and analytic physics ticks.
- Existing diagnostics: `harness/planner/diagnostics.py`, `harness/evaluation/`, `development/replays/`, and output JSONL/summary files under `data/`.

## Measurement method

1. Pick a set of comparable runs and verify config, seed, evidence mode, software commit, and whether each run completed or timed out.
2. Parse monotonic event durations where available. Report sample count, median, p90, and total per stage; distinguish client-observed latency from server inference and simulator time.
3. Reconcile stage totals with wall-clock elapsed time. Identify uninstrumented intervals explicitly; never assign unexplained time to SAM 3, Opus, or the policy without direct timing evidence.
4. Use offline replay or injected fake clients to validate timing attribution without GPU use. Keep live tests serial and bounded.
5. Recommend one change at a time and include an expected metric, tradeoff, and regression check. Do not delete observability just to reduce measured overhead.

## Constraints

Never log `ANTHROPIC_API_KEY`, `HF_TOKEN`, authorization headers, raw provider errors, or private payloads. Keep robot simulator/CuRobo local and policy inference on the existing Modal app. Do not modify policy weights or run concurrent simulator jobs. Validation traces, videos, images, and reports under `data/` remain local; PRs should include a short sanitized result summary rather than raw run artifacts.

## Branch and PR

Use a clean clone/worktree; the shared workstation has uncommitted code on `main`.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c latency/<short-topic>
python -m unittest tests.test_local_simulator_diagnostics tests.test_baseline_compatibility -v
python -m unittest discover -s tests -q
git diff --check
git add <only-your-files>
git commit -m "Improve latency instrumentation"
git push -u origin latency/<short-topic>
gh pr create --base main --head latency/<short-topic>
```

If the checkout is dirty, use a separate clean worktree instead of stashing or resetting another contributor’s work. The PR should include run selection, counts, median/p90 and wall-time reconciliation, missing instrumentation, test results, and a concrete next experiment. Do not commit full logs, videos, checkpoints, weights, or credentials.
