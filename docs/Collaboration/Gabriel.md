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

## Implemented first task: saved-episode latency report

Branch: `gabriel-latency-analysis`.

`development/replays/latency.py` reads a saved supervised run without importing the
simulator, Modal SDK, planner SDK or NumPy. It produces `report.md` for reading and
`report.json` for comparisons. The command itself uses only Python's standard library.
It does not call any provider or execute robot actions. Finish or stop the episode
before analyzing it so the input files are stable.

For each measured stage it reports count, median, p90 and total **in seconds**.
p90 uses linear interpolation; event samples from one episode are not independent
episode trials. Source-file SHA-256 hashes bind each report to the files it read.
Reports include seed, evidence settings, outcome, action/tick counts and a settings
fingerprint. New local runs also record the config name/hash, source commit and
dirty status in `manifest.json`. A dirty checkout needs its diff as well as its
commit to reproduce it; keep that diff locally with your experiment notes. Older
runs and Modal containers without `.git` have unknown source revisions; the
analyzer never substitutes its own current revision.

The policy adapter now preserves safe numeric packing, RPC, server compute and
inference measurements in `chunk_received.policy_timing`, and reset round-trip
time in `policy_history_reset.reset_ms`. Older runs still work but cannot recover
details that were discarded when they were recorded. Policy weights and action
selection are unchanged.

### 1. Try the report immediately, without a GPU

Run from the repository root:

```bash
conda activate robotwin
python -m development.replays.latency --help
python -m development.replays.latency \
  --episode tests/fixtures/latency_episode \
  --output data/latency/my_synthetic_check_01
cat data/latency/my_synthetic_check_01/report.md
```

The fixture has deliberately invented durations. The output must be labelled
**SYNTHETIC VALIDATION**, and its reconciliation must be:

```text
20 seconds controller elapsed = 6 seconds planner + 11 seconds skills + 3 seconds unexplained
```

Use a new report output directory each time. The CLI refuses to overwrite an
existing directory or write reports inside the source episode. It leaves source
files untouched. Missing measurements appear as `unmeasured` in Markdown and
`null` in JSON, rather than zero. Interrupted traces may lack elapsed time;
the report then says reconciliation is unavailable. An incomplete trailing JSONL
record is ignored with a warning; malformed complete records are rejected.

### 2. Collect your first real local episode

Complete README route 2 first: a working NVIDIA driver and Vulkan, the pinned
RoboTwin checkout/assets/patches, CuRobo and the deployed Modal policy. Check:

```bash
conda activate robotwin
nvidia-smi
vulkaninfo --summary
modal token info
```

Then run one bounded scripted episode with an explicit seed:

```bash
timeout -s INT 15m python run_skills.py \
  --config configs/skill_runtime.yaml \
  --seed 3 --development-oracle \
  --output data/runs/gabriel_latency_smoke_01
```

The README documents `step_budget` as an expected smoke outcome. The runner
returns a nonzero exit status for that outcome; inspect `summary.json` and run
the analysis as a separate command even then. A watchdog interruption may return
124 and have no controller elapsed time. If initialization fails, the report can
inspect available diagnostics but cannot reconstruct an episode that never ran.

This scripted run does not measure SAM 3 or Opus calls. Use the README's full-task
public-perception command for that later, once the basic pipeline works. Keep
local and cloud simulator measurements separate, and record Modal deployment/GPU,
warm versus cold conditions, driver version and any external setup changes in
your local experiment notes. The manifest does not establish deployed worker
identity or GPU model by itself. Never run two simulator jobs concurrently.

### 3. Analyze any saved episode later

```bash
python -m development.replays.latency \
  --episode data/runs/gabriel_latency_smoke_01 \
  --output data/latency/gabriel_latency_smoke_01
cat data/latency/gabriel_latency_smoke_01/report.md
```

The analyzer reads `manifest.json`, `summary.json`, `assessments.jsonl`,
`skill_events.jsonl` and `simulator_timing.jsonl`. It does not need videos or
observation archives. Missing files generate warnings; at least one recognized
source must exist. The command applies to downloaded Modal episode folders too,
but use the local simulator for the live measurements assigned in this guide.

### 4. Interpret the report correctly

Only planner selections and inclusive skill durations are added to reconcile
`summary.elapsed_seconds`. Those spans execute serially inside
`SkillOrchestrator.run`. The same skill result also appears in planner-turn logs;
the report uses canonical `skill_events.jsonl` results once. Duplicate identities,
invalid sibling durations and totals exceeding elapsed time disable reconciliation.

Policy requests, server inference, action round trips and simulator phases are
nested diagnostic rows. **Do not sum the whole stage table.** Simulator reset is
outside the controller's elapsed scope. Setup, resource cleanup and full process
wall time are not measured by that summary. The residual is unexplained time
within the controller, not a measurement of network, SAM 3 or Opus latency.

Current gaps are explicit: dedicated perception, predicate, analytic-tick and
artifact/video-writing wall times are unavailable. Simulator observation timings
cover request-generated observations, not every direct local `observe()` call.
The RPC-minus-compute residual can include queueing and cold starts as well as
transport, so it must not be called network transfer time. Failed/in-flight
requests have no completed-span duration and remain in the unexplained portion.

For a future comparison, select runs with matching config/settings hashes, seed,
evidence mode, source revision, worker deployments and warm/cold conditions.
Keep timeouts and successful completions as separate groups. Generate one report
per episode; do not average event-level p90 values into a campaign p90. This
version does not aggregate multiple episodes automatically.

### 5. Verification and initial analysis report

Validation performed on 2026-10-29:

- Ran the CLI on `tests/fixtures/latency_episode`; local reports are under
  `data/latency/synthetic_validation_01/`.
- Ran the actual `run_skills.main` wiring with the CPU fakes from
  `tests/test_run_skills_episode.py`, saved its output and analyzed it. Local
  artifacts are under `data/latency/cpu_validation_01/`. The fake episode reached
  `acquisition_confirmed`, with one planner turn, one skill call, one VLA chunk
  and two actions. Its timings describe CPU fakes only, and its report is marked
  synthetic. This verifies serialization and timing attribution across the runner
  and analyzer, not the simulator or remote policy.
- Full CPU suite: 464 tests passed. Coverage includes statistics/units, nested
  spans, missing measurements, invalid durations, interrupted JSONL, duplicate
  results, inconsistent totals, output protection, provenance, policy timing
  filtering, runner integration and baseline compatibility.
- Live GPU validation was unavailable: `nvidia-smi` could not communicate with
  the NVIDIA driver, and no local RoboTwin checkout or saved live runs existed.
  No real robot performance bottleneck or speedup is established by these checks.

Known-duration fixture results (invented values used to verify the analyzer):

| Stage | Count | Median (s) | p90 (s) | Total (s) |
|---|---:|---:|---:|---:|
| Planner selection | 2 | 3.0 | 3.8 | 6.0 |
| Inclusive skill execution | 2 | 5.5 | 7.5 | 11.0 |
| VLA client request, inside skill | 1 | 3.0 | 3.0 | 3.0 |
| Policy RPC, inside request | 1 | 2.5 | 2.5 | 2.5 |
| Server inference, inside RPC | 1 | 1.5 | 1.5 | 1.5 |

The fixture accounts for 17/20 seconds (85%) of controller elapsed time through
non-overlapping sibling spans, leaving 3 seconds explicitly unexplained. Adding
VLA, RPC or inference again would overstate the measured total. Its reset takes
5 seconds outside the controller scope and is not added to that reconciliation.

The concrete change delivered here is retaining policy timing details that the
supervised adapter previously dropped. The next experiment is to collect the
bounded live smoke run above and inspect the largest measured contributor. If
unexplained time is substantial, add dedicated monotonic timing for direct
observations and predicate/perception boundaries before proposing a speed change.
Expected metric: reduce unexplained coverage by attributing existing work, without
changing action counts or outcomes. Tradeoff: extra timer and log-write overhead.
Check overhead with injected clients and rerun baseline compatibility tests. A
performance optimization is deferred until live measurements support it.

Re-run verification with:

```bash
python -m unittest tests.test_latency_report tests.test_skills tests.test_run_skills_episode -q
python -m unittest tests.test_local_simulator_diagnostics tests.test_baseline_compatibility -v
python -m unittest discover -s tests -q
git diff --check
git branch --show-current
```

When writing your later result summary, include run IDs and matching conditions,
outcomes/timeouts, counts, median/p90/total for relevant stages, the controller
reconciliation, missing measurements, one proposed change with an expected metric
and tradeoff, and the regression checks. Keep full generated reports under ignored
`data/`; include only a sanitized result summary in the PR.
