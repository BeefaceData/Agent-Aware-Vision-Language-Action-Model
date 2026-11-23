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

The default smoke can also stop during analytic staging before making a policy
request. A report with zero VLA samples therefore does not validate policy timing.
Use the bounded policy probe below to exercise the new transport measurements.

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

## Post-rebase review and real measurement protocol (2026-10-30)

`main` was updated to `ca49175`, including the perception and sampling changes,
and the latency commit was rebased onto it. The import conflict in
`harness/vla/modal_client.py` was resolved by retaining both upstream policy-call
timeouts and latency validation. The combined CPU suite passed **566 tests**;
42 focused latency/runner/diagnostics/baseline/transport tests also passed. The
synthetic CLI report was regenerated under
`data/latency/rebased_validation_20261030_01/`.

A GPU check outside the sandbox succeeded: NVIDIA GeForce RTX 2050, driver
595.99.02, 4096 MiB. Earlier `nvidia-smi` failures were specific to the sandbox.
The local RoboTwin checkout/assets are still absent, and no local
`ANTHROPIC_API_KEY` was configured during this check. No live runs were collected.

The offline reporting contribution is implemented and CPU-tested. The full
latency-analysis assignment is not complete: no live run has established a
bottleneck or speedup. Dedicated per-request perception, predicate, analytic-tick,
artifact-write and full-process timings still need work. Rack/table initialization
events already carry `latency_seconds` in `assessments.jsonl`, but this analyzer
does not yet consume those initialization measurements. They occur outside the
controller reconciliation. Do not call the report a complete end-to-end profile.

### A. Exercise the actual policy, without Opus or SAM 3

After completing the local README setup, create an ignored probe config from the
existing scripted config. This starts directly with a bounded VLA skill instead
of analytic staging, permits at most 3 chunks / 24 policy actions, and disables
video for this particular transport test:

```bash
conda activate robotwin
python - <<'PY'
from pathlib import Path
import yaml

settings = yaml.safe_load(Path('configs/skill_runtime.yaml').read_text())
settings['record_video'] = False
settings['max_turns'] = 1
settings['recipe'].update(stage_before_first_vla=False, max_retries=0)
settings['recipe']['vla'].update(max_chunks=3, chunk_size=8, max_policy_actions=24)
settings['budget'].update(max_skill_calls=1, max_policy_actions=24, max_vla_chunks=3)
directory = Path('data/latency')
directory.mkdir(parents=True, exist_ok=True)
(directory / 'policy_probe.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
PY

timeout -s INT 15m python run_skills.py \
  --config data/latency/policy_probe.yaml --seed 3 --development-oracle \
  --output data/runs/gabriel_policy_probe_01

python -m development.replays.latency \
  --episode data/runs/gabriel_policy_probe_01 \
  --output data/latency/gabriel_policy_probe_01
```

Run the analysis as a separate command after the episode, even if a budget
outcome gives a nonzero exit status. Reaching a skill or turn budget is acceptable
for a bounded latency probe; a setup error or policy timeout does not establish
completed inference latency.

Check these items in `data/latency/gabriel_policy_probe_01/report.json`:

- `metadata.synthetic` is false; config hash, seed, evidence mode, source revision
  and outcome describe the actual run. Keep the checkout clean for reproducibility.
- `vla_request`, `policy_pack`, `policy_rpc`, `policy_server_inference` and
  `action_round_trip` have at least one sample. Confirm `chunk_received` records
  in the source trace contain `policy_timing`. Zero counts mean that path was
  not measured, regardless of whether a video was produced.
- Reconciliation has a numeric controller elapsed and unexplained duration, or
  an explicit explanation for their absence. No negative residual or double-counted
  server inference should appear.
- Inspect warnings and completion counts. An aborted partial call is not a slow
  completed sample and must not be dropped silently from the written conclusion.

After the first probe works, repeat serially using new run/report names, for
example `_02` and `_03`, with the same config and seed. Record first-call cold
starts separately from warmed calls. The probe verifies real policy round trips;
it does not evaluate grasp success or public-perception latency.

### B. Measure a public-perception episode

Configure `ANTHROPIC_API_KEY` locally using README L1; ensure your account has
access to the configured model. Deploy the current policy and SAM 3 apps using
the README setup, including approved Hugging Face SAM 3 access. Then run:

```bash
timeout -s INT 75m python run_skills.py \
  --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml \
  --seed 3 --planner anthropic \
  --perception-factory harness.perception.modal_sam3:create_detector \
  --output data/runs/gabriel_public_latency_01

python -m development.replays.latency \
  --episode data/runs/gabriel_public_latency_01 \
  --output data/latency/gabriel_public_latency_01
```

This makes paid remote policy, segmentation and planner calls. Follow the
local-simulator requirement in this task. Do not compare its times directly with
the short oracle probe: config, video recording, evidence mode and call mix differ.
For repeated public runs, retain the same config/seed/deployments and use new
output names. Keep completed and timed-out episodes separate in the analysis.

The new perception-only camera is available to capture tools, but is not yet
enabled by a `run_skills.py` episode setting. Describe the actual head/wrist
camera setup used by these commands; do not claim the capture tool's extra view
was used. Preserve `render_once_per_step` when comparing runs; its new opt-in
setting changes rendering cost and defaults to false. Almond's sampling replay
counts do not establish live API or latency savings, and that scheduler is not
automatically wired into these episodes.

### C. Decide what the PR can claim

For a reporting/instrumentation PR, provide the passing CPU checks plus at least
one bounded real probe that confirms nonzero policy timing samples. State which
stages remain unmeasured and attach a sanitized summary instead of raw artifacts.
That supports the claim that the instrumentation records real request timings.

To close the broader assignment, additionally collect comparable public episodes,
measure the missing stages sufficiently to support attribution, and recommend one
change using direct event evidence. A speedup claim requires a controlled before /
after comparison with outcomes and action counts as regression checks. The current
supported conclusion is: **reporting and timing retention are validated on CPU;
live bottleneck attribution and performance improvements remain unverified.**
