# Agent-Aware Vision-Language-Action Model

Investigating an AI supervisor that observes a frozen vision-language-action (VLA) policy, detects failures or stalled progress, and eventually intervenes with targeted corrections—without retraining the underlying policy.

The current implementation runs **SmolVLA in LIBERO simulation**, recording an episode video, every executed action, and the final task outcome. It establishes the baseline for developing and evaluating the supervisor.

**Status:** the reusable harness now includes supervisor decision handling,
failure-assessment signals, checked recovery and action-adjustment mechanisms,
recording/replay, and most cross-episode memory features. These are verified
primarily through deterministic behavioral and complete-episode replay tests.
The standalone command still runs the frozen baseline; live supervised integration,
detector/correction calibration, and comparative evaluation remain to be completed.
No measured improvement from supervision or memory is claimed, and no GRPO
training or policy fine-tuning has been performed.

## What is included

- `run_smolvla_episode.py`: standalone runner for one selected LIBERO task.
- `README.md`: environment setup, usage, output descriptions, and initial results.
- `.gitignore`: excludes generated runs, model weights, caches, and local environments.
- `episode_harness.py`: shared episode execution with optional supervision and memory.
- `tests/`: behavioral tests using public interfaces and complete replay episodes.
- `docs/`: implemented workflows, controller evidence, and accepted design decisions.
- [Client interface handoff packet](docs/client-interface-handoff.md): unsent checklist for Boniface to coordinate the client contract, timing and permissions; missing answers remain explicit.

The runner loads [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero), freezes its weights, uses LeRobot's observation and action processors, and steps one simulation environment until success, termination, truncation, or the step limit.

This is a custom adaptation of LeRobot's evaluation pipeline, not an official LeRobot example.

## Tested environment

| Component | Observed configuration |
|---|---|
| Operating system | Linux |
| GPU | NVIDIA GeForce RTX 2050, 4 GiB VRAM |
| Python | 3.10 |
| LeRobot | 0.4.3 |
| PyTorch | 2.7.1 |
| Gymnasium | 1.3.0 |
| hf-libero | 0.1.4 |
| Rendering | MuJoCo EGL, offscreen |

A standalone CUDA rollout was completed on this machine. Runtime and memory requirements depend on the environment and workload. The script checks for LeRobot **0.4.3**; compatibility with later releases has not been established.

## Setup

For a clean-environment import and task/state check before an authorized
episode, follow the [baseline startup preflight](docs/baseline-startup.md).
For an end-to-end replay and artifact inspection walkthrough, follow the
[baseline-to-report guide](docs/baseline-workflow.md).

Run these commands in a terminal. Conda, Git, and a working NVIDIA driver are prerequisites for the tested CUDA setup.

### 1. Clone the repository

```bash
git clone https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model.git
cd Agent-Aware-Vision-Language-Action-Model
```

### 2. Create the environment

```bash
conda create -n smolvla_libero python=3.10 -y
conda activate smolvla_libero
conda install -c conda-forge ffmpeg=7.1.1 -y
python -m pip install --upgrade pip
python -m pip install "lerobot[smolvla,libero]==0.4.3"
```

These are the installation commands used for the project. They pin LeRobot and FFmpeg, but are not a complete dependency lock: a new installation may resolve different transitive package versions.

### 3. Verify installation

```bash
python -m pip check
python -c "import torch; from importlib.metadata import version; print('LeRobot:', version('lerobot')); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For the default GPU run, CUDA availability should be `True` and LeRobot should be `0.4.3`. The first policy load requires internet access to download the checkpoint and its dependencies.

## Run one episode

From the repository directory, with `smolvla_libero` activated:

```bash
python run_smolvla_episode.py --suite libero_10 --task-id 0 --seed 0
```

Task 0 in the tested configuration is **“put both the alphabet soup and the tomato sauce in the basket.”** The default LIBERO-10 horizon is 520 actions. The script uses an offscreen renderer and saves a video; it does not open a live simulation window.

To run task 9, which succeeded in the earlier suite baseline:

```bash
python run_smolvla_episode.py --suite libero_10 --task-id 9 --seed 0
```

That earlier success does not guarantee every subsequent run will succeed.

To choose a destination:

```bash
python run_smolvla_episode.py --suite libero_10 --task-id 0 --seed 0 --output-dir outputs/task0_run1
```

The destination must not already exist, preventing accidental overwrites. Relative paths are resolved from the directory where the command is launched.

### Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--suite` | `libero_10` | LIBERO task suite |
| `--task-id` | `0` | Zero-based task index within the suite |
| `--initial-state-id` | `0` | Explicit index in the selected task's initial-state catalog |
| `--seed` | `0` | Random seed |
| `--policy` | `HuggingFaceVLA/smolvla_libero` | Frozen baseline repository; other values are rejected |
| `--device` | `cuda` | `cuda` or `cpu`; recorded run used CUDA |
| `--max-steps` | Suite horizon | Override the episode action limit |
| `--video-fps` | `20` | Saved-video playback rate |
| `--output-dir` | Timestamped directory | New directory for this episode |

Supported suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_90`. Video FPS changes playback speed, not simulation control frequency. The script defaults `MUJOCO_GL` to `egl` before simulator imports, while respecting an existing setting.

The baseline enforces checkpoint revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
and verifies weights, configuration, processor state and separate backbone/tokenizer
assets before loading. See [frozen asset resolution](docs/pinned-baseline-assets.md)
for the lock, recorded identities, download requirements and offline checks.

Before reset, the runner reads the created environment's action horizon and
requires a positive integer. An explicit `--max-steps` must match that readback;
an ignored override fails startup. `result.json` records `requested_max_steps`
(null for the default), `max_steps` (the effective limit), and `action_horizon`
with `effective`, `requested_override`, and `source` (`environment_default` or
`explicit_override`). The replay manifest retains the effective `config.max_steps`.
No action beyond this limit is proposed or executed. Success returned by the
last allowed action takes precedence over simultaneous terminal/truncation flags;
otherwise an environment terminal signal retains its reason, or exhaustion is
reported as `step_limit` with `success: false`.

```bash
python run_smolvla_episode.py --help
```

## Saved outputs

By default, each episode creates `outputs/smolvla_<suite>_task<id>_<timestamp>/`.

| File | Contents |
|---|---|
| `episode.mp4` | Main-camera recording: available frames from the initial and returned observations |
| `episode_wrist.mp4` | Wrist-camera recording from the same returned observations; absent if no wrist frames were available |
| `frames.jsonl` | One reference per camera per observation, including camera identity, observation sequence, capture timestamps, synchronization status, video path, and frame index; missing views have null paths and indices |
| `steps.jsonl` | One JSON record per action: observation identity, action, outcome, and monotonic step timing |
| `result.json` | Instruction, configuration, selected package versions, outcome, stop reason, and timing |
| `replay/` | Full JSON observation payloads and recorded decisions, with a checksum manifest published only after successful finalization |

The script also prints progress and output paths to the terminal. It does not automatically save terminal output to a separate log file.

`status: "completed"` means the script completed its rollout; **`success` is the task outcome**. A run can complete normally with `success: false` and `stop_reason: "step_limit"`. The loop's step limit can be reached even when the last logged termination and truncation flags are false.

### Reusable episode interface

`episode_harness.run_episode(EpisodeConfig(seed, max_steps), policy, environment, recorder)` runs one attempt and returns `EpisodeOutcome` with success, executed step count, stop reason, reward total, rollout time, cumulative supervisor wait, per-step timing, and artifact paths. The policy adapter supplies `reset()` and `act(observation)`; the environment adapter supplies `reset(seed, episode_id)` and `step(action)` returning a `StepResult`; the recorder supplies `begin(observation)`, `record_step(step, source, action, result, ingestion)`, `record_failure(step, source, action, failure)`, and `finish()` returning artifact references. The caller releases any adapter resources after completion or failure. The existing CLI supplies the LeRobot/LIBERO and file-recording adapters; the interface itself imports no simulator or model packages.

Every call creates a fresh attempt identity, including calls that fail during
reset. The harness calls `policy.reset()` first, then
`environment.reset(seed, episode_id)`, and accepts only a valid sequence-zero
observation for that new identity before starting recording or inference.
Policy adapters must discard queued actions and other per-attempt state on
reset; environment adapters must replace terminal state. The LIBERO adapter
disables stepping before reset and enables it only after successful initial
packet capture. A failed reset therefore cannot resume the previous attempt.

Reset failures re-raise the original exception with an `episode_interruption`
record containing `pre_start_failure`: the stage (`policy_reset`,
`environment_reset`, or `initial_observation`), requested seed, and flags
acknowledging which reset contracts returned. These failures have zero executed
actions, no accepted observation, unknown task status and incomplete artifacts.
The CLI persists this record in `result.json` under `partial_evidence` and uses
`status: "pre_start_failure"`; its stop reason still distinguishes timeout,
interruption and infrastructure failure. Recording never begins for an invalid
initial packet. Retry by calling `run_episode` again, which resets both adapters
and allocates another identity. Adapter implementations remain responsible for
actually clearing their internal state; the harness cannot inspect hidden queues.

For simulated external overrides, a policy must also implement
`resume(observation)` (`ResumablePolicyAdapter`). The harness checks this capability
before executing an override. After the overridden action returns an accepted,
nonterminal observation, and while action budget remains, it calls `resume` with
a detached copy of that packet before the next `act`. Resume must invalidate
obsolete queued actions and synchronize policy state without executing an action.
It preserves the episode identity, environment state, instruction and consumed
action budget. Pass decisions do not call resume; terminal, truncated, rejected
or exhausted episodes do not request another policy action.

The harness never substitutes `reset()` for a missing resume implementation.
`policy_adapter.ResetOnResumePolicyAdapter(reset, infer)` is an explicit opt-in
for policies whose declared resumption behavior is to clear policy state and
infer again from the next packet. The SmolVLA CLI uses this adapter: its native
reset clears the action queue in the pinned
[LeRobot v0.4.3 implementation](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/modeling_smolvla.py).
Other policies can implement a different resume transition. Scripted replay
adapters validate the next observation while preserving their replay cursor.
A resume exception stops the episode before another proposal, retaining the
acknowledged correction and last accepted observation in `episode_interruption`.
Resume time is included in rollout time, not supervisor decision latency.

Run the adapter and complete queued-policy override fixture without live resources:

```powershell
python -m unittest discover -s tests -p test_policy_resume.py -v
```

`libero_adapter.LiberoEnvironmentAdapter` handles a single vector environment. On same-step automatic reset it selects the unbatched `final_observation` or `final_obs` envelope, restores the one-environment batch dimension, and copies the terminal payload before recording. The returned reset frame is discarded. Terminal evaluator information comes only from `final_info`; reset-episode success fields cannot determine the ending outcome. Missing or masked terminal evidence raises an error instead of producing a completed outcome. Next-step reset environments supply the terminal observation directly. Custom wrappers must expose one of these contracts; an undocumented reset cannot be detected from pixels alone.

`EpisodeOutcome.terminal_observation` and the CLI's `result.json` contain the ending packet's `episode_id` and `sequence`, linking to `steps.jsonl` and `frames.jsonl`. This reference is set for success, termination, or truncation; it is null for a harness step limit or rejected proposal without an environment terminal signal. The adapter refuses further steps after termination until explicitly reset. Capture timestamps describe when the adapter receives the evidence, not camera exposure time.

Each observation packet carries a wall-clock `captured_at` for provenance and `captured_monotonic` for durations. The environment and harness must sample the same monotonic clock; tests can inject a controlled `clock` into both. The optional `supervisor(packet, proposed_action)` callback observes a copy of the proposal before execution and cannot replace the executed action. Observation age ends when that request starts, after policy inference. Decision latency is the supervisor request-to-response interval; for a failed request it ends when the exception is observed. It is zero when no supervisor is configured. Cumulative wait sums those intervals, independently of the simulator action count. Execution duration covers `environment.step`; rollout duration includes per-step recording but excludes reset and artifact finalization. A supervisor or environment exception produces a separate failure record with its stage, request end, optional response and execution-call boundaries, and cumulative wait. It does not produce a completed step or outcome. The CLI writes that record to `steps.jsonl` and retains cumulative wait in `result.json` when the run fails. These measurements describe paused simulation and do not establish physical control timing.

LIBERO camera references use `pixels.image` (main) and `pixels.image2` (wrist). The runner timestamps the paired environment return, so `co_observed` means both views came from one observation; it does not assert measured exposure synchronization. Adapters with per-camera capture clocks may supply timestamps, observation associations, and an explicit skew limit to verify synchronization. A missing view is marked `missing` and its available counterpart `unpaired`; a stale or skewed pair is marked `unsynchronized`. Neither is silently filled with another frame. Video frame indices in `frames.jsonl` are per camera and can differ when a view is missing.

`supervisor_observation(packet)` constructs a detached supervisor packet and is always applied by `run_episode` before the supervisor callback. Its allowlist includes a string `task`, numeric `pixels.image`/`image2`, and numeric robot measurements: `position`, `orientation`, `joint_positions`, `joint_velocities`, `eef.pos`/`quat`, and `gripper.qpos`/`qvel`/`closed`. Numeric arrays become plain lists; invalid measurement payloads become unavailable (`None`). Unknown fields and nested metadata are omitted, including simulator object poses, rewards, and success predicates. Identity, capture times, and typed camera/state references remain available for freshness checks. Adapters must explicitly map additional deployable sensors to this contract; arbitrary raw observations are not supervisor inputs. The policy and recorder retain the original packet, and evaluator-only `StepResult` fields still control scoring and termination. This boundary trusts adapters to provide actual camera/state measurements and the declared task instruction; it cannot detect truth deliberately encoded as pixels or mislabeled measurements.

For an explicit pass decision, supply `supervisor_decider(proposal)` to
`run_episode`. It receives an `ActionProposal` with a detached native action and
the same sanitized observation boundary described above. Return a typed response:

```python
from episode_harness import SupervisorPass

def pass_current(proposal):
    return SupervisorPass(
        episode_id=proposal.observation.episode_id,
        observation_sequence=proposal.observation.sequence,
        proposal_id=proposal.proposal_id,
    )

# run_episode(config, policy, environment, recorder,
#             supervisor_decider=pass_current)
```

The harness checks all three identities before executing the original proposal
unchanged. Missing, untyped, stale or foreign responses produce a supervisor
failure without executing that proposal. This callback has exclusive decision
ownership and cannot be combined with `supervisor`, `window_supervisor` or
`action_selector`. Accepted responses appear in `ActionRecord.supervisor_pass`
and recorded trace evidence, including when the subsequent environment call
fails; acknowledgement still requires a returned step result. Baseline execution
leaves this field `None`. Sealed replay validates and retains historical pass
evidence; replay execution uses the recorded action choices without calling a
supervisor model. This interface makes no active-correction claim.

For an uncertain diagnosis, return `SupervisorAbstention` through the same
callback. It requires the same three identities, a nonempty `reason`, and a
nonempty `evidence_availability` dictionary mapping assessed input names to
`available`, `missing`, `stale`, or `unknown`. For example:

```python
from episode_harness import SupervisorAbstention

def abstain_current(proposal):
    return SupervisorAbstention(
        proposal.observation.episode_id, proposal.observation.sequence,
        proposal.proposal_id, 'Grasp is occluded; insufficient temporal evidence',
        {'main': 'available', 'wrist': 'missing', 'history': 'unknown'},
    )
```

Abstention records `kind='abstain'` and `diagnosis='unknown'` in the separate
`ActionRecord.supervisor_abstention` field, with a detached snapshot of its reason
and reported evidence availability. Availability is the supervisor's assessment,
not independently verified sensor freshness. Abstention requests no correction:
the declared healthy-baseline behavior executes the original proposal unchanged
within the usual horizon and terminal checks. Execution or observation faults end
the attempt; abstention does not retry or bypass those faults. Malformed responses
are supervisor failures before execution, not valid abstentions. Sealed traces
validate uncertainty evidence and retain it for inspection and model-free replay.

For temporal supervision, pass `window_supervisor(window, proposed_action)` and
optionally `window_settings=WindowSettings(max_observations=8, max_actions=8)`
from `observation_window` to `run_episode`. Choose either this callback or the
single-packet `supervisor`. Both are observation-only and share the same wait and
failure accounting. The temporal callback requires a nonempty string `task` in
the initial observation; subsequent missing task fields retain that instruction,
and a changed instruction is rejected.

`ObservationWindow.observations` contains the most recent N allowlisted packets,
oldest first by sequence, with at most 2*N camera payloads and N robot-state
samples. Capture times and missing/stale view references remain attached; views
are never filled from other observations. `actions` independently retains the
most recent M acknowledged executions with their source sequence, proposal ID,
proposed and executed values, and execution completion time. The current proposal
is supplied separately; neither unconfirmed actions nor evaluator outcomes enter
the history. M may be zero. Limits bound sample counts, not image bytes or model
tokens; image resizing and provider budgets belong to the consuming adapter.

Each window declares its settings and ordering. `omitted_prefix` is the inclusive
sequence range before the oldest retained packet, whether unavailable or evicted;
`missing_intervals` identifies gaps between retained packets, and
`omitted_action_count` counts acknowledged actions evicted by the action limit.
The public `ObservationWindowBuilder(episode_id, task, settings)` supports offline
history through `append(packet)`, `record_action(WindowAction(...))`, and
`snapshot()`. It accepts forward sequence gaps explicitly but rejects duplicates,
reversed sequences, regressing timestamps, and foreign episodes without changing
the history. The live harness retains its stricter contiguous-stream requirement.
Builder inputs and returned snapshots are detached, and each episode gets a new
builder. These windows provide evidence; they do not establish detector readiness
or measured task improvement.

For a synthetic successful replay without a model, simulator, or files, call the same interface with the public fixture:

```python
from episode_harness import run_episode
from replay_adapters import successful_replay

fixture = successful_replay()
outcome = run_episode(fixture.config, fixture.policy,
                      fixture.environment, fixture.recorder)
assert outcome.success and fixture.environment.actions == [
    ('reach', 0.25), ('place', 0.75)]
```

Each fixture starts a scripted attempt; its in-memory recorder returns no artifact paths. The replay outcome verifies software behavior, not physical or LIBERO task performance.

### Replay a recorded episode

New CLI episodes retain a portable `replay/` directory. Run it without LeRobot,
NumPy, model downloads, or simulator access:

```bash
python recorded_replay.py outputs/my_episode/replay
python recorded_replay.py tests/fixtures/recorded_episode
```

The bundled synthetic fixture executes an override followed by a pass decision
and ends unsuccessfully by termination. It verifies recording and execution
contracts; it is not a measured LIBERO attempt. The command reports the original
episode identity and reconstructed outcome.

`load_recorded_replay(directory).run(recorder)` uses the public harness with
fresh replay-only policy/environment adapters and the recorded action selector.
The optional recorder receives every reconstructed observation, action and result.
Loading validates all required files and SHA-256 hashes, contiguous observation
and decision order, capture chronology, source/proposal/acknowledgement identities,
camera availability, terminal boundaries and outcome consistency before execution.
Missing/corrupt payloads raise `TraceError`; no frame or state is fabricated.
Checksums detect artifact changes, not the authenticity of the manifest's author.

To record another public adapter, wrap its recorder with
`TraceRecorder(new_directory, config, recorder)`, pass that wrapper to
`run_episode`, then call `trace.seal(outcome)` on the returned outcome. Sealing
requires completed finalization and a consistent full episode. Success, task
failure, truncation, step-limit and rejected-proposal episodes are supported.
Interruptions retain partial files without a manifest and are explicitly refused
as complete replay inputs. Older video/action-only bundles lack full observations
and cannot be replayed by this format.

The JSON payloads preserve array values and shapes as lists, not framework dtypes
or objects. They include both cameras and raw evaluator fields and can occupy
substantial disk space; keep them in private artifact storage. Videos are auxiliary
viewing artifacts and are not required by replay because the original pixel values
are embedded in the observation stream. Unsupported values fail serialization
instead of falling back to string representations. Replay gets a new episode ID,
retains the source ID separately and preserves historical capture timestamps;
its synthetic clock does not reproduce source inference latency or establish
performance. This replays recorded decisions, not the original model's reasoning.

### Recover execution progress after a process crash

New CLI recordings also write `replay/execution.jsonl`. Each returned environment
step is recorded with its proposed, selected and acknowledged action. The journal
flushes and calls `os.fsync` after its header and every action, before writing the
full observation or invoking downstream video/log sinks. Read a stopped recording:

```powershell
python execution_progress.py outputs/my_episode/replay/execution.jsonl
```

The same public API is `read_execution_progress(path)` in `execution_progress.py`.
It checks record digests, ordering, chain links and acknowledgement semantics,
retaining the valid prefix and reporting the first incomplete or invalid record
with its line number. `valid_bytes` marks the end of that prefix; readback never
repairs or overwrites evidence. Unsupported journal versions are rejected in the
diagnostic. Missing files raise an I/O error; older bundles have no journal.

The guarantee begins when the journal append returns after a successful disk sync,
subject to the filesystem/device honoring that operation. A crash between an
environment action and that sync can leave execution unknown. A returned adapter
step acknowledges the call, not independently measured physical actuation.
Rejected proposals and calls that never return are not counted as executed.
Disk-sync failures propagate as recording failures rather than silently continuing.
The abrupt-process-exit tests bypass cleanup and verify readback, including a torn
final record; they do not simulate power loss or certify storage hardware.

This separate journal is partial execution evidence, not a full observation trace,
an episode-completion marker, or authority to resume robot execution. Even a clean
EOF reports completion and later execution as unknown. Use the sealed replay
manifest for completed-episode verification; its existing format is unchanged and
does not include this auxiliary journal. Checksums detect damage, not malicious
rewriting. Keep the journal with private episode evidence under the same access rules.

To inspect the intervention timeline locally, generate a standalone HTML report
and open the resulting file in a browser:

```bash
python episode_timeline.py tests/fixtures/recorded_episode timeline.html
python episode_timeline.py outputs/my_episode/replay my_episode_timeline.html
```

The output must be a new file. No server, JavaScript, model or simulator is needed.
The synthetic failed fixture shows decision 1 changing `[0.1]` to `[0.5]`, then
decision 2 executing `[0.2]` unchanged and terminating unsuccessfully. Follow each
observation link to its original episode/sequence, timestamp, camera availability
and expandable full payload; decision links show the original acknowledgement
and JSONL line. Observation 1 has an unavailable wrist view. This verifies trace
inspection, not the quality or causal benefit of a model intervention.

The report validates checksums and evidence identities before rendering. Missing
or corrupt required files produce `Report unavailable` and a nonzero exit, without
creating a report or claiming an outcome. Missing camera views and unrecorded
video artifacts are marked in the report. The format does not retain exact
decision/execution timestamps or model diagnoses; those remain explicitly
unavailable rather than inferred from observation times or task failure.
Interrupted/unsealed traces are not supported. Reports embed raw evaluator
payloads, so retain them with the original private artifacts; they are not
supervisor inputs. For programmatic use, `render_episode_timeline(directory)`
returns HTML and `load_recorded_replay(directory).evidence()` returns a detached
JSON-compatible copy of the validated historical evidence.

Interrupted attempts re-raise the original exception with an
`episode_interruption` (`EpisodeInterruption`) record. It retains the episode ID,
termination reason (`interrupted` for cancellation, `timeout` for `TimeoutError`,
and `infrastructure_failure` for other ordinary exceptions),
exception category, acknowledged action count and records, reward sum, and a detached
copy of the last accepted observation. A command whose environment call raises is
not acknowledged. Recorder finalization is attempted on failure; recording and
cleanup diagnostics cannot replace the initiating exception. Successful cleanup
does not make the interrupted attempt a completed episode. The CLI saves this
partial evidence in `result.json`, including native actions and observation data;
these are raw evaluator artifacts, not supervisor inputs or a shareable report.
The caller still owns environment/policy teardown. This covers caught exceptions
and cancellation, not process kill, power loss, or incremental crash durability.

Task outcome and evidence completion are independent. `EpisodeOutcome.artifact_status`
is `completed` only after the recorder's `finish()` returns successfully; an ordinary
finalization exception returns `incomplete`, empty finalized artifact references and
`artifact_diagnostics`, while retaining task success, stop reason, reward, timing and
terminal observation identity. Callers must check this status before treating the
evidence package as complete. The CLI records these fields in `result.json` and exits
with an error for incomplete artifacts. Paths retained there after failure identify
partial evidence, not verified usable files. Video and log close failures are collected
across all sinks; repeated cleanup cannot clear a failed finalization. A task failure
can still have a completely finalized evidence package.

### Terminal categories and report denominators

`EpisodeOutcome`, `EpisodeInterruption`, and CLI `result.json` expose separate
`stop_reason`, `task_status`, and `artifact_status` fields. Retain all attempts
and these categories for later protocol-specific denominators.

| Stop reason | Task status | Meaning |
|---|---|---|
| `success` | `success` | Evaluator success takes precedence over terminal flags. |
| `terminated` | `failure` | Environment ended the task without success. |
| `step_limit` | `failure` | Action budget exhausted without success. |
| `truncated` | `unknown` | Environment cut execution short. |
| `proposal_rejected` | `unknown` | Executor rejected a proposal before sending it. |
| `timeout` | Last established status, otherwise `unknown` | An adapter/callback raised `TimeoutError`. |
| `interrupted` | Last established status, otherwise `unknown` | Cancellation or another non-`Exception` interruption. |
| `infrastructure_failure` | Last established status, otherwise `unknown` | Ordinary execution, observation, policy, or recording exception. |

Task status comes from accepted evaluator observations or budget exhaustion.
Recording/callback faults after an accepted terminal result preserve its task
status alongside the fault's reason. Interrupted attempts have incomplete
artifacts even if cleanup succeeds. Ordinary finalization or sealing failures
after a completed rollout retain its original reason and task status, marking
artifacts incomplete. The legacy `success: false` alone does not establish policy
failure: inspect `task_status` and `stop_reason`. These fields do not choose
exclusions or approve a scientific denominator. Truncation is not automatically
a timeout. Adapters must raise `TimeoutError` for deadline failures; this change
adds no deadlines or retries.

Version-1 sealed bundles retain their existing outcome fields (`success` and
`stop_reason` included); replay returns the derived `task_status`. Immutable
fixtures need no rewriting. Public checks are in
`tests/test_episode_termination.py` and `tests/test_episode_interruption.py`.

### Recorded recovery request interface

The [static recovery registry](docs/recovery-registry.md) adds executor-side
selection against immutable tool declarations and trusted adapter capabilities.
It resolves bounded requests without granting execution authority.

`supervisor_recovery.RecoveryRequestDecoder` decodes a JSON object into an
immutable `SupervisorRecoveryRequest`. The caller supplies the allowed tool
name and inclusive scalar `RecoveryParameter` bounds from trusted configuration;
the response cannot supply code, definitions, limits or extra fields. For example:

```python
from supervisor_recovery import RecoveryParameter, RecoveryRequestDecoder

# Synthetic interface example only; these are not calibrated robot limits.
decoder = RecoveryRequestDecoder("reopen_and_retreat", (
    RecoveryParameter("retreat_m", 0.0, 0.03),
))
request = decoder.decode(recorded_response, current_proposal)
```

The recorded JSON fixture is [recovery_response.json](tests/fixtures/recovery_response.json).
It carries episode, observation, proposal and decision identities, the tool name,
required numeric parameters and bounded evidence references. The decoder checks
the current proposal identity and rejects nonfinite/out-of-range numbers, booleans,
future/duplicate references and unknown fields. References identify observation
sequences and named sources; their existence and diagnostic validity remain for
the evidence resolver to establish.

This interface returns data only. It neither executes a recovery nor establishes
robot compatibility, calibration, eligibility or physical readiness. The current
`run_episode` decision callback accepts only pass and abstention; returning a recovery
request there fails before execution. Tool registry and execution integration are
separate work. Run the public recorded-response and no-execution checks with
`python -m unittest discover -s tests -p test_supervisor_recovery.py -v`.

### Recorded numerical adjustment interface

`supervisor_adjustment.AdjustmentRequestDecoder` parses a recorded response for
exactly one current policy proposal. Configure its target arm/group, frame and
named components from trusted adapter settings:

```python
from dataclasses import asdict
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder

# Synthetic interface bounds only, not calibrated robot limits.
decoder = AdjustmentRequestDecoder("left_arm", "world", (
    AdjustmentComponent("translation_x", "m", -0.03, 0.03),
    AdjustmentComponent("rotation_z", "rad", -0.05, 0.05),
))
request = decoder.decode(recorded_response, current_proposal)
proposed_adjustment_record = asdict(request)  # JSON-serializable evidence
```

See [adjustment_response.json](tests/fixtures/adjustment_response.json) for the
wire format. The response carries episode/observation/proposal/decision identities,
target, frame, per-component units and numerical residuals. Every configured
component is required, including an explicit zero for an unchanged component.
Residuals are additive changes, not replacement actions. `kind: adjustment` and
`scope: single_action` distinguish this request from recovery and reject persistent
or chunk-wide adjustments. Unknown fields, stale identities, mismatched semantics,
booleans, nonfinite values and values outside caller-owned bounds fail decoding.

The immutable result records the proposed adjustment without modifying the policy
action. Recording via `asdict` produces ordered component/value pairs for units and
residuals; the wire input uses objects. This decoder does not map native action
indices, convert frames/units, validate the resulting action, or establish robot
readiness. A configured group name does not establish coordinated execution.
Executor integration remains separate: returning this request directly from
`run_episode`'s decision callback is rejected before execution. Public fixture,
recording and rejection checks:
`python -m unittest discover -s tests -p test_supervisor_adjustment.py -v`.

### Structured supervisor response validation

`supervisor_response.SupervisorResponseDecoder` is the shared boundary for
JSON-decoded provider/recorded objects. It accepts only `pass`, `abstain`,
`recovery` and `adjustment`. Pass requires `kind`, `episode_id`,
`observation_sequence` and `proposal_id`; abstention additionally requires
`diagnosis: "unknown"`, a nonempty `reason` and `evidence_availability` using
the statuses documented above. Correction variants use the wire contracts above
and require caller-supplied `recovery`/`adjustment` decoders with trusted bounds.
All variants reject missing/unsupported fields and foreign proposal identities;
numeric parameters reject non-finite values and booleans. Tool code, replacement
instructions and undeclared action fields are unsupported.

```python
from supervisor_response import SupervisorResponseDecoder

decoder = SupervisorResponseDecoder()  # pass/abstain only
# In supervisor_decider(current_proposal):
# return decoder.decode(recorded_response, current_proposal)
```

Invalid objects raise `episode_harness.SupervisorResponseError` with a schema
reason without echoing provider payloads. When decoding in `supervisor_decider`,
the harness records a supervisor failure and an `ActionRecord` with disposition
`rejected`, `rejection_reason`, and no selected/executed action or acknowledgement.
It finalizes the incomplete attempt and re-raises the error; this does not enable
baseline fallback. Valid recovery/adjustment results remain request data and
cannot authorize execution through this callback. Failed attempts retain their
decision records but cannot be sealed as completed replay bundles.

Responses must match all three active identities: episode, observation sequence
and proposal. A response from a completed attempt remains invalid after reset,
even when the new observation sequence matches. A previous decision cannot
authorize the next proposal within an episode. The harness checks typed
pass/abstention responses too and records identity mismatches as explicit
rejections with the same retained failure evidence, before any environment step.

Verify with
`python -m unittest discover -s tests -p test_supervisor_response.py -v`.

### Bounded provider requests

`BoundedSupervisorProvider(provider, timeout_seconds)` in `supervisor_provider.py`
wraps a callable `provider(proposal, deadline, cancellation_event)`. The deadline
uses `time.monotonic()`; configure the transport's own timeout from its remaining
time and honor cancellation. Inputs are detached and supervisor-allowlisted, and
responses pass through `SupervisorResponseDecoder` before acceptance.

`request(proposal, cancellation=None)` returns a typed `ProviderResult` with
episode/observation/proposal identity, request/deadline/completion times, and a
`response`, `timeout`, `error`, `cancelled`, `busy`, or `rejected` status. Only
`response` carries a decoded decision. Supply an Event for external cancellation.
No automatic retries occur. Each request has a separate result mailbox; expired
or cancelled replies cannot become a later decision.

Use the adapter directly as `run_episode(..., supervisor_decider=adapter)`.
Failures raise `ProviderRequestError` carrying the typed result; the harness
retains their safe status/reason and measured wait in failed-attempt evidence,
executes no action for that decision, and finalizes through its existing failure
path. Provider exception text is omitted. This boundary does not implement healthy
baseline fallback or grant correction execution authority.

Caller waiting is bounded even if a provider ignores cancellation. Such a daemon
worker cannot be forcibly killed by Python and may continue remote work; the
adapter returns `busy` for new requests until it exits, preventing worker buildup
within that adapter. This is not a guarantee of remote request cancellation or a
resource allowance. Provider-specific transports must enforce their own limits.
Public controlled-provider and complete-episode checks:
`python -m unittest discover -s tests -p test_supervisor_provider.py -v`.

### Synchronous simulation scheduling

`run_episode(..., supervisor_decider=BoundedSupervisorProvider(provider, 2.0))`
waits for one decision on the current proposal before calling `environment.step`
or requesting another policy action. The decider has exclusive decision ownership;
other supervision callbacks and action selectors cannot be combined with it.
The simulation adapter must advance only when stepped. This path does not issue
physical hold/stop commands or pause an independently advancing environment.

A validated pass or abstention resumes through the normal execution interface.
`StepTiming.decision_latency_seconds` records each wait, and the outcome records
their sum in `cumulative_wait_seconds`. A timeout retains failed-step wait and
rejection evidence, finalizes the recorder, and raises `ProviderRequestError`
without executing the pending action. The incomplete trace remains unsealed;
late responses cannot resume it.

Run the controlled-delay fixture with
`python -m unittest discover -s tests -p test_synchronous_supervision.py -v`.
It holds both decisions of a two-action episode, verifies no simulation progress
while pending and rejects competing provider requests, then seals and replays the
successful branch. Its timeout branch preserves the first action and records the
second proposal's rejection, including waiting time and no late execution.

### Periodic and event-triggered assessments

Set `EpisodeConfig(seed=17, max_steps=20, supervisor_interval_actions=4)`
and pass `supervisor_decider` to assess before action 1, then after 4, 8, 12,
and 16 acknowledged actions. The positive integer interval defaults to 1.
Its units are executed actions, not seconds or policy chunks; `TraceRecorder`
retains `config.supervisor_interval_actions` in the sealed manifest. Older
manifests without this field use the default. Cadence restarts each episode.

An optional `assessment_trigger(proposal) -> bool` receives a detached,
sanitized current proposal on every action and can request an extra assessment.
Event assessments do not shift periodic boundaries. A coincident event and
periodic boundary produce one decider call. The synchronous loop cannot advance
or enqueue another assessment while that call is pending. Use the bounded
provider above for deadlines. This action-based schedule makes no wall-clock
latency guarantee and does not enable physical operation.

On unscheduled actions, the baseline proposal executes and the action record
contains no supervisor response; a skipped assessment is not a model pass or
abstention. Trigger errors terminate the attempt through supervisor-failure
recording. The cadence option applies to `supervisor_decider`; legacy packet
and window observation callbacks still run on every action and reject a
nondefault interval. Baseline-only runs make no supervisor calls.

Run `python -m unittest discover -s tests -p test_assessment_schedule.py -v`
for a 20-action quiet episode, extra/overlapping events, pending-call ordering,
episode reset, configuration validation, and complete sealed replay.

### Repeated proposal evidence

`RepeatedProposalTrigger(RepetitionSettings(window=4, action_tolerance=0.0,
position_tolerance=0.0))` from `repeated_proposals` plugs into
`assessment_trigger`. It requests inspection when a full consecutive window
has action and observed position spans at or below the configured tolerances
in every coordinate. Position uses sanitized `robot_state.eef.pos`, falling
back to `robot_state.position` when absent. Configure tolerances in the
adapter's declared units and consistent frame; defaults are exact equality,
not calibrated detector-readiness evidence.

The callable returns a boolean and exposes an immutable `signal` on the
emitting call: episode, supporting proposal IDs, observation sequences,
action/position vectors, criterion and settings. A caller wrapping the trigger
can retain that signal in its evidence sink before returning the boolean.
The signal is cleared on the next call; only bounded history is retained.
The existing sealed episode records the resulting assessments and actions,
not this optional signal sink. Proposals are not proof of executed actions.

One signal is emitted per continuous matching period. Observed motion beyond
the threshold suppresses the signal; invalid/missing vectors, dimensional
changes and sequence gaps break the window, and a new episode resets history.
Stationary productive work can still match: repetition requests assessment,
never declares task failure or executes a correction. Periodic assessments
remain available when this evidence is missing. Verify boundary cases,
productive motion, episode isolation and complete sealed replay with
`python -m unittest discover -s tests -p test_repeated_proposals.py -v`.

## Initial baseline results

The following results were recorded on September 25, 2026, using the pretrained policy without a supervisor. Suite evaluations used LeRobot's evaluator with one episode per task and serial execution; the standalone runner was tested separately.

| Suite | Episodes | Successful | Observed success rate | Evaluation time |
|---|---:|---:|---:|---:|
| LIBERO-Spatial | 10 | 8 | 80% | 31.8 min |
| LIBERO-Long / LIBERO-10 | 10 | 6 | 60% | 77.1 min |

| Suite | Successful task IDs | Failed task IDs |
|---|---|---|
| `libero_spatial` | 0, 1, 4, 5, 6, 7, 8, 9 | 2, 3 |
| `libero_10` | 2, 3, 5, 6, 7, 9 | 0, 1, 4, 8 |

These are preliminary local observations from **one episode per task**, not statistically robust benchmark estimates or claims of improvement.

### Standalone task-0 validation

| Measurement | Result |
|---|---|
| Task | Put both the alphabet soup and tomato sauce in the basket |
| Suite / task ID | `libero_10` / `0` |
| Actions executed | 520 |
| Task success | False |
| Total reward | 0.0 |
| Stop reason | Step limit |
| Rollout time | 557.0 s, approximately 9.3 min |
| Total time including setup | 598.9 s, approximately 10.0 min |
| Recorded video | 521 frames at 20 FPS, 26.05 s |

The uploaded action log contained 520 consecutive entries with seven finite action values each. Visual review suggested the first container reached the basket, followed by repeated unsuccessful attempts to pick up the second. This is a qualitative observation; the simulator's final outcome confirms only that the complete task was not achieved.

## Recorded supervisor responses

`RecordedSupervisor(bundle)` in `supervisor_replay.py` is an offline
`supervisor_decider` callback. Load a JSON bundle with `json.loads`, then pass
the adapter to `run_episode`. It has no provider connection or fallback.

```json
{
  "schema_version": 1,
  "fixtures": [
    {
      "episode_id": "attempt-id",
      "observation_sequence": 0,
      "proposal_id": "attempt-id:1",
      "response": {
        "kind": "pass",
        "episode_id": "attempt-id",
        "observation_sequence": 0,
        "proposal_id": "attempt-id:1"
      }
    }
  ]
}
```

Each fixture matches the full episode/observation/proposal identity, including
the identity inside its structured response. A new harness attempt generates a
new episode ID; author fixtures explicitly for that ID from the public reset
packet before decisions begin. The adapter never silently rebinds old evidence.
The complete-episode test demonstrates this using the recorder's `begin` hook.
Repeated requests for an identical key return detached decisions. Duplicate keys
or unsupported bundle versions fail at construction; missing keys and malformed
responses fail visibly at request time.

Use the same structured abstention/recovery/adjustment response schemas described
above (corrections still require caller-configured decoders and grant no execution
authority). To reproduce a provider failure, replace `response` with
`"error": "provider_error"`. Errors follow the harness's recorded rejection and
finalization path without executing the rejected proposal or contacting a model.
These synthetic fixtures verify software behavior, not model diagnosis quality.

Verify with
`python -m unittest discover -s tests -p test_supervisor_replay.py -v`.

## Gripper transition assessment

`GripperTransitionTrigger()` from `gripper_transitions` plugs into
`run_episode(..., assessment_trigger=trigger, supervisor_decider=decide)`.
It requests inspection on contiguous observed open-to-closed and closed-to-open
changes. It accepts only the adapter-mapped scalar `robot_state.gripper.closed`
boolean or numeric 0/1; native joint positions and action commands do not define
closure automatically. The adapter owns the mapping for its gripper contract.

Inspect `trigger.evidence` after every call and retain it in the caller's
evidence sink, including calls that return false. The immutable record contains
up to two samples with proposal/observation identities, packet capture times,
available robot-state capture metadata, observed states, transition direction,
and missing-state limitations. The trigger retains only the current record and
previous valid sample; it does not automatically append evidence to the episode
bundle. Missing/invalid state, observation sequence gaps and episode changes
break continuity. Missing sensor capture metadata is explicitly unverified;
packet timestamps alone do not establish sensor freshness. Use the required
sensing gate described above when fresh sensor evidence is necessary.

A normal grasp and a missed grasp can have identical closure transitions, and
opening alone cannot establish object loss. All such transitions request temporal
assessment without asserting success, failure, or intervention eligibility.
Stable closure produces no repeated event. Periodic assessment continues during
missing gripper state; this trigger does not change action execution authority.

Verify suspected missed grasps, normal grasps, unavailable state and complete
sealed replay with
`python -m unittest discover -s tests -p test_gripper_transitions.py -v`.
These synthetic cases establish software behavior, not detector calibration.

## Supervisor roadmap

The [chronological VLM adapter guide](docs/supervisor-vlm.md) describes the
configurable Anthropic transport, camera ordering, bounded observation history,
credential isolation and offline verification. Live calls remain resource gated.

1. **Offline observation:** process task instructions and chronological frame windows from recorded episodes.
2. **Progress detection:** log completed subtasks, remaining work, and evidence of repeated failed attempts. Use only information available up to each observation time.
3. **Detection evaluation:** inspect failed and successful episodes, measuring false alarms as well as detected failures.
4. **Live observation:** connect the observer to the existing loop while leaving policy actions unchanged.
5. **Bounded intervention:** introduce corrections and compare against matched baseline runs, including intervention counts and overhead.

The immediate test case is detecting repeated pickup attempts without progress after the first placement. A successful task-9 recording provides an initial false-alarm check; broader testing will require more episodes and tasks.

## Implementation notes and limitations

- The runner uses the checkpoint's processors and LeRobot's LIBERO processors for camera and state preprocessing and action conversion.
- Selection of one task uses `task_ids` inside the Python environment factory's `gym_kwargs`. The previously attempted `--env.task_ids` evaluator CLI option was unsupported in the tested release.
- Video frames come from adapter-selected observations, including the terminal envelope when the vector environment resets in the same step.
- The policy is frozen; the script does not train an observer or implement corrective actions.
- The historical standalone run used initial-state index 0. Changing the seed alone does not select another catalog state.
- Current runs enforce the checkpoint and backbone revisions in `policy-assets.lock.json`; the earlier recorded results do not establish those asset identities or a complete environment lock.
- Generated runs and model weights are excluded from Git. The result summaries above are documented observations; raw run artifacts are not bundled here.

## Sources and acknowledgments

The implementation was checked against the published LeRobot 0.4.3 source during development and subsequently exercised in the user's Linux/CUDA environment.

- [LeRobot 0.4.3 installation](https://huggingface.co/docs/lerobot/v0.4.3/en/installation)
- [SmolVLA documentation](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla)
- [LeRobot LIBERO integration](https://huggingface.co/docs/lerobot/v0.4.3/en/libero)
- [SmolVLA LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- [LeRobot evaluation pipeline, v0.4.3](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/scripts/lerobot_eval.py)
- [LeRobot LIBERO wrapper and factory, v0.4.3](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/envs/libero.py)
- [LeRobot environment processors, v0.4.3](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/processor/env_processor.py)
- [Original LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO)

Built on the work of the SmolVLA, LeRobot, LIBERO, robosuite, and MuJoCo contributors.

## Development trace annotation

Use the [offline annotation workflow](docs/development-annotations.md) to record
reviewed failure or productive-motion examples with trace identity, onset
intervals, evidence and explicit uncertainty. A synthetic fixture and CLI
round-trip run without live inference or client demonstrations.

## End-effector progress assessment

`EndEffectorProgressTrigger(ProgressSettings(frame='world', position_units='m'))`
from `end_effector_progress` plugs into `run_episode`'s `assessment_trigger`.
The adapter must explicitly supply `robot_state.eef.pos` (finite xyz),
`eef.frame`, and `eef.position_units` on every observation. The sanitizer retains
these two string metadata fields. Use actual adapter semantics; this trigger
neither assumes LIBERO coordinates nor converts between frames or units.
Missing or incompatible metadata/state is unknown, never zero displacement.

The default window contains four consecutive samples spanning at least 0.1 and
at most 2 seconds, with maximum pairwise Euclidean displacement <= 0.001 in the
declared units. Configure these values for the task and retain the frozen
settings with the run configuration. Pairwise distance counts an excursion even
when the end effector returns to its starting position. Sequence gaps, episode
changes, invalid samples and non-increasing packet times break the history.
Expired samples are discarded, and storage is bounded by the sample count.

Inspect and retain `trigger.evidence` after each call: it contains proposal and
observation identities, positions, packet monotonic times, settings, elapsed
interval, displacement (or `None`), status and limitations. Packet time does
not establish sensor freshness; use the existing observation eligibility gate
when verified state age is required. Missing camera views do not become motion
measurements; camera eligibility remains a separate assessment requirement.

A low-progress window emits once until the condition clears. Stationary
productive work or an intentional pause can produce the same evidence as a
stall. Only supervisor assessment can interpret it using additional context;
the trigger cannot diagnose task failure or authorize correction. Periodic
assessment continues while evidence is unknown. The caller's evidence sink
retains these diagnostic records; the sealed trace retains the resulting
supervisor decisions and executed actions.

`python -m unittest discover -s tests -p test_end_effector_progress.py -v`
checks distance/interval boundaries, frame and state loss, episode isolation,
and complete sealed replays of stall candidates, intentional pauses, motion,
and unknown observations with unchanged policy execution.

## Visual-change assessment

`VisualChangeTrigger(VisualChangeSettings(view='main', pixel_max=255))` from
`visual_change` plugs into `run_episode`'s `assessment_trigger`. Select `main`
or `wrist` explicitly. The image contract is a rectangular HW grayscale or HWC
image with one or three channels and finite values from zero to `pixel_max`.
Configure the actual adapter's scale (for example, 1 for normalized pixels);
the trigger does not infer scale or transpose channel-first images.

Each image is sampled on an endpoint-inclusive grid of at most 16 by 16 points
by default (configurable from 2 to 64 per dimension). The metric is mean absolute
channel difference divided by `pixel_max`, comparing consecutive same-view,
same-shape samples. Defaults request assessment for change <= 0.01 or >= 0.5
over intervals from 0.1 to 2 seconds, inclusive. Intermediate change does not
request assessment. An extreme condition emits once until it clears or changes
regime; periodic assessment remains independent. These defaults are uncalibrated
software settings, not measured detection thresholds.

Only the previous grid and current immutable evidence record are retained,
with at most two grids (at most `2 * grid_size**2 * 3` channel values).
Sanitization still visits the incoming image; this is a retained-evidence bound,
not a bound on input decoding or sanitizer cost. Grid sampling can miss local
changes. Missing views, stale associations, sequence gaps, episode changes,
shape/time-basis changes and invalid intervals prevent temporal comparison.
Missing pixels are never replaced with a blank image. Observation-return timing
is explicitly distinguished from camera-capture timing; use the separate
observation eligibility gate when verified sensing freshness is required.

Retain `trigger.evidence` after each call in the caller's evidence sink. It
records the selected view/settings, proposal and observation identities, sampled
values, capture times/time basis, interval, computed change or `None`, status,
and limitations. These records are not automatically appended to the episode
bundle; resulting assessments and executed actions use the existing trace.
Camera motion, lighting, occlusion and productive stationary work can all cause
these signals. Neither extreme establishes task failure, progress, grasp state
or correction eligibility; those interpretations remain with the supervisor.

`python -m unittest discover -s tests -p test_visual_change.py -v` verifies
thresholds, view selection, bounded samples, missing/incompatible evidence,
episode isolation and successful/unsuccessful complete sealed replays with
unchanged execution. No live model calls or calibration runs are involved.

## Offline detector metrics

Use [the detector metrics workflow](docs/detector-metrics.md) to compare declared
event outputs with reviewed development annotations on a sealed trace. It reports
per-family precision/recall, onset-interval delay bounds, unmatched events,
unknown/abstention counts and executed interventions during reviewed productive
intervals. Synthetic fixtures verify the calculations without establishing
detector calibration or intervention readiness.

## Simulator controller verification

For controller scaling and reference-frame evidence required by correction
conversion, see [the controller verification workflow](docs/simulator-controller.md).
It uses an isolated WSL environment and a simulator-only probe.
