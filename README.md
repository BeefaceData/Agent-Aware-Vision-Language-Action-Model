# Agent-Aware Vision-Language-Action Model

Investigating an AI supervisor that observes a frozen vision-language-action (VLA) policy, detects failures or stalled progress, and eventually intervenes with targeted corrections—without retraining the underlying policy.

The current implementation runs **SmolVLA in LIBERO simulation**, recording an episode video, every executed action, and the final task outcome. It establishes the baseline for developing and evaluating the supervisor.

**Status:** baseline inference and simulation are working. The observation-only supervisor, corrective interventions, and memory are planned; they are not implemented in this repository yet. No GRPO training or policy fine-tuning has been performed.

## What is included

- `run_smolvla_episode.py`: standalone runner for one selected LIBERO task.
- `README.md`: environment setup, usage, output descriptions, and initial results.
- `.gitignore`: excludes generated runs, model weights, caches, and local environments.

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
| `--seed` | `0` | Random seed |
| `--policy` | `HuggingFaceVLA/smolvla_libero` | Compatible SmolVLA checkpoint |
| `--device` | `cuda` | `cuda` or `cpu`; recorded run used CUDA |
| `--max-steps` | Suite horizon | Override the episode action limit |
| `--video-fps` | `20` | Saved-video playback rate |
| `--output-dir` | Timestamped directory | New directory for this episode |

Supported suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_90`. Video FPS changes playback speed, not simulation control frequency. The script defaults `MUJOCO_GL` to `egl` before simulator imports, while respecting an existing setting.

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

The script also prints progress and output paths to the terminal. It does not automatically save terminal output to a separate log file.

`status: "completed"` means the script completed its rollout; **`success` is the task outcome**. A run can complete normally with `success: false` and `stop_reason: "step_limit"`. The loop's step limit can be reached even when the last logged termination and truncation flags are false.

### Reusable episode interface

`episode_harness.run_episode(EpisodeConfig(seed, max_steps), policy, environment, recorder)` runs one attempt and returns `EpisodeOutcome` with success, executed step count, stop reason, reward total, rollout time, cumulative supervisor wait, per-step timing, and artifact paths. The policy adapter supplies `reset()` and `act(observation)`; the environment adapter supplies `reset(seed, episode_id)` and `step(action)` returning a `StepResult`; the recorder supplies `begin(observation)`, `record_step(step, source, action, result, ingestion)`, `record_failure(step, source, action, failure)`, and `finish()` returning artifact references. The caller releases any adapter resources after completion or failure. The existing CLI supplies the LeRobot/LIBERO and file-recording adapters; the interface itself imports no simulator or model packages.

`libero_adapter.LiberoEnvironmentAdapter` handles a single vector environment. On same-step automatic reset it selects the unbatched `final_observation` or `final_obs` envelope, restores the one-environment batch dimension, and copies the terminal payload before recording. The returned reset frame is discarded. Terminal evaluator information comes only from `final_info`; reset-episode success fields cannot determine the ending outcome. Missing or masked terminal evidence raises an error instead of producing a completed outcome. Next-step reset environments supply the terminal observation directly. Custom wrappers must expose one of these contracts; an undocumented reset cannot be detected from pixels alone.

`EpisodeOutcome.terminal_observation` and the CLI's `result.json` contain the ending packet's `episode_id` and `sequence`, linking to `steps.jsonl` and `frames.jsonl`. This reference is set for success, termination, or truncation; it is null for a harness step limit or rejected proposal without an environment terminal signal. The adapter refuses further steps after termination until explicitly reset. Capture timestamps describe when the adapter receives the evidence, not camera exposure time.

Each observation packet carries a wall-clock `captured_at` for provenance and `captured_monotonic` for durations. The environment and harness must sample the same monotonic clock; tests can inject a controlled `clock` into both. The optional `supervisor(packet, proposed_action)` callback observes a copy of the proposal before execution and cannot replace the executed action. Observation age ends when that request starts, after policy inference. Decision latency is the supervisor request-to-response interval; for a failed request it ends when the exception is observed. It is zero when no supervisor is configured. Cumulative wait sums those intervals, independently of the simulator action count. Execution duration covers `environment.step`; rollout duration includes per-step recording but excludes reset and artifact finalization. A supervisor or environment exception produces a separate failure record with its stage, request end, optional response and execution-call boundaries, and cumulative wait. It does not produce a completed step or outcome. The CLI writes that record to `steps.jsonl` and retains cumulative wait in `result.json` when the run fails. These measurements describe paused simulation and do not establish physical control timing.

LIBERO camera references use `pixels.image` (main) and `pixels.image2` (wrist). The runner timestamps the paired environment return, so `co_observed` means both views came from one observation; it does not assert measured exposure synchronization. Adapters with per-camera capture clocks may supply timestamps, observation associations, and an explicit skew limit to verify synchronization. A missing view is marked `missing` and its available counterpart `unpaired`; a stale or skewed pair is marked `unsynchronized`. Neither is silently filled with another frame. Video frame indices in `frames.jsonl` are per camera and can differ when a view is missing.

`supervisor_observation(packet)` constructs a detached supervisor packet and is always applied by `run_episode` before the supervisor callback. Its allowlist includes a string `task`, numeric `pixels.image`/`image2`, and numeric robot measurements: `position`, `orientation`, `joint_positions`, `joint_velocities`, `eef.pos`/`quat`, and `gripper.qpos`/`qvel`/`closed`. Numeric arrays become plain lists; invalid measurement payloads become unavailable (`None`). Unknown fields and nested metadata are omitted, including simulator object poses, rewards, and success predicates. Identity, capture times, and typed camera/state references remain available for freshness checks. Adapters must explicitly map additional deployable sensors to this contract; arbitrary raw observations are not supervisor inputs. The policy and recorder retain the original packet, and evaluator-only `StepResult` fields still control scoring and termination. This boundary trusts adapters to provide actual camera/state measurements and the declared task instruction; it cannot detect truth deliberately encoded as pixels or mislabeled measurements.

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

## Supervisor roadmap

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
- The recorded initial-state index is 0. Changing the seed alone should not be interpreted as a comprehensive sweep of LIBERO initial states.
- The checkpoint is referenced by repository name without an immutable revision. Exact replication also requires recording checkpoint revisions and a full environment lock.
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
