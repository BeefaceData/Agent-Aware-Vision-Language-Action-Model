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
| `episode.mp4` | Main-camera recording: initial observation and one frame per executed action |
| `steps.jsonl` | One JSON record per action: step, seven action values, reward, success, terminated, and truncated |
| `result.json` | Instruction, configuration, selected package versions, outcome, stop reason, and timing |

The script also prints progress and output paths to the terminal. It does not automatically save terminal output to a separate log file.

`status: "completed"` means the script completed its rollout; **`success` is the task outcome**. A run can complete normally with `success: false` and `stop_reason: "step_limit"`. The loop's step limit can be reached even when the last logged termination and truncation flags are false.

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
- Video frames come from returned observations, preserving the final observation when the environment wrapper resets internally.
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
