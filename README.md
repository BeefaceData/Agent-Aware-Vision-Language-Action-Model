# Agent-Aware Vision-Language-Action Model

A runtime supervision harness for the frozen Neotix policy on RoboTwin's
`hanging_mug` task. The optional Anthropic planner (Claude Opus) selects one
bounded skill per turn and never generates joint actions. This project does not
retrain the policy.

The baseline inference path remains available independently. The supervised
runtime adds typed skills, public RGB-D perception, task predicates, structured
logs, and offline replay tools. Live exploration is still in progress; success
of individual phases is not evidence of reliable full-task hanging.

For the component map, data flow, perception boundary, and contributor
workflow, see [CONTRIBUTE_README.md](CONTRIBUTE_README.md). For a beginner-level
walkthrough with diagrams, see `docs/v1/robot_mug_guide.pdf`.

## Contents

- [What runs where](#what-runs-where)
- [Shared setup (both routes)](#shared-setup-both-routes)
- [Route 1: run on Modal](#route-1-run-on-modal)
- [Route 2: run locally](#route-2-run-locally)
- [Reference](#reference): outputs, policy and SAM 3 GPUs, costs, troubleshooting

## What runs where

| Component | Code | Runs on |
|---|---|---|
| Episode runner: RoboTwin simulator, CuRobo, harness | `run_skills.py`, `harness/` | a Modal GPU (route 1) **or** your NVIDIA GPU (route 2) |
| Frozen Neotix policy | `modal_apps/policy.py` | Modal, T4, both routes |
| SAM 3 segmentation (optional) | `modal_apps/sam3.py` | Modal, T4, both routes |
| Opus planner and perception prompts (optional) | `harness/planner/`, `harness/perception/` | Anthropic API, both routes |

Both routes take the same `run_skills.py` options and produce the same output
folder layout.

- **Route 1, Modal.** The simulator runs in a fresh cloud container for each
  episode, and your machine needs no GPU. On an L4, camera rendering is about
  7× faster than on a 4 GB laptop GPU.
- **Route 2, local.** You get the simulator in reach for debugging. It needs
  Linux, an NVIDIA GPU with Vulkan, and about 10 GB of disk.

You can set up both routes.

## Shared setup (both routes)

### Python environment

From the repository root:

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin
python -m pip install -r requirements/sim.txt -r requirements/planner.txt 'modal>=1,<2'
```

Keep local NumPy at 1.26.4; the Modal workers use their own images. If you only
use route 1, `requirements/planner.txt` plus `modal` is enough. Your machine
then only validates options and talks to Modal.

### Modal account and frozen policy

```bash
modal setup                        # log in once
modal deploy modal_apps/policy.py  # frozen policy; weights download on first use
```

### SAM 3 (needed for `--perception-factory`)

1. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) and
   wait for approval.
2. Create a Hugging Face read token, store it on Modal, then deploy and test SAM 3:

   ```bash
   modal secret create huggingface HF_TOKEN=hf_...   # add --force to replace it
   modal deploy modal_apps/sam3.py
   modal run modal_apps/sam3.py                      # smoke test: "SAM 3 ... ready"
   ```

`GatedRepoError ... not in the authorized list` means the account behind the
token is not approved yet, or the token belongs to another account.

### Tests (CPU only, no keys, about 10 seconds)

```bash
python -m unittest discover -s tests -q
```

## Route 1: run on Modal

The simulator image is defined in `modal_apps/simulator.py`. It contains:

- RoboTwin and CuRobo at the pinned commits;
- the renderer patch and RoboTwin's SAPIEN and mplib patches;
- an NVIDIA-only Vulkan setting.

Your machine only launches the episode and receives the results.

### M1. Anthropic secret

Every `modal run modal_apps/simulator.py ...` needs this secret to exist, even
scripted runs and the spike. Never commit the key.

```bash
modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...
```

### M2. Build the image and copy the assets (once)

```bash
modal run modal_apps/simulator.py::download_assets
```

- **First run:** builds the image, about 20–40 minutes, mostly compiling CuRobo.
  Later runs use the cached image.
- **Assets:** copies RoboTwin's `embodiments` and `objects` bundles into the
  `robotwin-assets` Volume. This step uses no GPU.
- **If the build fails:** run `modal image logs <id>` with the newest image ID
  from the error message.

### M3. Check the GPU (optional)

```bash
modal run modal_apps/simulator.py::spike
```

The spike does one reset, renders 20 RGB-D observations, runs 200 physics steps
and makes one CuRobo plan. It writes `data/modal_spike/<GPU>.json` and a
head-camera image. The simulator works on that GPU when:

- `"ok": true`;
- `camera_pixel_std` is nonzero for all three cameras;
- `curobo_plan_status` is `"Success"`.

### M4. Run episodes

```bash
# Scripted smoke run: no Opus calls, a few cents of GPU time
modal run modal_apps/simulator.py::episode \
  --config configs/skill_runtime.yaml --development-oracle \
  --output data/runs/modal_smoke_01 --time-limit-minutes 15

# Full task: Opus planner and public RGB-D perception
modal run modal_apps/simulator.py::episode \
  --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml \
  --planner anthropic \
  --perception-factory harness.perception.modal_sam3:create_detector \
  --output data/runs/modal_full_01 --time-limit-minutes 75
```

The smoke run is expected to end with `"outcome": "step_budget"` after about
720 physics ticks. The first planned move needs slightly more than its
1000-tick allowance. The run still proves the simulator, CuRobo and logging work.

What `::episode` does:

1. **Checks options locally.** It runs the same checks as `run_skills.py` on your
   machine, so a bad combination fails before any GPU starts. It uploads the
   `--config` and `--memory-snapshot` files.
2. **Runs the episode remotely.** `run_skills.py` runs unchanged in a single-use
   container on the chosen GPU. It calls the policy and SAM 3 inside Modal, and
   Anthropic with the `anthropic` secret.
3. **Saves and downloads the results.** They are written to the `robotwin-runs`
   Volume as `<output name>-<UTC timestamp>`, then copied into your `--output`
   folder. Partial results are copied even after a failure.

`::episode` also has two options of its own:

| Option | Default | Effect |
|---|---|---|
| `--time-limit-minutes N` | 90 (max 210) | sends Ctrl+C to the episode after N minutes; the outcome is `interrupted` |
| `--skip-observations` | off | leaves the per-step `observations/*.npz` on the Volume instead of downloading them |

Limitations today:

- **Keep the terminal open.** Stopping `modal run` stops the episode. To run
  without your terminal, use `modal run --detach ...`, then fetch the results
  with `modal volume get robotwin-runs <run_id> data/runs/`.
- **Run one Modal episode at a time.** The policy worker keeps a single
  observation history (`max_containers=1`).
- **Frozen evaluation and campaigns run locally.** `--evaluation-manifest` is
  rejected, and `run_exploration.py` and `run_evaluation.py` use route 2.

### M5. Choose the simulator GPU

Set `SIM_GPU` on the same command line as `modal run`. The default is `L4`.

```bash
SIM_GPU=T4 modal run modal_apps/simulator.py::episode --config ... --output data/runs/NAME
SIM_GPU=A10G modal run modal_apps/simulator.py::spike    # measure before switching
```

`SIM_GPU` is read when `modal run` starts, so it applies to that run only. To
change the default, edit `GPU = os.environ.get("SIM_GPU", "L4")` in
`modal_apps/simulator.py`. The run's `summary.json` does not record the GPU, so
name the output folder after it when you compare GPUs.

Measured with the spike (seed 3; three RGB-D cameras):

| GPU | Render 3 RGB-D cameras | CuRobo plan | Approx. $/h* | Notes |
|---|---|---|---|---|
| T4 | 344 ms | 88 ms | ~0.59 | cheapest; only about 1 min slower per full episode |
| A10G | 153 ms | 73 ms | ~1.10 | slower than L4 and pricier |
| L4 | 110 ms | 86 ms | ~0.80 | default; fastest rendering tested |
| Laptop RTX 2050 (route 2) | 800 ms | 322 ms | — | for comparison |

\*Approximate; check Modal's pricing page.

**Which to pick:** a full Opus episode spends most of its time waiting for Opus,
the policy and SAM 3, so the T4 is usually the cheapest. The L4 pays off when
rendering dominates, for example with few Opus turns per policy step.

**Do the images match?** All three tested GPUs render the same images as the local
GPU (mean difference 0.03/255). Still record which GPU produced a result.

**Other GPUs:**

- CuRobo in the image is compiled for compute capabilities `7.5;8.0;8.6;8.9`, which
  covers T4, A100, A10G, L4 and L40S.
- For an H100 or another architecture, add it to `CUDA_ARCHES` in
  `modal_apps/simulator.py` (for example `9.0`). The next run rebuilds the CuRobo layer.
- The A100 has no ray-tracing cores and is untested for RoboTwin's ray-traced
  cameras. Run the spike first.

### M6. Manage remote runs

```bash
modal volume ls robotwin-runs                          # list runs
modal volume get robotwin-runs <run_id> data/runs/     # download again
modal volume rm -r robotwin-runs <run_id>              # delete
modal app list                                         # deployed apps and running containers
```

## Route 2: run locally

### L1. Anthropic key

Needed for `--planner anthropic`. It is kept in the conda environment; never commit it.

```bash
conda env config vars set -n robotwin ANTHROPIC_API_KEY=sk-ant-...
conda deactivate && conda activate robotwin
```

### L2. RoboTwin and assets

Run these from the repository root.

**Assets:** `hanging_mug` with `demo_clean` needs only the `embodiments` and
`objects` bundles, about 4 GB. RoboTwin's `scripts/_download_assets.sh` also
fetches 11 GB of background textures, which are only used with random
backgrounds and need about 30 GB of free space while unzipping.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
git -C RoboTwin checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
git -C RoboTwin apply ../patches/robotwin-renderer.patch

cd RoboTwin/assets
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as get; \
  get('TianxingChen/RoboTwin2.0', repo_type='dataset', local_dir='.', \
  allow_patterns=['embodiments.zip', 'objects.zip'])"
unzip -q embodiments.zip && rm embodiments.zip
unzip -q objects.zip && rm objects.zip && rm -rf __MACOSX
cd .. && python scripts/update_embodiment_config_path.py && cd ..
```

If the download fails with `Permission denied` under another drive, `HF_HOME`
points somewhere unwritable. Prefix the command with `HF_HOME=$HOME/.cache/huggingface`.

**Source patches:** apply RoboTwin's SAPIEN and mplib patches from
[`RoboTwin/scripts/_install.sh`](https://github.com/RoboTwin-Platform/RoboTwin).
These are the `urdf_loader.py` UTF-8 edit and the `planner.py` collision-check
edit. The Modal image applies the same two edits automatically.

### L3. CuRobo

CuRobo compiles CUDA code, and CUDA 12.1's `nvcc` rejects gcc newer than 12.

- **Compiler:** pin the build tools in the environment once. This needs `g++-12`.
- **GPU architecture:** set `TORCH_CUDA_ARCH_LIST` to your GPU's compute
  capability, for example 8.6 for RTX 30-series and RTX 2050, or 8.9 for RTX 40-series.
- **Where to clone:** from the repository root, not from inside `RoboTwin/`.

```bash
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git RoboTwin/envs/curobo
git -C RoboTwin/envs/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
conda env config vars set CUDA_HOME="$CONDA_PREFIX" CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
  CUDAHOSTCXX=/usr/bin/g++-12 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12" TORCH_CUDA_ARCH_LIST=8.6
conda deactivate && conda activate robotwin
python -m pip install -e RoboTwin/envs/curobo --no-build-isolation   # 5-15 minutes
python -m pip install warp-lang==1.12.0
python -c "import curobo; print('curobo ok')"
```

### L4. Vulkan

On laptops with an integrated GPU as well as the NVIDIA GPU, let Vulkan load
only the NVIDIA driver. Otherwise the renderer can freeze mid-episode with the
GPU at 0%.

```bash
conda env config vars set VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
conda deactivate && conda activate robotwin
vulkaninfo --summary | grep deviceName   # should list only the NVIDIA GPU
```

To review or remove these environment settings, use
`conda env config vars list -n robotwin` or `conda env config vars unset -n robotwin NAME`.

### L5. Look at the scene (simulator only)

Before running episodes, check that RoboTwin renders and see what each camera
sees. This resets the simulator once. It starts no policy, planner or Modal
client, and it never moves the robot.

```bash
python -m development.capture.scene_views --seed 3 --output data/scenes/seed_3
xdg-open data/scenes/seed_3/overview.png

# Compare seeds: each seed moves the mug and rack and picks one of 10 mug models
for s in 0 1 2 4; do python -m development.capture.scene_views --seed $s --output data/scenes/seed_$s; done

# Interactive 3D window (needs a display; close it to exit)
python -m development.capture.scene_views --seed 3 --output data/scenes/seed_3_viewer --viewer
```

Each folder gets colour and depth PNGs for every camera RoboTwin renders, an
`overview.png` contact sheet and `cameras.json` (size, position, depth range).

| Camera | Mounted | Used by the harness and policy? |
|---|---|---|
| `head_camera` | fixed, above the table | yes |
| `left_camera` | left wrist | yes |
| `right_camera` | right wrist | yes |
| `front_camera` | fixed, low and in front | no |
| `observer_camera` | third-person view | no |

### L6. Run episodes

Use a new `--output` folder for each run. `timeout -s INT` acts like Ctrl+C, so
a stuck run still writes `summary.json` and its video.

```bash
# Scripted smoke run: no Opus calls; expected outcome "step_budget" (see M4)
python run_skills.py --config configs/skill_runtime.yaml \
  --development-oracle --output data/runs/smoke_01

# Full task: Opus planner and public RGB-D perception
timeout -s INT 75m python run_skills.py \
  --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml \
  --planner anthropic \
  --perception-factory harness.perception.modal_sam3:create_detector \
  --output data/runs/full_task_01
```

Development-oracle mode is for internal debugging. Do not combine it with a
public perception factory. Run one RoboTwin job at a time on GPUs with little VRAM.

### L7. Campaigns, evaluation and the baseline

```bash
# Bounded scripted exploration campaign (one process per attempt)
python run_exploration.py --config configs/explore_natural_acquisition.yaml \
  --development-oracle --output data/runs/acquisition_01

# Frozen-policy baseline, no harness: two terminals
python run_inference_chunked.py sim --robotwin-root ./RoboTwin
python run_inference_chunked.py policy --episodes 1 --steps 900 --chunk-size 8
```

RoboTwin's own step limit for `hanging_mug` is 900 policy actions. Frozen
evaluation (`run_evaluation.py freeze` / `run`) also runs on this route.

### L8. Local GPU

Local runs use whichever NVIDIA GPU Vulkan and CUDA select. After changing
hardware, update `TORCH_CUDA_ARCH_LIST`, reinstall CuRobo (L3), and keep the
NVIDIA-only Vulkan setting (L4).

## Reference

### Outputs

Each run folder (both routes) contains:

- `manifest.json`: settings, seed and prompt hash
- `assessments.jsonl`: planner choices and turn results
- `skill_events.jsonl`: every action, tick and check
- `simulator_timing.jsonl`
- `observations/`
- `episode.mp4`, when `record_video: true`
- `summary.json`: outcome and counts
- `candidate_catalog.json`: memory candidates, never applied automatically

Exit code 0 means a success outcome such as `acquisition_confirmed` or
`task_success`. Run outputs live under the ignored `data/` folder.

### Policy and SAM 3 GPUs

Both routes use the same deployed workers. Their GPU is fixed in the app
definition: `gpu="T4"` in `modal_apps/policy.py` and `modal_apps/sam3.py`. To
change it, edit that value and redeploy:

```bash
modal deploy modal_apps/policy.py   # or modal_apps/sam3.py
```

Changing the policy GPU can change its outputs slightly, so treat runs before and
after as different conditions.

### Costs

- **Idle shutdown:** Modal containers stop when idle. The policy stops after 10
  minutes, SAM 3 after 2, and each episode container after its episode.
- **Cheapest checks:** the tests, then the scripted `--development-oracle` smoke
  run, which makes no Opus calls.
- **Capping long runs:**
  - Modal: `--time-limit-minutes`.
  - Local: `timeout -s INT`.
  - Opus calls: copy a config and lower `max_turns` / `anthropic.max_calls`.
- **Storage:** remote runs stay on the `robotwin-runs` Volume until you delete them (M6).

### Troubleshooting

| Symptom | Route | Fix |
|---|---|---|
| SAM 3 `GatedRepoError` | both | access not approved yet, or wrong token; replace with `modal secret create huggingface ... --force` |
| `missing pytorch3d` at start-up | both | harmless RoboTwin message |
| Smoke run ends with `step_budget` | both | expected, see M4 |
| `... already exists; choose a new --output` | both | each run needs a new output folder |
| Modal run fails with a missing secret | Modal | create the `anthropic` secret (M1) |
| `ImageBuildError` | Modal | `modal image logs <id>` with the newest image ID |
| `Permission denied` under `/run/media/...` during downloads | local | `HF_HOME` points at an unmounted drive; set it to `$HOME/.cache/huggingface` |
| `Not enough free disk space` | local | download only `embodiments` and `objects` (L2); `conda clean -a` frees space |
| `unsupported GNU version! gcc versions later than 12` | local | set the CuRobo build variables (L3) and reactivate the environment |
| CuRobo cloned into `RoboTwin/RoboTwin/envs/` | local | run the clone from the repository root |
| Episode freezes, GPU at 0%, run files stop changing | local | apply L4, stop with Ctrl+C, rerun with `timeout -s INT` |
