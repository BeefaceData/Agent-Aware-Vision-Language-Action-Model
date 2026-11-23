# Windows + Modal setup

All commands use **PowerShell**. Your laptop launches commands; Modal runs Linux,
RoboTwin, CuRobo and the GPU workers.

- **Section A: From scratch** — one-time installation on a new machine.
- **Section B: Everyday commands** — use after reopening PowerShell or restarting.

On a new machine start at section A. Once section A has been completed, use
section B.

## A. From scratch — run once

### A1. Clone the repository

Install Git and Miniconda first. Open PowerShell in the parent folder where you
want the project, then run:

```powershell
git clone https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model.git RoboTwin-Modal
Set-Location RoboTwin-Modal
```

If this repository is already on your laptop, open its folder and skip A1.

### A2. Create Python environment and install dependencies

```powershell
conda create -n robotwin python=3.10 -y
conda run -n robotwin python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/planner.txt 'modal>=1,<2' 'numpy==1.26.4'
.\.venv\Scripts\python.exe -m pip install 'imageio[ffmpeg]'
```

Skip `conda create` if `robotwin` already exists. Skip all of A2 if this checkout's
`.venv` already works. Do not install `requirements/sim.txt` on native Windows.
The explicit `.venv\Scripts\...` commands do not require environment activation.

### A3. Set terminal settings and sign in

```powershell
$env:MODAL_IMAGE_BUILDER_VERSION = '2025.06'
$env:SIM_GPU = 'T4'
$env:MODAL_ENVIRONMENT = 'main'
.\.venv\Scripts\modal.exe setup
.\.venv\Scripts\modal.exe profile list
```

Complete browser login to workspace **beefacedata**. Complete account verification
in the Modal dashboard if requested. These terminal settings must also be restored
each day using B1. The older image builder is required for simulator Python 3.10.

### A4. Create the Anthropic secret in Modal

In the Modal dashboard, choose workspace **beefacedata**, environment **main**,
then **Secrets → Create new secret**:

| Setting | Exact value |
|---|---|
| Secret name | `anthropic` |
| Field name | `ANTHROPIC_API_KEY` |
| Field value | Your Anthropic API key |

The current simulator module requires this secret even for scripted runs.
Use `anthropic`, not `anthropic-secret`. Keep the key out of source files and chat.
Skip this step if the secret already exists.

### A5. Check local dependencies

The repository's `.gitattributes` keeps `patches/*.patch` in LF line endings, which
the Linux image build needs; no manual conversion is required after cloning.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest tests.test_episode_video tests.test_episode_request tests.test_runtime_assembly -q
```

On Windows the full suite has known symlink, path and process-handling
limitations, plus tests requiring local PyTorch or matplotlib.

### A6. Deploy the policy and download simulator assets

```powershell
.\.venv\Scripts\modal.exe deploy modal_apps/policy.py
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::download_assets
```

Keep your laptop awake until completion. The first image build can take 20–40
minutes; completed layers are cached. Success ends with `App completed`, after
unpacking the assets and updating their configuration paths. The NVIDIA-driver
warning during this CPU-only download step is expected.

Deployment creates the policy service; it does not yet verify model loading.
Check model loading and reset separately:

```powershell
.\.venv\Scripts\modal.exe run modal_apps/policy.py
```

This may download weights on first use. It does not test action inference.

### A7. Check the simulator GPU

```powershell
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::spike
```

Success means `"ok": true`, all three camera pixel-standard-deviation values are
nonzero, and `"curobo_plan_status": "Success"`. The T4 passed in this setup.
The `missing pytorch3d` message did not prevent the check.

### A8. Run the first smoke episode

Use the command in **B2** below. After this, use section B for everyday work.

### A9. Optional setup for full Claude + SAM 3 episodes

Only needed for B3. Obtain approved access to [facebook/sam3](https://huggingface.co/facebook/sam3).
In Modal environment **main**, create a secret named `huggingface`, containing
field `HF_TOKEN` with a token from the approved Hugging Face account. Then run:

```powershell
.\.venv\Scripts\modal.exe deploy modal_apps/sam3.py
.\.venv\Scripts\modal.exe run modal_apps/sam3.py
```

Wait for the SAM 3 smoke test to succeed before B3. Full episodes use Anthropic
API billing in addition to Modal GPU/storage usage.

## B. Everyday commands — after opening PowerShell

### Claude-free frozen-policy baseline (normal project folder)

Your normal project folder now uses branch `hang/stage-evidence`. This command calls
the frozen Neotix worker directly and records RGB-D, actions, video and summary.
It does not call Claude or SAM 3 and applies no analytic recovery. The simulator
module still declares an Anthropic secret for its separate supervised function,
so keep the existing secret configured.

From the main repository folder, first run a short integration check:

```powershell
$env:MODAL_IMAGE_BUILDER_VERSION = '2025.06'
$env:SIM_GPU = 'T4'
$env:MODAL_ENVIRONMENT = 'main'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::baseline --seed 3 --max-actions 64 --chunk-size 8 --time-limit-minutes 15
```

After verifying nonzero policy actions and a usable video, repeat with
`--max-actions 900 --time-limit-minutes 30` for the full baseline. GPU and model
worker usage are billable. Keep the terminal open and laptop awake; use
`run --detach` if disconnecting after remote submission. Run only one episode.

The command prints its local output directory (inside this project's `data/runs/`)
and remote `robotwin-runs/<run_id>` recovery path. Outputs include `episode.mp4`,
`observations/`, `actions.jsonl`, `manifest.json`, and `summary.json`. Action-limit
completion is an unsuccessful bounded attempt, not an execution error. Native
success does not independently verify sustained hanging. The soft time limit is
checked between blocking simulator/policy calls; the remote function has a hard
90-minute cap. A preempted attempt restarts from reset with prior artifacts kept.

The short T4 run completed: 64 actions, eight chunks, no Claude calls, no reported
execution errors. Outcome `action_limit` was expected. A full trajectory and
hanging success have not yet been validated. All 108 focused tests pass locally.

#### Optional fourth camera and SAM 3 replay

The baseline supports a fixed perception-only camera. It is recorded as RGB-D
alongside the three original views and displayed in a four-panel video. Neotix
still receives only its original three views. The camera pose is recorded in
`manifest.json`; it is added after scene placement to preserve the seed layout.
This camera integration has passed CPU tests but still needs a live recording check.

From this normal project folder, after setting the environment variables above:

```powershell
$runFolder = 'data/runs/baseline_camera_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::baseline --seed 3 --max-actions 64 --chunk-size 8 --time-limit-minutes 15 --perception-camera default --output $runFolder
```

After completion, query SAM 3 on at most five saved fourth-camera frames:

```powershell
.\.venv\Scripts\python.exe -m development.replays.sam3_cache --frames $runFolder --camera perception_camera --every 16 --max-frames 5
```

This second command uses the deployed SAM 3 GPU worker (billable), but does not
run the simulator or Claude. Results, including encoded masks, are written to
`$runFolder/sam3_cache.json`; source observations/video remain unchanged. Subsequent
replay through `CachedSam3Worker` can use that cache without further API calls.
This is post-run segmentation, not online control or proof of rack support.
The interval counts saved archives, not seconds. To use a different selection,
provide `--output` with a new cache filename; existing files are not overwritten.
If reopening PowerShell, assign `$runFolder` to the printed run folder again.
Old three-camera recordings cannot acquire a fourth view retrospectively.

SAM 3 detections in the cache are model outputs, not manually verified masks or
evidence of contact or hanging. Cache creation refuses an existing output file.
If `invalid_sam3_request` appears after updating local camera support, verify the
actual worker using its `capabilities` method; local changes alone do not establish
which server code handled a request. The worker now returns field-specific errors.

### B1. Start here every day

```powershell
Set-Location '<path-to>\RoboTwin-Modal'
$env:MODAL_IMAGE_BUILDER_VERSION = '2025.06'
$env:SIM_GPU = 'T4'
$env:MODAL_ENVIRONMENT = 'main'
```

Replace `<path-to>` with the folder you cloned into. **No reinstall, secret
creation, asset download or policy deployment is needed just because you restarted.**

After B1, choose **one** of B2 or B3. Run one episode at a time because the policy
worker shares observation history. Timestamped output names avoid overwriting runs.

### B2. Run a scripted smoke episode

```powershell
$runOutput = 'data/runs/T4_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::episode --config configs/skill_runtime.yaml --development-oracle --output $runOutput --time-limit-minutes 15
```

Keep the laptop awake and terminal open. Results download automatically to
`$runOutput`. The expected outcome is `step_budget`, with `phase: FAILED` and
`exit_code: 1`. This checks execution and logging; it does not prove hanging
success or policy inference. Your first smoke run executed 724 analytic ticks
and zero policy actions, with no exception or timeout.

Read the summary after the command finishes:

```powershell
Get-Content (Join-Path $runOutput 'summary.json')
```

### B3. Run a full Claude + public-perception episode

Requires completed A9 and a working Anthropic key. This is experimental; full-task
success is not guaranteed.

```powershell
$runOutput = 'data/runs/T4_full_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\modal.exe run modal_apps/simulator.py::episode --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml --planner anthropic --perception-factory harness.perception.modal_sam3:create_detector --output $runOutput --time-limit-minutes 75
```

Keep the laptop awake and terminal open. Results download automatically.
Do not combine public perception with `--development-oracle`.

### B4. Run while your laptop is closed

Use this **instead of** B2 if you want a detached smoke episode:

```powershell
$runOutput = 'data/runs/T4_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\modal.exe run --detach modal_apps/simulator.py::episode --config configs/skill_runtime.yaml --development-oracle --output $runOutput --time-limit-minutes 15
```

For a full episode, add `--detach` immediately after `run` in the B3 command.
Wait until the remote call is submitted before closing the laptop. Save the app
URL and the `Running <run_id> on T4` line. The remote episode can continue after
disconnection, subject to timeouts/errors, but local downloads cannot run while
the laptop sleeps. Use B5 to retrieve results later. Usage continues while running.

[Modal's detach documentation](https://modal.com/docs/cli/latest/run).

### B5. Download an earlier run after reconnecting

Run B1 first, then list saved runs:

```powershell
.\.venv\Scripts\modal.exe volume ls robotwin-runs
```

Copy the actual directory name from that list into `$remoteRun` below:

```powershell
$remoteRun = 'REPLACE_WITH_DIRECTORY_NAME_FROM_LIST'
$localRun = Join-Path 'data/runs' $remoteRun
.\.venv\Scripts\modal.exe volume get robotwin-runs $remoteRun $localRun
Get-Content (Join-Path $localRun 'summary.json')
```

Remote names look like `modal_smoke_01-20261029T093454Z`. Wait for the episode to
finish and commit results before downloading. Files may include the summary,
manifest, event logs, observations and a video when recording is enabled.

### B6. Check status or diagnose a problem

```powershell
.\.venv\Scripts\modal.exe app list
.\.venv\Scripts\modal.exe secret list
.\.venv\Scripts\modal.exe volume list
```

| Problem | Fix |
|---|---|
| Unsupported Python 3.10 | Run B1 to select builder `2025.06` |
| Waiting for L4 capacity | Run B1 to select T4; stop the old attempt before relaunching |
| Secret not found | Check exact name `anthropic`, field `ANTHROPIC_API_KEY`, environment `main` |
| Renderer patch fails | Check `git ls-files --eol patches/` shows `w/lf`; if not, run `git add --renormalize .` |
| Image build externally shut down | Retry; if repeated, inspect logs below |
| Output directory exists | Generate a new `$runOutput`; keep previous results |
| SAM 3 access denied | Check gated-model approval and the account behind `HF_TOKEN` |
| Missing login on a new machine | Run `.\.venv\Scripts\modal.exe setup` |

For image-build logs, replace the placeholder with the image ID from the error:

```powershell
.\.venv\Scripts\modal.exe image logs IMAGE_ID_FROM_ERROR
```

### B7. What to keep and back up

Closing your laptop preserves the repository, `.venv`, downloaded files, Modal
secrets, deployed policy and saved volumes. Only the terminal settings in B1 need
restoring. A normal attached run may stop on disconnection; use B4 to avoid that.
An interrupted episode does not resume its physics state; start a new episode.
A crash before saving/committing may leave partial artifacts.

Back up source changes, this guide and important `data/runs/` results. Git ignores
`data/` and `.venv/`; this guide is eligible for Git tracking but must be committed
and pushed to be backed up on GitHub. Never commit keys or `~/.modal.toml`.

The current Modal launcher supports individual episodes. Exploration campaigns
and frozen evaluation runners still require the repository's local Linux route.
