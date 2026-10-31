"""RoboTwin simulator and supervised episodes on Modal.

Builds the simulator image (RoboTwin + CuRobo at the README's pinned commits),
fills the asset Volume once, measures a GPU (spike), and runs whole
``run_skills.py`` episodes remotely with the same arguments as locally.

    modal run modal_apps/simulator.py::download_assets   # once; CPU only
    modal run modal_apps/simulator.py::spike             # default GPU: T4
    SIM_GPU=T4 modal run modal_apps/simulator.py::spike
    modal run modal_apps/simulator.py::episode --config ... --output data/runs/NAME [...]
    SIM_GPU=L4 modal run modal_apps/simulator.py::frames --seeds 0,1,2,3,4   # labelled perception frames
    SIM_GPU=L4 modal run modal_apps/simulator.py::frames --seeds 0 --perception-camera default \
      --output-root data/perception/frames_cam                              # ... with the perception camera

Episode-only options: ``--time-limit-minutes N`` (default 90) and
``--skip-observations`` (leave observations on the Volume, not downloaded).
Every ``modal run`` of this file needs the ``anthropic`` Modal secret to exist.
Spike reports and one head-camera frame per GPU are saved to data/modal_spike/.
"""

import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time

import modal

APP_NAME = "robotwin-simulator"
REPO = Path(__file__).resolve().parents[1]

ROBOTWIN = "/opt/RoboTwin"
ROBOTWIN_COMMIT = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
CUROBO_COMMIT = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

ASSETS = "/robotwin-assets"
ASSET_REPO = "TianxingChen/RoboTwin2.0"
# demo_clean disables random backgrounds, so background_texture (11 GB) is not needed.
ASSET_DIRS = ("embodiments", "objects")

RUNS = "/runs"
EPISODE_INPUTS = "/tmp/episode-inputs"
# After the time limit, SIGINT lets run_skills write summary.json and close the video.
SHUTDOWN_GRACE_SECONDS = 5 * 60
# Covers episode_request.MAX_TIME_LIMIT_MINUTES (210) plus shutdown and Volume commit.
EPISODE_FUNCTION_TIMEOUT_SECONDS = 4 * 60 * 60
# One 900-action labelled capture takes roughly 10 minutes on an L4.
CAPTURE_FUNCTION_TIMEOUT_SECONDS = 60 * 60
# Files committed inside a container can take seconds to appear in this
# machine's listing of the Volume; downloads wait up to a minute for them.
VOLUME_SETTLE_ATTEMPTS = 12
VOLUME_SETTLE_SECONDS = 5

# T4, A100, A10G, L4. Compiling CuRobo for all of them lets one image serve any GPU.
CUDA_ARCHES = "7.5;8.0;8.6;8.9"
# SIM_GPU may list fallbacks, e.g. "L4,A10G,T4": Modal uses the first type with free
# capacity instead of queueing for one. GPU is the label used in names and messages.
GPU_CHOICES = tuple(name.strip() for name in os.environ.get("SIM_GPU", "T4").split(",") if name.strip())
GPU = "-".join(GPU_CHOICES)
GPU_REQUEST = list(GPU_CHOICES) if len(GPU_CHOICES) > 1 else GPU_CHOICES[0]

# Fallback Vulkan loader entry for when the container runtime mounts NVIDIA's
# graphics libraries but not their ICD file (ManiSkill's documented fix).
NVIDIA_ICD = {"file_format_version": "1.0.0",
              "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155"}}


def _configure_image():
    """Apply RoboTwin's SAPIEN/mplib source patches and add the Vulkan ICD fallback.

    Mirrors RoboTwin/scripts/_install.sh, which edits installed packages in place.
    """
    import re
    import sysconfig

    site = Path(sysconfig.get_paths()["purelib"])
    loader = site / "sapien/wrapper/urdf_loader.py"
    loader.write_text(re.sub(r'("r")(\))( as)', r'\1, encoding="utf-8") as', loader.read_text()))
    planner = site / "mplib/planner.py"
    planner.write_text(planner.read_text().replace(
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"))

    icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if not icd.exists():
        icd.parent.mkdir(parents=True, exist_ok=True)
        icd.write_text(json.dumps(NVIDIA_ICD))


app = modal.App(APP_NAME)
assets_volume = modal.Volume.from_name("robotwin-assets", create_if_missing=True)
runs_volume = modal.Volume.from_name("robotwin-runs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "unzip", "build-essential", "libvulkan1", "vulkan-tools",
                 "libgl1", "libegl1", "libglib2.0-0", "libx11-6", "libxext6", "libxrender1")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "CUDA_HOME": "/usr/local/cuda",
          "TORCH_CUDA_ARCH_LIST": CUDA_ARCHES, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "requirements", "/opt/requirements", copy=True)
    .run_commands("python -m pip install -r /opt/requirements/sim.txt "
                  "-r /opt/requirements/planner.txt 'modal>=1,<2'")
    .run_commands(f"git clone https://github.com/RoboTwin-Platform/RoboTwin.git {ROBOTWIN}",
                  f"git -C {ROBOTWIN} checkout {ROBOTWIN_COMMIT}")
    .add_local_file(REPO / "patches/robotwin-renderer.patch", "/opt/robotwin-renderer.patch", copy=True)
    .run_commands(f"git -C {ROBOTWIN} apply /opt/robotwin-renderer.patch")
    # CUDA kernels compile for CUDA_ARCHES without a GPU attached (slow, cached once).
    # --no-build-isolation uses installed build tools, so add CuRobo's build-system
    # requirements first; ninja replaces the much slower serial compile fallback.
    # Modal's standalone Python records clang as its compiler and linker; this
    # image only has GCC, so native extension builds must be told to use it.
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -shared", "LDCXXSHARED": "g++ -shared"})
    .run_commands("python -m pip install wheel 'setuptools_scm>=6.2' ninja",
                  f"git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git {ROBOTWIN}/envs/curobo",
                  f"git -C {ROBOTWIN}/envs/curobo checkout {CUROBO_COMMIT}",
                  f"python -m pip install -e {ROBOTWIN}/envs/curobo --no-build-isolation",
                  "python -m pip install warp-lang==1.12.0")
    .run_function(_configure_image)
    # open3d loads libusb on import.
    .apt_install("libusb-1.0-0")
    # The container also exposes Mesa's llvmpipe CPU renderer; load only NVIDIA's
    # Vulkan driver (the same pin that stopped renderer hangs on the hybrid laptop).
    .env({"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
          "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json"})
    # RoboTwin reads ./assets/<dir>; the directories themselves live on the Volume.
    .run_commands(*(f"ln -s {ASSETS}/{name} {ROBOTWIN}/assets/{name}" for name in ASSET_DIRS))
    # Project code last, so code edits do not rebuild the layers above.
    .add_local_dir(REPO / "harness", "/root/harness", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "development", "/root/development", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "configs", "/root/configs")
    .add_local_file(REPO / "run_skills.py", "/root/run_skills.py")
    .add_local_file(REPO / "modal_apps/episode_request.py", "/root/episode_request.py")
)


@app.function(image=image, volumes={ASSETS: assets_volume}, timeout=60 * 60)
def download_assets():
    """Download and unpack the RoboTwin assets into the Volume; skip parts already present."""
    import shutil
    import zipfile
    from huggingface_hub import hf_hub_download

    for name in ASSET_DIRS:
        target = Path(ASSETS) / name
        if target.is_dir():
            print(f"{name}: already present")
            continue
        archive = hf_hub_download(ASSET_REPO, f"{name}.zip", repo_type="dataset", local_dir="/tmp/assets")
        # Unpack beside the target, then rename, so an interrupted run never looks complete.
        staging = Path(ASSETS) / f".partial-{name}"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.namelist() if not m.startswith("__MACOSX/")]
            bundle.extractall(staging, members=members)
        (staging / name).rename(target)
        shutil.rmtree(staging)
        os.remove(archive)
        print(f"{name}: unpacked")

    # Writes the CuRobo configs with absolute paths for ROBOTWIN inside the container.
    subprocess.run([sys.executable, "scripts/update_embodiment_config_path.py"],
                   cwd=ROBOTWIN, check=True, input="n\n", text=True)
    assets_volume.commit()


def _command_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=120).stdout[-4000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"


def _timed(function, repeats=1):
    started = time.perf_counter()
    for _ in range(repeats):
        result = function()
    return result, (time.perf_counter() - started) * 1000.0 / repeats


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume}, timeout=30 * 60)
def measure(seed: int = 3, observations: int = 20, physics_steps: int = 200):
    """Report driver, Vulkan, render, physics and CuRobo timings for one simulator."""
    import io
    import numpy as np
    from PIL import Image

    sys.path.insert(0, "/root")
    report = {"nvidia_smi": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                              "--format=csv,noheader"]),
              "vulkan": _command_output(["vulkaninfo", "--summary"])}
    head_png = None
    try:
        from development.simulator.local_simulator import LocalSimulator
        from harness.environment.local import LocalEnvironmentClient

        environment = LocalEnvironmentClient(LocalSimulator(ROBOTWIN, rgbd_enabled=True))
        observation, report["reset_ms"] = _timed(lambda: environment.reset(seed))
        _, report["observe_rgbd_ms"] = _timed(environment.observe, observations)

        images = observation.images
        # A missing camera falls back to a black frame, so check the pixels actually vary.
        report["camera_pixel_std"] = {name: float(np.std(image)) for name, image in images.items()}
        buffer = io.BytesIO()
        Image.fromarray(images["head_camera"]).save(buffer, format="PNG")
        head_png = buffer.getvalue()

        scene = environment.simulator.task.scene
        _, report["physics_step_ms"] = _timed(scene.step, physics_steps)

        pose = np.asarray(environment.arm_state("left")["pose"], dtype=float)
        pose[2] += 0.05
        plan, report["curobo_plan_ms"] = _timed(lambda: environment.plan("left", pose))
        report["curobo_plan_status"] = plan.get("status") if isinstance(plan, dict) else repr(plan)[:200]
        environment.close()
        report["ok"] = True
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:2000])
    return report, head_png


@app.local_entrypoint()
def spike(seed: int = 3):
    report, head_png = measure.remote(seed=seed)
    # SIM_GPU is only set on this machine; inside the container GPU reads the default.
    report = {"gpu_requested": GPU, **report}
    output = REPO / "data/modal_spike"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{GPU}.json").write_text(json.dumps(report, indent=2) + "\n")
    if head_png:
        (output / f"{GPU}_head_camera.png").write_bytes(head_png)
    print(json.dumps({key: value for key, value in report.items() if key != "vulkan"}, indent=2))
    print("vulkan summary saved in", output / f"{GPU}.json")


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume, RUNS: runs_volume},
              secrets=[modal.Secret.from_name("anthropic")],
              timeout=EPISODE_FUNCTION_TIMEOUT_SECONDS, single_use_containers=True)
def run_episode(argv: list[str], uploads: dict[str, bytes], time_limit_minutes: int) -> dict:
    """Run one run_skills.py episode and keep its artifacts on the runs Volume.

    A single-use container replaces the fresh process each local attempt gets, so
    no CUDA, renderer or planner state survives between episodes. If Modal
    preempted an earlier attempt, its folder is kept beside this one and the
    episode starts again from reset.
    """
    from episode_request import set_aside_previous_attempt

    output = Path(argv[argv.index("--output") + 1])
    set_aside = set_aside_previous_attempt(output)
    if set_aside is not None:
        print(f"Earlier attempt was interrupted; kept it as {set_aside.name} and restarting the episode",
              flush=True)
    inputs = Path(EPISODE_INPUTS)
    inputs.mkdir(parents=True, exist_ok=True)
    for name, content in uploads.items():
        (inputs / name).write_bytes(content)

    process = subprocess.Popen([sys.executable, "/root/run_skills.py", *argv], cwd="/root")
    timed_out = False
    try:
        exit_code = process.wait(timeout=time_limit_minutes * 60)
    except subprocess.TimeoutExpired:
        # Same as `timeout -s INT` locally: the runner records an interrupted outcome.
        timed_out = True
        process.send_signal(signal.SIGINT)
        try:
            exit_code = process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait()
    runs_volume.commit()

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return {"exit_code": exit_code, "timed_out": timed_out, "summary": summary,
            "interrupted_attempt": None if set_aside is None else set_aside.name}


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume, RUNS: runs_volume},
              timeout=CAPTURE_FUNCTION_TIMEOUT_SECONDS, single_use_containers=True)
def capture_frames(run_id: str, seed: int, every: int, max_actions: int, chunk_size: int,
                   perception_camera: str = "") -> dict:
    """Run development.capture.labeled_frames in a fresh container; frames go to the runs Volume."""
    sys.path.insert(0, "/root")
    from episode_request import set_aside_previous_attempt
    from development.capture.labeled_frames import capture
    from harness.environment.perception_cameras import parse_camera_spec

    output = Path(RUNS) / run_id
    set_aside_previous_attempt(output)  # a preempted attempt restarts the rollout cleanly
    try:
        manifest = capture(ROBOTWIN, output, seed, every=every, max_actions=max_actions, chunk_size=chunk_size,
                           perception_camera=parse_camera_spec(perception_camera) if perception_camera else None)
        # With GPU fallbacks the type that actually ran is only known here.
        manifest["gpu"] = _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        runs_volume.commit()


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume, RUNS: runs_volume},
              timeout=CAPTURE_FUNCTION_TIMEOUT_SECONDS + 30 * 60, single_use_containers=True)
def run_baseline(run_id: str, seed: int, max_actions: int, chunk_size: int,
                 time_limit_minutes: int, perception_camera: str = ""):
    """One frozen Neotix rollout; no Anthropic secret or API calls in this function."""
    sys.path.insert(0, "/root")
    from episode_request import set_aside_previous_attempt
    from development.hanging.baseline import run

    output = Path(RUNS) / run_id
    set_aside_previous_attempt(output)
    try:
        return run(ROBOTWIN, output, seed=seed, max_actions=max_actions,
                   chunk_size=chunk_size, time_limit_minutes=time_limit_minutes,
                   perception_camera=perception_camera)
    finally:
        runs_volume.commit()


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume, RUNS: runs_volume},
              timeout=30 * 60, retries=0, max_containers=1, single_use_containers=True)
def run_entry660_validation(run_id: str, uploads: dict[str, bytes], allow_unverified: bool):
    import tempfile
    sys.path.insert(0, "/root")
    from development.checkpoints.schemas import relative_path
    from development.hanging.run_entry660 import run

    try:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for name, data in uploads.items():
                relative_path(name)
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return run(package, Path(RUNS) / run_id, ROBOTWIN,
                       allow_unverified=allow_unverified)
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def hanging_entry660(package: str = "data/runs/hang_replay_candidate_660",
                     allow_unverified: bool = False):
    """One serial, policy-free entry comparison; does not establish grasp/support."""
    from development.checkpoints.bank import CheckpointBank
    from development.hanging.run_entry660 import validate_package

    checkpoint = validate_package(package, allow_unverified=allow_unverified)
    bank = CheckpointBank(package)
    paths = {*checkpoint.artifact_hashes, "records/unverified_hang_660.json"}
    uploads = {name: bank.resolve_artifact(name).read_bytes() for name in paths}
    run_id = f"entry660_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.time_ns()}"
    target = REPO / "data/runs" / run_id
    print(f"Unverified entry 660 on {GPU}; exclusive serial run; hard timeout 30 min")
    try:
        report = run_entry660_validation.remote(run_id, uploads, allow_unverified)
        print(json.dumps(report, indent=2))
    finally:
        if _wait_for_volume(run_id):
            _download_run(run_id, target, skip_observations=False)
        print(f"Results: {target}; remote robotwin-runs/{run_id}")


grounding_preflight_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26,<3", "pillow", "pydantic>=2.13,<3", "anthropic")
    .add_local_dir(REPO / "harness", "/root/harness", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "development", "/root/development", ignore=["**/__pycache__"])
)


@app.function(image=grounding_preflight_image, cpu=1, timeout=300, retries=0,
              max_containers=1, secrets=[modal.Secret.from_name("anthropic")])
def preflight_hanging_recovery(uploads: dict[str, bytes], model: str):
    import tempfile
    from development.checkpoints.schemas import relative_path
    from development.hanging.grounding_preflight import preflight
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package, output = root / "package", root / "preflight"
        for name, data in uploads.items():
            relative_path(name)
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        report = preflight(package, output, model)
        files = {path.relative_to(output).as_posix(): path.read_bytes()
                 for path in output.rglob("*") if path.is_file()}
        return report, files


@app.function(image=image, gpu=GPU_REQUEST, volumes={ASSETS: assets_volume, RUNS: runs_volume},
              timeout=45 * 60, retries=0, max_containers=1, single_use_containers=True,
              secrets=[modal.Secret.from_name("anthropic")])
def run_hanging_recovery(run_id: str, uploads: dict[str, bytes], model: str,
                         allow_unverified: bool, preflight_result: dict):
    import tempfile
    sys.path.insert(0, "/root")
    from development.checkpoints.schemas import relative_path
    from development.hanging.recovery import run
    try:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for name, data in uploads.items():
                relative_path(name)
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return run(package, Path(RUNS) / run_id, ROBOTWIN,
                       model=model, allow_unverified=allow_unverified,
                       preflight_result=preflight_result)
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def hanging_recovery(model: str, package: str = "data/runs/hang_replay_candidate_660",
                     allow_unverified: bool = False, allow_entry660_demo_override: bool = False,
                     aperture_annotation: str = "", peg_annotation: str = "",
                     reviewed_archive: str = ""):
    from development.checkpoints.bank import CheckpointBank
    from development.hanging.run_entry660 import validate_package
    if not model.startswith("claude-") or "MODEL" in model:
        raise ValueError("real_claude_model_id_required")
    checkpoint = validate_package(package, allow_unverified=allow_unverified)
    bank = CheckpointBank(package)
    paths = {*checkpoint.artifact_hashes, "records/unverified_hang_660.json"}
    uploads = {name: bank.resolve_artifact(name).read_bytes() for name in paths}
    run_id = f"hanging_recovery_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.time_ns()}"
    target = REPO / "data/runs" / run_id
    if allow_entry660_demo_override:
        from development.hanging.recovery import entry660_demo_preflight
        if not all((aperture_annotation, peg_annotation, reviewed_archive)):
            raise ValueError("reviewed_archive_and_both_annotations_required")
        print("LOCAL reviewed Entry660 demo preflight; no model/API calls", flush=True)
        preflight_result = entry660_demo_preflight(reviewed_archive,
            json.loads(Path(aperture_annotation).read_text()),
            json.loads(Path(peg_annotation).read_text()), allow_override=True)
        preflight_result["model"] = model
        files = {"report.json": json.dumps(preflight_result, indent=2).encode()}
        print(json.dumps({key: preflight_result[key] for key in
                          ("prepared_targets", "clearance_terms", "mating_prior")}, indent=2))
    else:
        print(f"CPU grounding preflight with {model}; simulator has not started", flush=True)
        preflight_result, files = preflight_hanging_recovery.remote(uploads, model)
    from development.checkpoints.schemas import relative_path
    for name, data in files.items():
        relative_path(name)
        artifact = target / "preflight" / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with artifact.open("xb") as stream:
            stream.write(data)
    if preflight_result["status"] != "passed":
        print(json.dumps(preflight_result, indent=2))
        print(f"Preflight stopped before GPU/simulator startup. Details: {target / 'preflight/report.json'}")
        return
    print(f"Recorded prefix to 660, then grounded recovery on {GPU}; model={model}")
    try:
        report = run_hanging_recovery.remote(run_id, uploads, model, allow_unverified, preflight_result)
        print(json.dumps(report, indent=2))
    finally:
        if _wait_for_volume(run_id):
            _download_run(run_id, target, skip_observations=False)
        print(f"Recovery video and report: {target}; remote robotwin-runs/{run_id}")


@app.local_entrypoint()
def baseline(seed: int = 3, max_actions: int = 900, chunk_size: int = 8,
             time_limit_minutes: int = 30, output: str = "", perception_camera: str = ""):
    """Record a single Claude-free baseline with RGB-D, actions, video and summary."""
    from development.hanging.baseline import validate_options

    validate_options(seed, max_actions, chunk_size, time_limit_minutes)
    if perception_camera:
        from harness.environment.perception_cameras import parse_camera_spec
        parse_camera_spec(perception_camera)  # Validate before starting a GPU.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"baseline_seed{seed}_{stamp}_{time.time_ns() % 1000000:06d}"
    target = Path(output).resolve() if output else REPO / "data/runs" / run_id
    if target.exists():
        raise SystemExit(f"baseline: {target} already exists; choose a new --output")
    print(f"Running {run_id} on {GPU}; no Claude calls", flush=True)
    result = None
    try:
        result = run_baseline.remote(run_id, seed, max_actions, chunk_size, time_limit_minutes,
                                     perception_camera)
    finally:
        if _wait_for_volume(run_id):
            count = _download_run(run_id, target, skip_observations=False)
            print(f"Downloaded {count} files to {target}")
        print(f"Remote results: robotwin-runs/{run_id}")
    print(json.dumps(result, indent=2))
    if result is None or result.get("outcome") in ("initialization_error", "execution_error"):
        raise SystemExit("baseline failed; inspect summary.json and action logs")


@app.local_entrypoint()
def frames(seeds: str = "0,1,2,3,4", every: int = 16, max_actions: int = 900, chunk_size: int = 16,
           output_root: str = "data/perception/frames", perception_camera: str = ""):
    """Capture labelled perception frames on Modal and download each seed to <output_root>/seed_<N>.

    ``--perception-camera default`` (or ``x,y,z/tx,ty,tz/fovy``) adds a
    perception-only camera to every frame; the policy still gets its three.

    Seeds run one after another: the deployed policy worker keeps a single
    observation history, so concurrent rollouts would mix their inputs.
    """
    seed_list = [int(value) for value in seeds.split(",") if value.strip()]
    if perception_camera:
        sys.path.insert(0, str(REPO))
        from harness.environment.perception_cameras import parse_camera_spec
        parse_camera_spec(perception_camera)  # reject a typo before any GPU starts
    root = Path(output_root).resolve()
    targets = {seed: root / f"seed_{seed}" for seed in seed_list}
    existing = [str(path) for path in targets.values() if path.exists()]
    if not seed_list or existing:
        raise SystemExit("frames: no seeds given" if not seed_list else
                         "frames: already exist (delete them or pass --output-root): " + ", ".join(existing))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    failed = {}
    for seed, target in targets.items():
        run_id = f"frames_{root.name}_seed{seed}_{stamp}"
        print(f"Seed {seed}: capturing on {GPU} (up to {max_actions} actions) ...", flush=True)
        try:
            manifest = capture_frames.remote(run_id, seed, every, max_actions, chunk_size, perception_camera)
            print(f"Seed {seed}: {manifest['frames_saved']} frames, {manifest['policy_actions']} actions, "
                  f"stopped by {manifest['stop_reason']}, on {manifest['gpu']}", flush=True)
        except Exception as exc:
            # One failed rollout should not cost the remaining seeds.
            manifest = None
            failed[seed] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"Seed {seed}: FAILED ({failed[seed]}); keeping its partial frames", flush=True)
        # Keep whatever was saved even if the rollout failed part-way.
        expected = None if manifest is None else manifest["frames_saved"]
        if _wait_for_volume(run_id, expected_observations=expected) == 0:
            print(f"Seed {seed}: nothing saved on the Volume", flush=True)
            continue
        count = _download_run(run_id, target, skip_observations=False)
        frames_downloaded = len(list((target / "observations").glob("observation_*.npz")))
        print(f"Seed {seed}: downloaded {count} files ({frames_downloaded} frames) to {target}", flush=True)
        if expected is not None and frames_downloaded < expected:
            print(f"Seed {seed}: WARNING only {frames_downloaded} of {expected} frames arrived; fetch again with "
                  f"modal volume get robotwin-runs {run_id} <dir>", flush=True)
    if failed:
        raise SystemExit(f"frames: seeds failed: {failed}")


def _wait_for_volume(run_id, *, expected_observations=None):
    """Wait until this machine's view of the runs Volume shows a finished run.

    With ``expected_observations`` wait for that many observation files;
    otherwise wait until two listings agree. Returns the number of files seen.
    """
    previous = None
    for attempt in range(VOLUME_SETTLE_ATTEMPTS):
        try:
            paths = [entry.path for entry in runs_volume.listdir(run_id, recursive=True)]
        except modal.exception.NotFoundError:
            paths = []
        observations = sum("/observations/observation_" in f"/{path}" for path in paths)
        if expected_observations is not None and observations >= expected_observations:
            return len(paths)
        if expected_observations is None and paths and len(paths) == previous:
            return len(paths)
        previous = len(paths)
        if attempt + 1 < VOLUME_SETTLE_ATTEMPTS:
            time.sleep(VOLUME_SETTLE_SECONDS)
    return previous or 0


def _run_exists_on_volume(run_id):
    try:
        return bool(runs_volume.listdir(run_id))
    except modal.exception.NotFoundError:
        return False


def _download_run(run_id, destination, *, skip_observations):
    """Copy one run directory from the runs Volume into ``destination``; return the file count."""
    from modal.volume import FileEntryType

    count = 0
    for entry in runs_volume.listdir(run_id, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = PurePosixPath(entry.path.lstrip("/")).relative_to(run_id)
        if skip_observations and relative.parts[0] == "observations":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            runs_volume.read_file_into_fileobj(entry.path, stream)
        count += 1
    return count


@app.local_entrypoint()
def episode(*argv):
    """Run one supervised episode on Modal with run_skills.py's own arguments."""
    sys.path.insert(0, str(REPO))
    import run_skills
    from modal_apps.episode_request import build_request

    # Without --output, name the run like local runs, marked with where it ran.
    known, _unknown = run_skills.build_parser(REPO).parse_known_args(list(argv))
    if known.output is None:
        argv = (*argv, "--output", str(run_skills.default_output(
            REPO, known.config, known.seed, where=f"modal-{GPU}")))
    try:
        request = build_request(argv, input_dir=EPISODE_INPUTS, runs_dir=RUNS, robotwin_root=ROBOTWIN)
    except ValueError as exc:
        raise SystemExit(f"episode: {exc}") from None
    # The same checks run_skills.main makes, before any GPU is allocated.
    run_skills.parse_arguments(run_skills.build_parser(REPO), list(request.local_argv))
    output = request.local_output.resolve()
    if output.exists():
        raise SystemExit(f"episode: {output} already exists; choose a new --output")
    if _run_exists_on_volume(request.run_id):
        raise SystemExit(f"episode: {request.run_id} already exists on the robotwin-runs Volume; "
                         "choose a new --output or delete it with modal volume rm -r")
    uploads = {name: path.read_bytes() for name, path in request.uploads.items()}

    print(f"Running {request.run_id} on {GPU} (limit {request.time_limit_minutes} min)")
    result = None
    try:
        result = run_episode.remote(list(request.remote_argv), uploads, request.time_limit_minutes)
    finally:
        # Partial artifacts are still worth having after a failed or interrupted run.
        try:
            _wait_for_volume(request.run_id)
            count = _download_run(request.run_id, output, skip_observations=request.skip_observations)
            print(f"Downloaded {count} files to {output}")
        except Exception as exc:
            print(f"No artifacts downloaded for {request.run_id}: {type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2))
    if result and result.get("interrupted_attempt"):
        print(f"Modal preempted an earlier attempt; its partial run is on the Volume as "
              f"{result['interrupted_attempt']} (modal volume get robotwin-runs "
              f"{result['interrupted_attempt']} data/runs/).")
