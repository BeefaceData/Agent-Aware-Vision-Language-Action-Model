"""RoboTwin simulator on Modal (migration phase 0: feasibility spike).

Builds the simulator image (RoboTwin + CuRobo at the README's pinned commits),
fills the asset Volume once, and measures whether a Modal GPU can render
RoboTwin's ray-traced RGB-D cameras and plan with CuRobo.

    modal run modal_apps/simulator.py::download_assets   # once; CPU only
    modal run modal_apps/simulator.py::spike             # default GPU: L4
    SIM_GPU=T4 modal run modal_apps/simulator.py::spike

Spike reports and one head-camera frame per GPU are saved to data/modal_spike/.
"""

import json
import os
from pathlib import Path
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

# T4, A100, A10G, L4. Compiling CuRobo for all of them lets one image serve any GPU.
CUDA_ARCHES = "7.5;8.0;8.6;8.9"
GPU = os.environ.get("SIM_GPU", "L4")

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


@app.function(image=image, gpu=GPU, volumes={ASSETS: assets_volume}, timeout=30 * 60)
def measure(seed: int = 3, observations: int = 20, physics_steps: int = 200):
    """Report driver, Vulkan, render, physics and CuRobo timings for one simulator."""
    import io
    import numpy as np
    from PIL import Image

    sys.path.insert(0, "/root")
    report = {"nvidia_smi":_command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
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
