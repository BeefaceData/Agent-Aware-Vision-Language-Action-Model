"""Local reconstruction provenance; this does not attest a remote worker/server."""

from importlib import metadata
from pathlib import Path

from development.checkpoints.bank import sha256


ROOT = Path(__file__).resolve().parents[2]


def local_provenance():
    paths = [ROOT / "run_inference_chunked.py", ROOT / "harness/environment/robotwin.py"]
    for directory, pattern in (("RoboTwin/envs", "*.py"), ("RoboTwin/env_cfg", "*.yml")):
        paths.extend(sorted((ROOT / directory).rglob(pattern)))
    paths.extend(sorted((ROOT / "RoboTwin/assets/embodiments").glob("*/config.yml")))
    versions = {}
    for package in ("numpy", "sapien", "torch", "mplib", "curobo"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "package_versions": versions,
        "scope": "local source/config subset and installed distributions; excludes assets, GPU/driver and remote processes",
    }
