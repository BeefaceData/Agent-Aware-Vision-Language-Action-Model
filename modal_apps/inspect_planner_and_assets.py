"""Static source/configuration and scaled asset audit; never initializes a robot."""

import hashlib
import json
import os
from pathlib import Path
import re
import sysconfig

import modal
import numpy as np

from modal_apps.simulator import ASSETS, ROBOTWIN, image


app = modal.App("robotwin-static-collision-audit")
assets = modal.Volume.from_name("robotwin-assets", create_if_missing=False).read_only()
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_MESH_BYTES = 32 * 1024 * 1024
MAX_VERTICES = 2_000_000
SOURCE_PATTERN = re.compile(
    r"plan_path|MotionGen|collision|collide|delta_twist|within_joint_limit|self_collision|attach|detach|update_world|"
    r"world_config|WorldConfig|obstacle|cuboid|mesh|table|rack|embodiment|robot_file|CONFIGS_PATH", re.I)


def source_evidence(path):
    if path.stat().st_size > MAX_SOURCE_BYTES:
        return dict(path=str(path), status="unavailable", reason="source_size_limit")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    selected = set(range(len(lines))) if path.suffix in (".yml", ".yaml") else set()
    for index, line in enumerate(lines):
        if SOURCE_PATTERN.search(line):
            selected.update(range(max(0, index-3), min(len(lines), index+4)))
    indexes = sorted(selected)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status="source_evidence_only", total_lines=len(lines), truncated=len(indexes)>240,
        lines=[dict(line=i+1, text=lines[i][:500]) for i in indexes[:240]])


def inspect_sources(robotwin_root, assets_root, mplib_path):
    root, assets_root = Path(robotwin_root), Path(assets_root)
    selected = []
    enumeration_truncated = False
    scanned = 0
    for base in (root/"envs", root/"task_config", root/"configs", assets_root/"embodiments"):
        if not base.is_dir():
            continue
        for directory, directories, files in os.walk(base, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in (".git", "__pycache__", "curobo"))
            for name in sorted(files):
                scanned += 1
                if scanned > 2000 or len(selected) >= 100:
                    enumeration_truncated = True
                    break
                path = Path(directory)/name
                if path.suffix not in (".py", ".yml", ".yaml"):
                    continue
                relative = path.relative_to(base)
                if (path.suffix == ".py" and relative != Path("__init__.py")
                        and not any(word in str(relative).lower()
                                    for word in ("robot", "plan", "motion", "base_task", "hanging_mug"))):
                    continue
                selected.append(path)
            if enumeration_truncated:
                break
        if enumeration_truncated:
            break
    evidence, missing = [], []
    remaining_lines = 2000
    for path in selected:
        try:
            item = source_evidence(path)
            if "lines" in item:
                if len(item["lines"]) > remaining_lines:
                    item["truncated"] = True
                item["lines"] = item["lines"][:remaining_lines]
                remaining_lines -= len(item["lines"])
            evidence.append(item)
        except OSError as exc:
            missing.append(dict(path=str(path), reason=str(exc)))
    mplib = dict(status="unavailable", path=str(mplib_path), reason="mplib_source_missing")
    path = Path(mplib_path)
    if path.is_file():
        mplib = source_evidence(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        mplib.update(patched_condition_present="if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:" in text,
            original_condition_present="if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:" in text,
            interpretation="Condition presence only; inspect enclosing function and actual callers. Not proof of global collision coverage.")
    return dict(files=evidence, errors=missing, enumeration_truncated=enumeration_truncated,
        missing_roots=[str(p) for p in (root/"envs", assets_root/"embodiments") if not p.is_dir()],
        mplib_patch=mplib,
        coverage={name: "unknown; static source requires call/configuration review and does not prove live synchronization"
                  for name in ("self_collision", "opposite_arm", "table", "rack", "attached_mug")})


def scaled_envelope(scene, scale):
    """Apply every GLB node transform, then the asset's outer scale."""
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid_asset_scale")
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    count, nodes = 0, 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        vertices = np.asarray(scene.geometry[name].vertices, dtype=float)
        transform = np.asarray(transform, dtype=float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], (0., 0., 0., 1.))):
            raise ValueError("invalid_mesh_vertices_or_node_transform")
        count += len(vertices)
        if count > MAX_VERTICES:
            raise ValueError("mesh_vertex_limit")
        points = (vertices@transform[:3, :3].T+transform[:3, 3])*scale
        lower, upper = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0))
        nodes += 1
    if not nodes:
        raise ValueError("mesh_scene_empty")
    center = (lower+upper)/2
    return dict(bounds_min_max_m=[lower.tolist(), upper.tolist()], extents_m=(upper-lower).tolist(),
        sphere_center_m=center.tolist(), sphere_radius_m=float(np.linalg.norm(upper-lower)/2),
        sphere_basis="conservative AABB circumsphere, not minimum sphere",
        transformed_nodes=nodes, vertices=count,
        frame="asset-local after GLB node transforms and metadata scale; not TCP/world registered")


def inspect_mugs(assets_root, *, mesh_loader=None):
    from development.hanging.public_provider import MUG_CLASS_VISUAL_HASHES, MUG_CLASS_COLLISION_HASHES, MUG_CLASS_SCALES
    if mesh_loader is None:
        import trimesh
        mesh_loader = lambda path: trimesh.load(str(path), force="scene", process=False)
    root = Path(assets_root)/"objects"/"039_mug"
    variants = []
    for index in range(10):
        row = dict(variant=index, status="incomplete", meshes={}, errors=[])
        metadata = root/f"model_data{index}.json"
        try:
            if metadata.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("metadata_size_limit")
            scale = json.loads(metadata.read_text())["scale"]
            numeric = np.asarray(scale, dtype=float)
            if numeric.shape != (3,) or not np.isfinite(numeric).all() or np.any(numeric <= 0):
                raise ValueError("invalid_asset_scale")
            row.update(scale=numeric.tolist(), metadata_path=str(metadata),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                scale_matches_verified_class=bool(np.allclose(numeric, MUG_CLASS_SCALES[index], rtol=0, atol=1e-12)))
        except (OSError, KeyError, ValueError) as exc:
            row["errors"].append(dict(path=str(metadata), reason=str(exc)))
            variants.append(row)
            continue
        for kind, expected in (("visual", MUG_CLASS_VISUAL_HASHES), ("collision", MUG_CLASS_COLLISION_HASHES)):
            path = root/kind/f"base{index}.glb"
            try:
                if path.stat().st_size > MAX_MESH_BYTES:
                    raise ValueError("mesh_file_size_limit")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                envelope = scaled_envelope(mesh_loader(path), numeric)
                row["meshes"][kind] = dict(path=str(path), sha256=digest,
                    hash_matches_verified_class=digest==expected[index], **envelope)
            except Exception as exc:
                row["errors"].append(dict(path=str(path), error_type=type(exc).__name__, reason=str(exc)))
        if len(row["meshes"]) == 2:
            row["status"] = "geometry_complete"
        variants.append(row)
    complete = len(variants)==10 and all(row["status"]=="geometry_complete" for row in variants)
    verified = complete and all(row["scale_matches_verified_class"] and all(
        mesh["hash_matches_verified_class"] for mesh in row["meshes"].values()) for row in variants)
    union = None
    if complete:
        bounds = np.array([mesh["bounds_min_max_m"] for row in variants for mesh in row["meshes"].values()])
        lower, upper = bounds[:,0].min(0), bounds[:,1].max(0)
        union = dict(bounds_min_max_m=[lower.tolist(), upper.tolist()],
            sphere_center_m=((lower+upper)/2).tolist(), sphere_radius_m=float(np.linalg.norm(upper-lower)/2),
            frame="union in nominal asset-local coordinates; requires public TCP registration before use")
    return dict(geometry_complete=complete, verified_asset_class=verified, variants=variants,
                whole_class_union=union, tcp_registration="unknown", slip_bound="unknown")


def inspect_local(robotwin_root, assets_root, mplib_path, *, mesh_loader=None):
    sources = inspect_sources(robotwin_root, assets_root, mplib_path)
    mugs = inspect_mugs(assets_root, mesh_loader=mesh_loader)
    return dict(status="static_audit_only", simulator_started=False, motion=False,
        native_planner=sources, mug_mesh_envelopes=mugs,
        active_viewing_authorized=False,
        limits=dict(source_files=100, source_bytes=MAX_SOURCE_BYTES, mesh_bytes=MAX_MESH_BYTES,
                    vertices_per_mesh=MAX_VERTICES, source_excerpt_lines=2000, enumerated_files=2000),
        limitations=["Keyword-selected excerpts are evidence, not a collision capability verdict.",
                     "Static files cannot establish live obstacle synchronization or grasp attachment.",
                     "Mesh bounds are not an observed mug-to-TCP transform or slip guarantee."])


@app.function(image=image, cpu=2, memory=4096, timeout=300, retries=0,
              max_containers=1, volumes={ASSETS: assets}, block_network=True)
def inspect_remote():
    return inspect_local(ROBOTWIN, ASSETS, Path(sysconfig.get_paths()["purelib"])/"mplib"/"planner.py")


@app.local_entrypoint()
def main():
    print(json.dumps(inspect_remote.remote(), indent=2, allow_nan=False))
