"""CPU-only Modal wrapper for two saved-image hanging-grounding requests."""

import json
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import modal


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    "data/runs/entry660_20261029T202054Z_1793305254857158600/"
    "observations/observation_00000660.npz"
)
RESULT_FILES = ("source.png", "rack_peg_proposal.png", "mug_handle_proposal.png", "results.json")
DEFAULT_OBSERVATIONS = tuple(range(416, 513, 16))

app = modal.App("robotwin-hanging-grounding660")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26,<3", "pillow", "pydantic>=2.13,<3", "anthropic")
    .add_local_dir(REPO / "harness", "/root/harness", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "development", "/root/development", ignore=["**/__pycache__"])
)


def grounding_files(archive_bytes, model, *, client=None):
    """Reuse the bounded checker; injectable client keeps CPU tests offline."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("explicit_model_required")
    from development.hanging.check_grounding660 import check_grounding

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "observation_00000660.npz"
        archive.write_bytes(archive_bytes)
        output = root / "review"
        check_grounding(archive, output, model=model, client=client)
        return {name: (output / name).read_bytes()
                for name in RESULT_FILES if (output / name).is_file()}


@app.function(image=image, cpu=1, timeout=180, retries=0, max_containers=1,
              secrets=[modal.Secret.from_name("anthropic")])
def check_remote(archive_bytes: bytes, model: str):
    # The checker constructs Anthropic with max_retries=0 and a 60s timeout.
    # Its only requests are the rack peg and mug handle, each at most once.
    return grounding_files(archive_bytes, model)


def launch(archive, output, model, remote):
    """Upload only saved RGB-D; local process never constructs an API client."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("explicit_model_required")
    data = Path(archive).read_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = remote(data, model)
    if not set(files) <= set(RESULT_FILES) or "results.json" not in files:
        raise ValueError("unexpected_grounding_results")
    for name, content in files.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    return json.loads((output / "results.json").read_text())


@app.local_entrypoint()
def main(model: str, archive: str = DEFAULT_ARCHIVE):
    output = REPO / "data/runs" / f"grounding660_modal_{uuid4().hex}"
    report = launch(archive, output, model, check_remote.remote)
    print(json.dumps({target: row["status"] for target, row in report["targets"].items()}, indent=2))
    print(f"Saved results and review overlays: {output}")


def parse_observations(value=""):
    """Comma-separated observation IDs; omission preserves the seven-frame check."""
    if value == "":
        return DEFAULT_OBSERVATIONS
    if not isinstance(value, str):
        raise ValueError("invalid_observation_selection")
    parts = value.split(",")
    if not 1 <= len(parts) <= 7 or any(not p.strip().isascii() or not p.strip().isdigit() for p in parts):
        raise ValueError("invalid_observation_selection")
    selected = tuple(int(p.strip()) for p in parts)
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate_observation_selection")
    return selected


def saved_geometry_gates(observation, report, *, agree=None, rack_reference=None):
    """Measure unreviewed proposals without entry, fixture or motion overrides.

    ``agree(aperture)`` returns the runtime's two-frame agreement for an aperture
    that passed; ``rack_reference`` is the provider's kept rack reference.
    """
    import numpy as np
    from development.hanging.aperture_frame import estimate, inset_material_samples
    from development.hanging.public_provider import (MINIMUM_PROPOSAL_CONFIDENCE, SELF_CONFIRMED_CONFIDENCE,
                                                     measured_bounds)
    from harness.perception.pose_grounding import grounded_frame_from_pixels, measured_peg_centerline, _pose_matrix

    camera = observation.camera_frames["perception_camera"]
    aperture = estimate(camera, report.contour_px, report.material_px)
    inset = dict(attempted=False)
    if aperture.get("reason") == "material_depth_discontinuity":
        inset = dict(attempted=True, initial_reason=aperture["reason"])
        try:
            material = inset_material_samples(camera, report.contour_px, report.material_px)
            report = report.model_copy(update={"material_px": material})
            inset["material_px"] = material
            aperture = estimate(camera, report.contour_px, material)
        except ValueError as exc:
            inset["reason"] = str(exc)
    aperture["material_sample_count"] = len(report.material_px or ())
    if aperture.get("frame_pose_world_wxyz") is not None:
        frame = _pose_matrix(aperture["frame_pose_world_wxyz"])
        aperture.update(centroid_world=frame[:3, 3].tolist(), normal_world=frame[:3, 0].tolist())
    peg = dict(status="unavailable", reason="visible_peg_axis_required",
               semantics="unreviewed_model_proposal", radius_status="unavailable")
    if report.peg_keypoints_px is not None:
        try:
            frame = grounded_frame_from_pixels(camera, report.peg_keypoints_px)
            seat, tip = (np.asarray(camera.unproject(*uv, radius=1)) for uv in report.peg_keypoints_px[:2])
            peg.update(status="validated", reason="public_axis_depth_geometry_only",
                axis_pose_world_wxyz=np.asarray(frame, dtype=float).tolist(),
                axis_length_m=float(np.linalg.norm(tip-seat)))
            if report.peg_width_px is not None:
                registration = measured_peg_centerline(camera, report.peg_keypoints_px, report.peg_width_px)
                registration["pose_world_wxyz"] = np.asarray(registration["pose_world_wxyz"], dtype=float).tolist()
                peg.update(radius_status="validated", registration=registration)
        except ValueError as exc:
            peg.update(status="rejected", reason=str(exc))
    clearance = dict(status="unavailable", clearance_validated=None,
                     reason="validated_aperture_required", fixture="not_supplied",
                     rack_reference="current_frame_only")
    aperture_ok = aperture["status"] == "candidate_not_motion_validated"
    # Same rule as the runtime: one frame at high model confidence, otherwise
    # agreement with an earlier frame's opening after the measured TCP motion.
    agreement = (agree(aperture) if aperture_ok and agree is not None
                 else dict(status="not_evaluated"))
    confident = (report.confidence >= SELF_CONFIRMED_CONFIDENCE
                 or (report.confidence >= MINIMUM_PROPOSAL_CONFIDENCE and agreement["status"] == "agreed"))
    semantics_ok = (confident and report.complete_visible_opening
                    and report.body_polygon_px is not None)
    if aperture_ok and semantics_ok:
        try:
            bounds = measured_bounds(observation, report, aperture, rack_reference=rack_reference)
            clearance.update(status="validated", clearance_validated=True,
                reason="visible_optical_clearance_only", bounds=bounds)
        except ValueError as exc:
            reason = str(exc)
            unavailable = reason in ("visible_thickness_and_peg_cross_section_required",
                                     "opposite_handle_faces_unresolved")
            clearance.update(status="unavailable" if unavailable else "rejected",
                             clearance_validated=None if unavailable else False, reason=reason)
    elif aperture_ok:
        clearance["reason"] = "complete_visible_aperture_unavailable"
    blockers = ["saved_frame_entry_not_verified", "model_geometry_requires_semantic_review"]
    if not aperture_ok:
        blockers.append(aperture.get("reason", aperture["status"]))
    if not semantics_ok:
        blockers.append("complete_visible_aperture_unavailable")
    if peg["status"] != "validated":
        blockers.append(peg["reason"])
    if clearance["status"] != "validated":
        blockers.append(clearance["reason"])
    return dict(aperture=aperture, material_inset=inset, peg=peg, clearance=clearance,
        two_frame_agreement=agreement,
        rack_reference_sequence=None if rack_reference is None else rack_reference[0].sequence,
        recovery_eligibility=dict(status="unavailable", blockers=list(dict.fromkeys(blockers))),
        motion_authorized=False, proposal_review="unreviewed_model_proposal")


def saved_crop_files(archives, model, *, client=None):
    """One to seven saved frames, at most one full-view and one crop request each."""
    from PIL import Image, ImageDraw
    from development.hanging.public_provider import BoundedVisionClient, PublicRGBDHangingProvider
    from development.hanging.grounding_preflight import sanitized_failure
    from harness.perception.replay import load_public_observation
    from harness.perception.observations import ObservationStore
    if not isinstance(model, str) or not model.startswith("claude-"):
        raise ValueError("explicit_claude_model_required")
    if (not isinstance(archives, dict) or not 1 <= len(archives) <= 7
            or any(type(seq) is not int or seq < 0 for seq in archives)):
        raise ValueError("one_to_seven_saved_frames_required")
    sequences = tuple(sorted(archives))
    maximum_requests = 2*len(sequences)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        observations = []
        for sequence in sequences:
            path = root/f"observation_{sequence:08d}.npz"
            path.write_bytes(archives[sequence])
            observation = load_public_observation(path)
            if "perception_camera" not in (observation.camera_frames or {}):
                raise ValueError("perception_rgbd_required")
            if observation.sequence != sequence:
                raise ValueError("archive_observation_sequence_mismatch")
            observations.append(observation)
        owned = client is None
        if owned:
            import anthropic
            client = anthropic.Anthropic(max_retries=0, timeout=45.)
        shared = BoundedVisionClient(client, model=model, deadline=time.monotonic()+600,
                                     maximum_requests=maximum_requests)
        events, rows = [], []
        provider = PublicRGBDHangingProvider(shared, model=model, episode_id="saved_frames_only",
            store=ObservationStore(root/"observations"), log=events.append, maximum_requests=maximum_requests)
        files = {}
        try:
            for observation in observations:
                sequence = observation.sequence
                start_requests = shared.requests
                row = dict(sequence=sequence, crop_needed=None, crop_attempted=False,
                           motion_authorized=False, proposal_review="unreviewed_model_proposal",
                           aperture=dict(status="unavailable", reason="proposal_unavailable",
                                         motion_authorized=False),
                           peg=dict(status="unavailable", reason="proposal_unavailable"),
                           recovery_eligibility=dict(status="unavailable", blockers=["proposal_unavailable"]),
                           clearance=dict(status="unavailable", clearance_validated=None,
                                          reason="proposal_unavailable"))
                try:
                    report = provider.observe(observation)
                    row["full_frame_proposal"] = report.model_dump(mode="json")
                    # This experiment requests geometry without assuming entry.
                    row["crop_needed"] = provider.needs_geometry_crop(report)
                    before_crop = shared.requests
                    result = provider.crop_geometry(observation, report) if row["crop_needed"] else report
                    row["crop_attempted"] = shared.requests > before_crop
                    row["mapped_proposal"] = result.model_dump(mode="json")
                    tcp = (observation.measured_tcp_poses or {}).get("right")
                    agree = None if tcp is None else (
                        lambda aperture, o=observation, t=tcp: provider._aperture_agrees(o, t, aperture))
                    row.update(saved_geometry_gates(observation, result, agree=agree,
                        rack_reference=provider._current_rack_reference(observation)))
                    row["peg_keypoints_available"] = result.peg_keypoints_px is not None
                    picture = Image.fromarray(observation.images["perception_camera"])
                    draw = ImageDraw.Draw(picture)
                    draw.text((5, 5), "UNREVIEWED MODEL PROPOSAL - NO MOTION", fill="red")
                    for name, color in (("contour_px", "red"), ("material_px", "lime"),
                                        ("peg_keypoints_px", "cyan")):
                        for u, v in getattr(result, name) or ():
                            draw.ellipse((u-2, v-2, u+2, v+2), outline=color)
                    for u, v in row["material_inset"].get("material_px", ()):
                        draw.ellipse((u-1, v-1, u+1, v+1), outline="blue")
                    import io
                    buffer = io.BytesIO()
                    picture.save(buffer, format="PNG")
                    files[f"review_{sequence:08d}.png"] = buffer.getvalue()
                except Exception as exc:
                    row["failure"] = sanitized_failure(exc)
                row["requests_attempted"] = shared.requests-start_requests
                events.append(dict(event_type="saved_geometry_validation", **row))
                rows.append(row)
        finally:
            if owned:
                client.close()
        report = dict(mode="saved_frame_crop_check", simulator_started=False, motion=False,
            maximum_requests=maximum_requests, requests_attempted=shared.requests, deadline_seconds=600,
            selected_observations=list(sequences), retries=0,
            human_review_required=True, maximum_requests_per_observation=2,
            crop_is_enlargement_not_new_sensor_detail=True, frames=rows)
        files["results.json"] = json.dumps(report, indent=2, allow_nan=False).encode()
        files["perception_events.jsonl"] = ("\n".join(json.dumps(e, allow_nan=False) for e in events)+"\n").encode()
        return files


@app.function(image=image, cpu=1, timeout=12*60, retries=0, max_containers=1,
              secrets=[modal.Secret.from_name("anthropic")])
def check_saved_crops_remote(archives: dict, model: str):
    return saved_crop_files(archives, model)


def launch_saved_crops(model, frames, output, *, observations="", remote):
    if not model.startswith("claude-"):
        raise ValueError("explicit_claude_model_required")
    source, target = Path(frames), Path(output)
    sequences = parse_observations(observations)
    archives = {seq: (source/f"observation_{seq:08d}.npz").read_bytes()
                for seq in sequences}
    target.mkdir(parents=True, exist_ok=False)
    files = remote(archives, model)
    allowed = {"results.json", "perception_events.jsonl"} | {
        f"review_{seq:08d}.png" for seq in sequences}
    if not set(files) <= allowed or "results.json" not in files:
        raise ValueError("unexpected_saved_crop_results")
    for name, data in files.items():
        with (target/name).open("xb") as stream:
            stream.write(data)
    print(f"Saved crop proposals, mapped overlays and validation results: {target.resolve()}")


@app.local_entrypoint()
def saved_crops(model: str, frames: str, output: str, observations: str = ""):
    launch_saved_crops(model, frames, output, observations=observations, remote=check_saved_crops_remote.remote)
