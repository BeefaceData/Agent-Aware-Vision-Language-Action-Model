"""Observation-driven hanging recovery; providers supply public evidence only.

Existing three-keypoint grounders do not establish a complete aperture contour.
Until a provider can validate that semantic input, the default is unavailable,
not a fallback to a recorded annotation. No simulator initialization occurs here.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol
import time

import numpy as np

from development.hanging.aperture_frame import Limits, estimate, _inside_opening, _material_points
from development.hanging.motion import aperture_targets, confirm_visible_handle, execute_stages
from harness.perception.anchors import Anchor, AnchorRegistry
from harness.perception.geometry import pose7
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.pose_grounding import _pose_matrix
from harness.perception.rgbd import calibration_sha256
from harness.predicates.rack_seat_gate import RackSeatReleaseGate


Status = Literal["confirmed", "rejected", "unknown"]


@dataclass(frozen=True)
class EntryAssessment:
    status: Status
    reason: str


@dataclass(frozen=True)
class PublicHangingProposal:
    episode_id: str
    sequence: int
    camera: str
    rgb_sha256: str
    calibration_sha256: str
    task_stage: str | None
    right_mug_held: bool | None
    grasp_reason: str
    source: str
    semantics_confirmed: bool | None
    complete_visible_contour: bool | None
    contour_px: object
    material_px: object
    body_polygon_px: object
    rack_report: dict
    mating_prior: dict
    rack_reference_observation: object = None


@dataclass(frozen=True)
class ProposalResult:
    status: Status
    reason: str
    proposal: PublicHangingProposal | None = None


class HangingEvidenceProvider(Protocol):
    def propose(self, observation, measured_tcp, *, episode_id: str) -> ProposalResult: ...


class UnavailableHangingProvider:
    def propose(self, observation, measured_tcp, *, episode_id):
        return ProposalResult("unknown", "validated_visible_aperture_provider_unavailable")


@dataclass(frozen=True)
class CurrentGeometry:
    proposal: PublicHangingProposal
    aperture: dict
    rack_pose: tuple[float, ...]


@dataclass(frozen=True)
class GroundingResult:
    status: Status
    reason: str
    geometry: CurrentGeometry | None = None


def assess_entry(proposal):
    """Stage and public grasp evidence are required; gripper closure is not evidence."""
    if proposal.source != "public_rgbd":
        return EntryAssessment("rejected", "public_grasp_evidence_required")
    if proposal.task_stage is None:
        return EntryAssessment("unknown", "hanging_stage_unknown")
    if proposal.task_stage != "hanging":
        return EntryAssessment("rejected", "not_hanging_stage")
    if proposal.right_mug_held is False:
        return EntryAssessment("rejected", "right_arm_not_holding_mug")
    if proposal.right_mug_held is not True or not proposal.grasp_reason:
        return EntryAssessment("unknown", "right_arm_grasp_unknown")
    return EntryAssessment("confirmed", "public_hanging_entry_confirmed")


def validate_current_geometry(observation, measured_tcp, result, *, episode_id):
    """Validate fresh proposal semantics and rendered depth, without oracle inputs."""
    if not isinstance(result, ProposalResult) or result.status not in ("confirmed", "rejected", "unknown"):
        return GroundingResult("rejected", "invalid_provider_result")
    if result.status != "confirmed":
        return GroundingResult(result.status, result.reason)
    proposal = result.proposal
    if not isinstance(proposal, PublicHangingProposal):
        return GroundingResult("rejected", "public_hanging_proposal_required")
    try:
        pose7(measured_tcp)
        if not episode_id or proposal.episode_id != episode_id:
            raise ValueError("hanging_evidence_wrong_episode")
        if proposal.sequence != observation.sequence:
            raise ValueError("hanging_evidence_stale")
        camera = (observation.camera_frames or {}).get(proposal.camera)
        image = observation.images.get(proposal.camera)
        if camera is None or image is None:
            return GroundingResult("unknown", "hanging_rgbd_unavailable")
        if (proposal.calibration_sha256 != calibration_sha256(camera)
                or proposal.rgb_sha256 != sha256(image.tobytes()).hexdigest()):
            raise ValueError("hanging_evidence_camera_or_image_mismatch")
        entry = assess_entry(proposal)
        if entry.status != "confirmed":
            return GroundingResult(entry.status, entry.reason)
        if proposal.semantics_confirmed is not True or proposal.complete_visible_contour is not True:
            return GroundingResult("unknown", "validated_complete_visible_contour_required")
        # Overrides and uncalibrated demo priors cannot silently enter runtime.
        if not isinstance(proposal.mating_prior, dict) or not isinstance(proposal.rack_report, dict):
            raise ValueError("invalid_geometry_bounds_or_rack_report")
        if any(key.startswith("allow_") and value for key, value in proposal.mating_prior.items()):
            raise ValueError("experimental_override_not_runtime_evidence")
        bounds = proposal.mating_prior
        if (bounds.get("dimensions_validated") is not True
                or not bounds.get("dimension_basis")):
            return GroundingResult("unknown", "validated_dimensional_bounds_required")
        peg_radius = float(bounds.get("peg_radius_m", np.nan))
        if not np.isfinite(peg_radius) or peg_radius <= 0:
            raise ValueError("validated_peg_radius_required")
        aperture = estimate(camera, proposal.contour_px, proposal.material_px)
        if aperture["status"] != "candidate_not_motion_validated":
            raise ValueError(aperture["reason"])
        aperture.update(sequence=observation.sequence, camera=proposal.camera,
                        rgb_sha256=proposal.rgb_sha256, calibration_sha256=proposal.calibration_sha256)
        aperture["material_points_world"] = _material_points(
            camera, proposal.material_px, Limits(), np.asarray(proposal.contour_px)).tolist()
        polygon = np.asarray(proposal.body_polygon_px, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3 or not np.isfinite(polygon).all():
            return GroundingResult("unknown", "visible_mug_body_evidence_required")
        height, width = camera.depth_m.shape
        if np.any(polygon < 0) or np.any(polygon[:, 0] >= width) or np.any(polygon[:, 1] >= height):
            raise ValueError("body_polygon_out_of_bounds")
        grid = np.array([(u, v) for v in range(height) for u in range(width)])
        body = []
        for u, v in grid[_inside_opening(grid, polygon)]:
            patch = camera.depth_m[max(0, v-1):v+2, max(0, u-1):u+2]
            if patch.shape == (3, 3) and np.isfinite(patch).all() and np.all(patch > 0) and np.ptp(patch) <= .06:
                body.append(camera.unproject(u, v, radius=1))
        if len(body) < 9:
            return GroundingResult("unknown", "visible_mug_body_depth_unavailable")
        aperture["visible_body_points_world"] = body
        radius = aperture["geometry_diagnostics"]["centroid_inscribed_radius_m"]
        if not np.isfinite(bounds.get("radial_clearance_m", np.nan)) or bounds["radial_clearance_m"] > radius-peg_radius:
            raise ValueError("radial_clearance_exceeds_observed_aperture")
        if proposal.rack_report.get("camera") != proposal.camera:
            raise ValueError("rack_camera_mismatch")
        # A provider-owned episode-static public reference can survive occlusion;
        # it is never treated as freshly observed geometry in the current image.
        reference = proposal.rack_reference_observation or observation
        if (reference.sequence > observation.sequence or calibration_sha256(
                reference.camera_frames[proposal.camera]) != calibration_sha256(camera)):
            raise ValueError("rack_reference_incompatible")
        minimum_confidence = .55
        if bounds.get("dimension_basis") == "verified fixture bounds plus current public RGB-D":
            # Recheck geometric admission against the original depth, not a
            # fabricated high model-confidence score on a cached proposal.
            from development.hanging.public_provider import fixture_axis
            fixture_axis(reference, proposal.rack_report.get("keypoints_px"))
            minimum_confidence = 0.
        cache = StaticRackFrameCache(minimum_confidence=minimum_confidence)
        cache.cache(reference, proposal.rack_report)
        return GroundingResult("confirmed", "current_public_geometry_validated",
                               CurrentGeometry(proposal, aperture, tuple(cache.resolve(observation))))
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return GroundingResult("rejected", str(exc))


def recover_current(environment, store, *, episode_id, deadline, provider=None, log=lambda event: None):
    """Run only after current public evidence confirms entry; no fixed-step trigger."""
    provider = provider if provider is not None else UnavailableHangingProvider()
    anchors = AnchorRegistry()
    gate = RackSeatReleaseGate(environment, position_tolerance_m=.005,
                              orientation_tolerance_rad=.05, unavailable_is_false=True)
    initial_material = initial_tcp = None
    assessment = dict(status="unknown", reason="entry_not_assessed")

    def prepare(command, observation):
        nonlocal initial_material, initial_tcp, assessment
        if command.action.value != "move_to":
            return None
        if time.monotonic() >= deadline:
            raise ValueError("recovery_time_limit")
        measured_tcp = environment.arm_state("right")["pose"]
        try:
            proposed = provider.propose(observation, np.asarray(measured_tcp).copy(), episode_id=episode_id)
        except Exception as exc:
            raise ValueError("hanging_evidence_provider_error") from exc
        grounded = validate_current_geometry(observation, measured_tcp, proposed, episode_id=episode_id)
        assessment = dict(status=grounded.status, reason=grounded.reason,
                          observation_sequence=observation.sequence)
        log(dict(event_type="hanging_entry_assessment", observation_sequence=observation.sequence,
                 status=grounded.status, reason=grounded.reason))
        if grounded.status != "confirmed":
            raise ValueError(grounded.reason)
        geometry = grounded.geometry
        current_tcp = _pose_matrix(measured_tcp)
        if initial_material is None:
            initial_material = np.asarray(geometry.aperture["material_points_world"])
            initial_tcp = current_tcp
        else:
            delta = current_tcp @ np.linalg.inv(initial_tcp)
            expected = initial_material @ delta[:3, :3].T + delta[:3, 3]
            camera = observation.camera_frames[geometry.proposal.camera]
            xyz = expected @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
            if np.any(xyz[:, 2] <= 0):
                raise ValueError("tracked_aperture_behind_camera")
            uv = xyz @ camera.intrinsic.T
            confirmation = confirm_visible_handle(camera, uv[:, :2]/uv[:, 2, None], expected)
            log(dict(event_type="fresh_handle_confirmation", observation_sequence=observation.sequence,
                     measured_tcp_pose=np.asarray(measured_tcp).tolist(), confirmation=confirmation))
            if confirmation["status"] != "confirmed_visible_subset":
                raise ValueError(confirmation["reason"])
        targets, evidence = aperture_targets(observation, geometry.aperture, geometry.rack_pose,
                                             measured_tcp, geometry.proposal.mating_prior,
                                             clearance_model="rss_sensitivity")
        for name, pose in targets.items():
            anchors.bind(name, "right", Anchor(tuple(pose), observation.sequence, source="perception"))
        log(dict(event_type="aperture_target_preparation", observation_sequence=observation.sequence,
                 evidence=evidence))
        return anchors.resolve(command.parameters.target, "right", observation)

    report = execute_stages(environment, prepare, anchors, gate, store, deadline=deadline, log=log)
    return {**report, "entry_assessment": assessment,
            "retention": dict(status="unknown", reason="public_support_evaluator_unavailable",
                              observed_ticks=report.get("post_retreat_observation_only_ticks", 0))}
