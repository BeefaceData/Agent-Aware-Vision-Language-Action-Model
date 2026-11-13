"""Bounded image proposals and rendered-depth measurements for hanging.

Model confidence is not calibrated semantic accuracy. Hidden surfaces require
an explicitly injected, independently verified fixture contract; the default
path requires visible faces and cross-section. Live reliability is unvalidated.
"""

from hashlib import sha256
from dataclasses import dataclass, replace
from pathlib import Path
import json
import time
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import Field, StrictBool, StrictFloat, StrictStr

from development.hanging.aperture_frame import estimate, inset_material_samples
from development.hanging.grounding_preflight import sanitized_failure
from development.hanging.runtime import ProposalResult, PublicHangingProposal
from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
from harness.perception.pose_grounding import _pose_matrix, measured_peg_centerline
from harness.perception.object_pose import (
    MUG, SAME_BODY_AXIS_DEG, SURFACE_TOLERANCE_M, masked_surface, object_pose,
)
from harness.perception.rgbd import calibration_sha256
from harness.perception.vision_request import image_block, png_bytes, request_tool_input
from harness.predicates.base import PredicateResult
from harness.schemas import Model


class SurfacePixel(Model):
    camera: StrictStr
    uv: tuple[StrictFloat, StrictFloat]


class VisibleHangingReport(Model):
    stage: Literal["grasp", "handoff", "hanging", "unknown"]
    left_held: StrictBool | None
    right_held: StrictBool | None
    mug_off_table: StrictBool | None
    transfer_complete: StrictBool | None
    confidence: StrictFloat = Field(ge=0, le=1)
    task_state_confidence: StrictFloat | None = Field(default=None, ge=0, le=1)
    evidence: StrictStr = Field(max_length=700)
    complete_visible_opening: StrictBool
    contour_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=32)
    material_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=32)
    body_polygon_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=24)
    peg_keypoints_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=3)
    peg_width_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=2)
    handle_front: SurfacePixel | None = None
    handle_back: SurfacePixel | None = None
    opposite_handle_faces_visible: StrictBool
    right_gripper_closed_visible: StrictBool | None = None
    table_material_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=32)
    rack_region_px: list[tuple[StrictFloat, StrictFloat]] | None = Field(default=None, max_length=16)
    peg_confidence: StrictFloat | None = Field(default=None, ge=0, le=1)


# Lift carried forward from an earlier corroborated frame, not this frame's evidence.
LIFT_PERSISTENCE_REASON = "bounded_temporal_lift_persistence"

# Model confidence is not calibrated. A geometry proposal scored at least
# SELF_CONFIRMED_CONFIDENCE stands on one frame, as before. A proposal scored
# between MINIMUM_PROPOSAL_CONFIDENCE and that must instead be confirmed by an
# independent second frame whose depth geometry agrees: the handle opening
# after allowing for the measured TCP motion (rigid grasp), the static peg
# directly. The agreement limits equal the estimator's own default 5 mm centre
# and 5 degree normal sensitivity limits.
SELF_CONFIRMED_CONFIDENCE = .85
MINIMUM_PROPOSAL_CONFIDENCE = .5
AGREEMENT_CENTER_M = .005
AGREEMENT_NORMAL_DEG = 5.

PUBLIC_PROMPT ="""Report only visible public image evidence; never robot actions,
world coordinates, object dimensions, hidden geometry or simulator state.
Images are labeled by camera, observation and pixel size. Full-resolution UV
origin is top-left. If uncertain use null/false and explain; gripper closure alone
does not prove holding. Distinguish left acquisition, table handoff and right-arm
hanging. Held means the mug is visibly between/attached to the named fingers;
off-table and completed transfer require visible evidence, not task expectations.
Arm names refer to the ROBOT's anatomical left/right, never image-left/right.
left_camera is mounted on the left wrist; right_camera on the right wrist.
perception_reverse_camera, when present, is a complementary external RGB-D
view, not an arm camera. Use it for an opposing handle face only if that face
is actually visible; another camera never proves visibility by itself.
perception_camera is an external view: either arm can appear on either image
side as it moves. Use labeled wrist views and projected measured TCP locations
when supplied to associate fingers; a wrist-camera close-up or a TCP projection
alone does not prove holding. If arm identity is ambiguous keep it null.
task_state_confidence rates only current stage, arm holding and off-table state,
independent of geometry visibility. Do not lower it because the handle is hidden.
transfer_complete describes historical transfer evidence and may remain null
even when the right arm visibly holds a lifted mug. Such current state supports
stage=hanging without inventing a transfer history.
For independent lift corroboration, report right_gripper_closed_visible only
when the named fingers visibly form a closed grasp (not from the command or
wrist-camera proximity). Otherwise null. Supply 6-32 widely spread visible
TABLE surface pixels as table_material_px, away from shadows' object edges,
objects, rack and grippers. Leave null if the tabletop cannot be identified.
Geometry uses perception_camera only: trace the entire visible handle OPENING
as an ordered convex contour (8-32 vertices). Do not outline the mug body or
handle exterior. Give >=12 separate solid handle-material pixels outside the
opening, spanning the visible arch, with room for a 3x3 neighborhood. Give a
visible mug-body polygon excluding gripper/handle/background.
The handle opening is a THROUGH-HOLE: background may be visible through the
loop. It is not the cup mouth/interior, a gripper recess, or a cylinder-body
patch. Delineate the visible inner aperture separately from the exterior arch.
A handle opening is always closed off on one side by the mug body: the visible
outer wall of the cylinder between the two handle attachments is part of the
opening's own boundary, not an occluder. When the inner arch and that body wall
are both visible all the way round, the opening is complete; it is incomplete
only when a gripper, the rack, the mug's near side or the image edge hides part
of that boundary.
Solid arch material samples must exclude black gripper fingers, the primary
mug cylinder and ambiguous attachment/junction pixels. Do not complete hidden
boundaries or substitute a gripper/body depth recess for the handle opening.
Give rack_region_px as a visible rack-region polygon independently of whether
the handle is visible; do not withhold visible peg landmarks just because the
handle opening is unavailable. Never invent the rack region from task layout.
peg_confidence rates only the visible peg landmarks and cross-section, independent
of handle visibility. Use null if the peg cannot be assessed.
Peg keypoints: the selected branch-to-trunk JUNCTION (never the vertical post
base, rack foot or table contact), its free endpoint along the branch, then a visible
rack-material point above the seat defining the vertical plane. Peg-width
endpoints must lie on opposite visible sides at ONE perpendicular MID-SHAFT
cross-section, away from the trunk junction, end cap, table and occluders.
Do not infer a hidden junction. A surface landmark is not automatically a
centerline point: withhold ambiguous or occluded axis landmarks.
Handle front/back must be VISIBLE opposite planar faces at the SAME material
cross-section, optionally from different labeled cameras; never infer a hidden
back face or substitute the inner/outer arch edges. If unavailable leave null.
confidence describes only reported geometry; do not claim complete visibility
when any opening boundary is occluded. Call report_public_hanging once."""


class BoundedVisionClient:
    """One shared request/deadline cap for planner, perception and preflight."""
    def __init__(self, client, *, model, deadline, maximum_requests=161):
        self.client, self.model, self.deadline = client, model, deadline
        self.maximum_requests, self.requests = maximum_requests, 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        if self.requests >= self.maximum_requests or time.monotonic() >= self.deadline:
            raise RuntimeError("vision_request_budget_exhausted")
        self.requests += 1  # Errors consume the request; never retry automatically.
        request.update(model=self.model, max_tokens=4096,
                       tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                       extra_body={"thinking": {"type": "adaptive"},
                                   "output_config": {"effort": "low"}})
        response = self.client.messages.create(**request)
        calls = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
        allowed = {tool["name"] for tool in request["tools"]}
        if len(calls) != 1 or calls[0].name not in allowed:
            raise ValueError("exactly_one_allowed_tool_required")
        return response


def read_surface(observation, pixel):
    if pixel.camera not in (*CAMERAS, *PERCEPTION_CAMERAS):
        raise ValueError("surface_camera_invalid")
    camera = (observation.camera_frames or {}).get(pixel.camera)
    if camera is None:
        raise ValueError("surface_depth_unavailable")
    u, v = pixel.uv
    x, y = np.rint((u, v)).astype(int)
    h, w = camera.depth_m.shape
    if not (1 <= x < w-1 and 1 <= y < h-1):
        raise ValueError("surface_neighborhood_out_of_bounds")
    patch = camera.depth_m[y-1:y+2, x-1:x+2]
    if not np.isfinite(patch).all() or np.any(patch <= 0) or np.ptp(patch) > .06:
        raise ValueError("surface_depth_discontinuous")
    point = np.asarray(camera.unproject(u, v, radius=1))
    # Local raster sensitivity only; not a calibration confidence interval.
    pixel_error = max(np.linalg.norm(np.asarray(camera.unproject(u+du, v+dv, radius=1))-point)
                      for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return point, float(pixel_error + np.ptp(patch))


@dataclass(frozen=True)
class FixtureBounds:
    """Trusted caller evidence, never populated from a model's dimension guesses.

    The measured 47.99 mm opening-side envelope alone does NOT establish full
    corridor coverage. A caller must verify that coverage for the selected asset.
    """
    episode_id: str
    sequence: int
    calibration_hash: str
    rack_mesh_hash: str
    mug_mesh_hash: str
    verification_source: str
    full_slab_coverage_verified: bool = False
    centerline_verified: bool = False
    rack_asset: str = "040_rack/base0"
    mug_asset: str = "039_mug"
    rack_scale: float = 1.1
    peg_radius_m: float = .0055
    free_length_m: float = .095897
    slab_depth_m: float = .047991
    radius_uncertainty_m: float = 0.
    length_uncertainty_m: float = 0.
    thickness_uncertainty_m: float = 0.
    topology_certificate: str | None = None
    asset_class_binding_verified: bool = False
    aperture_registration_error_m: float | None = None
    aperture_axis_error_rad: float | None = None
    mug_scale: float | None = None
    asset_class_manifest_verified: bool = False
    public_rack_registration: dict | None = None


@dataclass(frozen=True)
class FixtureRegistrationResult:
    status: Literal["confirmed", "unknown", "rejected"]
    reason: str
    fixture: FixtureBounds | None
    evidence: dict


def register_public_fixture(observation, report, fixture, *, episode_id,
                            rack_reference=None, reference_episode_id=None):
    """Register the observed shaft without inventing CAD-aperture correspondence.

    Opposing mid-shaft surfaces supply a centerline offset and sensitivity.
    A front aperture plane alone does not register the canonical CAD corridor:
    absent independently supplied aperture bounds remain explicitly unknown.
    """
    evidence = dict(rack_centerline_status="unknown", aperture_registration_status="unknown",
                    observation_sequence=observation.sequence, motion_authorized=False)
    try:
        if not isinstance(fixture, FixtureBounds):
            raise ValueError("verified_fixture_bounds_required")
        camera = observation.camera_frames["perception_camera"]
        if (not episode_id or fixture.episode_id != episode_id
                or fixture.sequence != observation.sequence
                or fixture.calibration_hash != calibration_sha256(camera)):
            raise ValueError("fixture_evidence_stale_or_incompatible")
        if (not fixture.asset_class_binding_verified or not fixture.asset_class_manifest_verified
                or fixture.mug_mesh_hash != MUG_CLASS_MANIFEST_HASH or fixture.rack_mesh_hash != RACK_VISUAL_HASH
                or fixture.rack_asset != "040_rack/base0" or fixture.rack_scale != 1.1
                or fixture.mug_asset != "039_mug"):
            return FixtureRegistrationResult("unknown", "fixture_mug_class_binding_unverified", fixture, evidence)
        reference, rack = rack_reference if rack_reference is not None else (observation, report)
        if rack_reference is not None and reference_episode_id != episode_id:
            raise ValueError("rack_registration_wrong_episode")
        reference_camera = reference.camera_frames["perception_camera"]
        if (reference.sequence > observation.sequence
                or calibration_sha256(reference_camera) != calibration_sha256(camera)):
            raise ValueError("rack_reference_incompatible")
        if not rack.peg_keypoints_px or not rack.peg_width_px:
            return FixtureRegistrationResult("unknown", "visible_peg_cross_section_required", fixture, evidence)
        fixture_axis(reference, rack.peg_keypoints_px)
        registration = measured_peg_centerline(reference_camera, rack.peg_keypoints_px, rack.peg_width_px)
        if abs(registration["radius_m"]-fixture.peg_radius_m) > (
                registration["radius_uncertainty_m"]+fixture.radius_uncertainty_m):
            raise ValueError("observed_peg_radius_inconsistent_with_fixture")
        registration = {**registration,
            "pose_world_wxyz": np.asarray(registration["pose_world_wxyz"]).tolist(),
            "episode_id": episode_id,
            "observation_sequence": reference.sequence,
            "calibration_sha256": calibration_sha256(reference_camera),
            "rgb_sha256": sha256(reference.images["perception_camera"].tobytes()).hexdigest()}
        fixture = replace(fixture, centerline_verified=True, public_rack_registration=registration)
        evidence.update(rack_centerline_status="confirmed", rack_registration=registration,
                        rack_reference_sequence=reference.sequence)
        errors = (fixture.aperture_registration_error_m, fixture.aperture_axis_error_rad)
        if any(error is None for error in errors):
            return FixtureRegistrationResult("unknown", "fixture_aperture_cad_registration_unavailable",
                                             fixture, evidence)
        if (any(not np.isfinite(error) or error < 0 for error in errors)
                or fixture.aperture_axis_error_rad > np.pi/2):
            raise ValueError("fixture_aperture_registration_invalid")
        evidence["aperture_registration_status"] = "independently_supplied_bounds_not_derived_from_front_plane"
        return FixtureRegistrationResult("confirmed", "public_shaft_and_supplied_aperture_registration",
                                         fixture, evidence)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return FixtureRegistrationResult("rejected", str(exc), None, evidence)


# Read-only CPU audit, 2026-11-03: projected every visual triangle and every
# collision-piece convex hull along asset X through each central-loop centroid.
# All ten scaled variants clear a 7.5 mm infinite cylinder. These are canonical
# corridors, not guarantees for arbitrary observed centers or fitted normals.
MUG_CLASS_CORRIDOR_CERTIFICATE = "039_mug/variants0-9/canonical-X/2026-11-03"
PHYSICAL_PEG_RADIUS_M = .0055
PEG_RADIAL_BUFFER_M = .001
MUG_CLASS_CORRIDOR_RADIUS_M = PHYSICAL_PEG_RADIUS_M + PEG_RADIAL_BUFFER_M
MUG_CLASS_COLLISION_CLEARANCES_M = (
    .013536778848876006, .010998632814711714, .012534100125281203,
    .010764374123484497, .013377379880890095, .010513011363462992,
    .013073812510780582, .011259218435020344, .011948522501984977,
    .017078076665112704)
# Visual-mesh SHA256 -> allowed runtime scales. Variant 0/1 share mesh bytes.
MUG_CLASS_MESH_SCALES = {
    "392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812": (.08,.065),
    "5ee1fcabdd67f41fc9a6af1841a5643091b1ba0701a808bddffb780cd6dbd60c": (.074,),
    "519108d6b3e83ef37e7124db2e5b1a189038cdb52f97aab67a17e12a7de5a2c9": (.06,),
    "b7b9f47810d157ff85639c0618df729f6769b6bb8e718f10eb6e2b4f09023daa": (.067,),
    "a311a4ccb76f67f864d77c395069acc96b5b38adba4ac940fdd42e328ec529ba": (.08,),
    "d35a5633d47c317acda2d8117df12094af471767915847893abd5d21f586dc30": (.065,),
    "0bf6dac42c1046efaf8bbc384f62b3e02cf3ed41b9af719a2cdff69eea835f0c": (.063,),
    "2c0cd95c90da022fead78cb488b63b5140a5cde17fffa6df84141be9488b63e9": (.067,),
    "05784795fb0f1f03bd19bbfd40aeceb954b97fef8abe580218e05b98fbeb227a": (.07,),
}

MUG_CLASS_VISUAL_HASHES = (
    "392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812",
    "392ec804651d7d3c2d38f158de6e3863d75c63d9e59acb02fd7c3ae3cc694812",
    "5ee1fcabdd67f41fc9a6af1841a5643091b1ba0701a808bddffb780cd6dbd60c",
    "519108d6b3e83ef37e7124db2e5b1a189038cdb52f97aab67a17e12a7de5a2c9",
    "b7b9f47810d157ff85639c0618df729f6769b6bb8e718f10eb6e2b4f09023daa",
    "a311a4ccb76f67f864d77c395069acc96b5b38adba4ac940fdd42e328ec529ba",
    "d35a5633d47c317acda2d8117df12094af471767915847893abd5d21f586dc30",
    "0bf6dac42c1046efaf8bbc384f62b3e02cf3ed41b9af719a2cdff69eea835f0c",
    "2c0cd95c90da022fead78cb488b63b5140a5cde17fffa6df84141be9488b63e9",
    "05784795fb0f1f03bd19bbfd40aeceb954b97fef8abe580218e05b98fbeb227a")
MUG_CLASS_SCALES = (.08,.065,.074,.06,.067,.08,.065,.063,.067,.07)
MUG_CLASS_COLLISION_HASHES = (
    "73bde98a69ce5f2908cc9640906d8025eec3a0c80e50728d9ea1bcfbb34f3320",
    "fee6e4406048ac3e5f24614db3361ff3ba3c52703ceece5033c65cec0f8cc98a",
    "d5640a7fac7f8df3f3ddd03956dfff0e3ce7b6ad6d0bb88dd42914361310ca4a",
    "4111965f73910c9b804acacbcfa324be12b920bc82b6c3fc65442c792e709b12",
    "fdcff38af19eb1ced3fa5b11e34e3d1ecfd8d660f50db0405673599ae89f11c5",
    "4db208fc86eb93afe373ce5cc9805ff2ff4839aef4d2ac96fa2b0fe3ca66ebfb",
    "a2dd0b0c9b90567c7764f25459c17d627c45ffb6c3582a89c5615496329068ca",
    "77fdbd6c0dc9f3331cd145627a4ba9efe95d102436d3dc777a08072821c0f94f",
    "8395eece85bfbe4223c36fbeb2133c839ae43e82712d609ae519a634b2784335",
    "08f0eb6e4af5b7a4642b4c198def3205ca430597d0c7560adcd300cc094b22d0")
MUG_CLASS_MANIFEST_HASH = sha256(json.dumps(list(zip(
    MUG_CLASS_VISUAL_HASHES,MUG_CLASS_COLLISION_HASHES,MUG_CLASS_SCALES)),
    separators=(",", ":")).encode()).hexdigest()
RACK_VISUAL_HASH = "e8becf94e0a51d3f70be5dc2587e486af1675e3b45a03309e04032510a9187c5"
RACK_COLLISION_HASH = "f277f6e7bf71555497b93c56c8b4a89a6d6b93706daa5dd11e7972c83571b205"


@dataclass(frozen=True)
class VerifiedFixtureAssets:
    rack_mesh_hash: str
    class_manifest_hash: str

    def bind(self, observation, *, episode_id):
        """Bind verified static assets, never invent current pose registration."""
        if (self.class_manifest_hash != MUG_CLASS_MANIFEST_HASH
                or self.rack_mesh_hash != RACK_VISUAL_HASH):
            raise ValueError("fixture_class_manifest_mismatch")
        return FixtureBounds(episode_id, observation.sequence,
            calibration_sha256(observation.camera_frames["perception_camera"]),
            self.rack_mesh_hash, self.class_manifest_hash,
            "read-only asset mesh hashes/scales; hanging_mug task class; current registration unknown",
            topology_certificate=MUG_CLASS_CORRIDOR_CERTIFICATE,
            asset_class_binding_verified=True, asset_class_manifest_verified=True)


def load_verified_fixture_assets(robotwin_root):
    """Read mesh/config files only, before simulator or service initialization."""
    root=Path(robotwin_root)/"assets/objects"
    try:
        for index,(visual,collision,scale) in enumerate(zip(
                MUG_CLASS_VISUAL_HASHES,MUG_CLASS_COLLISION_HASHES,MUG_CLASS_SCALES)):
            mug=root/"039_mug"
            if (sha256((mug/"visual"/f"base{index}.glb").read_bytes()).hexdigest()!=visual
                    or sha256((mug/"collision"/f"base{index}.glb").read_bytes()).hexdigest()!=collision
                    or json.loads((mug/f"model_data{index}.json").read_text())["scale"] != [scale]*3):
                raise ValueError("fixture_mug_asset_audit_mismatch")
        rack=root/"040_rack"
        if json.loads((rack/"model_data0.json").read_text())["scale"] != [1.1]*3:
            raise ValueError("fixture_rack_scale_mismatch")
        rack_hash=sha256((rack/"visual/base0.glb").read_bytes()).hexdigest()
        if (rack_hash != RACK_VISUAL_HASH
                or sha256((rack/"collision/base0.glb").read_bytes()).hexdigest() != RACK_COLLISION_HASH):
            raise ValueError("fixture_rack_asset_audit_mismatch")
    except (OSError,KeyError,json.JSONDecodeError) as exc:
        raise ValueError("fixture_asset_verification_unavailable") from exc
    return VerifiedFixtureAssets(rack_hash,MUG_CLASS_MANIFEST_HASH)


def topology_slab_coverage(fixture, aperture, corridor, usable_length):
    """Transfer the audited class corridor only with bounded registration.

    Class topology supplies hidden continuity. Current RGB-D supplies collision
    vetoes; it does not supply an unobserved CAD-to-aperture registration.
    """
    if fixture.topology_certificate != MUG_CLASS_CORRIDOR_CERTIFICATE:
        raise ValueError("fixture_topology_certificate_required")
    verified_class = (fixture.asset_class_manifest_verified
                      and fixture.mug_mesh_hash == MUG_CLASS_MANIFEST_HASH)
    verified_instance = (fixture.mug_mesh_hash in MUG_CLASS_MESH_SCALES
                         and fixture.mug_scale in MUG_CLASS_MESH_SCALES[fixture.mug_mesh_hash])
    if not fixture.asset_class_binding_verified or not (verified_class or verified_instance):
        raise ValueError("fixture_mug_class_binding_unverified")
    errors = (fixture.aperture_registration_error_m, fixture.aperture_axis_error_rad)
    if any(e is None or not np.isfinite(e) or e < 0 for e in errors):
        raise ValueError("fixture_aperture_registration_unverified")
    if fixture.aperture_axis_error_rad > np.pi/2:
        raise ValueError("fixture_aperture_registration_invalid")
    uncertainty = aperture["uncertainty"]
    angular_error = fixture.aperture_axis_error_rad+np.radians(uncertainty["maximum_normal_sensitivity_deg"])
    if not np.isfinite(angular_error) or not 0 <= angular_error <= np.pi/2:
        raise ValueError("fixture_aperture_registration_invalid")
    displacement = (fixture.aperture_registration_error_m + uncertainty["center_sensitivity_m"]
        + .05*np.sin(angular_error))
    canonical_margin = min(MUG_CLASS_COLLISION_CLEARANCES_M)-MUG_CLASS_CORRIDOR_RADIUS_M
    if not np.isfinite(displacement) or displacement >= canonical_margin:
        raise ValueError("fixture_topology_registration_margin_insufficient")
    if (corridor["foreground_collisions"] or corridor["minimum_radial_margin_m"] is None
            or corridor["minimum_radial_margin_m"] <= .005):
        raise ValueError("fixture_topology_visible_clearance_insufficient")
    if usable_length <= fixture.slab_depth_m+fixture.thickness_uncertainty_m:
        raise ValueError("fixture_axial_clearance_insufficient")
    return dict(full_slab_coverage_verified=True, certificate=fixture.topology_certificate,
        canonical_margin_m=canonical_margin, registration_displacement_m=float(displacement),
        basis="audited infinite CAD corridor transferred with bounded registration; not depth-ray visibility")


def corridor_visibility(observation, aperture, *, radius_m=MUG_CLASS_CORRIDOR_RADIUS_M, depth_m=.05):
    """Sample depth-ray free space, never certify the continuous hidden volume.

    A sample is observed free only before a continuous 3x3 depth neighborhood
    in at least one camera, with 2 mm depth margin. Behind a first return is
    unknown, not free. A finite 2.5 mm grid cannot establish full slab coverage.
    """
    pose = _pose_matrix(aperture["frame_pose_world_wxyz"])
    offsets = np.arange(-radius_m, radius_m+.000001, .0025)
    disk = [(y, z) for y in offsets for z in offsets if y*y+z*z <= radius_m*radius_m+1e-12]
    axial = np.linspace(0., depth_m, int(np.ceil(depth_m/.0025))+1)
    points = np.array([pose[:3, 3]-pose[:3, 0]*x+pose[:3, 1]*y+pose[:3, 2]*z
                       for x in axial for y, z in disk])
    free = np.zeros(len(points), dtype=bool)
    for camera in (observation.camera_frames or {}).values():
        cp = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
        projected = cp @ camera.intrinsic.T
        positive = cp[:, 2] > 0
        uv = np.zeros((len(points), 2), dtype=int)
        uv[positive] = np.rint(projected[positive, :2]/projected[positive, 2:]).astype(int)
        h, w = camera.depth_m.shape
        inside = positive & (uv[:, 0] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] >= 1) & (uv[:, 1] < h-1)
        for index in np.flatnonzero(inside & ~free):
            u, v = uv[index]
            patch = camera.depth_m[v-1:v+2, u-1:u+2]
            if patch.min() > 0 and np.ptp(patch) <= .06 and cp[index, 2]+.002 < patch.min():
                free[index] = True
    counts = free.reshape(len(axial), len(disk)).sum(axis=1)
    return dict(sample_spacing_m=.0025, depth_margin_m=.002, sampled_points=len(points),
        observed_free_samples=int(free.sum()), unknown_or_surface_samples=int((~free).sum()),
        axial_depths_m=axial.tolist(), observed_free_per_slice=counts.tolist(),
        samples_per_slice=len(disk), full_slab_coverage_verified=False,
        coverage_basis="finite public depth-ray samples; hidden volume and calibration uncertainty not certified")


def observed_corridor(observation, aperture, *, radius_m=MUG_CLASS_CORRIDOR_RADIUS_M, depth_m=.05):
    """Collision veto only: unobserved space is not certified free space.

    The plane normal faces the camera, so hidden material extends in -normal.
    Include every valid raw depth return; do not filter away collision points.
    """
    pose = _pose_matrix(aperture["frame_pose_world_wxyz"])
    center, normal = pose[:3, 3], pose[:3, 0]
    count, nearest = 0, float("inf")
    for camera in (observation.camera_frames or {}).values():
        depth = camera.depth_m
        v, u = np.indices(depth.shape)
        pixels = np.stack((u, v, np.ones_like(u)), axis=-1)[depth > 0]
        camera_points = (pixels @ np.linalg.inv(camera.intrinsic).T)*depth[depth > 0, None]
        transform = np.linalg.inv(camera.world_to_camera)
        delta = camera_points @ transform[:3, :3].T + transform[:3, 3] - center
        axial = -(delta @ normal)
        radial = np.linalg.norm(delta + axial[:, None]*normal, axis=1)
        slab = (axial >= 0) & (axial <= depth_m)
        count += int(np.count_nonzero(slab & (radial <= radius_m)))
        if slab.any():
            nearest = min(nearest, float(radial[slab].min()-radius_m))
    return dict(foreground_collisions=count, minimum_radial_margin_m=(
        nearest if np.isfinite(nearest) else None), hidden_space="unknown",
        axial_interval_m=[0., depth_m], radius_m=radius_m,
        visibility=corridor_visibility(observation, aperture, radius_m=radius_m, depth_m=depth_m))


RACK_INTERIOR_CENTERLINE_M = .130512
RACK_EXPOSED_CROSS_SECTION_LENGTH_M = .09590
EXPOSED_LANDMARK_TOLERANCE_M = .012


def fixture_axis(observation, keypoints):
    """Exposed-surface length consistency, not certified centerline recovery.

    The 12 mm tolerance bounds the entire measured length discrepancy; local
    sensitivity is reported separately and does not widen this admission gate.
    130.512 mm is the junction-centerline-to-tip dimension including the trunk
    intersection. 95.90 mm bounds the fully exposed branch cross-section.
    Comparing surface landmarks with that exposed length is a fixture-specific
    admission hypothesis, not proof that the hidden tip or centerline was seen.
    """
    if not keypoints or len(keypoints) != 3:
        raise ValueError("visible_peg_axis_required")
    samples = [read_surface(observation, SurfacePixel(camera="perception_camera", uv=uv))
               for uv in keypoints]
    seat, seat_error = samples[0]
    tip, tip_error = samples[1]
    length = float(np.linalg.norm(tip-seat))
    if abs(length-RACK_EXPOSED_CROSS_SECTION_LENGTH_M) > EXPOSED_LANDMARK_TOLERANCE_M+1e-12:
        raise ValueError("peg_axis_length_inconsistent_with_fixture")
    return seat, tip, seat_error, tip_error, length


def fixture_bounds(observation, report, aperture, fixture, *, episode_id, rack_reference=None):
    if not isinstance(fixture, FixtureBounds):
        raise ValueError("verified_fixture_contract_required")
    camera = observation.camera_frames["perception_camera"]
    if (not episode_id or fixture.episode_id != episode_id or fixture.sequence != observation.sequence
            or fixture.calibration_hash != calibration_sha256(camera)):
        raise ValueError("fixture_evidence_stale_or_incompatible")
    if (fixture.rack_asset != "040_rack/base0" or fixture.mug_asset != "039_mug"
            or fixture.rack_scale != 1.1 or not fixture.verification_source
            or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                   for h in (fixture.rack_mesh_hash, fixture.mug_mesh_hash))):
        raise ValueError("fixture_identity_unverified")
    if not fixture.full_slab_coverage_verified and fixture.topology_certificate is None:
        raise ValueError("fixture_full_slab_coverage_unverified")
    if not fixture.centerline_verified:
        raise ValueError("peg_junction_centerline_unverified")
    values = np.array([fixture.peg_radius_m, fixture.free_length_m, fixture.slab_depth_m,
                       fixture.radius_uncertainty_m, fixture.length_uncertainty_m,
                       fixture.thickness_uncertainty_m])
    if (not np.isfinite(values).all() or np.any(values[:3] <= 0) or np.any(values[3:] < 0)
            or fixture.peg_radius_m < .0055 or fixture.free_length_m > .095897
            or fixture.slab_depth_m < .047991):
        raise ValueError("fixture_bounds_not_conservative")
    if aperture.get("status") != "candidate_not_motion_validated":
        raise ValueError("validated_aperture_required")
    uncertainty = aperture["uncertainty"]
    span = aperture["geometry_diagnostics"].get("principal_rms_span_m", [])
    plane_metrics = np.asarray([*span, uncertainty["maximum_plane_residual_m"],
        uncertainty["center_sensitivity_m"], aperture["geometry_diagnostics"]["centroid_inscribed_radius_m"]])
    if (len(span) < 2 or not np.isfinite(plane_metrics).all() or np.any(plane_metrics < 0)
            or span[1] < .008 or uncertainty["maximum_plane_residual_m"] > .003):
        raise ValueError("fixture_aperture_plane_invalid")
    reference, rack = rack_reference if rack_reference is not None else (observation, report)
    if (reference.sequence > observation.sequence or calibration_sha256(
            reference.camera_frames["perception_camera"]) != calibration_sha256(camera)):
        raise ValueError("rack_reference_incompatible")
    registration = None
    if fixture.public_rack_registration is not None:
        source = fixture.public_rack_registration
        reference_camera = reference.camera_frames["perception_camera"]
        if (source.get("episode_id") != episode_id or source.get("observation_sequence") != reference.sequence
                or source.get("calibration_sha256") != calibration_sha256(reference_camera)
                or source.get("rgb_sha256") != sha256(reference.images["perception_camera"].tobytes()).hexdigest()):
            raise ValueError("rack_registration_stale_or_incompatible")
        # Recompute from the source pixels; do not trust a supplied world pose.
        registration = measured_peg_centerline(reference_camera, rack.peg_keypoints_px, rack.peg_width_px)
        registration["pose_world_wxyz"] = np.asarray(registration["pose_world_wxyz"]).tolist()
    seat, tip, seat_error, tip_error, length = fixture_axis(reference, rack.peg_keypoints_px)
    radius = fixture.peg_radius_m+fixture.radius_uncertainty_m
    gap = (aperture["geometry_diagnostics"]["centroid_inscribed_radius_m"]-radius
           -uncertainty["center_sensitivity_m"])
    if gap <= .004 or radius+PEG_RADIAL_BUFFER_M > MUG_CLASS_CORRIDOR_RADIUS_M+1e-12:
        raise ValueError("fixture_radial_clearance_insufficient")
    usable_length = min(fixture.free_length_m-fixture.length_uncertainty_m,
                        length-seat_error-tip_error)
    axial = usable_length-fixture.slab_depth_m-fixture.thickness_uncertainty_m
    if axial <= 0 or fixture.slab_depth_m+fixture.thickness_uncertainty_m > .05:
        raise ValueError("fixture_axial_clearance_insufficient")
    corridor = observed_corridor(observation, aperture)
    if corridor["foreground_collisions"]:
        raise ValueError("fixture_corridor_foreground_collision")
    coverage = (topology_slab_coverage(fixture, aperture, corridor, usable_length)
                if fixture.topology_certificate is not None else
                dict(full_slab_coverage_verified=True, basis="independent caller-verified full slab coverage"))
    return dict(handle_thickness_m=fixture.slab_depth_m, thickness_uncertainty_m=fixture.thickness_uncertainty_m,
        peg_radius_m=radius, peg_radius_uncertainty_m=fixture.radius_uncertainty_m,
        peg_length_m=usable_length, engaged_length_m=usable_length,
        radial_clearance_m=gap, axial_clearance_m=axial, position_uncertainty_m=(max(seat_error, tip_error)
            + (registration["position_sensitivity_m"] if registration is not None else 0.)),
        orientation_uncertainty_rad=float(np.arctan2(seat_error+tip_error, length)),
        margin_m=.0005, roll_min_rad=-np.pi, roll_max_rad=np.pi,
        minimum_handle_height_m=.01, body_clearance_m=.005, dimensions_validated=True,
        clearance_validated=True, dimension_basis="verified fixture bounds plus current public RGB-D",
        fixture_provenance=fixture.__dict__.copy(), corridor=corridor,
        slab_coverage=coverage, full_slab_coverage_verified=True,
        **({"public_registration": registration} if registration is not None else {}),
        rack_reference_sequence=reference.sequence, measured_axis_length_m=length,
        axis_landmark_basis="exposed surface; centerline semantics require independent verification",
        axial_length_compatible=True,
        basis="asset-bounded slab; observed collision veto; pose calibration remains an assumption")


def measured_bounds(observation, report, aperture, *, rack_reference=None, fixture=None, episode_id=None):
    """Use optical measurements or explicit trusted fixture bounds; never guess."""
    rack_observation, rack = rack_reference if rack_reference is not None else (observation, report)
    if fixture is not None and not report.opposite_handle_faces_visible:
        return fixture_bounds(observation, report, aperture, fixture,
                              episode_id=episode_id, rack_reference=rack_reference)
    if (not report.opposite_handle_faces_visible or report.handle_front is None
            or report.handle_back is None or not rack.peg_width_px or len(rack.peg_width_px) != 2
            or not rack.peg_keypoints_px or len(rack.peg_keypoints_px) != 3):
        raise ValueError("visible_thickness_and_peg_cross_section_required")
    sample = lambda uv: read_surface(rack_observation, SurfacePixel(camera="perception_camera", uv=uv))
    seat, seat_error = sample(rack.peg_keypoints_px[0])
    tip, tip_error = sample(rack.peg_keypoints_px[1])
    axis = tip-seat
    length = float(np.linalg.norm(axis))
    if length <= seat_error+tip_error:
        raise ValueError("peg_axis_measurement_unresolved")
    axis /= length
    first, first_error = sample(rack.peg_width_px[0])
    second, second_error = sample(rack.peg_width_px[1])
    width_vector = second-first
    width = float(np.linalg.norm(width_vector-axis*(width_vector@axis)))
    radius = width/2
    radius_error = (first_error+second_error)/2
    if radius <= radius_error or abs(width_vector@axis) > first_error+second_error:
        raise ValueError("peg_radius_subpixel_or_wrong_cross_section")
    # Edge depth tends to read a thin peg narrow. Never let a measurement below
    # the rack fixture's physical radius enlarge the computed clearance.
    measured_radius = radius
    radius = max(radius, PHYSICAL_PEG_RADIUS_M)
    front, front_error = read_surface(observation, report.handle_front)
    back, back_error = read_surface(observation, report.handle_back)
    frame = _pose_matrix(aperture["frame_pose_world_wxyz"])
    normal, center = frame[:3, 0], frame[:3, 3]
    # The optical path requires opposing faces, not two depths along one
    # opaque line of sight. Check camera origins against the measured plane.
    face_views = []
    for pixel, point, error in ((report.handle_front, front, front_error),
                                (report.handle_back, back, back_error)):
        face_camera = observation.camera_frames[pixel.camera]
        image = observation.images.get(pixel.camera)
        if image is None or image.shape[:2] != face_camera.depth_m.shape:
            raise ValueError("opposite_handle_face_images_unavailable")
        origin = np.linalg.inv(face_camera.world_to_camera)[:3, 3]
        face_views.append(float((origin-point) @ normal))
    if face_views[0] <= front_error or face_views[1] >= -back_error:
        raise ValueError("opposite_handle_face_viewpoints_unresolved")
    thickness = float((front-back)@normal)
    thickness_error = front_error+back_error+aperture["uncertainty"]["maximum_plane_residual_m"]
    lateral = np.linalg.norm(front-back-thickness*normal)
    if (thickness <= thickness_error or lateral > front_error+back_error
            or abs((front-center)@normal) > front_error+.003):
        raise ValueError("opposite_handle_faces_unresolved")
    gap = aperture["geometry_diagnostics"]["centroid_inscribed_radius_m"]-radius-radius_error
    axial = length-seat_error-tip_error-(thickness+thickness_error)/2
    if gap <= 0 or axial <= 0:
        raise ValueError("observed_mating_clearance_insufficient")
    registration = measured_peg_centerline(rack_observation.camera_frames["perception_camera"],
                                          rack.peg_keypoints_px, rack.peg_width_px)
    registration["pose_world_wxyz"] = np.asarray(registration["pose_world_wxyz"]).tolist()
    return dict(handle_thickness_m=thickness, thickness_uncertainty_m=thickness_error,
        peg_radius_m=radius+radius_error, peg_radius_uncertainty_m=radius_error,
        measured_peg_radius_m=measured_radius,
        peg_radius_basis=("measured" if measured_radius >= PHYSICAL_PEG_RADIUS_M
                          else "physical_fixture_radius_floor"),
        peg_length_m=length, radial_clearance_m=gap, axial_clearance_m=axial,
        position_uncertainty_m=max(seat_error, tip_error)+registration["position_sensitivity_m"],
        orientation_uncertainty_rad=float(np.arctan2(seat_error+tip_error, length)),
        engaged_length_m=length, margin_m=.0005, roll_min_rad=-np.pi, roll_max_rad=np.pi,
        minimum_handle_height_m=.01, body_clearance_m=.005,
        dimensions_validated=True, dimension_basis="visible opposing surfaces and peg cross-section in rendered RGB-D",
        public_registration=registration,
        measurement_evidence=dict(observation_sequence=observation.sequence,
            rack_reference_sequence=rack_observation.sequence,
            front_camera=report.handle_front.camera, back_camera=report.handle_back.camera,
            front_calibration_sha256=calibration_sha256(observation.camera_frames[report.handle_front.camera]),
            back_calibration_sha256=calibration_sha256(observation.camera_frames[report.handle_back.camera]),
            signed_camera_to_face_distances_m=face_views,
            information_source="public_rgbd", cad_registration_used=False),
        basis="local one-pixel/depth-spread sensitivity; systematic calibration error unknown",
        systematic_calibration_error="unknown")


class PublicRGBDHangingProvider:
    source = "perception"

    def __init__(self, client, *, model, episode_id, store, log, maximum_requests=80, fixture=None,
                 lift_evidence_max_gap=32, table_plane_max_age=128):
        if (type(lift_evidence_max_gap) is not int or lift_evidence_max_gap <= 0
                or type(table_plane_max_age) is not int or table_plane_max_age <= 0):
            raise ValueError("invalid_public_lift_evidence_age")
        self.client, self.model, self.episode_id = client, model, episode_id
        self.store, self.log, self.maximum_requests = store, log, maximum_requests
        self.requests = 0
        self.fixture = fixture
        self._key = self._report = self._previous = None
        self._corroborated_count = 0
        self._geometry_retry_key = self._geometry_retry_report = None
        self._rack_reference = None
        self._rack_reference_episode = None
        self._peg_candidate = self._aperture_candidate = None
        self._lift_corroboration = {"status": "unknown", "reason": "not_evaluated"}
        self.lift_evidence_max_gap = lift_evidence_max_gap
        self.table_plane_max_age = table_plane_max_age
        self._table_plane = self._mug_track = None
        self._mug_body_track = None
        self._last_lift_sequence = None
        self._lift_identity = episode_id
        self._lift_motion_supported = False

    def _lift_table_plane(self, observation, report):
        camera = observation.camera_frames["perception_camera"]
        calibration = calibration_sha256(camera)
        cached = self._table_plane
        if cached is not None and (cached[0] != calibration or
                not 0 <= observation.sequence-cached[1] <= self.table_plane_max_age):
            cached = self._table_plane = None
        try:
            if not report.table_material_px or len(report.table_material_px) < 6:
                raise ValueError("table_samples_unavailable")
            table = np.array([read_surface(observation, SurfacePixel(
                camera="perception_camera", uv=uv))[0] for uv in report.table_material_px])
            center = table.mean(axis=0)
            _, singular, axes = np.linalg.svd(table-center, full_matrices=False)
            normal = axes[-1]
            if normal[2] < 0:
                normal = -normal
            if (singular[1] < .03 or singular[0]/singular[1] > 10 or normal[2] < .95
                    or np.max(np.abs((table-center)@normal)) > .003):
                raise ValueError("table_plane_invalid")
        except (ValueError, TypeError, np.linalg.LinAlgError):
            if cached is None:
                raise ValueError("table_plane_invalid")
            self._lift_corroboration["table_plane_source"] = "episode_static_cache"
            return cached[2], cached[3]
        # Static-table hypothesis: incompatible valid planes revoke the cache;
        # invalid/occluded proposals cannot overwrite a previously validated plane.
        if cached is not None and (normal@cached[3] < .995 or
                abs((center-cached[2])@cached[3]) > .003):
            self._table_plane = None
            raise ValueError("table_plane_inconsistent")
        self._table_plane = (calibration, observation.sequence, center, normal)
        self._lift_corroboration["table_plane_source"] = "current_rgbd"
        return center, normal

    def _track_mug_surface(self, observation, points):
        tcp = _pose_matrix(observation.measured_tcp_poses["right"])
        # Row-vector inverse rigid transform includes measured wrist rotation.
        local = (points-tcp[:3, 3])@tcp[:3, :3]
        previous = self._mug_track
        if previous is not None:
            sequence, old_local, old_tcp = previous
            if 0 < observation.sequence-sequence <= self.lift_evidence_max_gap:
                distances = np.linalg.norm(local[:, None, :]-old_local[None, :, :], axis=2)
                # Partial visibility is allowed, but an isolated matching pixel is
                # not object identity. 15 mm is a surface correspondence bound,
                # not an aperture, motion or release tolerance.
                overlap = min(np.mean(distances.min(axis=0) <= .015),
                              np.mean(distances.min(axis=1) <= .015))
                translation = np.linalg.norm(tcp[:3, 3]-old_tcp[:3, 3])
                angle = np.arccos(np.clip((np.trace(old_tcp[:3, :3].T@tcp[:3, :3])-1)/2, -1, 1))
                excited = translation >= .01 or angle >= .1
                self._lift_corroboration.update(surface_overlap_fraction=float(overlap),
                    measured_tcp_translation_m=float(translation), measured_tcp_rotation_rad=float(angle),
                    co_motion="supported" if excited and overlap >= .6 else
                              "inconsistent" if excited else "insufficient_motion",
                    surface_match_bound_m=.015)
                if excited and overlap < .6:
                    self._mug_track = None
                    self._lift_motion_supported = False
                    self._lift_corroboration.update(status="unknown", reason="mug_tcp_motion_inconsistent")
                    return False
                if excited:
                    self._lift_motion_supported = True
        # Bound pairwise matching cost without selecting a seed-specific region.
        self._mug_track = (observation.sequence, local[::max(1, len(local)//256)], tcp)
        return True

    def _track_mug_body(self, observation, report):
        """Supplement surface tracking; a cylinder fit never verifies grasp or a handle.

        The polygon is an unreviewed model proposal. Fitter quality describes
        geometry only, so neither agreement nor disagreement changes the
        existing surface, closure, clearance or motion-authorization guards.
        """
        from PIL import Image, ImageDraw

        previous, self._mug_body_track = self._mug_body_track, None
        evidence = dict(status="unknown", reason="body_geometry_unavailable",
                        source="unreviewed_body_polygon_and_public_depth",
                        motion_authorized=False, grasp_verified=False)
        camera = (observation.camera_frames or {}).get("perception_camera")
        poses = observation.measured_tcp_poses
        if camera is None or not poses or "right" not in poses:
            return evidence
        try:
            polygon = np.asarray(report.body_polygon_px, dtype=float)
            h, w = camera.depth_m.shape
            if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
                    or not np.isfinite(polygon).all() or (polygon < 0).any()
                    or (polygon[:, 0] >= w).any() or (polygon[:, 1] >= h).any()):
                evidence["reason"] = "invalid_or_missing_body_polygon"
                return evidence
            mask = Image.new("1", (w, h))
            ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], fill=1)
            surface = masked_surface(camera, np.asarray(mask, dtype=bool))
            pose, reason = object_pose({"perception_camera": surface}, MUG,
                table_height_m=None, observation_sequence=observation.sequence)
            evidence.update(reason=reason, surface_points=len(surface.points),
                            fit_trusted=pose is not None and pose.trusted)
            if pose is None:
                return evidence
            evidence["fit_inlier_fraction"] = pose.quality
            if not pose.trusted or not pose.base_known:
                evidence["reason"] = reason if not pose.trusted else "body_reference_end_unknown"
                return evidence
            tcp = _pose_matrix(poses["right"])
            identity = (self.episode_id, calibration_sha256(camera))
            self._mug_body_track = (identity, observation.sequence, pose, tcp)
            if previous is None or previous[0] != identity:
                evidence["reason"] = "body_tracking_history_unavailable"
                return evidence
            _, sequence, old_pose, old_tcp = previous
            if not 0 < observation.sequence-sequence <= self.lift_evidence_max_gap:
                evidence["reason"] = "body_tracking_history_stale_or_duplicate"
                return evidence
            delta = tcp @ np.linalg.inv(old_tcp)
            expected = delta[:3, :3] @ old_pose.position + delta[:3, 3]
            position_error = float(np.linalg.norm(np.asarray(pose.position)-expected))
            axis_error = float(np.degrees(np.arccos(np.clip(abs(
                (delta[:3, :3] @ old_pose.axis) @ pose.axis), 0., 1.))))
            radius_error = abs(pose.radius_m-old_pose.radius_m)
            translation = float(np.linalg.norm(tcp[:3, 3]-old_tcp[:3, 3]))
            rotation = float(np.arccos(np.clip((np.trace(
                old_tcp[:3, :3].T @ tcp[:3, :3])-1)/2, -1., 1.)))
            # Reuse the surface correspondence and fitter's existing body
            # agreement tolerances; these are diagnostics, not calibrated errors.
            consistent = (position_error <= .015 and axis_error <= SAME_BODY_AXIS_DEG
                          and radius_error <= SURFACE_TOLERANCE_M)
            excited = translation >= .01 or rotation >= .1
            evidence.update(status="unknown" if not excited else
                "supported" if consistent else "inconsistent",
                reason="insufficient_measured_tcp_motion" if not excited else
                "body_pose_co_motion_supported" if consistent else "body_pose_motion_inconsistent",
                previous_sequence=sequence, position_residual_m=position_error,
                axis_line_residual_deg=axis_error, radius_change_m=radius_error)
            return evidence
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            self._mug_body_track = None
            evidence["reason"] = "body_pose_evaluation_unavailable"
            return evidence

    def _fixture_for(self, observation):
        # A static asset verifier can bind fresh observation identity without
        # reusing stale registration evidence. Fixed per-frame contracts retain
        # their original freshness checks.
        if isinstance(self.fixture, VerifiedFixtureAssets):
            return self.fixture.bind(observation, episode_id=self.episode_id)
        return self.fixture

    def _current_rack_reference(self, observation):
        # Episode-static rack hypothesis; fixed-camera calibration changes invalidate
        # the reference. No claim that an occluded rack's motion is observable.
        if self._rack_reference is not None:
            old, _ = self._rack_reference
            if (self._rack_reference_episode != self.episode_id or observation.sequence < old.sequence
                    or calibration_sha256(observation.camera_frames["perception_camera"])
                    != calibration_sha256(old.camera_frames["perception_camera"])):
                self._rack_reference = None
                self._rack_reference_episode = None
        return self._rack_reference

    def _cache_rack(self, observation, report):
        from harness.perception.hang_targets import StaticRackFrameCache
        self._current_rack_reference(observation)
        confidence = report.peg_confidence if report.peg_confidence is not None else report.confidence
        fixture = self._fixture_for(observation)
        try:
            if fixture is not None:
                if (fixture.episode_id != self.episode_id or fixture.rack_asset != "040_rack/base0"
                        or fixture.rack_scale != 1.1 or fixture.calibration_hash != calibration_sha256(
                            observation.camera_frames["perception_camera"])):
                    return
                fixture_axis(observation, report.peg_keypoints_px)
                minimum_confidence = 0.
            else:
                if (confidence < MINIMUM_PROPOSAL_CONFIDENCE
                        or not report.peg_keypoints_px or not report.peg_width_px):
                    return
                for uv in (*report.peg_keypoints_px, *report.peg_width_px):
                    read_surface(observation, SurfacePixel(camera="perception_camera", uv=uv))
                camera = observation.camera_frames["perception_camera"]
                # The width must give a real mid-shaft cross-section here, not
                # only later in measured_bounds, before this frame can be kept.
                measured_peg_centerline(camera, report.peg_keypoints_px, report.peg_width_px)
                if self._rack_reference is not None:
                    return  # The rack is static: keep the first validated reference.
                if confidence < SELF_CONFIRMED_CONFIDENCE and not self._peg_agrees(observation, report):
                    return
                minimum_confidence = MINIMUM_PROPOSAL_CONFIDENCE
            StaticRackFrameCache(minimum_confidence=minimum_confidence).cache(
                observation, self._rack_report(observation, report))
        except (ValueError, TypeError):
            return
        self._rack_reference = (observation.copy(), report)
        self._rack_reference_episode = self.episode_id
        self.log(dict(event_type="rack_reference_kept", observation_sequence=observation.sequence,
                      peg_confidence=confidence))

    def _peg_agrees(self, observation, report):
        """Two frames' peg seat and tip coincide in the world (the rack is static)."""
        camera = observation.camera_frames["perception_camera"]
        seat, tip = (np.asarray(camera.unproject(*uv, radius=1), dtype=float)
                     for uv in report.peg_keypoints_px[:2])
        identity = (self.episode_id, calibration_sha256(camera))
        previous, self._peg_candidate = self._peg_candidate, (identity, observation.sequence, seat, tip)
        if previous is None or previous[0] != identity or not 0 < observation.sequence-previous[1]:
            return False
        error = max(float(np.linalg.norm(seat-previous[2])), float(np.linalg.norm(tip-previous[3])))
        agreed = error <= AGREEMENT_CENTER_M
        self.log(dict(event_type="peg_two_frame_agreement", observation_sequence=observation.sequence,
                      previous_sequence=previous[1], landmark_error_m=error, agreed=agreed))
        return agreed

    def _aperture_agrees(self, observation, measured_tcp, aperture):
        """This frame's opening matches the previous one moved by the measured TCP motion."""
        camera = observation.camera_frames["perception_camera"]
        if aperture.get("frame_pose_world_wxyz") is None:
            return dict(status="aperture_frame_unavailable", observation_sequence=observation.sequence)
        frame = _pose_matrix(aperture["frame_pose_world_wxyz"])
        tcp = _pose_matrix(measured_tcp)
        identity = (self.episode_id, calibration_sha256(camera))
        previous = self._aperture_candidate
        self._aperture_candidate = (identity, observation.sequence, tcp, frame[:3, 3].copy(), frame[:3, 0].copy())
        result = dict(status="awaiting_second_frame", observation_sequence=observation.sequence)
        if (previous is None or previous[0] != identity
                or not 0 < observation.sequence-previous[1] <= self.lift_evidence_max_gap):
            return result
        _, sequence, old_tcp, old_center, old_normal = previous
        delta = tcp @ np.linalg.inv(old_tcp)
        predicted_center = delta[:3, :3] @ old_center + delta[:3, 3]
        predicted_normal = delta[:3, :3] @ old_normal
        center_error = float(np.linalg.norm(frame[:3, 3]-predicted_center))
        normal_error = float(np.degrees(np.arccos(np.clip(abs(predicted_normal @ frame[:3, 0]), 0., 1.))))
        agreed = center_error <= AGREEMENT_CENTER_M and normal_error <= AGREEMENT_NORMAL_DEG
        result.update(status="agreed" if agreed else "disagreed", previous_sequence=sequence,
                      center_error_m=center_error, normal_error_deg=normal_error)
        return result

    @staticmethod
    def _rack_report(observation, report):
        image = observation.images["perception_camera"]
        return dict(status="grounded", target="rack_peg", camera="perception_camera",
            observation_sequence=observation.sequence, image_shape_px=list(image.shape[:2]),
            image_sha256=sha256(png_bytes(image)).hexdigest(), confidence=(
                report.peg_confidence if report.peg_confidence is not None else report.confidence),
            keypoints_px=report.peg_keypoints_px)

    def crop_geometry(self, observation, report):
        """One current-region proposal retry; enlargement adds no sensor detail."""
        from PIL import Image
        from development.hanging.grounding_preflight import CROP_PADDING_PX, CROP_LONG_SIDE_PX
        if not isinstance(observation.images, dict):
            return report
        image = observation.images.get("perception_camera")
        if not isinstance(image, np.ndarray):
            return report
        key = (self.episode_id, observation.sequence, sha256(image.tobytes()).hexdigest(),
               calibration_sha256(observation.camera_frames["perception_camera"]))
        rack_reference = self._current_rack_reference(observation)
        if key == self._geometry_retry_key:
            return self._geometry_retry_report
        points = np.asarray(report.body_polygon_px, dtype=float)
        h, w = image.shape[:2]
        crop_source = "mug_body_region"
        padding = CROP_PADDING_PX*max(1, w/320, h/240)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= w).any() or (points[:, 1] >= h).any()):
            poses = observation.measured_tcp_poses
            if not poses:
                return report
            camera = observation.camera_frames["perception_camera"]
            point = camera.world_to_camera @ np.array([*poses["right"][:3], 1.])
            if point[2] <= 0:
                return report
            pixel = camera.intrinsic @ point[:3]
            points = np.array([pixel[:2]/pixel[2]])
            if not (0 <= points[0, 0] < w and 0 <= points[0, 1] < h):
                return report
            # TCP is behind the fingers; without a body extent retain the wider
            # existing fallback rather than assuming a tightly centered mug.
            padding *= 2
            crop_source = "measured_right_tcp"
        lower = np.maximum(0, np.floor(points.min(axis=0)-padding)).astype(int)
        upper = np.minimum([w, h], np.ceil(points.max(axis=0)+padding)).astype(int)
        left, top = lower
        right, bottom = upper
        if right-left < 3 or bottom-top < 3 or self.requests >= self.maximum_requests:
            return report
        crop = Image.fromarray(image).crop((left, top, right, bottom))
        scale = CROP_LONG_SIDE_PX/max(crop.size)
        shown = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.LANCZOS)
        self._geometry_retry_key, self._geometry_retry_report = key, report
        self.requests += 1
        try:
            data = request_tool_input(self.client, model=self.model,
                system=PUBLIC_PROMPT+"\nThis request contains ONLY a crop of perception_camera. "
                    "All perception_camera pixels must use the displayed CROP coordinates. "
                    "Focus ONLY on mug opening, solid handle material and mug body. "
                    "Rack grounding is separate: leave peg_keypoints_px, peg_width_px, "
                    "rack_region_px and peg_confidence null. "
                    "Do not complete hidden opening boundaries. State may remain unknown.",
                tool={"name": "report_public_hanging", "description": "Visible cropped geometry only",
                      "input_schema": VisibleHangingReport.model_json_schema()},
                content=[{"type": "text", "text": f"perception_camera crop; sequence={observation.sequence}; "
                          f"pixels={shown.width}x{shown.height}; enlarged original pixels, not new depth"},
                         image_block(png_bytes(np.asarray(shown)))], max_tokens=4096)
            proposed = VisibleHangingReport.model_validate(data)
            def mapped(values):
                if values is None:
                    return None
                pixels = np.asarray(values, dtype=float)
                if (pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all()
                        or (pixels < 0).any() or (pixels[:, 0] >= shown.width).any()
                        or (pixels[:, 1] >= shown.height).any()):
                    raise ValueError("crop_pixels_out_of_bounds")
                original = (pixels+.5)*[crop.width/shown.width, crop.height/shown.height]-.5+[left, top]
                if (original < 0).any() or (original[:, 0] >= w).any() or (original[:, 1] >= h).any():
                    raise ValueError("mapped_crop_pixels_out_of_bounds")
                return [tuple(p) for p in original]
            fields = {name: mapped(getattr(proposed, name)) for name in (
                "contour_px", "material_px", "body_polygon_px")}
            for name in ("handle_front", "handle_back"):
                pixel = getattr(proposed, name)
                if pixel is not None and pixel.camera != "perception_camera":
                    raise ValueError("crop_surface_wrong_camera")
                fields[name] = (SurfacePixel(camera="perception_camera", uv=mapped([pixel.uv])[0])
                                if pixel is not None else None)
            fields.update(confidence=proposed.confidence,
                          complete_visible_opening=proposed.complete_visible_opening,
                          opposite_handle_faces_visible=proposed.opposite_handle_faces_visible)
            result = report.model_copy(update=fields)  # Never change current-state history from a crop.
            self._geometry_retry_report = result
            self.log(dict(event_type="public_geometry_crop", observation_sequence=observation.sequence,
                crop_box=[int(v) for v in (left, top, right, bottom)],
                shown_shape_px=[shown.height, shown.width], mapping="pixel_center",
                crop_source=crop_source, padding_px=float(padding), rack_grounding="separate_full_frame_or_cache",
                rack_reference_sequence=(rack_reference[0].sequence if rack_reference is not None else None),
                rack_reference_episode=(self._rack_reference_episode if rack_reference is not None else None),
                geometry_confidence=result.confidence, complete_visible_opening=result.complete_visible_opening,
                report=result.model_dump(mode="json")))
            return result
        except Exception as exc:
            self.log(dict(event_type="public_grounding_failure", observation_sequence=observation.sequence,
                          failure=sanitized_failure(exc)))
            return report

    def _corroborates_lift(self, observation, report):
        """Corroborate visual holding with closure intent and public surfaces.

        These conservative association/clearance bounds are evidence checks,
        not motion tolerances or calibrated confidence intervals. Commanded
        closure is intent, never a measured grasp or physical support claim.
        """
        closure_source = "visible_fingers"
        closed = report.right_gripper_closed_visible
        if closed is None:
            command = float(observation.command_state[13])
            closed = bool(np.isfinite(command) and 0 <= command <= .05)
            closure_source = "commanded_closure_not_measured"
        self._lift_corroboration = dict(status="unknown", reason="closure_not_confirmed",
                                        closure_source=closure_source)
        if closed is not True:
            return False
        self._lift_corroboration["reason"] = "public_lift_geometry_unavailable"
        if not observation.measured_tcp_poses or not report.body_polygon_px:
            return False
        from PIL import Image, ImageDraw
        camera = observation.camera_frames["perception_camera"]
        try:
            self._lift_corroboration["reason"] = "table_plane_invalid"
            try:
                center, normal = self._lift_table_plane(observation, report)
            except ValueError as exc:
                self._lift_corroboration["reason"] = str(exc)
                return False
            h, w = camera.depth_m.shape
            polygon = np.asarray(report.body_polygon_px, dtype=float)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all()
                    or (polygon < 0).any() or (polygon[:, 0] >= w).any() or (polygon[:, 1] >= h).any()):
                return False
            mask_image = Image.new("1", (w, h))
            ImageDraw.Draw(mask_image).polygon([tuple(p) for p in polygon], fill=1)
            mask = np.asarray(mask_image, dtype=bool)
            inner = mask.copy()
            inner[1:] &= mask[:-1]
            inner[:-1] &= mask[1:]
            inner[:, 1:] &= mask[:, :-1]
            inner[:, :-1] &= mask[:, 1:]
            pixels = np.argwhere(inner)[::4]
            self._lift_corroboration["reason"] = "insufficient_mug_surface_samples"
            if len(pixels) < 6:
                return False
            self._lift_corroboration["reason"] = "mug_surface_depth_invalid"
            samples, depths, coordinates, sensitivities = [], [], [], []
            for v, u in pixels:
                try:
                    point, sensitivity = read_surface(observation, SurfacePixel(
                        camera="perception_camera", uv=(float(u), float(v))))
                except ValueError:
                    continue  # A discontinuous boundary never becomes a surface sample.
                samples.append(point)
                sensitivities.append(sensitivity)
                depths.append(camera.sample_depth(float(u), float(v), radius=1))
                coordinates.append((v, u))
            if len(samples) < 6 or len(samples) < len(pixels)*.5:
                return False
            # Retain the dominant local depth band, not table/background points
            # admitted by an approximate image polygon. This is not aperture fitting.
            keep = np.abs(np.asarray(depths)-np.median(depths)) <= .03
            from development.hanging.aperture_frame import depth_components
            band = inner & (camera.depth_m > 0) & (np.abs(camera.depth_m-np.median(depths)) <= .03)
            components = list(depth_components(band))
            if not components:
                return False
            component = max(components, key=len)
            connected = np.zeros_like(band)
            connected[component[:,0],component[:,1]] = True
            keep &= np.array([connected[v,u] for v,u in coordinates])
            points = np.asarray(samples)[keep]
            if len(points) < 6 or len(points) < len(samples)*.5:
                return False
            mug = np.median(points, axis=0)
            right, left = (np.asarray(observation.measured_tcp_poses[arm][:3]) for arm in ("right", "left"))
            distance = np.linalg.norm(mug-right)
            margin = np.linalg.norm(mug-left)-distance
            # Visible surface centroids move as the mug rotates/occludes. Require
            # a supported patch (10th percentile), rather than one nearest pixel.
            surface_distance = float(np.quantile(np.linalg.norm(points-right, axis=1), .1))
            surface_uncertainty = float(np.quantile(np.asarray(sensitivities)[keep], .9))
            clearance = np.min((points-center)@normal)
            tcp_clearance = (right-center)@normal
            # TCP is behind the fingers. The existing RGB-D grounder documents
            # 200-220 mm mug offsets (and uses 240 mm); use the tighter 220 mm
            # envelope here, with independent lift and arm-separation checks.
            confirmed = bool(surface_uncertainty <= .015 and surface_distance+surface_uncertainty <= .22 and margin >= .05
                             and clearance >= .02 and tcp_clearance >= .02)
            self._lift_corroboration.update(status="confirmed" if confirmed else "unknown",
                reason="public_rgbd_and_closure" if confirmed else "association_or_clearance_failed",
                right_tcp_distance_m=float(distance), right_surface_distance_m=surface_distance,
                surface_raster_sensitivity_m=surface_uncertainty,
                uncertainty_basis="90th_percentile_local_pixel_depth_sensitivity; calibration_error_unknown",
                surface_association_basis="10th_percentile_connected_public_surface",
                right_association_margin_m=float(margin),
                minimum_mug_clearance_m=float(clearance), tcp_clearance_m=float(tcp_clearance))
            self._lift_corroboration.update(valid_surface_samples=len(samples),
                retained_surface_samples=len(points), rejected_boundary_samples=len(pixels)-len(samples),
                association_limit_m=.22)
            if clearance < .02:
                self._lift_corroboration["reason"] = "mug_on_table_or_clearance_insufficient"
            tracked = confirmed and self._track_mug_surface(observation, points)
            if tracked:
                self._lift_corroboration["body_pose_tracking"] = self._track_mug_body(observation, report)
            else:
                self._mug_body_track = None
            return tracked
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return False

    def _update_temporal_lift(self, observation, report, key):
        compatible = (self._key is not None and 0 < key[0]-self._key[0] <= self.lift_evidence_max_gap
                      and self._key[2] == key[2] and self._lift_identity == self.episode_id)
        if not compatible:
            self._corroborated_count = 0
            self._table_plane = self._mug_track = None
            self._mug_body_track = None
            self._last_lift_sequence = None
            self._lift_motion_supported = False
        self._lift_identity = self.episode_id
        self._lift_corroboration = dict(status="unknown", reason="state_report_not_eligible")
        command = float(observation.command_state[13])
        contradiction = (report.right_held is False or report.mug_off_table is False
                         or report.right_gripper_closed_visible is False
                         or not np.isfinite(command) or command > .05)
        if contradiction:
            self._corroborated_count = 0
            self._mug_track = None
            self._mug_body_track = None
            self._last_lift_sequence = None
            self._lift_motion_supported = False
            self._lift_corroboration["reason"] = "grasp_contradiction_or_gripper_open"
            return
        qualifies = (report.task_state_confidence is not None and report.task_state_confidence >= .70
                     and report.right_held is True and report.mug_off_table is True
                     and self._corroborates_lift(observation, report))
        if qualifies:
            self._corroborated_count += 1
            self._last_lift_sequence = observation.sequence
        elif (self._lift_corroboration["reason"] in
              ("mug_tcp_motion_inconsistent", "association_or_clearance_failed", "table_plane_inconsistent",
               "mug_on_table_or_clearance_insufficient")
              or self._last_lift_sequence is None
              or observation.sequence-self._last_lift_sequence > self.lift_evidence_max_gap):
            self._corroborated_count = 0
            self._mug_track = None
            self._lift_motion_supported = False
        if not qualifies:
            self._mug_body_track = None
        self._lift_corroboration["last_positive_sequence"] = self._last_lift_sequence

    def observe(self, observation):
        camera = (observation.camera_frames or {}).get("perception_camera")
        image = observation.images.get("perception_camera")
        if camera is None or image is None:
            raise ValueError("perception_camera_rgbd_required")
        key = (observation.sequence, sha256(image.tobytes()).hexdigest(), calibration_sha256(camera))
        if key == self._key and self._lift_identity == self.episode_id:
            return self._report
        if self.requests >= self.maximum_requests:
            raise RuntimeError("public_grounding_request_budget_exhausted")
        self.store.save(observation)
        content = []
        for name, rgb in observation.images.items():
            label = f"Current camera={name}; sequence={observation.sequence}; pixels={rgb.shape[1]}x{rgb.shape[0]}"
            frame = observation.camera_frames.get(name)
            if frame is not None and observation.measured_tcp_poses:
                for arm, pose in observation.measured_tcp_poses.items():
                    point = frame.world_to_camera @ np.array([*pose[:3], 1.])
                    if point[2] <= 0:
                        continue
                    pixel = frame.intrinsic @ point[:3]
                    u, v = pixel[:2]/pixel[2]
                    if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                        label += f"; robot_{arm}_measured_TCP_uv=({u:.1f},{v:.1f}) (not a contact measurement)"
            content += [{"type": "text", "text": label}, image_block(png_bytes(rgb))]
        self.requests += 1
        try:
            data = request_tool_input(self.client, model=self.model, system=PUBLIC_PROMPT,
                tool={"name": "report_public_hanging", "description": "Visible task state and hanging landmarks only",
                      "input_schema": VisibleHangingReport.model_json_schema()}, content=content, max_tokens=4096)
            report = VisibleHangingReport.model_validate(data)
            self._cache_rack(observation, report)
        except Exception as exc:
            self.log(dict(event_type="public_grounding_failure", observation_sequence=observation.sequence,
                          failure=sanitized_failure(exc)))
            raise
        same_episode = self._lift_identity == self.episode_id
        self._update_temporal_lift(observation, report, key)
        self._previous = self._report if (self._key is not None and self._key[0] < observation.sequence
                                        and observation.sequence-self._key[0] <= self.lift_evidence_max_gap
                                        and same_episode and self._key[2] == key[2]) else None
        self._key, self._report = key, report
        self.log(dict(event_type="public_hanging_proposal", episode_id=self.episode_id,
                      observation_sequence=observation.sequence, rgb_sha256=key[1], calibration_sha256=key[2],
                      corroborated_lift_count=self._corroborated_count,
                      lift_corroboration=dict(self._lift_corroboration),
                      semantic_source="model_visual_proposal_not_calibrated", report=report.model_dump(mode="json")))
        return report

    def _confirmed(self, report, attribute):
        # Require two separate current-scene observations; never count rereads.
        previous = self._previous
        return bool(report.task_state_confidence is not None and report.task_state_confidence >= .85
                    and previous is not None and previous.task_state_confidence is not None
                    and previous.task_state_confidence >= .85
                    and getattr(report, attribute) is True and getattr(previous, attribute) is True)

    def evaluate(self, name, observation, *, arm="auto"):
        report = self.observe(observation)
        value = None
        state_confident = report.task_state_confidence is not None and report.task_state_confidence >= .85
        if name == "mug_lifted":
            value = True if self._confirmed(report, "mug_off_table") and (
                self._confirmed(report, "left_held") or self._confirmed(report, "right_held")) else None
            if state_confident and report.mug_off_table is False:
                value = False
        if name in ("right_mug_lifted", "handoff_confirmed"):
            attributes = ("right_held", "mug_off_table")
            if name == "handoff_confirmed":
                attributes += ("transfer_complete",)
            value = True if all(self._confirmed(report, attribute) for attribute in attributes) else None
            if state_confident and any(getattr(report, attribute) is False for attribute in attributes):
                value = False
        temporal = self._corroborated_count >= 3 and self._lift_motion_supported
        if temporal and name in ("right_mug_lifted", "mug_lifted"):
            value = True
        if name in ("right_mug_lifted", "handoff_confirmed") and self._lift_corroboration["reason"] in (
                "grasp_contradiction_or_gripper_open", "mug_tcp_motion_inconsistent",
                "mug_on_table_or_clearance_insufficient", "table_plane_inconsistent",
                "association_or_clearance_failed"):
            value = False if state_confident and report.right_held is False else None
        return PredicateResult(name=name, value=value, source="perception", observation_sequence=observation.sequence,
                               reason=(LIFT_PERSISTENCE_REASON if temporal and value is True
                                       and self._lift_corroboration["status"] != "confirmed" else
                                       "three_corroborated_public_reports" if temporal and value is True
                                       and name in ("right_mug_lifted", "mug_lifted") else
                                       "two_public_visual_reports" if value is True else
                                       "public_visual_contradiction" if value is False else "public_evidence_unknown"))

    def lift_current(self, observation):
        """Whether this exact observation corroborated the right-hand lift.

        ``evaluate`` may keep a lift True for a bounded gap after the last
        corroborated frame; starting a takeover needs this frame's own evidence.
        """
        result = self.evaluate("right_mug_lifted", observation, arm="right")
        return result.value is True and result.reason != LIFT_PERSISTENCE_REASON

    def planner_summary(self, observation):
        report = self.observe(observation)
        return dict(stage=report.stage, visual_evidence=report.evidence,
                    evidence_mode="public_rgbd_model_proposal", confidence=report.confidence,
                    task_state_confidence=report.task_state_confidence,
                    body_pose_tracking=self._lift_corroboration.get("body_pose_tracking",
                        dict(status="unknown", reason="body_tracking_not_evaluated")))

    @staticmethod
    def needs_geometry_crop(report):
        return (report.confidence < .85 or not report.complete_visible_opening
                or report.contour_px is None or report.material_px is None
                or report.peg_keypoints_px is None)

    def propose(self, observation, measured_tcp, *, episode_id):
        if episode_id != self.episode_id:
            return ProposalResult("rejected", "hanging_evidence_wrong_episode")
        report = self.observe(observation)
        if (report.task_state_confidence is not None and report.task_state_confidence >= .85
                and report.right_held is False):
            return ProposalResult("rejected", "right_arm_not_holding_mug")
        if self.evaluate("right_mug_lifted", observation, arm="right").value is not True:
            return ProposalResult("unknown", "public_hanging_entry_not_confirmed")
        if self.fixture is None and self._current_rack_reference(observation) is None:
            # observe() normally offers each new frame's peg already; a crop
            # below carries no peg fields, so offer the full-frame report here.
            self._cache_rack(observation, report)
        if self.needs_geometry_crop(report):
            report = self.crop_geometry(observation, report)
        if (report.confidence < MINIMUM_PROPOSAL_CONFIDENCE or not report.complete_visible_opening
                or report.contour_px is None or report.material_px is None or report.body_polygon_px is None):
            return ProposalResult("unknown", "complete_visible_aperture_unavailable")
        camera = observation.camera_frames["perception_camera"]
        aperture = estimate(camera, report.contour_px, report.material_px)
        if aperture.get("reason") == "material_depth_discontinuity":
            try:
                material = inset_material_samples(camera, report.contour_px, report.material_px)
                report = report.model_copy(update={"material_px": material})
                aperture = estimate(camera, report.contour_px, material)
            except ValueError as exc:
                return ProposalResult("rejected", str(exc))
        if aperture["status"] != "candidate_not_motion_validated":
            return ProposalResult("rejected", aperture["reason"])
        agreement = self._aperture_agrees(observation, measured_tcp, aperture)
        self.log(dict(event_type="aperture_two_frame_agreement", geometry_confidence=report.confidence,
                      self_confirmed=report.confidence >= SELF_CONFIRMED_CONFIDENCE, **agreement))
        if report.confidence < SELF_CONFIRMED_CONFIDENCE and agreement["status"] != "agreed":
            return ProposalResult("unknown", "aperture_awaiting_two_frame_agreement")
        try:
            cached_rack = self._current_rack_reference(observation)
            if self.fixture is None and cached_rack is None:
                # Only a validated, kept rack reference supplies the peg; the
                # current frame's own peg qualifies through _cache_rack first.
                return ProposalResult("unknown", "validated_rack_reference_unavailable")
            rack_reference = cached_rack
            fixture = self._fixture_for(observation)
            if isinstance(self.fixture, VerifiedFixtureAssets) and not report.opposite_handle_faces_visible:
                registration = register_public_fixture(observation, report, fixture,
                    episode_id=episode_id, rack_reference=rack_reference,
                    reference_episode_id=self._rack_reference_episode if rack_reference is not None else None)
                self.log(dict(event_type="public_fixture_registration", observation_sequence=observation.sequence,
                              status=registration.status, reason=registration.reason, evidence=registration.evidence))
                if registration.status != "confirmed":
                    return ProposalResult(registration.status, registration.reason)
                fixture = registration.fixture
            bounds = measured_bounds(observation, report, aperture, rack_reference=rack_reference,
                                     fixture=fixture, episode_id=episode_id)
        except ValueError as exc:
            return ProposalResult("unknown", str(exc))
        image = observation.images["perception_camera"]
        rack_observation, rack_proposal = rack_reference if rack_reference is not None else (observation, report)
        rack = self._rack_report(rack_observation, rack_proposal)
        if "public_registration" in bounds:
            # The runtime/cache recomputes the metric frame from these original
            # pixels; do not trust a supplied world pose as registration proof.
            rack.update(centerline_from_cross_section=True, peg_width_px=rack_proposal.peg_width_px)
        return ProposalResult("confirmed", "visible_geometry_proposal", PublicHangingProposal(
            episode_id, observation.sequence, "perception_camera", sha256(image.tobytes()).hexdigest(),
            calibration_sha256(camera), "hanging", True, report.evidence, "public_rgbd", True, True,
            report.contour_px, report.material_px, report.body_polygon_px, rack, bounds,
            rack_observation if rack_reference is not None else None))
