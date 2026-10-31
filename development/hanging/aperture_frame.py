"""Offline visible-aperture hypothesis; deliberately not a motion target API."""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from harness.perception.pose_grounding import _rotation_to_quaternion
from harness.perception.replay import load_public_observation
from harness.perception.rgbd import RGBDCamera, calibration_sha256


@dataclass(frozen=True)
class Limits:
    """Exploratory rejection limits, not validated peg-clearance tolerances."""

    minimum_samples: int = 12
    minimum_plane_span_m: float = .008
    maximum_depth_gap_m: float = .06
    maximum_plane_residual_m: float = .003
    maximum_normal_sensitivity_deg: float = 5.
    maximum_center_sensitivity_m: float = .005
    pixel_uncertainty_px: float = 1.


def _pixels(values, minimum, shape):
    points = np.asarray(values, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
            or not np.isfinite(points).all()):
        raise ValueError("insufficient_or_invalid_pixels")
    height, width = shape
    if (np.any(points < 1) or np.any(points[:, 0] >= width-1)
            or np.any(points[:, 1] >= height-1)):
        raise ValueError("pixels_require_in_bounds_neighborhoods")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("duplicate_pixels")
    return points


def _plane(points, minimum_span):
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
    if singular[1] / np.sqrt(len(points)) < minimum_span:
        raise ValueError("plane_support_degenerate")
    return center, axes[-1]


def _intersections(camera, pixels, plane_center, normal):
    camera_to_world = np.linalg.inv(camera.world_to_camera)
    origin = camera_to_world[:3, 3]
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(camera.intrinsic).T
    rays = rays @ camera_to_world[:3, :3].T
    denominators = rays @ normal
    if np.any(np.abs(denominators) < .1):
        raise ValueError("aperture_plane_view_too_oblique")
    distances = ((plane_center-origin) @ normal) / denominators
    if np.any(distances <= 0):
        raise ValueError("aperture_plane_behind_camera")
    return origin + rays * distances[:, None]


def _centroid(camera, contour, center, normal):
    vertices = _intersections(camera, contour, center, normal)
    # A visible opening must be supplied as an ordered convex polygon. Reject
    # concave/occluded/self-crossing contours rather than completing them.
    edges = np.roll(vertices, -1, axis=0)-vertices
    turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
    sides = np.cross(edges[:, None, :], vertices[None, :, :]-vertices[:, None, :]) @ normal
    if not ((np.all(turns > 1e-12) and np.all(sides >= -1e-12))
            or (np.all(turns < -1e-12) and np.all(sides <= 1e-12))):
        raise ValueError("contour_requires_ordered_strictly_convex_visible_polygon")
    reference = vertices.mean(axis=0)
    cross = np.cross(vertices-reference, np.roll(vertices, -1, axis=0)-reference) @ normal
    if abs(cross.sum()) < 1e-8:
        raise ValueError("aperture_area_too_small")
    triangles = (reference + vertices + np.roll(vertices, -1, axis=0))/3
    return (triangles * cross[:, None]).sum(axis=0)/cross.sum()


def _inside_opening(pixels, contour):
    """Ray-crossing membership, including the polygon boundary."""
    pixels = np.asarray(pixels, dtype=float)
    inside = np.zeros(len(pixels), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(contour, np.roll(contour, -1, axis=0)):
        edge = b-a
        relative = pixels-a
        cross = edge[0]*relative[:, 1]-edge[1]*relative[:, 0]
        along = relative @ edge
        boundary |= ((np.abs(cross) < 1e-9) & (along >= 0) & (along <= edge @ edge))
        if b[1] != a[1]:
            crosses = (a[1] > pixels[:, 1]) != (b[1] > pixels[:, 1])
            intersection = a[0] + (pixels[:, 1]-a[1])*edge[0]/edge[1]
            inside ^= crosses & (pixels[:, 0] < intersection)
    return inside | boundary


def _connected_surface_gap(camera, u, v, contour):
    """Compare only the selected continuous surface, excluding the opening.

    Limited to explicitly reviewed aperture samples. The generic RGB-D gap
    check is unchanged. A 10 mm neighbor jump prevents connecting to the nearer
    body across the observed depth edge; it is not a larger allowed depth gap.
    """
    x, y = np.rint((u, v)).astype(int)
    height, width = camera.depth_m.shape
    left, right = max(0, x-8), min(width, x+9)
    top, bottom = max(0, y-8), min(height, y+9)
    patch = camera.depth_m[top:bottom, left:right]
    xs, ys = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
    opening = _inside_opening(np.column_stack((xs.ravel(), ys.ravel())), contour).reshape(patch.shape)
    # All nine samples used by unprojection must be on reviewed material,
    # never a contour edge or the empty opening, even if background is smooth.
    if opening[y-top-1:y-top+2, x-left-1:x-left+2].any():
        raise ValueError("material_patch_overlaps_opening")
    seen = {(y-top, x-left)}
    queue = deque(seen)
    while queue:
        sy, sx = queue.popleft()
        for ny, nx in ((sy-1, sx), (sy+1, sx), (sy, sx-1), (sy, sx+1)):
            if (0 <= ny < patch.shape[0] and 0 <= nx < patch.shape[1]
                    and (ny, nx) not in seen and not opening[ny, nx]
                    and patch[ny, nx] > 0
                    and abs(float(patch[ny, nx]-patch[sy, sx])) <= .01):
                seen.add((ny, nx))
                queue.append((ny, nx))
    if len(seen) < 9:
        raise ValueError("material_surface_support_insufficient")
    return max(0., camera.sample_depth(u, v, radius=1)
               - min(float(patch[sy, sx]) for sy, sx in seen))


def _material_points(camera, pixels, limits, contour=None):
    points = []
    for u, v in pixels:
        if contour is not None and _inside_opening([[u, v]], contour)[0]:
            raise ValueError("material_sample_inside_opening")
        x, y = np.rint((u, v)).astype(int)
        patch = camera.depth_m[y-1:y+2, x-1:x+2]
        if patch.shape != (3, 3) or np.any(patch <= 0):
            raise ValueError("material_depth_missing")
        if np.ptp(patch) > limits.maximum_depth_gap_m:
            raise ValueError("material_depth_discontinuity")
        gap = (camera.local_depth_gap(u, v, search_radius=8) if contour is None
               else _connected_surface_gap(camera, u, v, contour))
        if gap > limits.maximum_depth_gap_m:
            raise ValueError("material_depth_discontinuity")
        points.append(camera.unproject(u, v, radius=1))
    return np.asarray(points)


def aperture_dimensions(camera, contour, plane_center, normal):
    """Raw polygon dimensions on the fitted plane, not certified clearance.

    Solve the convex polygon's largest inscribed circle by enumerating active
    triples of edge constraints. This uses only visible contour rays; peg size
    and systematic measurement error are explicitly not filled in.
    """
    center = _centroid(camera, contour, plane_center, normal)
    vertices = _intersections(camera, contour, plane_center, normal)
    u = vertices[1]-vertices[0]
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    polygon = np.column_stack(((vertices-center) @ u, (vertices-center) @ v))
    edges = np.roll(polygon, -1, axis=0)-polygon
    signed_area = np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                         - polygon[:, 1]*np.roll(polygon[:, 0], -1))
    inward = (np.column_stack((-edges[:, 1], edges[:, 0]))
              / np.linalg.norm(edges, axis=1)[:, None] * np.sign(signed_area))
    constraints = np.column_stack((-inward, np.ones(len(polygon))))
    bounds = np.sum(-inward*polygon, axis=1)
    best = None
    for indices in combinations(range(len(polygon)), 3):
        selected = list(indices)
        try:
            circle = np.linalg.solve(constraints[selected], bounds[selected])
        except np.linalg.LinAlgError:
            continue
        if (circle[2] > 0 and np.all(constraints @ circle <= bounds+1e-10)
                and (best is None or circle[2] > best[2])):
            best = circle
    if best is None:
        raise ValueError("aperture_inscribed_circle_unavailable")
    return dict(centroid_world=center.tolist(),
        centroid_inscribed_radius_m=float(bounds.min()),
        maximum_inscribed_circle_radius_m=float(best[2]),
        circle_center_world=(center+u*best[0]+v*best[1]).tolist(),
        clearance_validated=False, peg_radius_m=None,
        systematic_measurement_error_bound_m=None)


def estimate(camera, contour_px, material_px, *, limits=Limits(), allow_demo_override=False):
    """Fit visible material, intersect contour rays, and quantify sensitivity.

    Inputs must be reviewed, complete visible contour vertices and separate
    visible handle-material samples surrounding the opening. A plane through
    the sampled front surface is an explicit hypothesis, not a hidden midplane.
    """
    result = dict(status="rejected", motion_authorized=False, support="unknown",
        limits=asdict(limits), assumptions=[
            "Input contour is complete, ordered, convex and visibly reviewed.",
            "Material samples belong to the visible handle, not body or gripper.",
            "Visible sampled material approximates one plane; no hidden midplane is inferred.",
            "Calibration and depth scale are supplied correctly, not independently verified.",
        ], uncertainty=dict(kind="deterministic_sensitivity_not_confidence_interval",
            systematic_depth_calibration_error="unknown", physical_clearance="unknown"))
    try:
        if not isinstance(camera, RGBDCamera):
            raise ValueError("rgbd_camera_required")
        numeric = np.asarray(list(asdict(limits).values()), dtype=float)
        if (not np.isfinite(numeric).all() or np.any(numeric <= 0)
                or type(limits.minimum_samples) is not int or limits.minimum_samples < 6):
            raise ValueError("invalid_estimator_limits")
        contour = _pixels(contour_px, 8, camera.depth_m.shape)
        pixels = _pixels(material_px, limits.minimum_samples, camera.depth_m.shape)
        # Duplicate subpixel locations rounding to one sample cannot create
        # independent support or inflate the leave-one-out test.
        if len(np.unique(np.rint(pixels), axis=0)) != len(pixels):
            raise ValueError("duplicate_depth_samples")
        points = _material_points(camera, pixels, limits, contour)
        center, normal = _plane(points, limits.minimum_plane_span_m)
        camera_to_world = np.linalg.inv(camera.world_to_camera)
        if normal @ (camera_to_world[:3, 3]-center) < 0:
            normal = -normal
        residual = float(np.max(np.abs((points-center) @ normal)))
        result["uncertainty"]["maximum_plane_residual_m"] = residual
        if residual > limits.maximum_plane_residual_m:
            raise ValueError("visible_material_not_planar")
        result["geometry_diagnostics"] = aperture_dimensions(camera, contour, center, normal)
        # Surrounding evidence in image space; one visible arc is insufficient.
        angles = np.sort(np.arctan2(*(pixels-contour.mean(axis=0)).T[::-1]))
        largest_gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
        result["uncertainty"]["material_angular_gap_deg"] = float(np.degrees(largest_gap))
        singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
        condition = float(singular[0]/singular[1])
        spread = 360.-float(np.degrees(largest_gap))
        result["uncertainty"].update(visible_arch_spread_deg=spread, plane_condition_number=condition)
        # A visible C-shaped arch need not enclose the hole with material.
        # Require 150 degrees of observed spread and a conditioned plane fit.
        if spread < 150. or condition > 10.:
            raise ValueError("material_does_not_surround_opening")
        aperture = _centroid(camera, contour, center, normal)
        edges = np.roll(contour, -1, axis=0)-contour
        relative = pixels[None, :, :]-contour[:, None, :]
        sides = edges[:, None, 0]*relative[:, :, 1]-edges[:, None, 1]*relative[:, :, 0]
        if np.any(np.all(sides >= 0, axis=0) | np.all(sides <= 0, axis=0)):
            raise ValueError("material_sample_inside_opening")
        normal_change = center_change = 0.
        for index in range(len(points)):
            # Removing an extremal sample reduces RMS span. The full fit above
            # must meet the original span; probes still require half that span.
            c, n = _plane(np.delete(points, index, axis=0), limits.minimum_plane_span_m/2)
            normal_change = max(normal_change, float(np.degrees(np.arccos(
                np.clip(abs(n @ normal), 0, 1)))))
            center_change = max(center_change, float(np.linalg.norm(
                _centroid(camera, contour, c, n)-aperture)))
        # Shift the entire contour by each one-pixel cardinal offset. This is
        # registration sensitivity, not an exhaustive contour-error bound.
        for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = contour + np.asarray(delta)*limits.pixel_uncertainty_px
            _pixels(shifted, 8, camera.depth_m.shape)
            center_change = max(center_change, float(np.linalg.norm(
                _centroid(camera, shifted, center, normal)-aperture)))
            shifted_material = pixels + np.asarray(delta)*limits.pixel_uncertainty_px
            _pixels(shifted_material, limits.minimum_samples, camera.depth_m.shape)
            try:
                shifted_points = _material_points(camera, shifted_material, limits, contour)
            except ValueError as exc:
                if not allow_demo_override:
                    raise
                # The actual selected patches passed above. An invalid probe
                # is unresolved annotation uncertainty, never a depth exemption
                # for the current observation's material samples.
                result["uncertainty"].setdefault("invalid_pixel_probes", []).append(
                    dict(shift=list(delta), reason=str(exc)))
                continue
            c, n = _plane(shifted_points, limits.minimum_plane_span_m/2)
            normal_change = max(normal_change, float(np.degrees(np.arccos(
                np.clip(abs(n @ normal), 0, 1)))))
            center_change = max(center_change, float(np.linalg.norm(
                _centroid(camera, contour, c, n)-aperture)))
        result["uncertainty"].update(maximum_normal_sensitivity_deg=normal_change,
            center_sensitivity_m=center_change,
            tested_pixel_error="four rigid contour shifts and four rigid material shifts; not exhaustive")
        if normal_change > limits.maximum_normal_sensitivity_deg + 1e-12:
            raise ValueError("plane_normal_sensitive")
        if center_change > limits.maximum_center_sensitivity_m + 1e-12:
            raise ValueError("aperture_center_sensitive")
        up = camera_to_world[:3, :3] @ np.array([0., -1., 0.])
        up -= normal*(up @ normal)
        if np.linalg.norm(up) < .1:
            raise ValueError("frame_roll_ambiguous")
        up /= np.linalg.norm(up)
        rotation = np.column_stack((normal, up, np.cross(normal, up)))
        result.update(status="candidate_not_motion_validated",
            frame_pose_world_wxyz=[*aperture.tolist(), *_rotation_to_quaternion(rotation).tolist()],
            frame_convention="X=camera-facing plane normal; Y=projected image-up; Z=X cross Y",
            insertion_direction="unknown", sample_count=len(points))
        result["allow_entry660_demo_override"] = bool(allow_demo_override)
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
    return result


def inspect_archive(archive, annotation=None, *, allow_demo_override=False):
    """Bind a reviewed annotation to exact archive bytes; never auto-complete it."""
    archive = Path(archive)
    digest = sha256(archive.read_bytes()).hexdigest()
    observation = load_public_observation(archive)
    rejected = dict(status="rejected", motion_authorized=False,
                    archive_sha256=digest, sequence=observation.sequence)
    if annotation is None:
        return {**rejected, "reason": "reviewed_contour_and_material_samples_required"}
    if (annotation.get("archive_sha256") != digest or annotation.get("reviewed") is not True
            or annotation.get("sequence") != observation.sequence
            or annotation.get("complete_visible_contour") is not True):
        return {**rejected, "reason": "annotation_unreviewed_incomplete_or_wrong_archive"}
    camera = (observation.camera_frames or {}).get(annotation.get("camera"))
    result = estimate(camera, annotation.get("contour_px"), annotation.get("material_px"),
                      allow_demo_override=allow_demo_override)
    return {**result, "archive_sha256": digest, "sequence": observation.sequence,
            "camera": annotation.get("camera"),
            "rgb_sha256": (sha256(observation.images[annotation["camera"]].tobytes()).hexdigest()
                           if annotation.get("camera") in getattr(observation, "images", {}) else None),
            "calibration_sha256": calibration_sha256(camera) if camera is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text()) if args.annotation else None
    report = inspect_archive(args.archive, annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
