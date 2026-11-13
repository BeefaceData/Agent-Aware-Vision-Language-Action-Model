"""Offline shaft fitting from reviewed public depth regions, never motion authority."""

from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from development.hanging.aperture_frame import Limits, surface_material_components
from harness.perception.object_pose import MIN_FIT_POINTS, masked_surface
from harness.perception.rgbd import calibration_sha256
from harness.perception.replay import load_public_observation


def seeded_shaft_surface(camera, polygon_px, seed_uv):
    """Keep the observed depth component containing the visible shaft seed.

    A smooth background patch can survive normal/depth checks inside an
    imperfect polygon. Never pool it with the shaft. The seed is caller
    evidence, not a semantic prediction or an invented hidden landmark.
    """
    polygon = np.asarray(polygon_px, dtype=float)
    seed = np.asarray(seed_uv, dtype=float)
    height, width = camera.depth_m.shape
    if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
            or seed.shape != (2,) or not np.isfinite(polygon).all()
            or not np.isfinite(seed).all() or np.any(polygon < 0)
            or np.any(polygon[:, 0] >= width) or np.any(polygon[:, 1] >= height)
            or not (0 <= seed[0] < width and 0 <= seed[1] < height)):
        raise ValueError("invalid_shaft_region_or_seed")
    image = Image.new("1", (width, height))
    ImageDraw.Draw(image).polygon([tuple(p) for p in polygon], fill=1)
    mask = np.asarray(image, dtype=bool)
    x, y = np.rint(seed).astype(int)
    if not (0 <= x < width and 0 <= y < height and mask[y, x]):
        raise ValueError("shaft_seed_outside_region")
    if not np.isfinite(camera.depth_m[y, x]) or camera.depth_m[y, x] <= 0:
        raise ValueError("shaft_seed_depth_unavailable")
    for component in surface_material_components(mask, camera.depth_m):
        if np.any(np.all(component == (y, x), axis=1)):
            selected = np.zeros(mask.shape, bool)
            selected[tuple(component.T)] = True
            return masked_surface(camera, selected)
    raise ValueError("shaft_seed_component_unavailable")


def _basis(axis):
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first)))


def _residual(points, center, axis, radius_m):
    delta = points-center
    radial = delta-np.outer(delta @ axis, axis)
    return np.linalg.norm(radial, axis=1)-radius_m


def fit_fixed_radius_cylinder(points, radius_m):
    """Bounded multi-start fit, conditional on the supplied radius assumption.

    Fit positions directly rather than unstable normals on a thin shaft.
    No outliers are trimmed. Returned residuals and conditioning are diagnostics,
    not calibrated uncertainty or verification of the radius/asset identity.
    """
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < MIN_FIT_POINTS
            or not np.isfinite(points).all() or not np.isfinite(radius_m) or radius_m <= 0):
        raise ValueError("invalid_or_insufficient_cylinder_points")
    mean = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-mean, full_matrices=False)
    if singular[1] <= 1e-9:
        raise ValueError("cylinder_surface_degenerate")
    initial_axis = axes[0]
    span = np.ptp((points-mean) @ initial_axis)
    initial_basis = _basis(initial_axis)
    solutions = []
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        axis = initial_axis.copy()
        center = mean+radius_m*(initial_basis @ np.array([np.cos(angle), np.sin(angle)]))
        for _ in range(60):
            delta = points-center
            along = delta @ axis
            radial = delta-np.outer(along, axis)
            unit = radial/np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
            basis = _basis(axis)
            projected = unit @ basis
            # Scale angular columns to displacement over the observed span.
            jacobian = np.column_stack((-projected, -along[:, None]*projected/span))
            residual = _residual(points, center, axis, radius_m)
            update = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            for scale in (1., .5, .25, .125, .0625):
                candidate_center = center+basis @ (update[:2]*scale)
                candidate_axis = axis+basis @ (update[2:]*scale/span)
                candidate_axis /= np.linalg.norm(candidate_axis)
                candidate_center += candidate_axis*((mean-candidate_center) @ candidate_axis)
                candidate_residual = _residual(points, candidate_center, candidate_axis, radius_m)
                if candidate_residual @ candidate_residual <= residual @ residual:
                    break
            else:
                break
            center, axis = candidate_center, candidate_axis
            if np.linalg.norm(update)*scale < 1e-9:
                break
        residual = _residual(points, center, axis, radius_m)
        solutions.append((float(np.mean(residual**2)), center, axis, residual))
    _, center, axis, residual = min(solutions, key=lambda result: result[0])
    return dict(center_world_m=center.tolist(), axis_world=axis.tolist(), radius_m=float(radius_m),
                rms_residual_m=float(np.sqrt(np.mean(residual**2))),
                maximum_residual_m=float(np.max(np.abs(residual))), sample_count=len(points),
                alternatives=[dict(center_world_m=c.tolist(), axis_world=a.tolist(),
                                   maximum_residual_m=float(np.max(np.abs(r))))
                              for _, c, a, r in solutions])


def assess_branch_regions(observation, regions, *, radius_m, radius_basis, limits=Limits()):
    """Compare current calibrated views and five pixel shifts; never enable motion.

    Regions map camera names to reviewed ``polygon_px`` and ``seed_uv``. No
    episode-specific pixels, radius, camera pose or annotation fallback lives
    in this implementation. Missing coverage/semantic evidence remains unknown.
    """
    result = dict(status="unavailable", motion_authorized=False, sequence=observation.sequence,
                  radius_basis=radius_basis, radius_m=radius_m, fits=[],
                  semantics="caller_reviewed_regions_not_autonomous_segmentation",
                  systematic_calibration_error="unknown", physical_clearance="unknown")
    try:
        if not regions or not radius_basis:
            raise ValueError("reviewed_shaft_regions_and_radius_basis_required")
        for shift in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            parts, sources = [], []
            for name, region in regions.items():
                camera = (observation.camera_frames or {}).get(name)
                if camera is None or name not in observation.images:
                    raise ValueError("shaft_camera_rgbd_unavailable")
                surface = seeded_shaft_surface(camera, np.asarray(region["polygon_px"])+shift,
                                               region["seed_uv"])
                parts.append(surface.points)
                sources.append(dict(camera=name, calibration_sha256=calibration_sha256(camera),
                                    points=len(surface.points)))
            fit = fit_fixed_radius_cylinder(np.concatenate(parts), radius_m)
            result["fits"].append(dict(shift_px=list(shift), sources=sources, **fit))
        base = result["fits"][0]
        axis, center = np.asarray(base["axis_world"]), np.asarray(base["center_world_m"])
        result["axis_sensitivity_deg"] = max(float(np.degrees(np.arccos(np.clip(
            abs(axis @ np.asarray(f["axis_world"])), -1, 1)))) for f in result["fits"])
        offsets = [np.asarray(f["center_world_m"])-center for f in result["fits"]]
        result["position_sensitivity_m"] = max(float(np.linalg.norm(v-axis*(v @ axis))) for v in offsets)
        if any(f["maximum_residual_m"] > limits.maximum_plane_residual_m for f in result["fits"]):
            raise ValueError("shaft_surface_residual_exceeds_existing_3mm_bound")
        if result["axis_sensitivity_deg"] > limits.maximum_normal_sensitivity_deg:
            raise ValueError("shaft_axis_sensitive")
        if result["position_sensitivity_m"] > limits.maximum_center_sensitivity_m:
            raise ValueError("shaft_center_sensitive")
        for fit in result["fits"]:
            reference_axis = np.asarray(fit["axis_world"])
            for alternative in fit["alternatives"]:
                if alternative["maximum_residual_m"] > limits.maximum_plane_residual_m:
                    continue
                angle = float(np.degrees(np.arccos(np.clip(abs(reference_axis @
                    np.asarray(alternative["axis_world"])), -1, 1))))
                delta = np.asarray(alternative["center_world_m"])-fit["center_world_m"]
                offset = np.linalg.norm(delta-reference_axis*(delta @ reference_axis))
                if angle > limits.maximum_normal_sensitivity_deg or offset > limits.maximum_center_sensitivity_m:
                    raise ValueError("shaft_fit_ambiguous")
        result.update(status="conditional_surface_fit_not_motion_validated",
                      reason="radius_provenance_semantics_and_fixture_registration_still_required")
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
    return result


def assess_reviewed_archive(archive, annotation):
    """Bind offline regions to their exact archive, images and calibration.

    A declared radius basis is recorded as a hypothesis, never upgraded to an
    asset certificate or permission to feed this diagnostic into recovery.
    """
    rejected = dict(status="unavailable", motion_authorized=False)
    path = Path(archive)
    try:
        if (annotation.get("reviewed") is not True
                or annotation.get("archive_sha256") != sha256(path.read_bytes()).hexdigest()):
            raise ValueError("shaft_annotation_unreviewed_or_wrong_archive")
        observation = load_public_observation(path)
        if annotation.get("sequence") != observation.sequence:
            raise ValueError("shaft_annotation_stale")
        regions = annotation["regions"]
        for name, region in regions.items():
            camera = observation.camera_frames.get(name)
            image = observation.images.get(name)
            if camera is None or image is None:
                raise ValueError("shaft_camera_rgbd_unavailable")
            if (region.get("calibration_sha256") != calibration_sha256(camera)
                    or region.get("rgb_sha256") != sha256(image.tobytes()).hexdigest()):
                raise ValueError("shaft_annotation_camera_or_image_mismatch")
        result = assess_branch_regions(observation, regions, radius_m=annotation["radius_m"],
                                       radius_basis=annotation["radius_basis"])
        result["archive_sha256"] = annotation["archive_sha256"]
        result["radius_provenance_verified"] = False
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {**rejected, "reason": str(exc)}
