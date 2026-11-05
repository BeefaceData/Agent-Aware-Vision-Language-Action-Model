"""Category-level object poses from public RGB-D: shape fits, not mesh lookups.

Every input is public: SAM 3 masks, same-frame depth and calibration. An object
is described by an ``ObjectSpec`` (size ranges of its cylindrical body), so a
new cylindrical object needs a spec, not new fitting code. A fit that cannot be
trusted returns ``None`` with a searchable reason instead of a guess.
"""

from dataclasses import dataclass

import numpy as np

from .rgbd import RGBDCamera


@dataclass(frozen=True)
class ObjectSpec:
    """What a category looks like to the fitter: a cylinder with bounded size."""

    label: str
    radius_m: tuple[float, float]
    height_m: tuple[float, float]
    rests_on_table: bool


# RoboTwin's ten hanging-task mugs have body radii 3.7-6.0 cm and heights
# 7.5-12.2 cm (meshes with scale applied); the ranges add a margin so other
# mugs fit too.
MUG = ObjectSpec("mug", radius_m=(0.03, 0.07), height_m=(0.06, 0.14), rests_on_table=True)


@dataclass(frozen=True)
class ObjectPose:
    label: str
    # The base centre when ``base_known``; otherwise the middle of the visible body.
    position: tuple[float, float, float]
    # Unit vector along the body. It points base -> top only when ``base_known``.
    axis: tuple[float, float, float]
    base_known: bool
    radius_m: float
    quality: float          # share of the chosen fit's points the cylinder explains, 0..1
    # Only a trusted pose may confirm anything (an upright mug, an anchor). An
    # untrusted one is still evidence against: a mug it reads as lying down
    # must not be released.
    trusted: bool
    reason: str
    cameras: tuple[str, ...]
    observation_sequence: int

    def tilt_deg(self):
        """Angle between the axis line and world up (0 upright, 90 lying down)."""
        return float(np.degrees(np.arccos(min(1.0, abs(self.axis[2])))))


@dataclass(frozen=True)
class CylinderFit:
    axis: np.ndarray             # unit vector
    centre: np.ndarray           # a point on the axis line
    radius_m: float
    extent: tuple[float, float]  # visible span of the points along the axis, from ``centre``
    inlier_fraction: float       # share of points on the fitted wall or a flat end
    points: int
    kind: str                    # "wall" or "end_cap"


# Normals within this angle of perpendicular to the axis count as wall.
WALL_TOLERANCE_DEG = 12.0
# A view is an end cap when most normals agree this closely and the points are flat.
END_CAP_NORMAL_DEG = 15.0
END_CAP_FRACTION = 0.8
END_CAP_FLATNESS_M = 0.004
END_CAP_ROUNDNESS = 0.6
SURFACE_TOLERANCE_M = 0.004
MIN_FIT_POINTS = 60
NORMAL_STEP_PX = 2
TABLE_CLEARANCE_M = 0.01
# On seeds 0-4 every per-camera axis more than 20 deg off (four held mugs, half
# hidden by the gripper) explained 0.38-0.44 of its points; all 159 good fits
# explained 0.70 or more. The floor sits in that gap.
MIN_INLIER_FRACTION = 0.60
# Two cameras see the same body when their fitted axes agree this well.
SAME_BODY_AXIS_DEG = 15.0
SAME_BODY_LINE_M = 0.03
# Same cut-off as the capture labels' TOPPLED_TILT_DEG.
UPRIGHT_MAX_TILT_DEG = 45.0
# A visible lower end this close to the table top stands on the table.
STANDING_GAP_M = 0.03


def object_pose(surfaces, spec, *, table_height_m, observation_sequence):
    """Fit each camera's view of one object and combine the fits into a pose.

    ``surfaces`` maps a camera name to the ``(points, normals)`` of its mask.
    Trusted fits are combined; trusted cameras that disagree make the pose
    unknown rather than picking one. With no trusted fit, the best untrusted
    one is returned with ``trusted=False``. Returns ``(ObjectPose, reason)`` or
    ``(None, reason)``.
    """
    fits, failures = {}, []
    for camera, (points, normals) in surfaces.items():
        fit, reason = fit_cylinder(points, normals, spec)
        if fit is None:
            failures.append(reason)
        else:
            fits[camera] = fit
    if not fits:
        return None, max(set(failures), key=failures.count) if failures else "object_not_seen"
    trusted = {camera: fit for camera, fit in fits.items() if fit.inlier_fraction >= MIN_INLIER_FRACTION}
    chosen = trusted or fits
    best = max(chosen.values(), key=lambda fit: fit.inlier_fraction * fit.points)
    if trusted and not all(_same_body(best, fit) for fit in trusted.values()):
        return None, "cameras_disagree"
    if trusted:
        reason = f"{best.kind}_fit_{len(trusted)}_camera" + ("s" if len(trusted) > 1 else "")
        cameras = tuple(sorted(trusted))
    else:
        reason = "cylinder_fit_inliers_low"
        cameras = (next(camera for camera, fit in fits.items() if fit is best),)
    position, axis, base_known = _placement(best, spec, table_height_m)
    return ObjectPose(spec.label, tuple(float(v) for v in position), tuple(float(v) for v in axis),
                      base_known, best.radius_m, best.inlier_fraction, bool(trusted), reason, cameras,
                      observation_sequence), reason


def _same_body(first, second):
    if abs(first.axis @ second.axis) < np.cos(np.radians(SAME_BODY_AXIS_DEG)):
        return False
    return _distance_to_axis(second.centre[None], first.centre, first.axis)[0] <= SAME_BODY_LINE_M


def _placement(fit, spec, table_height_m):
    """Position, axis and whether the base is known, from one fit.

    An upright body whose visible lower end reaches the table stands on it, so
    its base centre is where the axis meets the table top, even though the base
    itself is hidden. Otherwise the base end is not known yet.
    """
    axis, (low, high) = fit.axis, fit.extent
    upright = abs(axis[2]) >= np.cos(np.radians(UPRIGHT_MAX_TILT_DEG))
    if upright and axis[2] < 0:
        axis, (low, high) = -axis, (-high, -low)
    bottom = fit.centre + low * axis
    if (upright and spec.rests_on_table and table_height_m is not None
            and bottom[2] - table_height_m <= STANDING_GAP_M):
        base = fit.centre + (table_height_m - fit.centre[2]) / axis[2] * axis
        return base, axis, True
    return fit.centre + (low + high) / 2 * axis, axis, False


def masked_surface(frame: RGBDCamera, mask, *, table_height_m=None):
    """World points and normals of one mask, ready for a shape fit.

    Points within ``TABLE_CLEARANCE_M`` of the table top are dropped: they are
    table seen past the object's edge, or a shadow the detector mistook for
    the object.
    """
    points, normals = frame.surface_points_and_normals(mask, step_px=NORMAL_STEP_PX)
    if table_height_m is None:
        return points, normals
    above = points[:, 2] > table_height_m + TABLE_CLEARANCE_M
    return points[above], normals[above]


def fit_cylinder(points, normals, spec, *, rng=None, iterations=200):
    """Fit a cylinder axis, centre line and radius; ``(None, reason)`` when not supported."""
    points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
    if len(points) < MIN_FIT_POINTS:
        return None, "too_few_surface_points"
    cap = _end_cap(points, normals, spec)
    if cap is not None:
        return cap, "end_cap"
    inliers = _wall_inliers(points, normals, spec, rng or np.random.default_rng(0), iterations)
    if inliers is None:
        return None, "no_consistent_cylinder"
    axis = _refined_axis(normals[inliers])
    centre, radius, on_circle = _centre_line(points[inliers], normals[inliers], axis)
    if centre is None:
        return None, "cylinder_centre_unresolved"
    if not spec.radius_m[0] <= radius <= spec.radius_m[1]:
        return None, "cylinder_radius_out_of_range"
    along = (points[inliers][on_circle] - centre) @ axis
    if along.max() - along.min() > spec.height_m[1]:
        return None, "cylinder_longer_than_object"
    explained = on_circle.sum() + _flat_end_points(points, normals, centre, axis, radius)
    return CylinderFit(axis, centre, radius, (float(along.min()), float(along.max())),
                       float(explained / len(points)), len(points), "wall"), "wall"


def _end_cap(points, normals, spec):
    """A flat disc of mug radius: the base or opening seen end-on, so the axis is its normal."""
    mean = normals.mean(axis=0)
    mean /= np.linalg.norm(mean)
    agree = normals @ mean >= np.cos(np.radians(END_CAP_NORMAL_DEG))
    if agree.mean() < END_CAP_FRACTION:
        return None
    centroid = points[agree].mean(axis=0)
    offsets = points[agree] - centroid
    if np.percentile(np.abs(offsets @ mean), 90) > END_CAP_FLATNESS_M:
        return None
    in_plane = offsets - np.outer(offsets @ mean, mean)
    # A disc spreads about evenly in both directions; a flat stripe along a
    # lying mug's side does not.
    spreads = np.sqrt(np.linalg.eigvalsh(in_plane.T @ in_plane / len(in_plane))[1:])
    if spreads[0] < END_CAP_ROUNDNESS * spreads[1]:
        return None
    radius = float(np.percentile(np.linalg.norm(in_plane, axis=1), 95))
    if not spec.radius_m[0] <= radius <= spec.radius_m[1]:
        return None
    return CylinderFit(mean, centroid, radius, (0.0, 0.0), float(agree.mean()), len(points), "end_cap")


def _wall_inliers(points, normals, spec, rng, iterations):
    """RANSAC over whole cylinders: two wall samples fix the axis, centre and radius.

    Scoring a hypothesis by its normals alone is not enough: from an oblique
    view, a horizontal axis is also perpendicular to an upright mug's front wall
    and its flat rim. Requiring the points to lie on the cylinder surface too
    rejects that reading.
    """
    limit = np.sin(np.radians(WALL_TOLERANCE_DEG))
    best = None
    for _ in range(iterations):
        first, second = rng.choice(len(normals), size=2, replace=False)
        axis = np.cross(normals[first], normals[second])
        length = np.linalg.norm(axis)
        if length < 0.3:  # nearly parallel normals give an unstable cross product
            continue
        axis /= length
        centre = _normal_lines_meet(points[[first, second]], normals[[first, second]], axis)
        if centre is None:
            continue
        radius = float(np.mean(_distance_to_axis(points[[first, second]], centre, axis)))
        if not spec.radius_m[0] <= radius <= spec.radius_m[1]:
            continue
        inliers = ((np.abs(normals @ axis) < limit)
                   & (np.abs(_distance_to_axis(points, centre, axis) - radius) <= SURFACE_TOLERANCE_M))
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < MIN_FIT_POINTS:
        return None
    return best


def _flat_end_points(points, normals, centre, axis, radius):
    """Points on a flat end of the cylinder (its base, or the floor inside the cup)."""
    facing_along = np.abs(normals @ axis) >= np.cos(np.radians(END_CAP_NORMAL_DEG))
    within = _distance_to_axis(points, centre, axis) <= radius + SURFACE_TOLERANCE_M
    return int(np.sum(facing_along & within))


def _refined_axis(normals):
    """The direction most perpendicular to every wall normal."""
    _values, vectors = np.linalg.eigh(normals.T @ normals)
    return vectors[:, 0]


def _distance_to_axis(points, centre, axis):
    offsets = points - centre
    return np.linalg.norm(offsets - np.outer(offsets @ axis, axis), axis=1)


def _normal_lines_meet(points, normals, axis):
    """Where two wall normal lines cross, in the plane across ``axis``; None if parallel."""
    across = np.eye(3) - np.outer(axis, axis)
    directions = normals @ across
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Solve p0 + s d0 = p1 + t d1 in the least-squares sense.
    matrix = np.stack((directions[0], -directions[1]), axis=1)
    if abs(np.linalg.det(matrix.T @ matrix)) < 1e-3:
        return None
    s, _t = np.linalg.lstsq(matrix, (points[1] - points[0]) @ across, rcond=None)[0]
    return (points[0] @ across) + s * directions[0]


def _centre_line(points, normals, axis, *, rounds=3, tolerance_m=0.01):
    """Least-squares meeting point of the wall normal lines, in the plane across the axis.

    Each wall normal points at the axis, so the lines meet at the centre even
    when only a narrow arc of the wall is visible.
    """
    across = np.eye(3) - np.outer(axis, axis)
    flat_points, flat_normals = points @ across, normals @ across
    lengths = np.linalg.norm(flat_normals, axis=1)
    usable = lengths > 0.5
    if usable.sum() < MIN_FIT_POINTS:
        return None, 0.0, usable
    directions = flat_normals[usable] / lengths[usable][:, None]
    flat_points = flat_points[usable]
    keep = np.ones(len(flat_points), dtype=bool)
    centre = radius = None
    for _ in range(rounds):
        projectors = np.eye(3)[None] - directions[keep][:, :, None] * directions[keep][:, None, :]
        system = projectors.sum(axis=0) + np.outer(axis, axis)  # pin the free axial direction
        target = np.einsum("nij,nj->i", projectors, flat_points[keep])
        centre = np.linalg.solve(system, target)
        distances = np.linalg.norm(flat_points - centre, axis=1)
        radius = float(np.median(distances[keep]))
        keep = np.abs(distances - radius) <= tolerance_m
        if keep.sum() < MIN_FIT_POINTS:
            return None, 0.0, usable
    on_circle = np.zeros(len(points), dtype=bool)
    on_circle[np.flatnonzero(usable)[keep]] = True
    return centre, radius, on_circle
