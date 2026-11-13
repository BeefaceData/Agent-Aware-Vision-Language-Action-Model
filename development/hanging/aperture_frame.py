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


def depth_components(mask):
    """Four-connected observed raster components; never fill missing pixels."""
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        first = remaining.pop()
        queue, found = deque([first]), [first]
        while queue:
            y, x = queue.popleft()
            for point in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
                    found.append(point)
        yield np.asarray(found, dtype=int)


def surface_material_components(mask, depth):
    """Separate observed material patches; continuity does not prove handle identity.

    Use the same 10 mm neighbor step as the reviewed-material gap check. Never
    pool disconnected patches into a plane or bridge an opening/body depth edge.
    Raster order makes sampling reproducible across repeated offline checks.
    """
    remaining = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    remaining = remaining.copy()
    for y, x in np.argwhere(remaining):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        queue, found = deque([(y, x)]), []
        while queue:
            sy, sx = queue.popleft()
            found.append((sy, sx))
            for ny, nx in ((sy-1, sx), (sy+1, sx), (sy, sx-1), (sy, sx+1)):
                if (0 <= ny < remaining.shape[0] and 0 <= nx < remaining.shape[1]
                        and remaining[ny, nx]
                        and abs(float(depth[ny, nx]-depth[sy, sx])) <= .01):
                    remaining[ny, nx] = False
                    queue.append((ny, nx))
        yield np.asarray(sorted(found), dtype=int)


def body_excluded_material_domain(camera, body_pose, *, observation_sequence):
    """Exclude the fitted cylinder from sample selection, not from the opening.

    This is an offline sampling aid. Geometric fit quality is not semantic
    identity, and points outside the body can still belong to robot fingers.
    No hidden handle surface or collision clearance is established here.
    """
    from harness.perception.object_pose import ObjectPose, SURFACE_TOLERANCE_M

    if (not isinstance(camera, RGBDCamera) or not isinstance(body_pose, ObjectPose)
            or body_pose.label != "mug" or not body_pose.trusted
            or body_pose.observation_sequence != observation_sequence):
        raise ValueError("current_trusted_body_pose_required")
    axis = np.asarray(body_pose.axis, dtype=float)
    origin = np.asarray(body_pose.position, dtype=float)
    radius = body_pose.radius_m
    if (axis.shape != (3,) or origin.shape != (3,)
            or not np.isfinite(axis).all() or not np.isfinite(origin).all()
            or not np.isclose(np.linalg.norm(axis), 1.)
            or radius is None or not np.isfinite(radius) or radius <= 0):
        raise ValueError("invalid_body_cylinder")
    height, width = camera.depth_m.shape
    y, x = np.mgrid[:height, :width]
    rays = np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(camera.intrinsic).T
    transform = np.linalg.inv(camera.world_to_camera)
    points = (rays*camera.depth_m[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    offsets = points-origin
    radial = offsets-np.einsum("...i,i->...", offsets, axis)[..., None]*axis
    return (np.isfinite(camera.depth_m) & (camera.depth_m > 0)
            & (np.linalg.norm(radial, axis=-1) > radius+SURFACE_TOLERANCE_M))


def _observed_convex_contour(hole, left, top):
    """An inscribed convex subset, never a completion of an occluded boundary."""
    from PIL import Image, ImageDraw

    core = hole.copy()
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            core[1:-1, 1:-1] &= hole[1+dy:hole.shape[0]-1+dy, 1+dx:hole.shape[1]-1+dx]
    points = sorted((int(x), int(y)) for y, x in np.argwhere(core))
    if len(points) < 8:
        return None
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    edges = []
    for ordered in (points, reversed(points)):
        edge = []
        for point in ordered:
            while len(edge) >= 2 and cross(edge[-2], edge[-1], point) <= 0:
                edge.pop()
            edge.append(point)
        edges.append(edge[:-1])
    hull = edges[0]+edges[1]
    if len(hull) < 8:
        return None
    # Every filled pixel must already be observed as opening. A convex hull
    # across a concavity or missing depth must not manufacture free space.
    image = Image.new("1", (hole.shape[1], hole.shape[0]))
    ImageDraw.Draw(image).polygon(hull, fill=1)
    if np.any(np.asarray(image, dtype=bool) & ~hole):
        return None
    if len(hull) > 16:
        hull = [hull[i] for i in np.linspace(0, len(hull)-1, 16, dtype=int)]
    return [[float(x+left), float(y+top)] for x, y in hull]


def depth_contrast_aperture(camera, region_xyxy, *, limits=Limits(), exclusion_mask=None,
                            background_opening=False, material_domain=None,
                            separate_material_surfaces=False):
    """Find enclosed recessed-wall candidates, independent of RGB contrast.

    A closed foreground arch and a 5-80 mm observed depth recess are required;
    background-opening mode permits a farther observed background return.
    A mug mouth/gripper can also produce this pattern: semantic identity remains
    unknown, and this offline detector never supplies motion authorization.
    Optional component isolation is conservative: disconnected but coplanar
    arch fragments may lose sufficient plane span. It is not used by recovery.
    """
    result = dict(status="unavailable", reason="no_enclosed_depth_contrast_loop",
                  motion_authorized=False, semantics="unknown", candidates=[])
    if not isinstance(camera, RGBDCamera):
        raise ValueError("rgbd_camera_required")
    box = np.asarray(region_xyxy)
    h,w = camera.depth_m.shape
    if (box.shape != (4,) or box.dtype.kind not in "iu" or not
            (1 <= box[0] < box[2] < w-1 and 1 <= box[1] < box[3] < h-1)):
        raise ValueError("depth_loop_region_invalid")
    left,top,right,bottom = map(int,box)
    if exclusion_mask is None:
        exclusion_mask = np.zeros((h,w),dtype=bool)
    if np.asarray(exclusion_mask).shape != (h,w) or np.asarray(exclusion_mask).dtype != bool:
        raise ValueError("aperture_exclusion_mask_invalid")
    excluded = exclusion_mask[top:bottom,left:right]
    if material_domain is None:
        material_domain=np.ones((h,w),dtype=bool)
    if np.asarray(material_domain).shape != (h,w) or np.asarray(material_domain).dtype != bool:
        raise ValueError("aperture_material_domain_invalid")
    depth = camera.depth_m[top:bottom,left:right]
    valid = depth > 0
    if valid.sum() < 64:
        result["reason"] = "depth_loop_insufficient_depth"
        return result
    thresholds = np.unique(np.quantile(depth[valid], np.linspace(.02,.75,24)))
    seen = set()
    for threshold in thresholds:
        near = valid & (depth <= threshold)
        for pixels in depth_components(~near):
            ys,xs = pixels.T
            if (len(pixels)<12 or len(pixels)>depth.size//3 or (ys==0).any()
                    or (xs==0).any() or (ys==depth.shape[0]-1).any() or (xs==depth.shape[1]-1).any()
                    or not valid[ys,xs].all()):
                continue
            if excluded[ys,xs].any():
                continue  # Removing body/fingers must not manufacture a hole.
            signature = (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
            if signature in seen:
                continue
            seen.add(signature)
            hole = np.zeros_like(near);hole[ys,xs]=True
            grown = hole.copy()
            for _ in range(12):
                previous = grown.copy()
                grown[1:] |= previous[:-1];grown[:-1] |= previous[1:]
                grown[:,1:] |= previous[:,:-1];grown[:,:-1] |= previous[:,1:]
            material = grown & near & ~excluded & material_domain[top:bottom,left:right]
            # Leave three pixels for 3x3 samples and the estimator's +/-1 px
            # sensitivity probes; diagonal edge pixels must also be excluded.
            for _ in range(3):
                previous=material.copy()
                material[1:] &= previous[:-1];material[:-1] &= previous[1:]
                material[:,1:] &= previous[:,:-1];material[:,:-1] &= previous[:,1:]
                material[1:,1:] &= previous[:-1,:-1];material[:-1,:-1] &= previous[1:,1:]
                material[1:,:-1] &= previous[:-1,1:];material[:-1,1:] &= previous[1:,:-1]
            # Ordered radial boundary samples must actually lie in the hole;
            # reject incomplete/nonconvex boundaries in the unchanged estimator.
            center=np.array([xs.mean(),ys.mean()])
            offsets=np.column_stack((xs,ys))-center
            angles=np.arctan2(offsets[:,1],offsets[:,0])
            contour=[]
            for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
                delta=np.abs(np.angle(np.exp(1j*(angles-angle))))
                choices=np.where(delta < np.pi/16)[0]
                if not len(choices):break
                index=choices[np.argmax(np.linalg.norm(offsets[choices],axis=1))]
                contour.append([float(xs[index]+left),float(ys[index]+top)])
            if len(contour)!=16:continue
            radial_contour = contour
            components = (surface_material_components(material, depth) if separate_material_surfaces
                          else (np.argwhere(material),))
            for component in components:
                if len(component) < limits.minimum_samples:
                    continue
                my, mx = component.T
                recess = float(np.median(depth[ys, xs])-np.median(depth[my, mx]))
                if recess < .005 or (not background_opening and recess > .08):
                    continue
                contour = radial_contour
                sample=np.linspace(0,len(mx)-1,min(24,len(mx))).astype(int)
                samples=np.column_stack((mx[sample]+left,my[sample]+top)).astype(float)
                fit=estimate(camera,contour,samples,limits=limits)
                contour_basis = "radial_observed_boundary"
                if fit.get("reason") == "contour_requires_ordered_strictly_convex_visible_polygon":
                    refined = _observed_convex_contour(hole, left, top)
                    if refined is not None:
                        contour = refined
                        contour_basis = "inscribed_observed_background_not_complete_boundary"
                        fit = estimate(camera, contour, samples, limits=limits)
                statistics = None
                try:
                    points = _material_points(camera,samples,limits,np.asarray(contour))
                    centered=points-points.mean(axis=0)
                    _,singular,axes=np.linalg.svd(centered,full_matrices=False)
                    statistics=dict(sample_count=len(points),singular_values_m=singular.tolist(),
                        principal_rms_span_m=(singular/np.sqrt(len(points))).tolist(),
                        bbox_dimensions_m=np.ptp(points,axis=0).tolist(),
                        maximum_plane_residual_m=float(np.max(np.abs(centered@axes[-1]))))
                except (ValueError,np.linalg.LinAlgError):
                    pass
                result["candidates"].append(dict(contour_px=contour,material_px=samples.tolist(),
                                                  depth_recess_m=recess,aperture=fit,sample_statistics=statistics,
                                                  contour_basis=contour_basis,
                                                  material_component_size=len(component),
                                                  sampling_basis=("single_depth_connected_surface"
                                                      if separate_material_surfaces else "combined_material_mask")))
                if len(result["candidates"])>=8:break
            if len(result["candidates"])>=8:break
        if len(result["candidates"])>=8:break
    accepted=[c for c in result["candidates"] if c["aperture"]["status"]=="candidate_not_motion_validated"]
    result["candidate_search_truncated"] = len(result["candidates"]) >= 8
    if len(accepted)==1:
        result.update(status="geometric_candidate_semantics_unknown",reason="visible_depth_loop_requires_identity_review")
    elif len(accepted)>1:
        result["reason"]="ambiguous_depth_loops"
    elif result["candidates"]:
        result["reason"]="depth_loop_plane_or_contour_rejected"
    return result


def extract_handle_arch(observation, body_polygon_px, *, finger_boxes_tcp=None,
                        finger_geometry_source=None, limits=Limits()):
    """Exclude known robot volumes and the public primary body depth component.

    Finger boxes are conservative [minimum,maximum] metre bounds expressed in
    measured TCP coordinates, supplied from calibrated robot geometry. Never
    infer these boxes from commanded closure or a TCP point. No simulator object
    geometry is read, and hidden/removed contours are never completed.
    """
    result=dict(status="unavailable",reason="finger_geometry_unavailable",motion_authorized=False,
                semantics="unknown",candidates=[],sample_count=0)
    if finger_boxes_tcp is None or not isinstance(finger_geometry_source,str) or not finger_geometry_source.strip():
        return result
    boxes=np.asarray(finger_boxes_tcp,dtype=float)
    if (boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:,0]>=boxes[:,1])):
        raise ValueError("finger_envelopes_invalid")
    camera=(observation.camera_frames or {}).get("perception_camera")
    if camera is None or not observation.measured_tcp_poses:
        result["reason"]="measured_tcp_or_public_rgbd_unavailable"
        return result
    from harness.perception.pose_grounding import _pose_matrix
    from PIL import Image,ImageDraw
    h,w=camera.depth_m.shape
    polygon=_pixels(body_polygon_px,3,(h,w))
    image=Image.new("1",(w,h))
    ImageDraw.Draw(image).polygon([tuple(p) for p in polygon],fill=1)
    body=np.asarray(image,dtype=bool)&(camera.depth_m>0)
    if body.sum()<12:
        result["reason"]="primary_body_depth_unavailable"
        return result
    body &= np.abs(camera.depth_m-np.median(camera.depth_m[body]))<=.03
    components=list(depth_components(body))
    if not components:
        result["reason"]="primary_body_depth_unavailable"
        return result
    largest=max(components,key=len)
    body[:]=False;body[largest[:,0],largest[:,1]]=True
    foreground=(camera.depth_m>0)&(np.abs(camera.depth_m-np.median(camera.depth_m[body]))<=.06)
    attached=np.zeros_like(body)
    for component in depth_components(foreground):
        if body[component[:,0],component[:,1]].any():
            attached[component[:,0],component[:,1]]=True
    y,x=np.mgrid[:h,:w]
    rays=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(camera.intrinsic).T
    camera_points=rays*camera.depth_m[...,None]
    camera_to_world=np.linalg.inv(camera.world_to_camera)
    world=camera_points@camera_to_world[:3,:3].T+camera_to_world[:3,3]
    world_to_tcp=np.linalg.inv(_pose_matrix(observation.measured_tcp_poses["right"]))
    local=world@world_to_tcp[:3,:3].T+world_to_tcp[:3,3]
    fingers=np.zeros((h,w),dtype=bool)
    for low,high in boxes:
        fingers |= np.all((local>=low)&(local<=high),axis=-1)&(camera.depth_m>0)
    # Dilate exclusions for sample neighborhoods and one-pixel perturbation.
    excluded=body|fingers
    for _ in range(2):
        previous=excluded.copy()
        excluded[1:] |= previous[:-1];excluded[:-1] |= previous[1:]
        excluded[:,1:] |= previous[:,:-1];excluded[:,:-1] |= previous[:,1:]
    padding=48*max(1,w/320,h/240)
    low=np.maximum([1,1],np.floor(polygon.min(0)-padding)).astype(int)
    high=np.minimum([w-2,h-2],np.ceil(polygon.max(0)+padding)).astype(int)
    found=depth_contrast_aperture(camera,tuple([*low,*high]),limits=limits,
                                  exclusion_mask=excluded,background_opening=True,material_domain=attached)
    found.update(finger_geometry_source=finger_geometry_source,
                 excluded_finger_pixels=int(fingers.sum()),excluded_body_pixels=int(body.sum()),
                 sample_count=sum((c.get("sample_statistics") or {}).get("sample_count",0)
                                  for c in found["candidates"]))
    return found


def _hull_indices(points):
    """Indices of the 2D convex hull, counter-clockwise, collinear points excluded."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    def cross(o, a, b):
        return ((points[a][0]-points[o][0])*(points[b][1]-points[o][1])
                - (points[a][1]-points[o][1])*(points[b][0]-points[o][0]))
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1]+upper[:-1]


def _drop_subpixel_dents(contour, tolerance_px):
    """Replace a contour by its convex hull when it deviates by <= ``tolerance_px``.

    Proposed contour vertices carry pixel-level placement noise, so vertices a
    fraction of a pixel inside the hull are not evidence of a concave opening.
    The hull is used only when every dropped vertex lies within the tolerance
    of the hull edge that replaces it, and the hull keeps the contour's own
    cyclic vertex order, so a scrambled or self-crossing ordering is never
    turned into a convex polygon. Otherwise the contour is returned unchanged
    and the strict convexity check decides. Returns the contour and the
    dropped vertices with their distances, for the report.
    """
    contour = np.asarray(contour, dtype=float)
    hull = _hull_indices(contour)
    if len(hull) < 3 or len(hull) == len(contour):
        return contour, []
    # The hull must visit the kept vertices in the contour's cyclic order
    # (either direction); otherwise the input ordering is not a polygon walk.
    start = hull.index(min(hull))
    forward = hull[start:]+hull[:start]
    backward = [forward[0]]+forward[1:][::-1]
    if forward != sorted(forward) and backward != sorted(backward):
        return contour, []
    kept = sorted(hull)
    removed = []
    for position, first in enumerate(kept):
        second = kept[(position+1) % len(kept)]
        span = range(first+1, second) if second > first else [*range(first+1, len(contour)), *range(0, second)]
        a, b = contour[first], contour[second]
        edge = b-a
        length = float(np.linalg.norm(edge))
        for index in span:
            offset = contour[index]-a
            distance = abs(edge[0]*offset[1]-edge[1]*offset[0])/length if length else float(np.linalg.norm(offset))
            if distance > tolerance_px:
                return contour, []
            removed.append(dict(vertex_px=contour[index].tolist(), dent_px=distance))
    return contour[kept], removed


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


def inset_material_samples(camera, contour_px, material_px, *, limits=Limits()):
    """Move at most two pixels into the local depth-consistent material interior.

    Do not change sensitivity probes or cross the opening. Every chosen patch
    must survive the original four cardinal probes on the same local surface.
    Geometry/semantic validation of the resulting complete fit remains required.
    """
    contour = _pixels(contour_px, 8, camera.depth_m.shape)
    pixels = _pixels(material_px, limits.minimum_samples, camera.depth_m.shape)
    offsets = sorted(((x, y) for x in range(-2, 3) for y in range(-2, 3)
                      if x*x+y*y <= 4), key=lambda p: (p[0]*p[0]+p[1]*p[1], p))
    adjusted = []
    for pixel in pixels:
        original_depth = camera.sample_depth(*pixel, radius=1)
        for offset in offsets:
            candidate = pixel+offset
            try:
                if abs(camera.sample_depth(*candidate, radius=1)-original_depth) > .003:
                    continue
                probes = np.array([candidate+delta for delta in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))])
                _material_points(camera, probes, limits, contour)
            except ValueError:
                continue
            adjusted.append(candidate)
            break
        else:
            raise ValueError("surface_consistent_material_inset_unavailable")
    if len(np.unique(np.rint(adjusted), axis=0)) != len(pixels):
        raise ValueError("duplicate_depth_samples")
    return np.asarray(adjusted).tolist()


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
        # Sub-pixel placement noise is not concavity; deeper dents still reject.
        contour, dropped = _drop_subpixel_dents(contour, limits.pixel_uncertainty_px)
        if dropped:
            result["contour_vertices_dropped"] = dropped
        contour = _pixels(contour, 8, camera.depth_m.shape)
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
        result["geometry_diagnostics"]["principal_rms_span_m"] = (singular/np.sqrt(len(points))).tolist()
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
