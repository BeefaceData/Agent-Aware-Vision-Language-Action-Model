"""Cylinder pose fits on synthetic depth images rendered from exact geometry; CPU only."""

import unittest

import numpy as np

from harness.perception.object_pose import (MUG, RACK, Surface, fit_cylinder, masked_surface, object_pose,
                                            pole_with_peg_pose)
from harness.perception.rgbd import RGBDCamera

TABLE_Z = 0.75
RADIUS = 0.045
HEIGHT = 0.10
INTRINSIC = np.array([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]])
IMAGE_HW = (240, 320)


def looking_at(eye, target):
    """OpenCV world-to-camera matrix for a camera at ``eye`` looking at ``target``."""
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, -rotation @ eye
    return matrix


def cylinder_hits(origin, rays, base, axis, radius, height, *, open_top=False):
    """Ray distances to a cylinder with a closed base; an open top shows the inside wall."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    start = origin - np.asarray(base, float)
    along_start, along_ray = start @ axis, rays @ axis
    flat_start, flat_ray = start - along_start * axis, rays - np.outer(along_ray, axis)
    a = np.sum(flat_ray ** 2, 1)
    b = 2 * flat_ray @ flat_start
    c = flat_start @ flat_start - radius ** 2
    disc = b ** 2 - 4 * a * c
    hits = np.full(len(rays), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        # The near wall, then (through an open top) the far wall seen from inside.
        for sign in ((-1, 1) if open_top else (-1,)):
            wall = (-b + sign * np.sqrt(disc)) / (2 * a)
            wall_height = along_start + wall * along_ray
            on_wall = (disc >= 0) & (wall > 0) & (wall_height >= 0) & (wall_height <= height)
            hits = np.where(on_wall & (wall < hits), wall, hits)
        for level in ((0.0,) if open_top else (0.0, height)):
            cap = (level - along_start) / along_ray
            spot = flat_start + cap[:, None] * flat_ray
            inside = (cap > 0) & (np.sum(spot ** 2, 1) <= radius ** 2)
            hits = np.where(inside & (cap < hits), cap, hits)
    return hits


def render_cylinders(world_to_camera, cylinders):
    """Depth (whole millimetres, like RoboTwin) and mask of a scene made of cylinders.

    Each cylinder is ``(base, axis, radius, height, open_top)``.
    """
    rotation, translation = world_to_camera[:3, :3], world_to_camera[:3, 3]
    origin = -rotation.T @ translation
    v, u = np.mgrid[:IMAGE_HW[0], :IMAGE_HW[1]]
    rays_camera = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3) @ np.linalg.inv(INTRINSIC).T
    rays = rays_camera @ rotation  # camera directions to world
    hits = np.full(len(rays), np.inf)
    for base, axis, radius, height, open_top in cylinders:
        hits = np.minimum(hits, cylinder_hits(origin, rays, base, axis, radius, height, open_top=open_top))
    hit = np.isfinite(hits)
    depth = np.zeros(len(rays))
    depth[hit] = np.round(hits[hit] * rays_camera[hit, 2] * 1000) / 1000  # metres along the optical axis
    return depth.reshape(IMAGE_HW).astype(np.float32), hit.reshape(IMAGE_HW)


def render(world_to_camera, base, axis, *, radius=RADIUS, height=HEIGHT, open_top=False, handle_towards=None):
    """A mug-like cylinder, optionally hollow, with an optional handle.

    The handle is a thin rod beside the body, parallel to the axis, on the side of ``handle_towards``.
    """
    cylinders = [(base, axis, radius, height, open_top)]
    if handle_towards is not None:
        unit_axis = np.asarray(axis, float) / np.linalg.norm(axis)
        rod_base = np.asarray(base, float) + (radius + 0.015) * np.asarray(handle_towards, float) + 0.02 * unit_axis
        cylinders.append((rod_base, axis, 0.006, 0.06, False))
    return render_cylinders(world_to_camera, cylinders)


def view(eye, target, base, axis, **shape):
    world_to_camera = looking_at(np.asarray(eye, float), target)
    depth, mask = render(world_to_camera, base, axis, **shape)
    return masked_surface(RGBDCamera(depth, INTRINSIC, world_to_camera), mask, table_height_m=TABLE_Z)


def angle_deg(first, second):
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(first, second)))))))


UPRIGHT_BASE = (0.0, 0.0, TABLE_Z)
LYING_BASE = (0.0, 0.0, TABLE_Z + RADIUS)
HEAD_EYE = (-0.5, 0.0, TABLE_Z + 0.45)  # elevated and oblique, like the head camera


class CylinderFitTests(unittest.TestCase):
    def test_upright_mug_seen_obliquely_gives_a_vertical_axis_and_its_base_on_the_table(self):
        surface = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1))

        pose, reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertEqual(reason, "wall_fit_1_camera")
        self.assertTrue(pose.trusted)
        self.assertLess(angle_deg(pose.axis, (0, 0, 1)), 3.0)
        self.assertTrue(pose.base_known)
        self.assertLess(np.linalg.norm(np.subtract(pose.position, UPRIGHT_BASE)), 0.005)
        self.assertAlmostEqual(pose.radius_m, RADIUS, delta=0.003)

    def test_mug_lying_side_on_gives_a_horizontal_axis(self):
        surface = view(HEAD_EYE, (0, 0, TABLE_Z), LYING_BASE, (0, 1, 0))

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertGreater(pose.tilt_deg(), 80.0)
        self.assertLess(angle_deg(pose.axis, (0, 1, 0)), 5.0)
        self.assertFalse(pose.base_known)

    def test_mug_lying_end_on_is_still_seen_as_lying_down(self):
        # Held-out seed 6: the flat base faces the camera and looks like an upright wall to a normal test.
        surface = view(HEAD_EYE, (0, 0, TABLE_Z), LYING_BASE, (1, 0, 0))

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertGreater(pose.tilt_deg(), 45.0)

    def test_a_flat_disc_alone_is_read_as_an_end_cap(self):
        eye = np.array((-0.5, 0.0, TABLE_Z + RADIUS))  # level with the axis: only the base is visible
        surface = view(eye, (0, 0, TABLE_Z + RADIUS), LYING_BASE, (1, 0, 0))

        fit, reason = fit_cylinder(surface.points, surface.normals, MUG)

        self.assertEqual(reason, "end_cap")
        self.assertLess(angle_deg(fit.axis, (1, 0, 0)), 5.0)

    def test_smallest_and_largest_mugs_fit(self):
        for radius in MUG.radius_m[0] + 0.005, MUG.radius_m[1] - 0.005:
            with self.subTest(radius=radius):
                surface = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1), radius=radius)

                fit, _reason = fit_cylinder(surface.points, surface.normals, MUG)

                self.assertAlmostEqual(fit.radius_m, radius, delta=0.003)

    def test_a_fit_explaining_too_few_points_is_returned_untrusted(self):
        points, normals, viewpoint = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1))
        # As many stray points again, all well clear of the mug body.
        clutter = np.random.default_rng(1).uniform((0.1, -0.1, TABLE_Z + 0.02), (0.2, 0.1, TABLE_Z + 0.2),
                                                   (len(points), 3))
        clutter_normals = np.tile((0.0, 0.0, 1.0), (len(points), 1))

        pose, reason = object_pose({"head_camera": Surface(np.vstack((points, clutter)),
                                                           np.vstack((normals, clutter_normals)), viewpoint)},
                                   MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertFalse(pose.trusted)
        self.assertEqual(reason, "cylinder_fit_inliers_low")

    def test_too_little_visible_surface_is_unknown(self):
        points, normals, viewpoint = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1))

        pose, reason = object_pose({"head_camera": Surface(points[:30], normals[:30], viewpoint)}, MUG,
                                   table_height_m=TABLE_Z, observation_sequence=4)

        self.assertIsNone(pose)
        self.assertEqual(reason, "too_few_surface_points")

    def test_a_flat_patch_on_the_table_is_removed_before_fitting(self):
        # A shadow masked as the mug: all of its points are table top.
        world_to_camera = looking_at(np.asarray(HEAD_EYE), (0, 0, TABLE_Z))
        depth, mask = render(world_to_camera, (0, 0, TABLE_Z - 0.01), (0, 0, 1), radius=0.2, height=0.01)
        frame = RGBDCamera(depth, INTRINSIC, world_to_camera)

        points = masked_surface(frame, mask, table_height_m=TABLE_Z).points

        self.assertEqual(len(points), 0)


class BaseEndAndHandleTests(unittest.TestCase):
    def test_a_lying_cup_open_towards_the_camera_has_its_base_at_the_far_end(self):
        # The inside wall is visible through the opening.
        surface = view(HEAD_EYE, (0, 0, TABLE_Z), (0.05, 0.0, TABLE_Z + RADIUS), (-1, 0, 0), open_top=True)

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertTrue(pose.base_known)
        self.assertGreater(float(np.dot(pose.axis, (-1, 0, 0))), 0.9)  # points base -> opening
        self.assertLess(np.linalg.norm(np.subtract(pose.position, (0.05, 0.0, TABLE_Z + RADIUS))), 0.015)

    def test_a_lying_cup_with_its_base_towards_the_camera_points_away_from_it(self):
        # Only the outside wall and the flat base are visible.
        surface = view(HEAD_EYE, (0, 0, TABLE_Z), (-0.05, 0.0, TABLE_Z + RADIUS), (1, 0, 0), open_top=True)

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertTrue(pose.base_known)
        self.assertGreater(float(np.dot(pose.axis, (1, 0, 0))), 0.9)

    def test_the_heading_points_from_the_axis_to_the_handle(self):
        towards = np.array((0.0, -1.0, 0.0))  # side-on to the head camera, so the handle is in view
        surface = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1), open_top=True,
                       handle_towards=towards)

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertLess(np.degrees(np.arccos(np.clip(np.dot(pose.heading, towards), -1, 1))), 10.0)
        self.assertIn("handle", pose.key_points)

    def test_no_handle_in_view_gives_no_heading(self):
        surface = view(HEAD_EYE, (0, 0, TABLE_Z + 0.05), UPRIGHT_BASE, (0, 0, 1), open_top=True)

        pose, _reason = object_pose({"head_camera": surface}, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertIsNone(pose.heading)
        self.assertNotIn("handle", pose.key_points)


POLE_FOOT = np.array((0.0, 0.0, TABLE_Z))
PEG_ROOT = POLE_FOOT + (0.0, 0.0, 0.13)
PEG_DIRECTION = np.array((np.cos(np.radians(38)), 0.0, np.sin(np.radians(38))))  # out along +x and up
PEG_LENGTH = 0.10


def rack_view(eye, *, with_peg=True):
    """Surface of a rack: base plate, pole and an angled peg, like RoboTwin's 040_rack."""
    world_to_camera = looking_at(np.asarray(eye, float), PEG_ROOT)
    cylinders = [(POLE_FOOT, (0, 0, 1), 0.06, 0.008, False),   # base plate
                 (POLE_FOOT, (0, 0, 1), 0.01, 0.23, False)]    # pole
    if with_peg:
        cylinders.append((PEG_ROOT, PEG_DIRECTION, 0.006, PEG_LENGTH, False))
    depth, mask = render_cylinders(world_to_camera, cylinders)
    return masked_surface(RGBDCamera(depth, INTRINSIC, world_to_camera), mask)


RACK_EYE = (0.05, -0.4, TABLE_Z + 0.25)  # in front of the peg's side, like the right wrist camera at reset


class RackPoseTests(unittest.TestCase):
    def test_pole_and_peg_give_the_rack_axis_heading_and_peg_tip(self):
        pose, reason = pole_with_peg_pose({"right_camera": rack_view(RACK_EYE)}, RACK, table_height_m=TABLE_Z,
                                          observation_sequence=0)

        self.assertEqual(reason, "pole_and_peg_fit_1_camera")
        self.assertLess(angle_deg(pose.axis, (0, 0, 1)), 3.0)
        self.assertLess(np.degrees(np.arccos(np.clip(np.dot(pose.heading, (1, 0, 0)), -1, 1))), 10.0)
        tip = PEG_ROOT + PEG_LENGTH * PEG_DIRECTION
        self.assertLess(np.linalg.norm(np.subtract(pose.key_points["peg_tip"], tip)), 0.015)
        self.assertLess(np.linalg.norm(np.subtract(pose.position[:2], POLE_FOOT[:2])), 0.015)

    def test_a_pole_without_a_visible_peg_is_unknown(self):
        pose, reason = pole_with_peg_pose({"right_camera": rack_view(RACK_EYE, with_peg=False)}, RACK,
                                          table_height_m=TABLE_Z, observation_sequence=0)

        self.assertIsNone(pose)
        self.assertEqual(reason, "peg_not_seen")


    def test_no_table_height_means_no_rack_pose(self):
        # The base plate could not be removed, and its edge could pass for the peg.
        pose, reason = pole_with_peg_pose({"right_camera": rack_view(RACK_EYE)}, RACK, table_height_m=None,
                                          observation_sequence=0)

        self.assertIsNone(pose)
        self.assertEqual(reason, "table_height_unavailable")

    def test_key_points_cannot_be_changed_after_the_fit(self):
        pose, _reason = pole_with_peg_pose({"right_camera": rack_view(RACK_EYE)}, RACK, table_height_m=TABLE_Z,
                                           observation_sequence=0)

        with self.assertRaises(TypeError):
            pose.key_points["peg_tip"] = (0.0, 0.0, 0.0)


class CameraAgreementTests(unittest.TestCase):
    def test_two_views_of_one_mug_agree(self):
        target = (0, 0, TABLE_Z + 0.05)
        surfaces = {"head_camera": view(HEAD_EYE, target, UPRIGHT_BASE, (0, 0, 1)),
                    "perception_camera": view((0.5, 0.3, TABLE_Z + 0.3), target, UPRIGHT_BASE, (0, 0, 1))}

        pose, reason = object_pose(surfaces, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertEqual(reason, "wall_fit_2_cameras")
        self.assertEqual(pose.cameras, ("head_camera", "perception_camera"))

    def test_views_of_mugs_in_different_places_make_the_pose_unknown(self):
        target = (0, 0, TABLE_Z + 0.05)
        shifted = (0.0, 0.08, TABLE_Z)
        surfaces = {"head_camera": view(HEAD_EYE, target, UPRIGHT_BASE, (0, 0, 1)),
                    "perception_camera": view((0.5, 0.3, TABLE_Z + 0.3), target, shifted, (0, 0, 1))}

        pose, reason = object_pose(surfaces, MUG, table_height_m=TABLE_Z, observation_sequence=4)

        self.assertIsNone(pose)
        self.assertEqual(reason, "cameras_disagree")


class SurfaceNormalStepTests(unittest.TestCase):
    def test_a_wider_normal_step_keeps_a_steep_but_smooth_slope(self):
        depth = np.tile(0.5 + 0.004 * np.arange(20, dtype=np.float32), (20, 1))  # 4 mm per pixel
        frame = RGBDCamera(depth, INTRINSIC, np.eye(4))
        mask = np.ones((20, 20), dtype=bool)

        narrow, _ = frame.surface_points_and_normals(mask, step_px=1)
        wide, _ = frame.surface_points_and_normals(mask, step_px=2)

        # Erosion stops at the image edge, so only the step's border is lost.
        self.assertEqual(len(narrow), 18 * 18)
        self.assertEqual(len(wide), 16 * 16)  # a 1 cm limit over 4 px would have dropped every pixel


if __name__ == "__main__":
    unittest.main()
