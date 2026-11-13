"""CPU surface-fit regressions; no simulation or perception service."""

from pathlib import Path
import unittest

import numpy as np

from development.hanging.rack_surface import (
    assess_branch_regions, assess_reviewed_archive, fit_fixed_radius_cylinder, seeded_shaft_surface,
)
from harness.perception.replay import load_public_observation
from harness.perception.rgbd import RGBDCamera


class RackSurfaceTests(unittest.TestCase):
    def test_seed_selection_excludes_smooth_disconnected_background(self):
        depth = np.ones((40, 60), np.float32)
        depth[:, 30:] = 2.
        camera = RGBDCamera(depth, np.array([[200., 0., 30.], [0., 200., 20.], [0., 0., 1.]]),
                            np.eye(4))
        polygon = [[2, 2], [57, 2], [57, 37], [2, 37]]
        surface = seeded_shaft_surface(camera, polygon, [15, 20])
        self.assertGreater(len(surface.points), 60)
        np.testing.assert_allclose(np.asarray(surface.points)[:, 2], 1.)
        other = seeded_shaft_surface(camera, polygon, [45, 20])
        np.testing.assert_allclose(np.asarray(other.points)[:, 2], 2.)
        for seed in ([0, 0], [-1, 10], [60, 10]):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seeded_shaft_surface(camera, polygon, seed)
        missing = depth.copy(); missing[20, 15] = 0.
        with self.assertRaisesRegex(ValueError, "shaft_seed_depth_unavailable"):
            seeded_shaft_surface(RGBDCamera(missing, camera.intrinsic, np.eye(4)), polygon, [15, 20])

    def test_fixed_radius_fit_recovers_rotated_translated_cylinder_without_normals(self):
        axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
        side = np.cross(axis, [0., 0., 1.]); side /= np.linalg.norm(side)
        up = np.cross(axis, side)
        center = np.array([.12, -.08, .9]); radius = .0055
        points = np.array([center+axis*t+radius*(np.cos(a)*side+np.sin(a)*up)
                           for t in np.linspace(-.04, .04, 15)
                           for a in np.linspace(-1.3, 1.3, 16)])
        fit = fit_fixed_radius_cylinder(points, radius)
        self.assertLess(fit["maximum_residual_m"], 1e-7)
        fitted_axis = np.asarray(fit["axis_world"])
        self.assertGreater(abs(fitted_axis @ axis), .99999)
        delta = np.asarray(fit["center_world_m"])-center
        self.assertLess(np.linalg.norm(delta-axis*(delta @ axis)), 1e-6)
        for invalid in (points[:4], np.full_like(points, np.nan), np.zeros_like(points)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                fit_fixed_radius_cylinder(invalid, radius)
        with self.assertRaisesRegex(ValueError, "cylinder_surface_degenerate"):
            fit_fixed_radius_cylinder(np.column_stack((np.linspace(0., .1, 100), np.zeros((100, 2)))), radius)

    def test_archive_binding_rejects_unreviewed_stale_and_incompatible_annotations(self):
        from hashlib import sha256
        from unittest.mock import patch
        import tempfile
        from types import SimpleNamespace

        camera = RGBDCamera(np.ones((10, 10), np.float32), np.eye(3), np.eye(4))
        image = np.zeros((10, 10, 3), np.uint8)
        observation = SimpleNamespace(sequence=17, camera_frames={"test": camera}, images={"test": image})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"record.npz"; path.write_bytes(b"test archive")
            valid_hash = sha256(path.read_bytes()).hexdigest()
            for annotation, reason in (
                ({}, "shaft_annotation_unreviewed_or_wrong_archive"),
                (dict(reviewed=True, archive_sha256=valid_hash, sequence=16), "shaft_annotation_stale"),
                (dict(reviewed=True, archive_sha256=valid_hash, sequence=17, regions={"test": {}}),
                 "shaft_annotation_camera_or_image_mismatch"),
            ):
                with self.subTest(reason=reason), patch(
                        "development.hanging.rack_surface.load_public_observation", return_value=observation):
                    result = assess_reviewed_archive(path, annotation)
                self.assertEqual(result["reason"], reason)
                self.assertFalse(result["motion_authorized"])

    def test_saved_two_view_shaft_remains_blocked_on_residual_not_just_low_rms(self):
        archive = Path("data/runs/seed4_perception640_20261103_084536_510/observations/observation_00000160.npz")
        if not archive.exists():
            self.skipTest("ignored saved rack evidence unavailable")
        observation = load_public_observation(archive)
        regions = {
            "perception_camera": dict(polygon_px=[[137,252],[144,257],[175,218],[170,213]], seed_uv=[155,235]),
            "head_camera": dict(polygon_px=[[225,6],[231,8],[245,28],[241,31]], seed_uv=[235,20]),
        }
        result = assess_branch_regions(observation, regions, radius_m=.0055,
            radius_basis="Existing nominal audit constant; asset provenance not reverified for this recording")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "shaft_surface_residual_exceeds_existing_3mm_bound")
        self.assertLess(result["axis_sensitivity_deg"], 5.)
        self.assertLess(result["position_sensitivity_m"], .005)
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["physical_clearance"], "unknown")
        self.assertTrue(any(f["maximum_residual_m"] > .003 for f in result["fits"]))

    def test_missing_inputs_do_not_become_geometry_or_motion_approval(self):
        from types import SimpleNamespace
        observation = SimpleNamespace(sequence=1, camera_frames={}, images={})
        result = assess_branch_regions(observation, {}, radius_m=.0055, radius_basis="test hypothesis")
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["motion_authorized"])
        result = assess_branch_regions(observation, {"missing": {}}, radius_m=.0055, radius_basis="test")
        self.assertEqual(result["reason"], "shaft_camera_rgbd_unavailable")
        self.assertFalse(result["motion_authorized"])
