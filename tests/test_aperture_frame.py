"""Synthetic public camera geometry only; no simulator or provider client."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from development.hanging.aperture_frame import (
    Limits, estimate, inspect_archive, _material_points, aperture_dimensions,
    depth_contrast_aperture,
    extract_handle_arch,
    body_excluded_material_domain, _observed_convex_contour,
    surface_material_components,
)
from harness.perception.pose_grounding import _pose_matrix
from harness.perception.rgbd import RGBDCamera


class ApertureFrameTests(unittest.TestCase):
    def test_material_components_do_not_pool_disconnected_or_depth_separated_surfaces(self):
        depth = np.ones((9, 12), np.float32)
        mask = np.zeros(depth.shape, bool)
        mask[1:4, 1:8] = True
        depth[1:4, 5:8] = 1.1  # Adjacent in the raster, separate in depth.
        mask[6:8, 1:5] = True  # Same depth, but disconnected observed support.
        depth[6, 1] = np.nan
        depth[7, 1] = 0.
        components = list(surface_material_components(mask, depth))
        self.assertEqual([len(c) for c in components], [12, 9, 6])
        for component in components:
            self.assertEqual(len(np.unique(depth[tuple(component.T)])), 1)
        again = list(surface_material_components(mask, depth))
        for first, second in zip(components, again):
            np.testing.assert_array_equal(first, second)

    def test_saved_entry_surface_components_stay_unavailable_without_plane_span(self):
        import json
        from harness.perception.replay import load_public_observation

        root = Path("data/runs")
        cases = (
            ("seed5_live_a209eedb8cb54725948b1e641a14b809", 440,
             "seed5_views_af4fd356b5074c2c9e1f87bee5aa42fc"),
            ("seed4_perception640_20261103_084536_510", 464,
             "seed4_obs464_semantic_97c95fb4a5ca4b9489c3851071345ef1"),
        )
        for recording, sequence, review in cases:
            with self.subTest(sequence=sequence):
                archive = root/recording/"observations"/f"observation_{sequence:08d}.npz"
                source = root/review/"results.json"
                if not archive.exists() or not source.exists():
                    self.skipTest("ignored saved entry evidence unavailable")
                proposal = next(f["mapped_proposal"] for f in json.loads(source.read_text())["frames"]
                                if f["sequence"] == sequence)
                camera = load_public_observation(archive).camera_frames["perception_camera"]
                uv = np.asarray(proposal["contour_px"]+proposal["material_px"])
                height, width = camera.depth_m.shape
                # Existing 12-pixel material-growth reach, not a new annotation.
                low = np.maximum(1, np.floor(uv.min(axis=0)-12).astype(int))
                high = np.minimum([width-2, height-2], np.ceil(uv.max(axis=0)+12).astype(int))
                result = depth_contrast_aperture(camera, [*low, *high], background_opening=True,
                                                separate_material_surfaces=True)
                self.assertEqual(result["status"], "unavailable")
                self.assertTrue(result["candidates"])
                self.assertTrue(result["candidate_search_truncated"])
                self.assertEqual(result["semantics"], "unknown")
                self.assertFalse(result["motion_authorized"])
                for candidate in result["candidates"]:
                    self.assertEqual(candidate["sampling_basis"], "single_depth_connected_surface")
                    self.assertEqual(candidate["aperture"]["reason"], "plane_support_degenerate")
                    self.assertFalse(candidate["aperture"]["motion_authorized"])

    def test_body_domain_excludes_only_observed_cylinder_and_rejects_stale_or_untrusted_fit(self):
        from harness.perception.object_pose import ObjectPose

        camera = RGBDCamera(np.ones((101, 101), np.float32),
            np.array([[200., 0., 50.], [0., 200., 50.], [0., 0., 1.]]), np.eye(4))
        pose = ObjectPose("mug", (0., 0., 1.), (0., 0., 1.), True, .045, None,
                          .9, True, "synthetic", ("perception_camera",), 7)
        mask = body_excluded_material_domain(camera, pose, observation_sequence=7)
        self.assertFalse(mask[50, 50])
        self.assertFalse(mask[50, 59])  # Body radius plus existing surface tolerance.
        self.assertTrue(mask[50, 70])  # Outside the body is not asserted to be handle.
        for bad in (replace(pose, trusted=False), replace(pose, observation_sequence=6)):
            with self.assertRaisesRegex(ValueError, "current_trusted_body_pose_required"):
                body_excluded_material_domain(camera, bad, observation_sequence=7)
        depth = camera.depth_m.copy(); depth[50, 70] = 0.
        missing = RGBDCamera(depth, camera.intrinsic, camera.world_to_camera)
        self.assertFalse(body_excluded_material_domain(missing, pose, observation_sequence=7)[50, 70])

    def test_refined_contour_is_convex_and_never_fills_unobserved_or_concave_space(self):
        from PIL import Image, ImageDraw

        y, x = np.mgrid[:61, :61]
        hole = ((x-30)/20)**2+((y-30)/15)**2 <= 1
        contour = _observed_convex_contour(hole, 0, 0)
        self.assertIsNotNone(contour)
        points = np.asarray(contour)
        edges = np.roll(points, -1, axis=0)-points
        cross = edges[:, 0]*np.roll(edges, -1, axis=0)[:, 1]-edges[:, 1]*np.roll(edges, -1, axis=0)[:, 0]
        self.assertTrue(np.all(cross > 0) or np.all(cross < 0))
        image = Image.new("1", (61, 61))
        ImageDraw.Draw(image).polygon([tuple(p) for p in contour], fill=1)
        self.assertFalse(np.any(np.asarray(image, bool) & ~hole))
        concave = hole.copy(); concave[27:34, 30:] = False
        self.assertIsNone(_observed_convex_contour(concave, 0, 0))

    def test_arch_extractor_requires_robot_geometry_and_excludes_body_and_fingers(self):
        from harness.environment.base import Observation
        y,x=np.mgrid[:121,:121]
        depth=np.full((121,121),1.3,np.float32)
        depth[(x-50)**2+(y-60)**2 <= 26**2]=1.
        depth[((x-50)/10)**2+((y-60)/8)**2 <= 1]=1.3
        depth[45:76,75:101]=1.  # Primary body joined to the handle arch.
        camera=RGBDCamera(depth,np.array([[200.,0.,60.],[0.,200.,60.],[0.,0.,1.]]),np.eye(4))
        observation=Observation(9,{"perception_camera":np.zeros((121,121,3),np.uint8)},tuple(np.zeros(14)),
            {"perception_camera":camera},{"right":(0.,0.,1.,1.,0.,0.,0.),"left":(.5,0.,1.,1.,0.,0.,0.)})
        polygon=[(80.,50.),(95.,50.),(95.,70.),(80.,70.)]
        missing=extract_handle_arch(observation,polygon)
        self.assertEqual(missing["reason"],"finger_geometry_unavailable")
        # Sourced fixture robot volume away from the loop; no simulator geometry.
        boxes=[[[.4,.4,-.1],[.5,.5,.1]]]
        result=extract_handle_arch(observation,polygon,finger_boxes_tcp=boxes,
                                   finger_geometry_source="known synthetic TCP-relative robot volume")
        self.assertGreater(result["excluded_body_pixels"],0)
        self.assertFalse(result["motion_authorized"])
        self.assertTrue(any(c["aperture"]["status"]=="candidate_not_motion_validated" for c in result["candidates"]))
        for candidate in result["candidates"]:
            for u,v in candidate["material_px"]:
                self.assertFalse(80<=u<=95 and 50<=v<=70)
        # A finger envelope covering the whole loop cannot become an aperture.
        blocked=extract_handle_arch(observation,polygon,finger_boxes_tcp=[[[-1.,-1.,-.5],[1.,1.,.5]]],
                                   finger_geometry_source="synthetic occluding robot volume")
        self.assertFalse(blocked["candidates"])
        self.assertGreater(blocked["excluded_finger_pixels"],0)

    def test_saved_464_arch_extractor_does_not_invent_finger_envelope(self):
        import json
        from harness.perception.replay import load_public_observation
        root=Path("data/runs/seed4_perception640_20261103_084536_510")
        if not root.exists():self.skipTest("saved recording unavailable")
        observation=load_public_observation(root/"observations/observation_00000464.npz")
        rows=[json.loads(l) for l in (root/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
        report=next(r["report"] for r in rows if r.get("event_type")=="public_hanging_proposal" and r["observation_sequence"]==464)
        result=extract_handle_arch(observation,report["body_polygon_px"])
        self.assertEqual(result["reason"],"finger_geometry_unavailable")
        self.assertEqual(result["sample_count"],0)
        self.assertFalse(result["candidates"])
        self.assertFalse(result["motion_authorized"])

    def test_depth_contrast_recess_without_rgb_is_candidate_not_semantic_proof(self):
        y,x=np.mgrid[:101,:101]
        depth=np.ones((101,101),np.float32)
        depth[((x-50)/10)**2+((y-50)/8)**2<=1]=1.025
        camera=RGBDCamera(depth,np.array([[200.,0.,50.],[0.,200.,50.],[0.,0.,1.]]),np.eye(4))
        result=depth_contrast_aperture(camera,(20,20,80,80))
        self.assertTrue(result["candidates"])
        self.assertTrue(any(c["aperture"]["status"]=="candidate_not_motion_validated" for c in result["candidates"]))
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["semantics"],"unknown")
        # A depth gap open to the region boundary is not a closed handle opening.
        depth[49:52,20:51]=1.025
        opened=RGBDCamera(depth,camera.intrinsic,np.eye(4))
        result=depth_contrast_aperture(opened,(20,20,80,80))
        self.assertEqual(result["status"],"unavailable")
        self.assertFalse(result["candidates"])

    def test_saved_464_depth_contrast_reports_geometry_without_inventing_handle_identity(self):
        import json
        from harness.perception.replay import load_public_observation
        root=Path("data/runs/seed4_perception640_20261103_084536_510")
        if not root.exists():self.skipTest("saved recording unavailable")
        observation=load_public_observation(root/"observations/observation_00000464.npz")
        rows=[json.loads(l) for l in (root/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
        report=next(r["report"] for r in rows if r.get("event_type")=="public_hanging_proposal" and r["observation_sequence"]==464)
        points=np.array(report["body_polygon_px"])
        camera=observation.camera_frames["perception_camera"]
        h,w=camera.depth_m.shape
        padding=48*max(w/320,h/240)
        low=np.maximum([1,1],np.floor(points.min(0)-padding)).astype(int)
        high=np.minimum([w-2,h-2],np.ceil(points.max(0)+padding)).astype(int)
        result=depth_contrast_aperture(camera,tuple([*low,*high]))
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["semantics"],"unknown")
        self.assertEqual(result["status"],"unavailable")
        self.assertTrue(result["candidates"])
        self.assertTrue(all(c["aperture"]["status"]=="rejected" for c in result["candidates"]))
        print("Saved 464 depth contrast:",json.dumps({"status":result["status"],"reason":result["reason"],
            "candidates":[{"depth_recess_m":c["depth_recess_m"],"status":c["aperture"]["status"],
                           "reason":c["aperture"].get("reason")} for c in result["candidates"]]}))

    def setUp(self):
        self.intrinsic = np.array([[200., 0., 50.], [0., 200., 50.], [0., 0., 1.]])
        self.depth = np.ones((101, 101), dtype=np.float32)
        angles = np.arange(16)*2*np.pi/16
        self.contour = np.column_stack((50+10*np.cos(angles), 50+8*np.sin(angles)))
        self.material = np.rint(np.column_stack((50+17*np.cos(angles), 50+15*np.sin(angles))))

    def camera(self, depth=None, extrinsic=None):
        return RGBDCamera(self.depth if depth is None else depth, self.intrinsic,
                          np.eye(4) if extrinsic is None else extrinsic)

    def test_plane_center_normal_and_handedness_without_sampling_hole_depth(self):
        depth = self.depth.copy()
        depth[44:57, 44:57] = 5.  # Background through the opening is not material.
        result = estimate(self.camera(depth), self.contour, self.material)
        self.assertEqual(result["status"], "candidate_not_motion_validated")
        transform = _pose_matrix(result["frame_pose_world_wxyz"])
        np.testing.assert_allclose(transform[:3, 3], [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(transform[:3, 0], [0, 0, -1], atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.)
        self.assertFalse(result["motion_authorized"])
        self.assertEqual(result["insertion_direction"], "unknown")
        self.assertEqual(result["uncertainty"]["systematic_depth_calibration_error"], "unknown")

    def test_inscribed_circle_is_measured_on_plane_not_background(self):
        # A regular octagon inscribed in a 50 mm radius circle has an inradius
        # of R*cos(pi/8), independently of background depth through the hole.
        angles = np.arange(8)*2*np.pi/8
        contour = np.column_stack((50+10*np.cos(angles), 50+10*np.sin(angles)))
        result = aperture_dimensions(self.camera(), contour, np.array([0.,0.,1.]),
                                     np.array([0.,0.,-1.]))
        self.assertAlmostEqual(result["maximum_inscribed_circle_radius_m"], .05*np.cos(np.pi/8))
        self.assertAlmostEqual(result["centroid_inscribed_radius_m"], .05*np.cos(np.pi/8))
        self.assertIsNone(result["peg_radius_m"])
        self.assertFalse(result["clearance_validated"])

    def test_saved_entry660_two_pixel_shift_has_conditioned_visible_arch_fit(self):
        import json
        from harness.perception.replay import load_public_observation
        root = Path(__file__).resolve().parents[1]
        archive = root / ("data/runs/entry660_20261029T202054Z_1793305254857158600/"
                          "observations/observation_00000660.npz")
        annotation = root / "data/runs/entry660_reviewed_demo_feasibility/handle_annotation.json"
        if not archive.is_file() or not annotation.is_file():
            self.skipTest("ignored local Entry 660 recording is unavailable")
        a = json.loads(annotation.read_text())
        pixels = [[111,49] if uv == [113,49] else uv for uv in a["material_px"]]
        camera = load_public_observation(archive).camera_frames["perception_camera"]
        result = estimate(camera, a["contour_px"], pixels)
        self.assertEqual(result["status"], "candidate_not_motion_validated")
        self.assertFalse(result["motion_authorized"])
        self.assertGreater(result["uncertainty"]["visible_arch_spread_deg"], 150.)
        self.assertLess(result["uncertainty"]["plane_condition_number"], 10.)
        self.assertLess(result["uncertainty"]["maximum_plane_residual_m"], .0012)
        self.assertLess(result["geometry_diagnostics"]["maximum_inscribed_circle_radius_m"], .015)

    def test_tilted_plane_uses_calibrated_rays(self):
        u = np.arange(101)[None, :]
        depth = np.broadcast_to(1/(1+.2*(u-50)/200), (101, 101)).astype(np.float32)
        result = estimate(self.camera(depth), self.contour, self.material,
                          limits=replace(Limits(), maximum_center_sensitivity_m=.006))
        self.assertEqual(result["status"], "candidate_not_motion_validated")
        normal = _pose_matrix(result["frame_pose_world_wxyz"])[:3, 0]
        expected = -np.array([.2, 0, 1])/np.linalg.norm([.2, 0, 1])
        np.testing.assert_allclose(normal, expected, atol=.003)

    def test_world_translation_changes_center_not_camera_facing_normal(self):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [-1., -2., -3.]
        result = estimate(self.camera(extrinsic=extrinsic), self.contour, self.material)
        np.testing.assert_allclose(result["frame_pose_world_wxyz"][:3], [1., 2., 4.], atol=1e-7)

    def test_missing_material_depth_rejects(self):
        self.depth[50, 67] = 0
        result = estimate(self.camera(), self.contour, self.material)
        self.assertEqual(result["reason"], "material_depth_missing")

    def test_nearer_unconnected_structure_does_not_reject_reviewed_material(self):
        depth = self.depth.copy()
        depth[45, 72] = .8  # Within radius eight, outside the sample's surface.
        camera = self.camera(depth)
        self.assertGreater(camera.local_depth_gap(67, 50, search_radius=8), .06)
        points = _material_points(camera, [[67, 50]], Limits(), self.contour)
        np.testing.assert_allclose(points[0], [.085, 0, 1])

    def test_refinement_keeps_background_and_mixed_patch_rejection(self):
        depth = self.depth.copy()
        depth[44:57, 44:57] = 5.
        with self.assertRaisesRegex(ValueError, "material_sample_inside_opening"):
            _material_points(self.camera(depth), [[50, 50]], Limits(), self.contour)
        depth = self.depth.copy()
        depth[51, 68] = 5.
        with self.assertRaisesRegex(ValueError, "material_depth_discontinuity"):
            _material_points(self.camera(depth), [[67, 50]], Limits(), self.contour)

    def test_saved_entry660_three_contiguous_samples_survive_nearer_body(self):
        from harness.perception.replay import load_public_observation
        archive = Path(__file__).resolve().parents[1] / (
            "data/runs/entry660_20261029T202054Z_1793305254857158600/"
            "observations/observation_00000660.npz")
        if not archive.is_file():
            self.skipTest("ignored local Entry 660 recording is unavailable")
        camera = load_public_observation(archive).camera_frames["perception_camera"]
        # Existing assistant-reviewed contour, not a new annotation for this test.
        contour = np.array([[112,39],[114,37.8],[118,38.2],[123,40.5],[126.5,44.5],
                            [122,47],[116,50],[113.5,47.5],[111.8,43]])
        for pixel in ([124,38], [126,40], [128,42]):
            with self.subTest(pixel=pixel):
                self.assertGreater(camera.local_depth_gap(*pixel, search_radius=8), .06)
                points = _material_points(camera, [pixel], Limits(), contour)
                np.testing.assert_allclose(points[0], camera.unproject(*pixel, radius=1))
        with self.assertRaisesRegex(ValueError, "material_sample_inside_opening"):
            _material_points(camera, [[118,43]], Limits(), contour)

    def test_mixed_patch_rejects_even_when_median_is_foreground(self):
        self.depth[51, 68] = 5
        result = estimate(self.camera(), self.contour, self.material)
        self.assertEqual(result["reason"], "material_depth_discontinuity")

    def test_only_three_points_cannot_establish_surrounding_surface(self):
        result = estimate(self.camera(), self.contour, self.material[:3])
        self.assertEqual(result["status"], "rejected")

    def test_one_visible_arc_is_insufficient(self):
        angles = np.linspace(-.6, .6, 16)
        material = np.column_stack((50+35*np.cos(angles), 50+35*np.sin(angles)))
        result = estimate(self.camera(), self.contour, material)
        self.assertEqual(result["reason"], "material_does_not_surround_opening")

    def test_self_crossing_contour_is_not_completed(self):
        result = estimate(self.camera(), self.contour[np.arange(16)*3 % 16], self.material)
        self.assertEqual(result["reason"], "contour_requires_ordered_strictly_convex_visible_polygon")

    def test_background_depth_used_as_material_does_not_silently_pass(self):
        material = self.material.copy()
        material[0] = [50, 50]
        result = estimate(self.camera(), self.contour, material)
        self.assertEqual(result["reason"], "material_sample_inside_opening")

    def test_registration_sensitivity_has_explicit_rejection_limit(self):
        limits = replace(Limits(), maximum_center_sensitivity_m=.001)
        result = estimate(self.camera(), self.contour, self.material, limits=limits)
        self.assertEqual(result["reason"], "aperture_center_sensitive")
        self.assertGreater(result["uncertainty"]["center_sensitivity_m"], .004)

    def test_nonplanar_samples_reject(self):
        depth = self.depth.copy()
        # A shallow curved surface remains locally continuous but is not planar.
        u, v = np.meshgrid(np.arange(101), np.arange(101))
        depth += .0001*((u-50)**2-(v-50)**2)
        result = estimate(self.camera(depth), self.contour, self.material)
        self.assertEqual(result["reason"], "visible_material_not_planar")

    def test_duplicate_samples_do_not_inflate_plane_evidence(self):
        material = self.material.copy()
        material[1] = material[0]+.01
        result = estimate(self.camera(), self.contour, material)
        self.assertEqual(result["reason"], "duplicate_depth_samples")

    def test_archive_identity_and_review_are_required_before_estimation(self):
        observation = SimpleNamespace(sequence=660, camera_frames={"perception_camera": self.camera()})
        annotation = dict(archive_sha256=sha256(b"synthetic archive identity").hexdigest(),
            sequence=660, camera="perception_camera", reviewed=True,
            complete_visible_contour=True, contour_px=self.contour.tolist(),
            material_px=self.material.tolist())
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/"fixture.npz"
            archive.write_bytes(b"synthetic archive identity")
            with patch("development.hanging.aperture_frame.load_public_observation",
                       return_value=observation):
                missing = inspect_archive(archive)
                self.assertEqual(missing["reason"], "reviewed_contour_and_material_samples_required")
                for change in (dict(reviewed=False), dict(complete_visible_contour=False),
                               dict(archive_sha256="different"), dict(sequence=740)):
                    with self.subTest(change=change):
                        result = inspect_archive(archive, {**annotation, **change})
                        self.assertEqual(result["reason"], "annotation_unreviewed_incomplete_or_wrong_archive")
                valid = inspect_archive(archive, annotation)
                self.assertEqual(valid["status"], "candidate_not_motion_validated")
                self.assertFalse(valid["motion_authorized"])


if __name__ == "__main__":
    unittest.main()
