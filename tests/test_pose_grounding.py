"""Public RGB-D keypoint-frame and dynamic hang-target geometry contracts."""

import unittest

import numpy as np

from harness.perception.pose_grounding import (
    frame_from_axes, grounded_frame_from_pixels, tcp_targets_for_mug_alignment, measured_peg_centerline,
)
from harness.perception.rgbd import RGBDCamera


def camera(depth=None):
    if depth is None:
        depth = np.ones((80, 80), dtype=np.float32)
    return RGBDCamera(depth, np.array([[100., 0., 40.], [0., 100., 40.],
                                       [0., 0., 1.]]), np.eye(4))


def matrix(pose):
    from harness.perception.pose_grounding import _pose_matrix
    return _pose_matrix(pose)


class PoseGroundingTests(unittest.TestCase):
    def shaft_camera(self):
        depth=np.ones((101,101),np.float32)
        for x,y in [(20,30),(70,30)]:
            depth[y-3:y+4,x-3:x+4]=1.004
        return RGBDCamera(depth,np.array([[1000.,0.,0.],[0.,1000.,0.],[0.,0.,1.]]),np.eye(4))

    def test_measured_cross_section_normalizes_surface_axis_to_centerline(self):
        rgbd=self.shaft_camera()
        points=[(20.,30.),(70.,30.),(20.,10.)]
        result=measured_peg_centerline(rgbd,points,[(45.,24.),(45.,36.)])
        pose=matrix(result["pose_world_wxyz"])
        self.assertAlmostEqual(pose[2,3],1.,places=6)
        self.assertGreater(result["centerline_offset_m"],.0039)
        self.assertAlmostEqual(result["radius_m"],.006,places=6)
        self.assertGreater(result["position_sensitivity_m"],0)
        self.assertEqual(result["systematic_calibration_error"],"unknown")

    def test_centerline_rejects_missing_wrong_axis_and_discontinuous_width(self):
        points=[(20.,30.),(70.,30.),(20.,10.)]
        with self.assertRaisesRegex(ValueError,"cross_section_required"):
            measured_peg_centerline(self.shaft_camera(),points,None)
        with self.assertRaisesRegex(ValueError,"wrong_cross_section"):
            measured_peg_centerline(self.shaft_camera(),points,[(45.,24.),(51.,36.)])
        rgbd=self.shaft_camera()
        depth=rgbd.depth_m.copy();depth[23,44]=2.
        broken=RGBDCamera(depth,rgbd.intrinsic,rgbd.world_to_camera)
        with self.assertRaisesRegex(ValueError,"depth_discontinuous"):
            measured_peg_centerline(broken,points,[(45.,24.),(45.,36.)])

    def test_rack_cache_recomputes_registration_from_pixels_not_supplied_pose(self):
        from harness.environment.base import Observation
        from harness.environment.layout import CAMERAS
        from harness.perception.hang_targets import StaticRackFrameCache
        rgbd=self.shaft_camera()
        observation=Observation(27,{name:np.zeros((101,101,3),np.uint8) for name in (*CAMERAS,"perception_camera")},
            tuple(np.zeros(14)),{"perception_camera":rgbd})
        report=dict(status="grounded",target="rack_peg",camera="perception_camera",observation_sequence=27,
            image_shape_px=[101,101],confidence=.95,keypoints_px=[(20.,30.),(70.,30.),(20.,10.)],
            peg_width_px=[(45.,24.),(45.,36.)],centerline_from_cross_section=True,
            pose_world_wxyz=[9.,9.,9.,1.,0.,0.,0.])
        cache=StaticRackFrameCache()
        cache.cache(observation,report)
        self.assertAlmostEqual(cache.resolve(observation)[2],1.,places=6)
        stale={**report,"observation_sequence":26}
        with self.assertRaisesRegex(ValueError,"stale"):
            cache.cache(observation,stale)

    def test_a_frame_from_axes_keeps_x_and_puts_y_in_the_given_plane(self):
        transform = matrix(frame_from_axes((1., 2., 3.), (2., 0., 0.), (1., 0., 1.)))

        np.testing.assert_allclose(transform[:3, 3], (1., 2., 3.))
        np.testing.assert_allclose(transform[:3, 0], (1., 0., 0.), atol=1e-9)
        np.testing.assert_allclose(transform[:3, 1], (0., 0., 1.), atol=1e-9)
        np.testing.assert_allclose(transform[:3, 2], (0., -1., 0.), atol=1e-9)  # x cross y

    def test_a_frame_from_parallel_axes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate_frame_axes"):
            frame_from_axes((0., 0., 0.), (1., 0., 0.), (2., 0., 0.))

    def test_ordered_rgbd_keypoints_make_a_right_handed_world_frame(self):
        frame = grounded_frame_from_pixels(camera(),
            [(40., 40.), (42., 40.), (40., 42.)])

        transform = matrix(frame)
        np.testing.assert_allclose(transform[:3, 3], [0., 0., 1.], atol=1e-7)
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)

    def test_depth_discontinuity_and_degenerate_keypoints_fail_closed(self):
        depths = np.ones((80, 80), dtype=np.float32)
        depths[32:49, 32:49] = 1.5
        with self.assertRaisesRegex(ValueError, "keypoint_depth_discontinuity"):
            grounded_frame_from_pixels(camera(depths),
                [(40., 40.), (42., 40.), (40., 42.)])
        with self.assertRaisesRegex(ValueError, "x_axis_baseline_out_of_bounds"):
            grounded_frame_from_pixels(camera(),
                [(40., 40.), (40.2, 40.), (40., 42.)])

    def test_median_cannot_hide_background_inside_selected_depth_patch(self):
        depth = np.ones((80, 80), dtype=np.float32)
        depth[39, 39] = 5.8
        rgbd = camera(depth)
        # The old foreground-gap check passes: median == nearby minimum == 1m.
        self.assertEqual(rgbd.local_depth_gap(40., 40., search_radius=8), 0.)
        with self.assertRaisesRegex(ValueError, "keypoint_depth_discontinuity"):
            grounded_frame_from_pixels(rgbd, [(40., 40.), (50., 40.), (40., 50.)])

    def test_tcp_targets_preserve_actual_grasp_and_follow_rack_transform(self):
        tcp = np.array([.2, -.1, 1., 1., 0., 0., 0.])
        mug = np.array([.23, -.1, 1.02, 1., 0., 0., 0.])
        rack = np.array([.4, .2, 1.1, 1., 0., 0., 0.])
        prehang, target = tcp_targets_for_mug_alignment(tcp, mug, rack)
        prehang_matrix, target_matrix = matrix(prehang), matrix(target)

        np.testing.assert_allclose(target_matrix[:3, 3], [.37, .2, 1.08], atol=1e-8)
        np.testing.assert_allclose(prehang_matrix[:3, 3], [.42, .2, 1.08], atol=1e-8)
        tcp_to_mug = np.linalg.inv(matrix(tcp)) @ matrix(mug)
        np.testing.assert_allclose(target_matrix @ tcp_to_mug, matrix(rack), atol=1e-8)
        self.assertGreater(float((prehang_matrix[:3, 3] - target_matrix[:3, 3])
                                 @ matrix(rack)[:3, 0]), 0.049)

        moved_rack = rack.copy()
        moved_rack[:3] += [.1, -.08, .04]
        shifted_pre, shifted_target = tcp_targets_for_mug_alignment(tcp, mug, moved_rack)
        np.testing.assert_allclose(matrix(shifted_target)[:3, 3] - target_matrix[:3, 3],
                                   [.1, -.08, .04], atol=1e-8)
        self.assertFalse(np.allclose(shifted_pre, prehang))

    def test_target_orientation_tracks_rotated_rack_frame(self):
        half = np.sqrt(.5)
        tcp = np.array([.1, .0, 1., 1., 0., 0., 0.])
        mug = np.array([.12, .0, 1., 1., 0., 0., 0.])
        rack = np.array([.4, .2, 1.1, half, 0., 0., half])

        _, target = tcp_targets_for_mug_alignment(tcp, mug, rack)

        np.testing.assert_allclose(matrix(target)[:3, :3], matrix(rack)[:3, :3], atol=1e-8)
        np.testing.assert_allclose(matrix(target) @ np.linalg.inv(matrix(tcp)) @ matrix(mug),
                                   matrix(rack), atol=1e-8)

    def test_invalid_poses_and_unsafe_approach_offsets_are_rejected(self):
        pose = [0., 0., 1., 1., 0., 0., 0.]
        for offset in (0., .101, float("nan")):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                tcp_targets_for_mug_alignment(pose, pose, pose,
                                              approach_offset_m=offset)
        with self.assertRaisesRegex(ValueError, "invalid_pose"):
            tcp_targets_for_mug_alignment(pose, [0., 0., 1., 0., 0., 0., 0.], pose)


if __name__ == "__main__":
    unittest.main()
