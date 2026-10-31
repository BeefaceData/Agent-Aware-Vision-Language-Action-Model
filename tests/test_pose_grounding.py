"""Public RGB-D keypoint-frame and dynamic hang-target geometry contracts."""

import unittest

import numpy as np

from harness.perception.pose_grounding import (
    grounded_frame_from_pixels, tcp_targets_for_mug_alignment,
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
