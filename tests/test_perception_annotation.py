"""CPU geometry/serialization checks for the manual RGB-D annotation tool."""

import unittest

import numpy as np

from development.capture.annotate_rgbd import encode_mask_rle, polygon_mask
from harness.perception.replay import decode_mask_rle


class PerceptionAnnotationTests(unittest.TestCase):
    def test_polygon_rasterization_and_rle_roundtrip(self):
        mask = polygon_mask((6, 7), [(1, 1), (4, 1), (4, 4), (1, 4)])
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])
        encoded = encode_mask_rle(mask)
        restored = decode_mask_rle(encoded["shape"], encoded["runs"])
        np.testing.assert_array_equal(restored, mask)

    def test_polygon_and_mask_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "polygon_requires_three_points"):
            polygon_mask((4, 4), [(0, 0), (3, 3)])
        with self.assertRaisesRegex(ValueError, "annotation_polygon_out_of_bounds"):
            polygon_mask((4, 4), [(0, 0), (3, 3), (4, 1)])
        with self.assertRaisesRegex(ValueError, "invalid_annotation_mask"):
            encode_mask_rle(np.zeros((3, 3), dtype=bool))


if __name__ == "__main__":
    unittest.main()
