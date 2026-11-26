"""Deployment defaults keep all GPU-backed Modal services on one GPU class."""

import unittest

from modal_apps.policy import DEFAULT_GPU as POLICY_GPU
from modal_apps.sam3 import DEFAULT_GPU as SAM3_GPU
from modal_apps.simulator import DEFAULT_GPU as SIMULATOR_GPU


class ModalGPUDefaultsTests(unittest.TestCase):
    def test_all_gpu_backed_apps_default_to_l4(self):
        self.assertEqual(
            {"policy": POLICY_GPU, "sam3": SAM3_GPU, "simulator": SIMULATOR_GPU},
            {"policy": "L4", "sam3": "L4", "simulator": "L4"},
        )


if __name__ == "__main__":
    unittest.main()
