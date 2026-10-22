import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from development.sam3_prompt_replay import PROMPTS, replay, validate_selection


class Remote:
    def __init__(self):
        self.calls = []

    def remote(self, png, *, observation_sequence, camera):
        self.calls.append((observation_sequence, camera))
        mask = BytesIO()
        Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(mask, format="PNG")
        return {"observation_sequence": observation_sequence, "camera": camera,
            "variants": [{"prompt": prompt, "confidence": 0.7,
                "mask_area_fraction": 1.0, "mask_png": mask.getvalue()}
                for prompt in PROMPTS]}


class Worker:
    def __init__(self):
        self.detect_rack_prompt_variants = Remote()


class Sam3PromptReplayTests(unittest.TestCase):
    def test_request_limit_and_camera_validation(self):
        self.assertEqual(validate_selection([0, 4], ["head_camera"]),
                         ((0, 4), ("head_camera",)))
        with self.assertRaisesRegex(ValueError, "request_limit"):
            validate_selection(list(range(5)), ["head_camera", "left_camera", "right_camera"])
        with self.assertRaisesRegex(ValueError, "camera_selection"):
            validate_selection([0], ["simulator_oracle"])

    def test_replays_saved_public_rgb_only_and_records_prompt_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations"
            observations.mkdir()
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            np.savez_compressed(observations / "observation_00000007.npz",
                head_camera=image, left_camera=image, right_camera=image,
                sequence=7, command_state=np.zeros(14))
            worker = Worker()
            output = root / "result"
            report = replay(observations, [7], ["head_camera"], output, worker=worker)
            self.assertFalse(report["robot_motion"])
            self.assertFalse(report["local_sam3_loaded"])
            self.assertEqual(worker.detect_rack_prompt_variants.calls, [(7, "head_camera")])
            self.assertTrue((output / "results.json").is_file())
            self.assertEqual(len(list(output.glob("*.png"))), len(PROMPTS))


if __name__ == "__main__":
    unittest.main()
