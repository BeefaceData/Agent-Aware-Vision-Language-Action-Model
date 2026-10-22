"""CPU contracts for one-time Opus RGB rack localization."""

from types import SimpleNamespace
import unittest

from harness.environment.base import Observation
from harness.perception.opus_rack import OpusRackGrounder
from tests.helpers import images


class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        block = SimpleNamespace(type="tool_use", name="report_rack_localization",
                                input=self.result)
        return SimpleNamespace(content=[block])


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


def observation():
    return Observation(0, images(), (0.0,) * 14)


class OpusRackGroundingTests(unittest.TestCase):
    def test_returns_normalized_public_reference_and_uses_one_rgb_image(self):
        client = FakeClient({"visible": True, "bbox_px": [1, 0, 3, 2],
            "peg_px": [2, 1], "description": "angled rack peg", "confidence": .62})
        grounder = OpusRackGrounder(client, "claude-test-model")

        result = grounder.locate(observation())

        self.assertEqual(result["status"], "grounded")
        self.assertEqual(result["source"], "opus_rgb")
        self.assertEqual(result["bbox_normalized"], [1/3, 0.0, 1.0, 1.0])
        self.assertEqual(result["peg_uv_normalized"], [1.0, 1.0])
        request = client.messages.requests[0]
        self.assertEqual(request["model"], "claude-test-model")
        self.assertEqual(request["tool_choice"],
                         {"type": "tool", "name": "report_rack_localization"})
        self.assertEqual(len(request["messages"][0]["content"]), 2)
        self.assertEqual(request["messages"][0]["content"][1]["type"], "image")

    def test_miss_is_a_nonfatal_structured_result(self):
        client = FakeClient({"visible": False, "bbox_px": None, "peg_px": None,
            "description": "rack unclear", "confidence": .2})
        result = OpusRackGrounder(client, "claude-test-model").locate(observation())
        self.assertEqual(result["status"], "not_detected")
        self.assertNotIn("peg_uv_normalized", result)

    def test_rejects_pixel_outside_box_or_image(self):
        client = FakeClient({"visible": True, "bbox_px": [0, 0, 1, 1],
            "peg_px": [2, 1], "description": "bad", "confidence": .5})
        with self.assertRaisesRegex(ValueError, "coordinates_out_of_bounds"):
            OpusRackGrounder(client, "claude-test-model").locate(observation())


if __name__ == "__main__":
    unittest.main()
