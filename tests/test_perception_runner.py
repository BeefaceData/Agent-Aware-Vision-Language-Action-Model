"""CPU tests for the public detector integration seam used by run_skills."""

import unittest
import sys

from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.runtime.assembly import load_perception_provider


class FakeDetector:
    def detect(self, images, *, observation_sequence):
        raise AssertionError("factory loading must not run inference")


class ConfigurableFakeDetector(FakeDetector):
    profile = "default"

    def configure_rack_prompt_profile(self, profile):
        self.profile = profile


def configurable_fake_detector_factory():
    return ConfigurableFakeDetector()


def fake_detector_factory():
    return FakeDetector()


class PerceptionRunnerTests(unittest.TestCase):
    def test_loads_explicit_rgb_only_detector_without_running_it(self):
        provider = load_perception_provider(
            f"{__name__}:fake_detector_factory", AnchorRegistry())
        self.assertIsInstance(provider, RGBDPerceptionProvider)
        self.assertIsInstance(provider.detector, FakeDetector)

    def test_rejects_malformed_or_non_detector_factories(self):
        for spec in ("bad", ":factory", f"{__name__}:missing"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                load_perception_provider(spec, AnchorRegistry())

        class InvalidFactory:
            pass

        module = sys.modules[__name__]
        module.invalid_factory = InvalidFactory
        try:
            with self.assertRaisesRegex(ValueError, "missing_detect"):
                load_perception_provider(
                    f"{__name__}:invalid_factory", AnchorRegistry())
        finally:
            del module.invalid_factory

    def test_exploration_profile_is_opt_in_and_only_lowers_rack_threshold(self):
        provider = load_perception_provider(
            f"{__name__}:configurable_fake_detector_factory", AnchorRegistry(),
            rack_prompt_profile="explore_mug_rack")
        self.assertEqual(provider.detector.profile, "explore_mug_rack")
        self.assertEqual(provider.min_confidence_by_label, {"mug": .40, "rack": .18})
        with self.assertRaisesRegex(ValueError, "unsupported_rack_prompt_profile"):
            load_perception_provider(f"{__name__}:configurable_fake_detector_factory",
                                     AnchorRegistry(), rack_prompt_profile="anything")

    def test_exploration_profile_requires_detector_support(self):
        with self.assertRaisesRegex(ValueError, "does_not_support_rack_prompt_profile"):
            load_perception_provider(f"{__name__}:fake_detector_factory", AnchorRegistry(),
                                     rack_prompt_profile="explore_mug_rack")


if __name__ == "__main__":
    unittest.main()
