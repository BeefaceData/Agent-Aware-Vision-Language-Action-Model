import json
import unittest

import numpy as np

from development.extract_checkpoint import extract_checkpoint
from development.checkpoint_bank import load_prefix
from development.replay import ReplayEnvironment
from tests import test_checkpoints as fixtures
from tests.test_baseline_compatibility import FakeRuntime


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.bank = self.fixture.bank
        self.source, _, _ = self.fixture.capture()

    def test_exact_prefix_images_and_replay_without_inherited_labels(self):
        source = self.bank.relabel(self.source.checkpoint_id, phase="hang_mug", failure_type="dropped_mug")
        child = extract_checkpoint(self.bank, source, 3, history=2)
        self.assertEqual(child.failure_type.value, "unknown")
        self.assertEqual(child.phase.value, "unknown")
        self.assertEqual(child.state, (3.0,) * 14)
        self.assertEqual(len(child.recent_states), 2)
        self.assertEqual(len(child.recent_actions), 2)
        self.assertEqual(load_prefix(self.bank, child), load_prefix(self.bank, source)[:4])
        for im in self.bank.load_images(child).values():
            np.testing.assert_array_equal(im, np.full((2, 3, 3), 3, dtype=np.uint8))
        self.assertEqual(child.restore(ReplayEnvironment(self.bank, FakeRuntime().sim)).status, "matched")
        self.bank.verify_artifacts(source)
        provenance = json.loads(self.bank.resolve_artifact(child.provenance_path).read_text())
        self.assertEqual(provenance["extraction"]["source_artifact_hashes"], source.artifact_hashes)
        with self.assertRaises(FileExistsError): extract_checkpoint(self.bank, source, 3)

    def test_reset_boundary_and_missing_camera_boundary(self):
        child = extract_checkpoint(self.bank, self.source, 0)
        self.assertEqual(child.recent_actions, ())
        self.assertEqual(child.restore(ReplayEnvironment(self.bank, FakeRuntime().sim)).status, "matched")
        with self.assertRaises(FileNotFoundError): extract_checkpoint(self.bank, self.source, 2)
        for step in (-1, 5, 6, True):
            with self.assertRaises(ValueError): extract_checkpoint(self.bank, self.source, step)

    def test_tampered_parent_and_malformed_image_rejected_before_publication(self):
        image = self.bank.resolve_artifact(self.source.trace_path).parent / "images_000003.npz"
        image.write_bytes(b"not an image archive")
        with self.assertRaises(ValueError): extract_checkpoint(self.bank, self.source, 3)
        self.assertFalse((self.bank.path / "extracted").exists())
        self.bank.resolve_artifact(self.source.trace_path).write_text("tampered")
        with self.assertRaises(ValueError): extract_checkpoint(self.bank, self.source, 0)
        self.assertEqual(len(self.bank.query()), 1)
