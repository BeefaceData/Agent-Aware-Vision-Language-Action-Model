import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from development.hanging.prepare_replay import prepare_replay
from development.checkpoints.bank import CheckpointBank, load_prefix
from development.checkpoints.replay import ReplayEnvironment
from harness.environment.layout import CAMERAS


class HangingReplayPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.original, self.recovered, self.output = [root / n for n in ("original", "recovered", "output")]
        for folder in (self.original, self.recovered):
            (folder / "observations").mkdir(parents=True)
        self.state = [0.] * 14
        for folder, sequence in ((self.recovered, 0), (self.original, 1)):
            np.savez(folder / "observations" / f"observation_{sequence:08d}.npz",
                     **{camera: np.zeros((4, 4, 3), dtype=np.uint8) for camera in CAMERAS},
                     sequence=sequence, command_state=self.state)
        (self.recovered / "manifest.json").write_text(json.dumps(
            dict(mode="frozen_policy_baseline", seed=3)))
        (self.original / "summary.json").write_text(json.dumps(dict(
            mode="frozen_policy_baseline", seed=3, policy_actions=1,
            execution_uncertain=False, error_type=None)))
        self.rows = [dict(event="action_requested", index=1, sequence=0, action=self.state),
                     dict(event="action_confirmed", index=1, sequence=1, command_state=self.state)]
        self.write_journal()

    def write_journal(self):
        (self.original / "actions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows))

    def prepare(self):
        return prepare_replay(self.original, self.recovered, self.output, step=1)

    def test_package_integrity_and_unknown_provenance_preserve_verified_gate(self):
        checkpoint = self.prepare()
        bank = CheckpointBank(self.output)
        bank.verify_artifacts(checkpoint)
        self.assertEqual(len(load_prefix(bank, checkpoint)), 2)
        provenance = json.loads((self.output / "provenance.json").read_text())
        self.assertIsNone(provenance["local"])
        self.assertFalse(provenance["verified_replay_eligible"])
        def forbidden_simulator(*args, **kwargs):
            self.fail("offline provenance rejection must precede simulator requests")
        with patch("development.checkpoints.replay.local_provenance", return_value={"test": True}):
            result = checkpoint.restore(ReplayEnvironment(bank, forbidden_simulator))
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.executed_steps, 0)
        self.assertFalse(result.full_state_restored)

    def test_reordered_confirmation_is_rejected_before_publication(self):
        self.rows[1]["sequence"] = 2
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "incomplete_or_reordered_prefix"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_nonfinite_action_and_boundary_mismatch_are_rejected(self):
        self.rows[0]["action"] = [float("nan")] * 14
        self.write_journal()
        with self.assertRaises(ValueError):
            self.prepare()
        self.rows[0]["action"] = self.state
        self.rows[1]["command_state"] = [1.] * 14
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "boundary_state_mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(), "unchanged")

    def test_unconfirmed_action_is_rejected_before_publication(self):
        self.rows.pop()
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "incomplete_or_reordered_prefix"):
            self.prepare()
        self.assertFalse(self.output.exists())
