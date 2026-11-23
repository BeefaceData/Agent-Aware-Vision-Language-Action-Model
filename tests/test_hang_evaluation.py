"""Offline evidence must not confuse alignment, release and rack support."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from development.hanging.evaluation import HangTrial, evaluate


def assessment(value, sequence):
    return dict(value=value, sequence=sequence, source="public",
                artifact=f"observations/{sequence}.npz", reason="reviewed_public_evidence")


def trial_data():
    return dict(run_id="synthetic_hang", seed=3, evidence_mode="public",
        entry=assessment(True, 0), approach=assessment(True, 1),
        seat_alignment=assessment(True, 2), contact=assessment(None, 2),
        release=assessment(True, 3), retreat=assessment(True, 4),
        support_samples=[dict(sequence=i+5, elapsed_seconds=t, supported=True,
            source="public", artifact=f"observations/{i+5}.npz")
            for i, t in enumerate((0.0, 0.5, 1.0))])


class HangEvaluationTests(unittest.TestCase):
    def test_post_retreat_support_can_verify_hang_without_claiming_contact_detection(self):
        report = evaluate(HangTrial.model_validate(trial_data()))
        self.assertEqual(report.outcome, "hang_verified")
        self.assertEqual(report.stages["contact"].status, "unknown")
        self.assertIsNone(report.native_success)

    def test_alignment_and_native_success_do_not_prove_release_or_hang(self):
        data = trial_data()
        data.update(release=None, retreat=None, support_samples=[], native_success=True)
        report = evaluate(HangTrial.model_validate(data))
        self.assertEqual(report.stages["seat_alignment"].status, "passed")
        self.assertEqual(report.stages["release"].status, "unknown")
        self.assertEqual(report.outcome, "hang_unknown")

    def test_successful_approach_and_failed_release_remain_separate(self):
        data = trial_data()
        data["release"] = assessment(False, 3)
        report = evaluate(HangTrial.model_validate(data))
        self.assertEqual(report.stages["approach"].status, "passed")
        self.assertEqual(report.stages["release"].status, "failed")
        self.assertEqual(report.outcome, "hang_failed")

    def test_support_loss_after_opening_or_during_retreat_is_failure(self):
        for failed_retreat in (False, True):
            with self.subTest(failed_retreat=failed_retreat):
                data = trial_data()
                if failed_retreat:
                    data["retreat"] = assessment(False, 4)
                else:
                    data["support_samples"][-1]["supported"] = False
                self.assertEqual(evaluate(HangTrial.model_validate(data)).outcome, "hang_failed")

    def test_incomplete_or_ambiguous_support_is_unknown(self):
        for mode in ("absent", "ambiguous", "short", "gap"):
            with self.subTest(mode=mode):
                data = trial_data()
                if mode == "absent":
                    data["support_samples"] = []
                elif mode == "ambiguous":
                    data["support_samples"][1]["supported"] = None
                elif mode == "short":
                    data["support_samples"][-1]["elapsed_seconds"] = 0.75
                else:
                    data["support_samples"][-1]["elapsed_seconds"] = 1.5
                self.assertEqual(evaluate(HangTrial.model_validate(data)).outcome, "hang_unknown")

    def test_bad_or_unknown_entry_is_not_scored_as_hang_failure(self):
        for value, expected in ((False, "invalid_entry"), (None, "entry_unknown")):
            data = trial_data()
            data["entry"] = assessment(value, 0)
            data["support_samples"][-1]["supported"] = False
            self.assertEqual(evaluate(HangTrial.model_validate(data)).outcome, expected)

    def test_mixed_oracle_and_public_evidence_is_rejected(self):
        for target in ("assessment", "sample"):
            data = trial_data()
            if target == "assessment":
                data["release"]["source"] = "development_oracle"
            else:
                data["support_samples"][0]["source"] = "development_oracle"
            with self.assertRaisesRegex(ValueError, "mixed_public_and_oracle"):
                HangTrial.model_validate(data)

    def test_oracle_report_stays_explicitly_oracle(self):
        data = trial_data()
        data["evidence_mode"] = "development_oracle"
        for name in ("entry", "approach", "seat_alignment", "contact", "release", "retreat"):
            data[name]["source"] = "development_oracle"
        for sample in data["support_samples"]:
            sample["source"] = "development_oracle"
        report = evaluate(HangTrial.model_validate(data))
        self.assertEqual(report.evidence_mode, "development_oracle")

    def test_stale_reordered_and_nonfinite_evidence_is_rejected(self):
        for mode in ("stage", "before_retreat", "duplicate", "backwards_time", "nan"):
            with self.subTest(mode=mode):
                data = trial_data()
                if mode == "stage":
                    data["release"]["sequence"] = 1
                elif mode == "before_retreat":
                    data["support_samples"][0]["sequence"] = 3
                elif mode == "duplicate":
                    data["support_samples"][1]["sequence"] = 5
                elif mode == "backwards_time":
                    data["support_samples"][1]["elapsed_seconds"] = 0.0
                else:
                    data["support_samples"][0]["elapsed_seconds"] = float("nan")
                with self.assertRaises(ValueError):
                    HangTrial.model_validate(data)

    def test_cli_roundtrip_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence.json"
            output = Path(tmp) / "report.json"
            evidence.write_text(json.dumps(trial_data()), encoding="utf-8")
            command = [sys.executable, "-m", "development.hanging.evaluation",
                       str(evidence), "--output", str(output)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["outcome"], "hang_verified")
            before = output.read_bytes()
            repeated = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
