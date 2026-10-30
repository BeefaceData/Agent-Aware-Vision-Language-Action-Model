"""Timing attribution contracts with known durations; no GPU or provider calls."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from development.replays.latency import analyze_episode, main, render_markdown


FIXTURE = Path(__file__).parent / "fixtures/latency_episode"


def stages(report):
    return {stage["name"]: stage for stage in report["stages"]}


class LatencyReportTests(unittest.TestCase):
    def test_known_statistics_units_and_nested_reconciliation(self):
        report = analyze_episode(FIXTURE)
        measured = stages(report)
        self.assertEqual(measured["planner_selection"]["count"], 2)
        self.assertEqual(measured["planner_selection"]["median_seconds"], 3.)
        self.assertAlmostEqual(measured["planner_selection"]["p90_seconds"], 3.8)
        self.assertAlmostEqual(measured["action_round_trip"]["p90_seconds"], .38)
        self.assertEqual(measured["vla_request"]["total_seconds"], 3.)
        self.assertEqual(measured["policy_server_inference"]["total_seconds"], 1.5)
        self.assertEqual(measured["policy_rpc_residual"]["total_seconds"], .5)
        # No double counting of turn-result copies, policy children or reset.
        self.assertEqual(report["reconciliation"]["measured_sibling_seconds"], 17.)
        self.assertEqual(report["reconciliation"]["unexplained_seconds"], 3.)
        self.assertEqual(report["reconciliation"]["measured_fraction"], .85)
        self.assertIsNone(report["reconciliation"]["full_process_elapsed_seconds"])

    def test_missing_measurements_are_unknown_not_zero(self):
        report = analyze_episode(FIXTURE)
        for name in ("perception", "predicate_evaluation", "analytic_ticks", "artifact_writing"):
            self.assertEqual(stages(report)[name]["count"], 0)
            self.assertIsNone(stages(report)[name]["total_seconds"])
        self.assertIsNone(report["metadata"]["source_commit"])
        self.assertTrue(report["metadata"]["synthetic"])
        self.assertIn("SYNTHETIC VALIDATION", render_markdown(report))

    def copy_fixture(self, directory):
        episode = Path(directory) / "episode"
        shutil.copytree(FIXTURE, episode)
        return episode

    def test_truncated_last_line_is_partial_but_malformed_middle_line_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            path = episode / "skill_events.jsonl"
            with path.open("a") as stream:
                stream.write('{"event_type":')
            report = analyze_episode(episode)
            self.assertTrue(any("incomplete final" in warning for warning in report["warnings"]))
            with path.open("a") as stream:
                stream.write('\n{}\n')
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                analyze_episode(episode)

    def test_interrupted_episode_without_elapsed_time_has_no_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            (episode / "summary.json").write_text('{"outcome":"interrupted"}')
            report = analyze_episode(episode)
            self.assertEqual(report["metadata"]["outcome"], "interrupted")
            self.assertEqual(report["reconciliation"]["status"], "unavailable")
            self.assertIsNone(report["reconciliation"]["unexplained_seconds"])

    def test_duplicate_skill_results_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            with (episode / "skill_events.jsonl").open("a") as stream:
                stream.write('{"event_type":"skill_result","call_id":1,"result":{"elapsed_seconds":8}}\n')
            report = analyze_episode(episode)
            self.assertEqual(stages(report)["skill_execution"]["total_seconds"], 11.)
            self.assertEqual(report["reconciliation"]["status"], "unavailable")

    def test_inconsistent_totals_do_not_produce_negative_unexplained_time(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            (episode / "summary.json").write_text('{"elapsed_seconds":1}')
            report = analyze_episode(episode)
            self.assertIsNone(report["reconciliation"]["unexplained_seconds"])
            self.assertIn("exceed", report["reconciliation"]["reason"])

    def test_invalid_durations_are_rejected_and_never_serialized_as_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            with (episode / "skill_events.jsonl").open("a") as stream:
                for value in ('NaN', '-1', 'true', '"3"'):
                    stream.write('{"event_type":"action_confirmed","environment_step_ms":' + value + '}\n')
            report = analyze_episode(episode)
            self.assertEqual(stages(report)["action_round_trip"]["invalid_samples"], 4)
            self.assertEqual(stages(report)["action_round_trip"]["count"], 2)
            json.dumps(report, allow_nan=False)

    def test_legacy_run_without_policy_details_or_simulator_file_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            (episode / "simulator_timing.jsonl").unlink()
            path = episode / "skill_events.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record.pop("policy_timing", None)
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            report = analyze_episode(episode)
            self.assertIsNone(stages(report)["policy_rpc"]["total_seconds"])
            self.assertEqual(stages(report)["vla_request"]["total_seconds"], 3.)
            self.assertTrue(any("simulator_timing" in warning for warning in report["warnings"]))

    def test_report_excludes_private_payloads_errors_and_unrelated_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            manifest = json.loads((episode / "manifest.json").read_text())
            manifest.update(ANTHROPIC_API_KEY="PRIVATE_KEY", settings={"private": "PRIVATE_SETTING"})
            (episode / "manifest.json").write_text(json.dumps(manifest))
            with (episode / "assessments.jsonl").open("a") as stream:
                stream.write('{"event_type":"provider_error","error":"PRIVATE_PROVIDER_PAYLOAD"}\n')
            rendered = render_markdown(analyze_episode(episode))
            self.assertNotIn("PRIVATE", rendered)

    def test_cli_writes_both_reports_without_mutating_source_or_overwriting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self.copy_fixture(directory)
            before = {p.name: p.read_bytes() for p in episode.iterdir()}
            output = Path(directory) / "report"
            args = ["--episode", str(episode), "--output", str(output)]
            self.assertEqual(main(args), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["reconciliation"]["unexplained_seconds"], 3.)
            self.assertTrue((output / "report.md").exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in episode.iterdir()})
            with self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            with self.assertRaises(SystemExit):
                main(["--episode", str(episode), "--output", str(episode / "report")])


if __name__ == "__main__":
    unittest.main()
