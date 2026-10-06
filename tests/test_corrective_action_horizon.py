"""Matched action budgets through public episodes and sealed synthetic replay."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from recorded_replay import TraceError, load_recorded_replay
import test_intervention_limits as fixtures


class CorrectiveActionHorizonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InterventionLimitTests()
        self.fixture.setUp()

    def episode(self, choices, horizon):
        config, *adapters = self.fixture.episode(choices)
        return (replace(config, max_steps=horizon), *adapters)

    def test_baseline_adjustments_and_recovery_share_exact_horizon(self):
        for choices, selections in (
            (['pass'] * 4, [0, 1, 2]),
            (['override'] * 4, [0, 1, 2]),
            (['pass', fixtures.TOOL, 'override'], [0, 1]),
            ([fixtures.TOOL, 'override', 'pass'], [0, 2]),
        ):
            with self.subTest(choices=choices), TemporaryDirectory() as tmp:
                episode = self.episode(choices, 3)
                resumes = []
                resume = episode[1].resume
                def record_resume(packet):
                    resumes.append(packet.sequence)
                    resume(packet)
                episode[1].resume = record_resume
                outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', episode)
                self.assertEqual((outcome.steps, outcome.stop_reason, outcome.success),
                                 (3, 'step_limit', False))
                self.assertEqual(len(episode[2].actions), 3)
                self.assertEqual(episode[4], selections)
                self.assertEqual(len(episode[1].observations), len(selections))
                self.assertNotIn(3, resumes)
                records = [row['action_record'] for row in evidence['decisions']]
                self.assertEqual(len(records), 3)
                self.assertEqual([r['executed_action'] for r in records],
                                 [list(action) for action in episode[2].actions])
                self.assertEqual([r['execution_acknowledgement']['result_sequence']
                                  for r in records], [1, 2, 3])

    def test_two_recoveries_consume_four_actions_not_two_interventions(self):
        with TemporaryDirectory() as tmp:
            episode = self.episode([fixtures.TOOL, fixtures.TOOL, 'pass'], 4)
            outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', episode)
            self.assertEqual((outcome.steps, outcome.stop_reason), (4, 'step_limit'))
            self.assertEqual(len(episode[2].actions), 4)
            self.assertEqual(episode[4], [0, 2])
            last = evidence['decisions'][-1]['action_record']
            self.assertEqual(last['intervention_budget']['interventions'], 2)
            self.assertEqual(last['recovery']['check']['status'], 'completed')

    def test_recovery_cannot_start_with_only_one_action_remaining(self):
        for prefix in ([], ['pass'], ['override'], [fixtures.TOOL]):
            consumed = 2 if prefix == [fixtures.TOOL] else len(prefix)
            with self.subTest(prefix=prefix), TemporaryDirectory() as tmp:
                episode = self.episode(prefix + [fixtures.TOOL], consumed + 1)
                outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', episode)
                self.assertEqual((outcome.steps, outcome.stop_reason),
                                 (consumed, 'proposal_rejected'))
                self.assertEqual(len(episode[2].actions), consumed)
                rejected = evidence['decisions'][-1]['action_record']
                self.assertEqual(rejected['rejection_reason'], 'insufficient recovery action horizon')
                self.assertIsNone(rejected['executed_action'])
                self.assertIsNone(rejected['intervention_budget'])

    def test_terminal_recovery_counts_only_commands_actually_executed(self):
        for terminal_step in (1, 2):
            for ending in ('success', 'terminated', 'truncated'):
                with self.subTest(step=terminal_step, ending=ending), TemporaryDirectory() as tmp:
                    episode = self.episode([fixtures.TOOL, 'pass'], 2)
                    step = episode[2].step
                    def terminal(action):
                        result = step(action)
                        if result.observation.sequence == terminal_step:
                            result = replace(result, **{ending: True})
                        return result
                    episode[2].step = terminal
                    outcome, _ = self.fixture.run_trace(Path(tmp) / 'trace', episode)
                    self.assertEqual((outcome.steps, outcome.stop_reason), (terminal_step, ending))
                    self.assertEqual(len(episode[2].actions), terminal_step)
                    self.assertEqual(episode[4], [0])

    def test_sealed_replay_rejects_changed_horizon(self):
        for choices in (['pass', fixtures.TOOL], [fixtures.TOOL, 'override']):
            for horizon in (2, 4):
                with self.subTest(choices=choices, horizon=horizon), TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'trace'
                    self.fixture.run_trace(path, self.episode(choices, 3))
                    manifest = json.loads((path / 'manifest.json').read_text())
                    manifest['config']['max_steps'] = horizon
                    (path / 'manifest.json').write_text(json.dumps(manifest))
                    with self.assertRaises(TraceError):
                        load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
