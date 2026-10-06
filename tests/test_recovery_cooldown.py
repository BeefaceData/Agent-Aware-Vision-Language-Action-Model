"""Acknowledged-action recovery cooldown through public episodes and sealed replay."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig
from recorded_replay import TraceError, load_recorded_replay
import test_intervention_limits as fixtures


class RecoveryCooldownTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InterventionLimitTests()
        self.fixture.setUp()

    def episode(self, choices, cooldown):
        config, *rest = self.fixture.episode(choices, limits=((fixtures.TOOL, 10),))
        return (replace(config, recovery_cooldown_actions=cooldown), *rest)

    def test_recovery_and_adjustment_rejected_before_exact_expiry(self):
        for following in (fixtures.TOOL, 'override'):
            for baseline_actions in (0, 1):
                with self.subTest(following=following, baseline_actions=baseline_actions), TemporaryDirectory() as tmp:
                    fixture = self.episode([fixtures.TOOL] + ['pass'] * baseline_actions + [following], 2)
                    outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', fixture)
                    self.assertEqual((outcome.steps, outcome.stop_reason),
                                     (2 + baseline_actions, 'proposal_rejected'))
                    self.assertEqual(len(fixture[2].actions), outcome.steps)
                    rejected = evidence['decisions'][-1]['action_record']
                    self.assertEqual(rejected['rejection_reason'],
                        f'recovery cooldown: {2 - baseline_actions} baseline actions remaining')
                    self.assertIsNone(rejected['executed_action'])
                    self.assertIsNone(rejected['intervention_budget'])
                    self.assertEqual(evidence['config']['recovery_cooldown_actions'], 2)

    def test_expiry_and_disabled_cooldown_allow_both_modes(self):
        for cooldown in (0, 1, 2):
            for following in (fixtures.TOOL, 'override'):
                with self.subTest(cooldown=cooldown, following=following), TemporaryDirectory() as tmp:
                    fixture = self.episode([fixtures.TOOL] + ['pass'] * cooldown + [following], cooldown)
                    outcome, _ = self.fixture.run_trace(Path(tmp) / 'trace', fixture)
                    self.assertEqual(outcome.stop_reason, 'step_limit')
                    self.assertEqual(outcome.steps, fixture[0].max_steps)

    def test_each_recovery_restarts_cooldown(self):
        with TemporaryDirectory() as tmp:
            fixture = self.episode([fixtures.TOOL, 'pass', fixtures.TOOL, 'override'], 1)
            outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', fixture)
            self.assertEqual((outcome.steps, outcome.stop_reason), (5, 'proposal_rejected'))
            self.assertIn('1 baseline actions remaining', evidence['decisions'][-1]['action_record']['rejection_reason'])

    def test_new_episode_clears_cooldown_with_reused_adapters(self):
        fixture = self.episode([fixtures.TOOL, 'override'], 2)
        with TemporaryDirectory() as tmp:
            first, _ = self.fixture.run_trace(Path(tmp) / 'first', fixture)
            second, _ = self.fixture.run_trace(Path(tmp) / 'second', fixture)
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual((first.steps, second.steps), (2, 2))

    def test_invalid_cooldowns(self):
        for value in (-1, True, 1.5, float('inf'), float('nan'), '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EpisodeConfig(17, 10, recovery_cooldown_actions=value)

    def test_replay_rejects_altered_cooldown_or_rejection_even_with_valid_checksums(self):
        for damage in ('execution_limit', 'rejection_limit', 'reason'):
            with self.subTest(damage=damage), TemporaryDirectory() as tmp:
                path = Path(tmp) / 'trace'
                cooldown = 0 if damage == 'execution_limit' else 2
                self.fixture.run_trace(path, self.episode([fixtures.TOOL, 'override'], cooldown))
                manifest = json.loads((path / 'manifest.json').read_text())
                if damage == 'reason':
                    rows = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
                    rows[-1]['action_record']['rejection_reason'] = 'recovery cooldown: 1 baseline actions remaining'
                    data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                    (path / 'decisions.jsonl').write_bytes(data)
                    manifest['files']['decisions.jsonl'] = sha256(data).hexdigest()
                else:
                    manifest['config']['recovery_cooldown_actions'] = 2 if cooldown == 0 else 0
                (path / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(TraceError):
                    load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
