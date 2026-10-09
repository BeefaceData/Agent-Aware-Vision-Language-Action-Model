"""Optional extended-horizon episodes through the public recorder and replay."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class UncappedActionHorizonTests(unittest.TestCase):
    def test_mode_requires_declared_reference_and_operational_caps(self):
        cases = [
            ({}, 'max_steps'),
            ({'standard_action_horizon': 2}, 'intervention'),
            ({'standard_action_horizon': 2, 'max_interventions': 0}, 'model-call'),
            ({'standard_action_horizon': 2, 'max_interventions': 0,
              'max_supervisor_calls': 0}, 'wall-clock'),
        ]
        for fields, error in cases:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, error):
                EpisodeConfig(17, None, **fields)

    def test_complete_episode_extends_past_reference_and_stops_at_time_cap(self):
        for terminal in (True, False):
            with self.subTest(terminal=terminal), TemporaryDirectory() as tmp:
                now = [10.]
                observation = {'task': 'place item'}
                config = EpisodeConfig(17, None, standard_action_horizon=2,
                                       max_interventions=0, max_supervisor_calls=0,
                                       max_episode_seconds=1.)
                environment = ReplayEnvironment(17, observation, [
                    ([0], ReplayStep(observation, 0, terminal and i == 3,
                                     terminal and i == 3, False))
                    for i in range(5)], clock=lambda: now[0])
                policy = ReplayPolicy([(observation, [0])] * 5)
                directory = Path(tmp) / 'trace'
                trace = TraceRecorder(directory, config, ReplayRecorder())

                def after_step(step, result):
                    if step == 4 and not terminal:
                        now[0] = 11.

                outcome = run_episode(config, policy, environment, trace,
                                      on_step=after_step, clock=lambda: now[0])
                trace.seal(outcome)
                replay = load_recorded_replay(directory)
                report = replay.report()
                manifest = json.loads((directory / 'manifest.json').read_text())

                self.assertEqual(outcome.steps, 4)
                self.assertEqual(outcome.evaluation_scope,
                                 'optional_uncapped_action_horizon')
                self.assertEqual(len(environment.actions), 4)
                self.assertEqual(outcome.stop_reason,
                                 'success' if terminal else 'wall_clock_limit')
                self.assertEqual(report['replay_outcome']['stop_reason'], outcome.stop_reason)
                self.assertEqual(report['standard_action_horizon'], 2)
                self.assertEqual(report['evaluation_scope'], 'optional_uncapped_action_horizon')
                self.assertTrue(report['optional_ablation'])
                self.assertFalse(report['primary_acceptance_evidence'])
                self.assertEqual(manifest['evaluation_scope'], report['evaluation_scope'])
                self.assertIsNone(manifest['config']['max_steps'])
                manifest['evaluation_scope'] = 'standard'
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(TraceError, 'evaluation scope'):
                    load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
