"""Observable baseline episode behavior through the public harness interface."""

import unittest

from episode_harness import EpisodeConfig, StepResult, run_episode
from run_smolvla_episode import parse_args


class ReplayPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.observations = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, observation):
        self.observations.append(observation)
        return next(self.actions)


class ReplayEnvironment:
    def __init__(self, results):
        self.results = iter(results)
        self.seeds = []
        self.actions = []

    def reset(self, seed):
        self.seeds.append(seed)
        return 'initial observation'

    def step(self, action):
        self.actions.append(action)
        return next(self.results)


class ReplayRecorder:
    def __init__(self):
        self.observations = []
        self.rows = []
        self.finalized = False

    def begin(self, observation):
        self.observations.append(observation)

    def record_step(self, step, action, result):
        self.rows.append((step, action, result.reward, result.success,
                          result.terminated, result.truncated))
        self.observations.append(result.observation)

    def finish(self):
        self.finalized = True
        return {'video_path': 'replay/episode.mp4',
                'steps_path': 'replay/steps.jsonl'}


class EpisodeHarnessTests(unittest.TestCase):
    def test_complete_episode_replays_observation_action_outcome_and_artifacts(self):
        policy = ReplayPolicy(['first action', 'second action'])
        environment = ReplayEnvironment([
            StepResult('next observation', 0.25, False, False, False),
            # This may be the terminal frame even if an underlying wrapper resets.
            StepResult('terminal observation', 1.0, True, True, False),
        ])
        recorder = ReplayRecorder()

        outcome = run_episode(EpisodeConfig(seed=17, max_steps=5),
                              policy, environment, recorder)

        self.assertEqual(environment.seeds, [17])
        self.assertEqual(policy.resets, 1)
        self.assertEqual(policy.observations,
                         ['initial observation', 'next observation'])
        self.assertEqual(environment.actions, ['first action', 'second action'])
        self.assertEqual(recorder.observations,
                         ['initial observation', 'next observation', 'terminal observation'])
        self.assertEqual(recorder.rows, [
            (1, 'first action', 0.25, False, False, False),
            (2, 'second action', 1.0, True, True, False),
        ])
        self.assertTrue(recorder.finalized)
        self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason,
                          outcome.sum_rewards), (True, 2, 'success', 1.25))
        self.assertGreaterEqual(outcome.rollout_seconds, 0)
        self.assertEqual(outcome.artifacts['video_path'], 'replay/episode.mp4')
        self.assertEqual(outcome.artifacts['steps_path'], 'replay/steps.jsonl')

    def test_terminal_flags_and_step_limit_stop_without_extra_actions(self):
        cases = [
            (StepResult('terminal', 0.0, False, True, False), 'terminated'),
            (StepResult('terminal', 0.0, False, False, True), 'truncated'),
            (StepResult('next', 0.0, False, False, False), 'step_limit'),
        ]
        for result, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                policy = ReplayPolicy(['only action', 'unwanted action'])
                environment = ReplayEnvironment([result])
                recorder = ReplayRecorder()
                outcome = run_episode(EpisodeConfig(0, 1), policy, environment,
                                      recorder)
                self.assertEqual(outcome.stop_reason, expected_reason)
                self.assertFalse(outcome.success)
                self.assertEqual(outcome.steps, 1)
                self.assertEqual(environment.actions, ['only action'])
                self.assertEqual(recorder.observations[-1], result.observation)

    def test_invalid_horizon_does_not_start_an_episode(self):
        policy = ReplayPolicy([])
        environment = ReplayEnvironment([])
        recorder = ReplayRecorder()
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            run_episode(EpisodeConfig(0, 0), policy, environment, recorder)
        self.assertEqual(policy.resets, 0)
        self.assertEqual(environment.seeds, [])
        self.assertFalse(recorder.finalized)

    def test_environment_failure_does_not_return_a_completed_outcome(self):
        class FailingEnvironment(ReplayEnvironment):
            def step(self, action):
                self.actions.append(action)
                raise TimeoutError('simulator step timed out')

        policy = ReplayPolicy(['proposed action'])
        environment = FailingEnvironment([])
        recorder = ReplayRecorder()
        with self.assertRaisesRegex(TimeoutError, 'timed out'):
            run_episode(EpisodeConfig(3, 2), policy, environment, recorder)
        self.assertEqual(environment.actions, ['proposed action'])
        self.assertEqual(recorder.observations, ['initial observation'])
        self.assertFalse(recorder.finalized)


class BaselineCliTests(unittest.TestCase):
    def test_existing_options_and_defaults(self):
        defaults = parse_args([])
        self.assertEqual((defaults.suite, defaults.task_id, defaults.seed,
                          defaults.policy, defaults.device, defaults.max_steps,
                          defaults.video_fps, defaults.output_dir),
                         ('libero_10', 0, 0, 'HuggingFaceVLA/smolvla_libero',
                          'cuda', None, 20, None))
        selected = parse_args(['--suite', 'libero_spatial', '--task-id', '9',
                               '--seed', '4', '--policy', 'local/checkpoint',
                               '--device', 'cpu', '--max-steps', '3',
                               '--video-fps', '80', '--output-dir', 'custom'])
        self.assertEqual((selected.suite, selected.task_id, selected.seed,
                          selected.policy, selected.device, selected.max_steps,
                          selected.video_fps, str(selected.output_dir)),
                         ('libero_spatial', 9, 4, 'local/checkpoint', 'cpu',
                          3, 80, 'custom'))


if __name__ == '__main__':
    unittest.main()
