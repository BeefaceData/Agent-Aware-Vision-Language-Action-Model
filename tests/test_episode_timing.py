"""Timing evidence through the public complete-episode interface."""

import unittest

from episode_harness import EpisodeConfig, run_episode
from replay_adapters import (ReplayEnvironment, ReplayPolicy, ReplayRecorder,
                             ReplayStep, successful_replay)


class ControlledClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class EpisodeTimingTests(unittest.TestCase):
    def test_delayed_supervisor_exception_preserves_wait_without_action(self):
        clock = ControlledClock()
        fixture = successful_replay(clock)

        def supervisor(observation, proposed_action):
            clock.advance((3.0, 7.0)[observation.sequence])
            if observation.sequence == 1:
                raise TimeoutError('supervisor timed out')

        with self.assertRaisesRegex(TimeoutError, 'supervisor timed out'):
            run_episode(fixture.config, fixture.policy, fixture.environment,
                        fixture.recorder, supervisor=supervisor, clock=clock)

        self.assertEqual(fixture.environment.actions, [('reach', 0.25)])
        self.assertEqual(len(fixture.recorder.steps), 1)
        self.assertEqual(fixture.recorder.steps[0][3].timing.cumulative_wait_seconds,
                         3.0)
        self.assertTrue(fixture.recorder.finalized)
        self.assertEqual(len(fixture.recorder.failures), 1)
        step, source, action, failure = fixture.recorder.failures[0]
        self.assertEqual((step, source.sequence, action, failure.stage,
                          failure.error_type),
                         (2, 1, ('place', 0.75), 'supervisor', 'TimeoutError'))
        self.assertEqual((failure.timing.capture_at, failure.timing.request_at,
                          failure.timing.request_finished_at,
                          failure.timing.response_at,
                          failure.timing.execution_started_at,
                          failure.timing.execution_finished_at),
                         (103.0, 103.0, 110.0, None, None, None))
        self.assertEqual(failure.timing.observation_age_seconds, 0.0)
        self.assertEqual(failure.timing.decision_latency_seconds, 7.0)
        self.assertEqual(failure.timing.cumulative_wait_seconds, 10.0)

    def test_delayed_response_preserves_wait_when_environment_raises(self):
        clock = ControlledClock()
        fixture = successful_replay(clock)
        attempted_actions = []

        def supervisor(observation, proposed_action):
            clock.advance((3.0, 7.0)[observation.sequence])

        class FailingEnvironment:
            def reset(self, seed, episode_id):
                return fixture.environment.reset(seed, episode_id)

            def step(self, action):
                if fixture.environment.actions:
                    attempted_actions.append(action)
                    clock.advance(2.0)
                    raise TimeoutError('environment timed out')
                return fixture.environment.step(action)

        with self.assertRaisesRegex(TimeoutError, 'environment timed out'):
            run_episode(fixture.config, fixture.policy, FailingEnvironment(),
                        fixture.recorder, supervisor=supervisor, clock=clock)

        self.assertEqual(fixture.environment.actions, [('reach', 0.25)])
        self.assertEqual(attempted_actions, [('place', 0.75)])
        self.assertEqual(len(fixture.recorder.steps), 1)
        self.assertTrue(fixture.recorder.finalized)
        self.assertEqual(len(fixture.recorder.failures), 1)
        step, source, action, failure = fixture.recorder.failures[0]
        self.assertEqual((step, source.sequence, action, failure.stage,
                          failure.error_type),
                         (2, 1, ('place', 0.75), 'execution', 'TimeoutError'))
        self.assertEqual((failure.timing.capture_at, failure.timing.request_at,
                          failure.timing.request_finished_at,
                          failure.timing.response_at,
                          failure.timing.execution_started_at,
                          failure.timing.execution_finished_at),
                         (103.0, 103.0, 110.0, 110.0, 110.0, 112.0))
        self.assertEqual(failure.timing.observation_age_seconds, 0.0)
        self.assertEqual(failure.timing.decision_latency_seconds, 7.0)
        self.assertEqual(failure.timing.cumulative_wait_seconds, 10.0)

    def test_observation_only_supervisor_cannot_mutate_executed_proposal(self):
        clock = ControlledClock()
        policy = ReplayPolicy((('initial', ['baseline']),))
        environment = ReplayEnvironment(17, 'initial', (
            (['baseline'], ReplayStep('terminal', 1.0, True, True, False)),
        ), clock)
        recorder = ReplayRecorder()

        def observe(packet, proposed_action):
            proposed_action.append('unexpected')
            clock.advance(2.0)
            return ['replacement']

        outcome = run_episode(EpisodeConfig(17, 1), policy, environment,
                              recorder, supervisor=observe, clock=clock)

        self.assertTrue(outcome.success)
        self.assertEqual(environment.actions, [['baseline']])
        self.assertEqual(outcome.cumulative_wait_seconds, 2.0)
        self.assertEqual(outcome.steps, 1)

    def test_delayed_response_records_age_latency_and_wait_without_extra_actions(self):
        clock = ControlledClock()
        fixture = successful_replay(clock)
        requests = []

        def delayed_supervisor(observation, proposed_action):
            requests.append((observation.sequence, observation.observation,
                             proposed_action))
            clock.advance((3.0, 7.0)[observation.sequence])

        class SlowEnvironment:
            def reset(self, seed, episode_id):
                return fixture.environment.reset(seed, episode_id)

            def step(self, action):
                clock.advance(2.0)
                return fixture.environment.step(action)

        class DelayedRecorder:
            def begin(self, observation):
                fixture.recorder.begin(observation)
                clock.advance(4.0)

            def record_step(self, step, source, action, result, ingestion):
                fixture.recorder.record_step(step, source, action, result, ingestion)
                clock.advance(1.0)

            def finish(self):
                return fixture.recorder.finish()

        outcome = run_episode(fixture.config, fixture.policy, SlowEnvironment(),
                              DelayedRecorder(), supervisor=delayed_supervisor,
                              clock=clock)

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(fixture.environment.actions,
                         [('reach', 0.25), ('place', 0.75)])
        self.assertEqual(requests, [(0, {}, ('reach', 0.25)),
                                    (1, {}, ('place', 0.75))])
        self.assertEqual(outcome.cumulative_wait_seconds, 10.0)
        self.assertEqual(outcome.rollout_seconds, 16.0)
        self.assertEqual(
            [(t.capture_at, t.request_at, t.response_at,
              t.execution_started_at, t.execution_finished_at)
             for t in outcome.step_timings],
            [(100.0, 104.0, 107.0, 107.0, 109.0),
             (109.0, 110.0, 117.0, 117.0, 119.0)],
        )
        self.assertEqual([t.observation_age_seconds for t in outcome.step_timings],
                         [4.0, 1.0])
        self.assertEqual([t.decision_latency_seconds for t in outcome.step_timings],
                         [3.0, 7.0])
        self.assertEqual([t.cumulative_wait_seconds for t in outcome.step_timings],
                         [3.0, 10.0])
        self.assertEqual([t.execution_seconds for t in outcome.step_timings],
                         [2.0, 2.0])
        self.assertEqual([row[3].timing for row in fixture.recorder.steps],
                         list(outcome.step_timings))


if __name__ == '__main__':
    unittest.main()
