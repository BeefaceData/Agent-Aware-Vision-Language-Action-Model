"""Override/resume through public adapters, without model or simulator calls."""

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from episode_harness import (ActionResolution, EpisodeConfig, ObservationPacket,
                             ObservationRejected, run_episode)
from policy_adapter import ResetOnResumePolicyAdapter
from replay_adapters import ReplayEnvironment, ReplayRecorder, ReplayStep


class QueuedModel:
    def __init__(self):
        self.queue = []
        self.inputs = []

    def reset(self):
        self.queue.clear()

    def infer(self, packet):
        self.inputs.append(packet)
        if not self.queue:
            self.queue.extend([packet.observation, 'obsolete queued action'])
        return self.queue.pop(0)


def override(proposal):
    return (ActionResolution('override', 'correction')
            if proposal.observation.sequence == 0 else ActionResolution('pass'))


def environment(terminal=False):
    return ReplayEnvironment(17, 'reach', (
        ('correction', ReplayStep('place', 0.0, terminal, terminal, False)),
        ('place', ReplayStep('done', 1.0, True, True, False)),
    ))


class PolicyResumeTests(unittest.TestCase):
    def test_adapter_discards_queue_and_infers_from_fresh_packet(self):
        model = QueuedModel()
        adapter = ResetOnResumePolicyAdapter(model.reset, model.infer)
        before = ObservationPacket('attempt', 0, datetime.now(timezone.utc), 'reach', 1.)
        after = replace(before, sequence=1, observation='place', captured_monotonic=2.)
        adapter.reset()
        self.assertEqual(adapter.act(before), 'reach')
        adapter.resume(after)
        self.assertEqual(adapter.act(after), 'place')
        self.assertEqual(model.inputs, [before, after])

    def test_complete_episode_resumes_after_override_without_resetting_environment(self):
        model = QueuedModel()
        adapter = ResetOnResumePolicyAdapter(model.reset, model.infer)
        env, recorder = environment(), ReplayRecorder()
        outcome = run_episode(EpisodeConfig(17, 3), adapter, env, recorder,
                              action_selector=override)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(env.actions, ['correction', 'place'])
        self.assertEqual([p.sequence for p in model.inputs], [0, 1])
        self.assertEqual({p.episode_id for p in model.inputs}, {outcome.episode_id})
        self.assertEqual([row[3].action_record.disposition for row in recorder.steps],
                         ['overridden', 'unmodified'])

    def test_policy_specific_resume_does_not_call_attempt_reset(self):
        class Policy(QueuedModel):
            def reset(self):
                super().reset()
                self.resumes = []
                self.attempt_resets = getattr(self, 'attempt_resets', 0) + 1

            act = QueuedModel.infer

            def resume(self, packet):
                self.queue.clear()
                self.resumes.append(packet)

        policy = Policy()
        outcome = run_episode(EpisodeConfig(17, 3), policy, environment(),
                              ReplayRecorder(), action_selector=override)
        self.assertTrue(outcome.success)
        self.assertEqual(policy.attempt_resets, 1)
        self.assertEqual([p.sequence for p in policy.resumes], [1])

    def test_unsupported_override_fails_before_execution(self):
        class Policy(QueuedModel):
            act = QueuedModel.infer

        env, recorder = environment(), ReplayRecorder()
        with self.assertRaisesRegex(NotImplementedError, 'override/resume') as caught:
            run_episode(EpisodeConfig(17, 3), Policy(), env, recorder,
                        action_selector=override)
        self.assertEqual(env.actions, [])
        self.assertEqual(caught.exception.episode_interruption.steps, 0)
        self.assertEqual(recorder.failures[0][3].stage, 'selection')

    def test_resume_failure_retains_executed_override_and_stops(self):
        for error in (RuntimeError('cannot resume'), KeyboardInterrupt('cancelled')):
            with self.subTest(error=type(error).__name__):
                class Policy(QueuedModel):
                    act = QueuedModel.infer

                    def resume(self, packet):
                        raise error

                policy, env, recorder = Policy(), environment(), ReplayRecorder()
                with self.assertRaises(type(error)) as caught:
                    run_episode(EpisodeConfig(17, 3), policy, env, recorder,
                                action_selector=override)
                self.assertIs(caught.exception, error)
                evidence = error.episode_interruption
                self.assertEqual(evidence.steps, 1)
                self.assertEqual(evidence.task_status, 'unknown')
                self.assertEqual(evidence.last_observation.sequence, 1)
                self.assertEqual(evidence.acknowledged_actions[0].executed_action,
                                 'correction')
                self.assertEqual(env.actions, ['correction'])
                self.assertEqual(len(policy.inputs), 1)
                self.assertTrue(recorder.finalized)

    def test_no_resume_after_terminal_horizon_or_rejected_observation(self):
        for mode in ('terminal', 'horizon', 'invalid'):
            with self.subTest(mode=mode):
                class Policy(QueuedModel):
                    act = QueuedModel.infer

                    def resume(self, packet):
                        raise AssertionError('must not resume')

                class Environment(ReplayEnvironment):
                    def step(self, action):
                        result = super().step(action)
                        return replace(result, observation=replace(
                            result.observation, sequence=0))

                env = environment(mode == 'terminal')
                if mode == 'invalid':
                    env = Environment(17, 'reach', (
                        ('correction', ReplayStep('place', 0., False, False, False)),))
                args = (EpisodeConfig(17, 1 if mode == 'horizon' else 3),
                        Policy(), env, ReplayRecorder())
                if mode == 'invalid':
                    with self.assertRaises(ObservationRejected):
                        run_episode(*args, action_selector=override)
                else:
                    result = run_episode(*args, action_selector=override)
                    self.assertEqual(result.steps, 1)
                    self.assertEqual(result.stop_reason,
                                     'success' if mode == 'terminal' else 'step_limit')


if __name__ == '__main__':
    unittest.main()
