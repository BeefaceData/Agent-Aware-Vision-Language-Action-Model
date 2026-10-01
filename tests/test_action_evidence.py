"""Action provenance through complete episodes and public replay adapters."""

import unittest

from episode_harness import (ActionResolution, EpisodeConfig, run_episode)
from replay_adapters import (ReplayEnvironment, ReplayPolicy, ReplayRecorder,
                             ReplayStep, successful_replay)


class ActionEvidenceTests(unittest.TestCase):
    def test_pass_through_episode_acknowledges_each_original_proposal(self):
        fixture = successful_replay()

        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder)

        self.assertTrue(outcome.success)
        self.assertEqual(len(fixture.recorder.steps), 2)
        for step, source, action, result, _ in fixture.recorder.steps:
            evidence = result.action_record
            self.assertEqual(evidence.proposal_id, f'{outcome.episode_id}:{step}')
            self.assertEqual(evidence.disposition, 'unmodified')
            self.assertEqual(evidence.proposed_action, action)
            self.assertEqual(evidence.selected_action, action)
            self.assertEqual(evidence.executed_action, action)
            self.assertEqual(evidence.execution_acknowledgement.proposal_id,
                             evidence.proposal_id)
            self.assertEqual(
                (evidence.execution_acknowledgement.result_episode_id,
                 evidence.execution_acknowledgement.result_sequence),
                (outcome.episode_id, source.sequence + 1))

    def test_override_is_distinct_and_does_not_mutate_original_proposal(self):
        original = ['reach', 0.25]
        policy = ReplayPolicy((('item visible', original),
                               ('item held', ['place', 0.75])))
        environment = ReplayEnvironment(17, 'item visible', (
            (['reach', 0.5], ReplayStep('item held', 0.0, False, False, False)),
            (['place', 0.75], ReplayStep('item placed', 1.0, True, True, False)),
        ))
        recorder = ReplayRecorder()
        seen_ids = []

        def select(proposal):
            seen_ids.append(proposal.proposal_id)
            if proposal.observation.sequence == 0:
                proposal.action[1] = 99.0  # Selector owns only a detached copy.
                return ActionResolution('override', ['reach', 0.5])
            return ActionResolution('pass')

        outcome = run_episode(EpisodeConfig(17, 3), policy, environment,
                              recorder, action_selector=select)

        self.assertTrue(outcome.success)
        self.assertEqual(original, ['reach', 0.25])
        self.assertEqual(environment.actions, [['reach', 0.5], ['place', 0.75]])
        first = recorder.steps[0][3].action_record
        second = recorder.steps[1][3].action_record
        self.assertEqual(seen_ids, [first.proposal_id, second.proposal_id])
        self.assertNotEqual(first.proposal_id, second.proposal_id)
        self.assertEqual((first.proposed_action, first.selected_action,
                          first.executed_action, first.disposition),
                         (['reach', 0.25], ['reach', 0.5], ['reach', 0.5],
                          'overridden'))
        self.assertEqual((second.proposed_action, second.executed_action,
                          second.disposition),
                         (['place', 0.75], ['place', 0.75], 'unmodified'))
        self.assertIsNotNone(first.execution_acknowledgement)

    def test_rejection_records_proposal_without_an_execution_claim(self):
        fixture = successful_replay()

        def select(proposal):
            if proposal.observation.sequence == 0:
                return ActionResolution('pass')
            return ActionResolution('reject', reason='stale observation')

        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder,
                              action_selector=select)

        self.assertEqual((outcome.steps, outcome.success, outcome.stop_reason),
                         (1, False, 'proposal_rejected'))
        self.assertEqual(fixture.environment.actions, [('reach', 0.25)])
        self.assertTrue(fixture.recorder.finalized)
        self.assertEqual(len(fixture.recorder.failures), 1)
        step, source, proposal, failure = fixture.recorder.failures[0]
        evidence = failure.action_record
        self.assertEqual((step, source.sequence, proposal),
                         (2, 1, ('place', 0.75)))
        self.assertEqual(evidence.proposal_id, f'{outcome.episode_id}:2')
        self.assertEqual(evidence.proposed_action, ('place', 0.75))
        self.assertEqual(evidence.disposition, 'rejected')
        self.assertEqual(evidence.rejection_reason, 'stale observation')
        self.assertIsNone(evidence.selected_action)
        self.assertIsNone(evidence.executed_action)
        self.assertIsNone(evidence.execution_acknowledgement)
        self.assertIsNone(failure.timing.execution_started_at)

    def test_environment_exception_retains_attempt_without_acknowledgement(self):
        fixture = successful_replay()

        class FailingEnvironment:
            def reset(self, seed, episode_id):
                return fixture.environment.reset(seed, episode_id)

            def step(self, action):
                raise TimeoutError('no step acknowledgement')

        with self.assertRaises(TimeoutError):
            run_episode(fixture.config, fixture.policy, FailingEnvironment(),
                        fixture.recorder)

        evidence = fixture.recorder.failures[0][3].action_record
        self.assertEqual(evidence.proposed_action, ('reach', 0.25))
        self.assertEqual(evidence.selected_action, ('reach', 0.25))
        self.assertEqual(evidence.disposition, 'unconfirmed')
        self.assertIsNone(evidence.executed_action)
        self.assertIsNone(evidence.execution_acknowledgement)
        self.assertFalse(fixture.recorder.finalized)

    def test_rejected_selection_wait_is_accounted_without_execution(self):
        class Clock:
            now = 100.0

            def __call__(self):
                return self.now

        clock = Clock()
        fixture = successful_replay(clock)

        def reject(proposal):
            clock.now += 3.0
            return ActionResolution('reject', reason='invalid target')

        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder,
                              clock=clock, action_selector=reject)

        self.assertEqual(outcome.steps, 0)
        self.assertEqual(outcome.stop_reason, 'proposal_rejected')
        self.assertEqual(outcome.cumulative_wait_seconds, 3.0)
        self.assertEqual(fixture.environment.actions, [])
        failure = fixture.recorder.failures[0][3]
        self.assertEqual(failure.timing.decision_latency_seconds, 3.0)
        self.assertEqual(failure.timing.cumulative_wait_seconds, 3.0)
        self.assertIsNone(failure.action_record.execution_acknowledgement)


if __name__ == '__main__':
    unittest.main()
