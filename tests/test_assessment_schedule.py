"""Action-unit scheduling through complete public episodes and sealed replay."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest

from episode_harness import EpisodeConfig, SupervisorPass, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


def fixture(length=20, interval=4):
    observations = [{'task': 'place item', 'robot_state': {'position': [i]},
                     'private_evaluator': 'hidden'} for i in range(length + 1)]
    actions = [[i] for i in range(length)]
    return (EpisodeConfig(17, length, interval),
            ReplayPolicy(list(zip(observations, actions))),
            ReplayEnvironment(17, observations[0], [
                (action, ReplayStep(observations[i + 1], 0,
                                    i == length - 1, i == length - 1, False))
                for i, action in enumerate(actions)]), ReplayRecorder())


def passed(proposal):
    return SupervisorPass(proposal.observation.episode_id,
                          proposal.observation.sequence, proposal.proposal_id)


class AssessmentScheduleTests(unittest.TestCase):
    def test_long_quiet_episode_and_event_overlap_replay(self):
        for events, expected in ((set(), [0, 4, 8, 12, 16]),
                                 ({2, 4, 11}, [0, 2, 4, 8, 11, 12, 16])):
            with self.subTest(events=events), TemporaryDirectory() as temporary:
                config, policy, environment, recorder = fixture()
                assessed, inspected = [], []

                def trigger(proposal):
                    inspected.append(proposal.observation.sequence)
                    self.assertNotIn('private_evaluator', proposal.observation.observation)
                    proposal.action.append('mutation')
                    return proposal.observation.sequence in events

                def decider(proposal):
                    assessed.append(proposal.observation.sequence)
                    self.assertEqual(proposal.action, [proposal.observation.sequence])
                    return passed(proposal)

                directory = Path(temporary) / 'trace'
                trace = TraceRecorder(directory, config, recorder)
                outcome = run_episode(config, policy, environment, trace,
                                      supervisor_decider=decider,
                                      assessment_trigger=trigger)
                self.assertTrue(outcome.success)
                self.assertEqual(assessed, expected)
                self.assertEqual(inspected, list(range(20)))
                trace.seal(outcome)
                manifest = json.loads((directory / 'manifest.json').read_text())
                self.assertEqual(manifest['config']['supervisor_interval_actions'], 4)
                replay = load_recorded_replay(directory)
                self.assertTrue(replay.run().success)
                rows = replay.evidence()['decisions']
                self.assertEqual([i for i, row in enumerate(rows)
                                  if row['action_record']['supervisor_pass']], expected)
                self.assertEqual(environment.actions, [[i] for i in range(20)])

    def test_pending_periodic_and_event_assessment_are_one_request(self):
        config, policy, environment, recorder = fixture(5, 2)
        entered, release = Event(), Event()
        calls, outcomes, errors = [], [], []

        def decider(proposal):
            calls.append(proposal.observation.sequence)
            if proposal.observation.sequence == 2:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test did not release assessment')
            return passed(proposal)

        def run():
            try:
                outcomes.append(run_episode(config, policy, environment, recorder,
                    supervisor_decider=decider,
                    assessment_trigger=lambda p: p.observation.sequence == 2))
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(calls, [0, 2])
            self.assertEqual(len(environment.actions), 2)
            self.assertEqual(len(policy.observations), 3)
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(outcomes[0].success)
        self.assertEqual(calls, [0, 2, 4])

    def test_new_episode_restarts_cadence_and_default_assesses_every_action(self):
        for interval, expected in ((1, [0, 1, 2]), (4, [0])):
            config, policy, environment, recorder = fixture(3, interval)
            calls = []

            def decider(proposal):
                calls.append(proposal.observation.sequence)
                return passed(proposal)

            for _ in range(2):
                self.assertTrue(run_episode(config, policy, environment, recorder,
                                           supervisor_decider=decider).success)
            self.assertEqual(calls, expected * 2)

    def test_invalid_cadence_and_trigger_fail_explicitly(self):
        for interval in (0, -1, True, 1.5, float('inf'), '4'):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                EpisodeConfig(17, 20, interval)
        config, policy, environment, recorder = fixture()
        with self.assertRaises(ValueError):
            run_episode(config, policy, environment, recorder,
                        assessment_trigger=lambda p: True)
        with self.assertRaises(ValueError):
            run_episode(config, policy, environment, recorder,
                        supervisor_decider=passed, assessment_trigger=lambda p: 1)
        self.assertEqual(environment.actions, [])
        self.assertEqual(recorder.failures[0][3].stage, 'supervisor')


if __name__ == '__main__':
    unittest.main()
