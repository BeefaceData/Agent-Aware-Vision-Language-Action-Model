"""Explicit supervisor pass through public interfaces and retained evidence."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionProposal, SupervisorPass, EpisodeConfig,
                             run_episode)
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


def pass_current(proposal):
    return SupervisorPass(proposal.observation.episode_id,
                          proposal.observation.sequence, proposal.proposal_id)


class SupervisorPassTests(unittest.TestCase):
    def fixture(self):
        observations = [{'task': 'place item', 'robot_state': {'position': [i]},
                         'private_evaluator': 'hidden'} for i in range(3)]
        actions = [[0.1], [0.2]]
        policy = ReplayPolicy(tuple(zip(observations, actions)))
        environment = ReplayEnvironment(17, observations[0], (
            (actions[0], ReplayStep(observations[1], 0, False, False, False)),
            (actions[1], ReplayStep(observations[2], 1, True, True, False))))
        return EpisodeConfig(17, 3), policy, environment, ReplayRecorder()

    def test_complete_pass_episode_records_identity_and_unchanged_actions(self):
        config, policy, environment, recorder = self.fixture()
        responses = []

        def decide(proposal):
            self.assertIsInstance(proposal, ActionProposal)
            self.assertNotIn('hidden', repr(proposal))
            self.assertEqual(proposal.observation.observation['task'], 'place item')
            response = pass_current(proposal)
            responses.append(response)
            proposal.action[0] = 99
            proposal.observation.observation['robot_state']['position'][0] = 99
            return response

        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            trace = TraceRecorder(directory, config, recorder)
            outcome = run_episode(config, policy, environment, trace,
                                  supervisor_decider=decide)
            trace.seal(outcome)
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            recorded = [row['action_record']['supervisor_pass']
                        for row in replay.evidence()['decisions']]
            self.assertEqual(recorded, [asdict(response) for response in responses])
            # Recompute the checksum to exercise identity validation, not just hashing.
            decisions_path = directory / 'decisions.jsonl'
            rows = [json.loads(line) for line in decisions_path.read_text().splitlines()]
            rows[0]['action_record']['supervisor_pass']['observation_sequence'] = 1
            raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            decisions_path.write_bytes(raw)
            manifest_path = directory / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TraceError, 'invalid supervisor pass evidence'):
                load_recorded_replay(directory)
        self.assertEqual((outcome.success, outcome.steps), (True, 2))
        self.assertEqual(environment.actions, [[0.1], [0.2]])
        self.assertEqual(len(responses), 2)
        for index, (_, source, action, result, _) in enumerate(recorder.steps):
            record = result.action_record
            self.assertEqual(record.supervisor_pass, responses[index])
            self.assertEqual(responses[index].observation_sequence, source.sequence)
            self.assertEqual(record.executed_action, record.proposed_action)
            self.assertEqual(record.disposition, 'unmodified')
        self.assertTrue(recorder.finalized)

    def test_invalid_or_foreign_responses_never_execute(self):
        for transform in (lambda p: None, lambda p: {'kind': 'pass'},
                          lambda p: replace(p, episode_id='foreign'),
                          lambda p: replace(p, observation_sequence=9),
                          lambda p: replace(p, observation_sequence=False),
                          lambda p: replace(p, proposal_id='foreign')):
            with self.subTest(transform=transform):
                config, policy, environment, recorder = self.fixture()
                with self.assertRaisesRegex(ValueError, 'current proposal'):
                    run_episode(config, policy, environment, recorder,
                                supervisor_decider=lambda p: transform(pass_current(p)))
                self.assertEqual(environment.actions, [])
                self.assertEqual(recorder.failures[0][3].stage, 'supervisor')
                self.assertTrue(recorder.finalized)

    def test_previous_pass_cannot_authorize_next_action(self):
        config, policy, environment, recorder = self.fixture()
        responses = []

        def stale(proposal):
            responses.append(pass_current(proposal))
            return responses[0]

        with self.assertRaisesRegex(ValueError, 'current proposal'):
            run_episode(config, policy, environment, recorder, supervisor_decider=stale)
        self.assertEqual(environment.actions, [[0.1]])
        self.assertIsNone(recorder.failures[0][3].action_record.execution_acknowledgement)

    def test_execution_failure_retains_accepted_pass_without_acknowledgement(self):
        config, policy, environment, recorder = self.fixture()

        class BrokenEnvironment:
            reset = environment.reset

            def step(self, action):
                raise TimeoutError('transport failed')

        with self.assertRaises(TimeoutError):
            run_episode(config, policy, BrokenEnvironment(), recorder,
                        supervisor_decider=pass_current)
        record = recorder.failures[0][3].action_record
        self.assertEqual(record.supervisor_pass.proposal_id, record.proposal_id)
        self.assertIsNone(record.execution_acknowledgement)
        self.assertIsNone(record.executed_action)

    def test_conflicting_callbacks_fail_before_reset(self):
        for kwargs in ({'supervisor': lambda *args: None},
                       {'window_supervisor': lambda *args: None},
                       {'action_selector': lambda *args: None}):
            with self.subTest(kwargs=kwargs):
                config, policy, environment, recorder = self.fixture()
                with self.assertRaisesRegex(ValueError, 'exclusive decision ownership'):
                    run_episode(config, policy, environment, recorder,
                                supervisor_decider=pass_current, **kwargs)
                self.assertEqual(recorder.observations, [])


if __name__ == '__main__':
    unittest.main()
