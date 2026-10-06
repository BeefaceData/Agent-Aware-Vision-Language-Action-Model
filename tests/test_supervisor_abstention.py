"""Uncertainty remains distinct from pass, malformed input and execution failure."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback

from episode_harness import EpisodeConfig, SupervisorAbstention, SupervisorPass, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep


def abstain(proposal):
    return SupervisorAbstention(
        proposal.observation.episode_id, proposal.observation.sequence,
        proposal.proposal_id, 'Occluded grasp; insufficient temporal evidence',
        {'main': 'available', 'wrist': 'missing', 'history': 'unknown'})


class SupervisorAbstentionTests(unittest.TestCase):
    def fixture(self):
        observations = [{'task': 'place item', 'robot_state': {'position': [i]}}
                        for i in range(3)]
        actions = [[0.1], [0.2]]
        return (EpisodeConfig(17, 3), ReplayPolicy(tuple(zip(observations, actions))),
                MeasuredReplayEnvironment(17, observations[0], (
                    (actions[0], ReplayStep(observations[1], 0, False, False, False)),
                    (actions[1], ReplayStep(observations[2], 1, True, True, False)))),
                ReplayRecorder())

    def test_complete_episode_distinguishes_abstention_from_pass_and_replays(self):
        config, policy, environment, recorder = self.fixture()
        responses = []

        def decide(proposal):
            if not responses:
                response = abstain(proposal)
            else:
                # Mutating the provider's old dictionary cannot rewrite evidence.
                responses[0].evidence_availability['wrist'] = 'available'
                response = SupervisorPass(proposal.observation.episode_id,
                                          proposal.observation.sequence, proposal.proposal_id)
            responses.append(response)
            proposal.action[0] = 99
            return response

        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            trace = TraceRecorder(directory, config, recorder)
            outcome = run_episode(config, policy, environment, trace, supervisor_decider=decide,
                                  baseline_fallback=healthy_fallback())
            trace.seal(outcome)
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            rows = replay.evidence()['decisions']
            first = rows[0]['action_record']
            self.assertEqual(first['supervisor_abstention']['diagnosis'], 'unknown')
            self.assertEqual(first['supervisor_abstention']['evidence_availability']['wrist'],
                             'missing')
            self.assertIsNone(first['supervisor_pass'])
            self.assertIsNone(rows[1]['action_record']['supervisor_abstention'])
            self.assertEqual(rows[1]['action_record']['supervisor_pass']['kind'], 'pass')
        self.assertEqual((outcome.steps, outcome.success), (2, True))
        self.assertEqual(environment.actions, [[0.1], [0.2]])
        self.assertTrue(recorder.finalized)
        for _, _, _, result, _ in recorder.steps:
            self.assertEqual(result.action_record.disposition, 'unmodified')
            self.assertEqual(result.action_record.executed_action,
                             result.action_record.proposed_action)

    def test_malformed_or_stale_abstention_never_executes(self):
        for transform in (lambda r: asdict(r), lambda r: replace(r, reason=' '),
                          lambda r: replace(r, reason=None),
                          lambda r: replace(r, evidence_availability={}),
                          lambda r: replace(r, evidence_availability={'main': True}),
                          lambda r: replace(r, evidence_availability={'': 'missing'}),
                          lambda r: replace(r, evidence_availability={'main': 'fresh'}),
                          lambda r: replace(r, episode_id='foreign'),
                          lambda r: replace(r, observation_sequence=False),
                          lambda r: replace(r, observation_sequence=9),
                          lambda r: replace(r, proposal_id='foreign')):
            with self.subTest(transform=transform):
                config, policy, environment, recorder = self.fixture()
                with self.assertRaises(ValueError):
                    run_episode(config, policy, environment, recorder,
                                supervisor_decider=lambda p: transform(abstain(p)))
                self.assertEqual(environment.actions, [])
                failure = recorder.failures[0][3]
                self.assertEqual(failure.stage, 'supervisor')
                self.assertIsNone(failure.action_record.supervisor_abstention)
                self.assertTrue(recorder.finalized)

    def test_previous_abstention_cannot_authorize_next_proposal(self):
        config, policy, environment, recorder = self.fixture()
        responses = []

        def stale(proposal):
            responses.append(abstain(proposal))
            return responses[0]

        outcome = run_episode(config, policy, environment, recorder, supervisor_decider=stale,
                              baseline_fallback=healthy_fallback())
        self.assertTrue(outcome.success)
        second = recorder.steps[1][3].action_record
        self.assertIsNone(second.supervisor_abstention)
        self.assertIn('SupervisorResponseError', second.fallback['cause'])
        self.assertEqual(environment.actions, [[0.1], [0.2]])

    def test_unhealthy_execution_stops_and_retains_uncertainty(self):
        config, policy, environment, recorder = self.fixture()
        calls = []

        class BrokenEnvironment:
            reset = environment.reset
            interruption_contract = environment.interruption_contract
            interrupt = environment.interrupt

            def step(self, action):
                calls.append(action)
                raise TimeoutError('transport unavailable')

        with self.assertRaises(TimeoutError):
            run_episode(config, policy, BrokenEnvironment(), recorder,
                        supervisor_decider=abstain,
                        baseline_fallback=healthy_fallback())
        self.assertEqual(calls, [[0.1]])
        failure = recorder.failures[0][3]
        self.assertEqual(failure.stage, 'execution')
        self.assertEqual(failure.action_record.supervisor_abstention.diagnosis, 'unknown')
        self.assertIsNone(failure.action_record.executed_action)
        self.assertIsNone(failure.action_record.execution_acknowledgement)
        self.assertTrue(recorder.finalized)

    def test_resealed_corrupt_abstention_is_rejected(self):
        config, policy, environment, recorder = self.fixture()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            trace = TraceRecorder(directory, config, recorder)
            outcome = run_episode(config, policy, environment, trace, supervisor_decider=abstain,
                                  baseline_fallback=healthy_fallback())
            trace.seal(outcome)
            original = (directory / 'decisions.jsonl').read_text()
            for changes in ({'reason': ''}, {'evidence_availability': {}},
                            {'observation_sequence': True}, {'episode_id': 'foreign'},
                            {'proposal_id': 'foreign'}, {'diagnosis': 'failure'},
                            {'kind': 'pass'}, {'action': [99]}):
                with self.subTest(changes=changes):
                    rows = [json.loads(line) for line in original.splitlines()]
                    rows[0]['action_record']['supervisor_abstention'].update(changes)
                    raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                    (directory / 'decisions.jsonl').write_bytes(raw)
                    manifest_path = directory / 'manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(TraceError, 'invalid supervisor abstention'):
                        load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
