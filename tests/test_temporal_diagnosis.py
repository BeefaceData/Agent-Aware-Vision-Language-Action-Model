"""Offline temporal diagnosis contracts and complete sealed episode replay."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback

from episode_harness import EpisodeConfig, SupervisorResponseError, run_episode
from observation_window import ObservationWindowBuilder
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider
from supervisor_response import SupervisorResponseDecoder
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings
from test_supervisor_vlm import PNG, message, proposal, raw

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/temporal_diagnoses.json').read_text())


class TemporalDiagnosisTests(unittest.TestCase):
    def test_categories_and_detached_evidence(self):
        for fixture in FIXTURES:
            with self.subTest(category=fixture['temporal_diagnosis']['category']):
                response = deepcopy(fixture)
                response['proposal_id'] = 'fixture:1'
                history = ObservationWindowBuilder('fixture', 'place item')
                for sequence in (0, 1):
                    history.append(proposal(sequence, 'fixture').observation)
                result = SupervisorResponseDecoder().decode(
                    response, proposal(1, 'fixture'), window=history.snapshot())
                self.assertEqual(result.temporal_diagnosis, fixture['temporal_diagnosis'])
                response['temporal_diagnosis']['evidence'].clear()
                self.assertEqual(len(result.temporal_diagnosis['evidence']), 2)

    def test_malformed_or_privileged_fields_are_rejected(self):
        base = deepcopy(FIXTURES[0])
        base['proposal_id'] = 'fixture:1'
        invalid = []
        for key, value in [('category', 'success'), ('evidence', []), ('summary', ''),
                           ('success', True), ('category', 'unknown')]:
            row = deepcopy(base)
            row['temporal_diagnosis'][key] = value
            invalid.append(row)
        for key, value in [('source', 'reward'), ('observation_sequence', True),
                           ('description', ''), ('hidden_object_pose', [0, 0, 0])]:
            row = deepcopy(base)
            row['temporal_diagnosis']['evidence'][0][key] = value
            invalid.append(row)
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(SupervisorResponseError):
                SupervisorResponseDecoder().decode(row, proposal(1, 'fixture'))

    def test_unknown_can_explain_unavailable_evidence_without_fabricating_it(self):
        row = deepcopy(FIXTURES[-1])
        row['proposal_id'] = 'fixture:1'
        row['temporal_diagnosis']['evidence'] = []
        result = SupervisorResponseDecoder().decode(row, proposal(1, 'fixture'))
        self.assertEqual(result.kind, 'abstain')
        self.assertEqual(result.temporal_diagnosis['evidence'], [])

    def test_categories_survive_sealed_episodes_with_unchanged_policy(self):
        for fixture in FIXTURES:
            with self.subTest(category=fixture['temporal_diagnosis']['category']), TemporaryDirectory() as tmp:
                observations = [raw(i) for i in range(3)]
                config = EpisodeConfig(17, 3)
                policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
                environment = MeasuredReplayEnvironment(17, observations[0], [
                    ([0], ReplayStep(observations[1], 0, False, False, False)),
                    ([1], ReplayStep(observations[2], 1, True, True, False))])
                requests = []

                def transport(payload, deadline, cancel):
                    requests.append(payload)
                    identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
                    if identity['observation_sequence'] == 0:
                        return message(dict(kind='pass', **identity))
                    return message({**deepcopy(fixture), **identity})

                provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                    VlmSettings('offline-fixture'), lambda image: PNG, transport), 2)
                trace = TraceRecorder(Path(tmp) / 'trace', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace, supervisor_decider=provider,
                                      baseline_fallback=healthy_fallback())
                trace.seal(outcome)
                self.assertTrue(load_recorded_replay(trace.directory).run().success)
                rows = [json.loads(line) for line in (trace.directory / 'decisions.jsonl').read_text().splitlines()]
                record = rows[-1]['action_record']
                decision = record['supervisor_pass'] or record['supervisor_abstention']
                self.assertEqual(decision['temporal_diagnosis'], fixture['temporal_diagnosis'])
                self.assertEqual(environment.actions, [[0], [1]])
                for request in requests:
                    self.assertEqual(json.loads(request['messages'][0]['content'][0]['text'])['task'], 'place item')
                    self.assertNotIn('HIDDEN', json.dumps(request))
                    self.assertNotIn('private_evaluator', json.dumps(request))
                self.assertIn('suspected_lost_grasp', requests[-1]['system'])


if __name__ == '__main__':
    unittest.main()
