"""Resolve model claims only against the context actually supplied."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, SupervisorResponseError, run_episode
from observation_window import ObservationWindowBuilder, WindowSettings
from recorded_replay import TraceRecorder
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError
from supervisor_response import SupervisorResponseDecoder
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings
from test_supervisor_vlm import PNG, message, proposal, raw


def response(current, sequence, source='main'):
    return dict(kind='pass', episode_id=current.observation.episode_id,
                observation_sequence=current.observation.sequence,
                proposal_id=current.proposal_id, temporal_diagnosis=dict(
                    category='stall', summary='An unverified model claim.', evidence=[dict(
                        observation_sequence=sequence, source=source,
                        description='PRIVATE model text must not enter rejection reasons')]))


class DiagnosisEvidenceTests(unittest.TestCase):
    def test_current_sources_resolve_without_history(self):
        for source in ('main', 'wrist', 'robot_state'):
            current = proposal(2)
            result = SupervisorResponseDecoder().decode(response(current, 2, source), current)
            self.assertEqual(result.temporal_diagnosis['evidence'][0]['source'], source)
        with self.assertRaisesRegex(SupervisorResponseError, 'outside active window'):
            SupervisorResponseDecoder().decode(response(current, 1), current)

    def test_unknown_assessment_cannot_cite_nonexistent_evidence(self):
        current = proposal()
        row = response(current, 1)
        row.update(kind='abstain', diagnosis='unknown', reason='Insufficient evidence',
                   evidence_availability={'main': 'unknown'})
        row['temporal_diagnosis']['category'] = 'unknown'
        with self.assertRaisesRegex(SupervisorResponseError, 'outside active window'):
            SupervisorResponseDecoder().decode(row, current)
        row['temporal_diagnosis']['evidence'] = []
        self.assertEqual(SupervisorResponseDecoder().decode(row, current).kind, 'abstain')

    def test_bounded_window_rejects_evicted_gap_future_and_foreign_context(self):
        history = ObservationWindowBuilder('episode', 'place item', WindowSettings(2, 0))
        for sequence in (0, 2, 4):
            history.append(proposal(sequence).observation)
        current = proposal(4)
        for sequence in (2, 4):
            self.assertEqual(SupervisorResponseDecoder().decode(
                response(current, sequence), current, window=history.snapshot()).kind, 'pass')
        for sequence in (0, 1, 3, 5):
            with self.subTest(sequence=sequence), self.assertRaisesRegex(
                    SupervisorResponseError, 'outside active window'):
                SupervisorResponseDecoder().decode(
                    response(current, sequence), current, window=history.snapshot())
        foreign = replace(history.snapshot(), observations=(proposal(2, 'foreign').observation,
                                                            current.observation))
        with self.assertRaisesRegex(SupervisorResponseError, 'invalid active'):
            SupervisorResponseDecoder().decode(response(current, 2), current, window=foreign)

    def test_unavailable_or_mismatched_sources_are_rejected(self):
        missing_wrist = proposal(0, wrist=False)
        current = proposal()
        empty_state = replace(current, observation=replace(current.observation,
            observation={'task': 'place item', 'robot_state': {'eef': {}}}))
        wrong_frame = replace(current, observation=replace(current.observation,
            frame_references=tuple(replace(ref, observation_sequence=99)
                                   for ref in current.observation.frame_references)))
        for item, source in ((missing_wrist, 'wrist'), (empty_state, 'robot_state'),
                             (wrong_frame, 'main')):
            with self.subTest(source=source), self.assertRaisesRegex(
                    SupervisorResponseError, 'unavailable or mismatched') as caught:
                SupervisorResponseDecoder().decode(response(item, 0, source), item)
            self.assertNotIn('PRIVATE', str(caught.exception))

    def test_provider_uses_exact_window_and_resets_episode_context(self):
        reference = [0]
        def transport(payload, deadline, cancel):
            identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
            row = response(proposal(identity['observation_sequence'], identity['episode_id']),
                           reference[0])
            return message(row)
        provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
            VlmSettings('offline', max_observations=2), lambda image: PNG, transport), 2)
        self.assertEqual(provider.request(proposal(0)).status, 'response')
        self.assertEqual(provider.request(proposal(1)).status, 'response')
        rejected = provider.request(proposal(2))
        self.assertEqual(rejected.status, 'rejected')
        self.assertIsNone(rejected.decision)
        self.assertIn('outside active window', rejected.reason)
        reference[0] = 1
        self.assertEqual(provider.request(proposal(2, 'next')).status, 'rejected')
        reference[0] = 2
        self.assertEqual(provider.request(proposal(3, 'next')).status, 'response')

    def test_failed_episode_retains_rejection_and_executes_no_fabricated_action(self):
        with TemporaryDirectory() as tmp:
            observations = [raw(i) for i in range(3)]
            config = EpisodeConfig(17, 3)
            policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
            environment = ReplayEnvironment(17, observations[0], [
                ([0], ReplayStep(observations[1], 0, False, False, False)),
                ([1], ReplayStep(observations[2], 1, True, True, False))])
            def transport(payload, deadline, cancel):
                identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
                if identity['observation_sequence'] == 0:
                    return message(dict(kind='pass', **identity))
                return message({**response(proposal(1), 2), **identity})
            provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                VlmSettings('offline'), lambda image: PNG, transport), 2)
            recorder = ReplayRecorder()
            trace = TraceRecorder(Path(tmp) / 'trace', config, recorder)
            with self.assertRaises(ProviderRequestError):
                run_episode(config, policy, environment, trace, supervisor_decider=provider)
            self.assertEqual(environment.actions, [[0]])
            self.assertTrue(recorder.finalized)
            rows = [json.loads(line) for line in
                    (trace.directory / 'decisions.jsonl').read_text().splitlines()]
            record = rows[-1]['action_record']
            self.assertEqual(record['disposition'], 'rejected')
            self.assertIn('temporal evidence reference 0', record['rejection_reason'])
            self.assertIsNone(record['executed_action'])
            self.assertNotIn('PRIVATE', json.dumps(rows))


if __name__ == '__main__':
    unittest.main()
