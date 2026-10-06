"""Observation-only complete episodes and offline metrics interoperability."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback

from development_annotations import annotation_template
from detector_metrics import detector_report
from episode_harness import EpisodeConfig, SupervisorResponseError, run_episode
from observation_only import export_diagnosis_events
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_response import SupervisorResponseDecoder
from test_episode_timing import ControlledClock
import test_supervisor_response as response_fixtures


class ObservationOnlyTests(unittest.TestCase):
    def setUp(self):
        contracts = response_fixtures.SupervisorResponseTests()
        contracts.setUp()
        self.decoder = replace(contracts.decoder, observation_only=True)
        self.responses = deepcopy(contracts.responses)

    def episode(self, directory, *, supervised=True, interval=1):
        clock = ControlledClock()
        categories = ('stall', 'stall', 'unknown', 'stall', 'progress',
                      'suspected_missed_grasp')
        kinds = ('recovery', 'adjustment', 'abstain', 'pass', 'pass', 'recovery')
        observations = [{'task': 'place item', 'robot_state': {'position': [i]},
                         'private_evaluator': 'HIDDEN'} for i in range(7)]
        actions = [[i / 10] for i in range(6)]
        config = EpisodeConfig(17, 6, supervisor_interval_actions=interval)
        policy = ReplayPolicy(list(zip(observations, actions)))
        environment = MeasuredReplayEnvironment(17, observations[0], [
            (action, ReplayStep(observations[i + 1], 0, False, i == 5, False))
            for i, action in enumerate(actions)], clock=clock)
        trace = TraceRecorder(directory, config, ReplayRecorder())

        def decide(proposal):
            sequence = proposal.observation.sequence
            self.assertNotIn('private_evaluator', proposal.observation.observation)
            response = deepcopy(next(row for row in self.responses
                                     if row['kind'] == kinds[sequence]))
            response.update(episode_id=proposal.observation.episode_id,
                            observation_sequence=sequence, proposal_id=proposal.proposal_id)
            if response['kind'] == 'recovery':
                response['evidence'] = [dict(observation_sequence=sequence,
                                             source='robot_state')]
            response['temporal_diagnosis'] = dict(
                category=categories[sequence], summary='Synthetic temporal assessment',
                evidence=[dict(observation_sequence=sequence, source='robot_state',
                               description='Synthetic recorded position')])
            clock.advance(0.25)
            return self.decoder.decode(response, proposal)

        outcome = run_episode(config, policy, environment, trace, clock=clock,
                              supervisor_decider=decide if supervised else None,
                              baseline_fallback=healthy_fallback())
        trace.seal(outcome)
        return environment.actions, outcome, load_recorded_replay(directory)

    def test_complete_failed_episode_parity_diagnoses_suppression_and_latency(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            actions, outcome, replay = self.episode(root / 'observed')
            baseline_actions, baseline, _ = self.episode(root / 'baseline', supervised=False)
            self.assertEqual(actions, baseline_actions)
            self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason),
                             (baseline.success, baseline.steps, baseline.stop_reason))
            self.assertFalse(replay.run().success)
            rows = replay.evidence()['decisions']
            for i, row in enumerate(rows):
                record = row['action_record']
                self.assertEqual(record['disposition'], 'unmodified')
                self.assertEqual(record['proposed_action'], record['executed_action'])
                response = record['supervisor_pass'] or record['supervisor_abstention']
                self.assertEqual(response['temporal_diagnosis']['evidence'][0]
                                 ['observation_sequence'], i)
                self.assertEqual(row['timing']['response_at'] - row['timing']['request_at'], .25)
                self.assertEqual(row['timing']['cumulative_wait_seconds'], .25 * (i + 1))
            self.assertEqual(rows[0]['action_record']['supervisor_pass']['suppressed_correction'],
                             'recovery')
            self.assertEqual(rows[1]['action_record']['supervisor_pass']['suppressed_correction'],
                             'adjustment')

    def test_export_segmentation_metrics_cli_and_immutable_source(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'observed'
            self.episode(directory)
            before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
            outputs = export_diagnosis_events(directory, configuration_id='synthetic-v1',
                                               provenance='Synthetic scripted diagnoses')
            self.assertEqual([(e['sequence'], e['kind'], e['family']) for e in outputs['events']],
                             [(0, 'detection', 'stall'), (2, 'unknown', None),
                              (2, 'abstention', None), (3, 'detection', 'stall'),
                              (5, 'detection', 'missed_grasp')])
            review = annotation_template(directory, reviewer='fixture author', provenance='Synthetic')
            review.update(status='reviewed', reviewed_at='2026-10-03T00:00:00+00:00', events=[
                dict(id='stall', family='stall', example='failed', onset=[0, 0], uncertain=False,
                     evidence=[dict(sequence=0, description='Synthetic position')], notes='Synthetic')])
            report = detector_report(directory, review, outputs)
            self.assertEqual(report['families']['stall']['recall'], 1)
            self.assertEqual(report['families']['stall']['precision'], .5)
            self.assertEqual(report['false_intervention_count'], 0)
            result = subprocess.run([sys.executable, 'observation_only.py', str(directory),
                                     '--configuration-id', 'synthetic-v1', '--provenance',
                                     'Synthetic scripted diagnoses'], capture_output=True,
                                    text=True, check=True)
            self.assertEqual(json.loads(result.stdout), outputs)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def test_skipped_assessments_end_events_and_baseline_has_no_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.episode(root / 'periodic', interval=3)
            self.episode(root / 'baseline', supervised=False)
            outputs = export_diagnosis_events(root / 'periodic', configuration_id='fixture',
                                               provenance='Synthetic')
            self.assertEqual([e['sequence'] for e in outputs['events']], [0, 3])
            self.assertEqual(export_diagnosis_events(root / 'baseline', configuration_id='fixture',
                                                     provenance='Synthetic')['events'], [])

    def test_mode_keeps_schema_identity_bounds_and_evidence_validation(self):
        contracts = response_fixtures.SupervisorResponseTests()
        contracts.setUp()
        for response, _ in contracts.invalid_cases():
            with self.subTest(response=response), self.assertRaises(SupervisorResponseError):
                self.decoder.decode(response, contracts.proposal)
        response = deepcopy(contracts.responses[2])
        response['temporal_diagnosis'] = dict(category='stall', summary='Not current', evidence=[
            dict(observation_sequence=99, source='robot_state', description='Future')])
        with self.assertRaises(SupervisorResponseError):
            self.decoder.decode(response, contracts.proposal)
        with self.assertRaises(ValueError):
            replace(self.decoder, observation_only='yes')

    def test_export_rejects_correction_trace_and_invalid_configuration(self):
        corrected = Path(__file__).parent / 'fixtures/recorded_episode'
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            export_diagnosis_events(corrected, configuration_id='fixture', provenance='Synthetic')
        for patch in ({'configuration_id': ''}, {'provenance': ''},
                      {'max_delay_sequences': True}, {'max_delay_sequences': -1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                export_diagnosis_events(corrected, **dict(
                    dict(configuration_id='fixture', provenance='Synthetic'), **patch))

    def test_conflicting_correction_still_abstains_and_preserves_evidence(self):
        contracts = response_fixtures.SupervisorResponseTests()
        contracts.setUp()
        proposal = replace(contracts.proposal, observation=replace(
            contracts.proposal.observation, observation={'robot_state': {'position': [1]}}))
        response = deepcopy(contracts.responses[2])
        # A valid request without a temporal diagnosis is still suppressed.
        self.assertEqual(self.decoder.decode(response, proposal).suppressed_correction, 'recovery')
        response['temporal_diagnosis'] = dict(category='stall', summary='Conflicting claims',
            evidence=[dict(observation_sequence=2, source='robot_state', description=text)
                      for text in ('Reported grasp', 'Reported loss')],
            conflicts=[dict(case='grasp_state_conflict', evidence_indices=[0, 1])])
        decision = self.decoder.decode(response, proposal)
        self.assertEqual(decision.kind, 'abstain')
        self.assertEqual(decision.temporal_diagnosis['category'], 'unknown')
        self.assertEqual(decision.temporal_diagnosis['evidence'],
                         response['temporal_diagnosis']['evidence'])


if __name__ == '__main__':
    unittest.main()
