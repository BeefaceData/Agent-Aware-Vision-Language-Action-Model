"""Recorded supervisor behavior through its public callback and episode traces."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionProposal, ObservationPacket, SupervisorAbstention,
                             SupervisorPass, SupervisorResponseError, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay
from supervisor_replay import RecordedSupervisor


def bundle(episode='episode', kinds=('pass', 'abstain')):
    entries = []
    for sequence, kind in enumerate(kinds):
        identity = dict(episode_id=episode, observation_sequence=sequence,
                        proposal_id=f'{episode}:{sequence + 1}')
        if kind == 'error':
            entries.append(dict(identity, error='provider_error'))
            continue
        response = dict(identity, kind=kind)
        if kind == 'abstain':
            response.update(diagnosis='unknown', reason='Occluded grasp',
                            evidence_availability={'wrist': 'missing'})
        entries.append(dict(identity, response=response))
    return dict(schema_version=1, fixtures=entries)


class RecordedSupervisorTests(unittest.TestCase):
    def proposal(self, sequence=0):
        return ActionProposal(f'episode:{sequence + 1}', ObservationPacket(
            'episode', sequence, datetime.now(timezone.utc), {'task': 'place'}), [.1])

    def test_pass_abstain_and_detached_repeatable_responses(self):
        source = bundle()
        adapter = RecordedSupervisor(source)
        source['fixtures'].clear()
        self.assertIsInstance(adapter(self.proposal()), SupervisorPass)
        decision = adapter(self.proposal(1))
        self.assertIsInstance(decision, SupervisorAbstention)
        decision.evidence_availability['wrist'] = 'available'
        self.assertEqual(adapter(self.proposal(1)).evidence_availability, {'wrist': 'missing'})

    def test_missing_and_each_foreign_identity_fail(self):
        adapter = RecordedSupervisor(bundle())
        current = self.proposal()
        for proposal in (self.proposal(2), replace(current, proposal_id='other'),
                         replace(current, observation=replace(current.observation, episode_id='other')),
                         replace(current, observation=replace(current.observation, sequence=1))):
            with self.subTest(proposal=proposal), self.assertRaisesRegex(SupervisorResponseError, 'missing'):
                adapter(proposal)

    def test_duplicate_and_invalid_bundle_fail_at_load(self):
        duplicate = bundle()
        duplicate['fixtures'].append(deepcopy(duplicate['fixtures'][0]))
        with self.assertRaisesRegex(SupervisorResponseError, 'duplicate'):
            RecordedSupervisor(duplicate)
        for change in (lambda b: b.update(schema_version=2),
                       lambda b: b.update(schema_version=True),
                       lambda b: b.update(fixtures={}),
                       lambda b: b['fixtures'][0].update(observation_sequence=True),
                       lambda b: b['fixtures'][0].update(episode_id=''),
                       lambda b: b['fixtures'][0].update(error='provider_error')):
            source = bundle()
            change(source)
            with self.assertRaises(SupervisorResponseError):
                RecordedSupervisor(source)

    def test_recorded_error_and_malformed_or_stale_payload_fail(self):
        with self.assertRaisesRegex(SupervisorResponseError, 'provider_error'):
            RecordedSupervisor(bundle(kinds=('error',)))(self.proposal())
        for payload in ({}, dict(bundle()['fixtures'][0]['response'], proposal_id='stale'),
                        dict(bundle()['fixtures'][0]['response'], code='SECRET')):
            source = bundle()
            source['fixtures'][0]['response'] = payload
            with self.assertRaises(SupervisorResponseError) as caught:
                RecordedSupervisor(source)(self.proposal())
            self.assertNotIn('SECRET', str(caught.exception))

    def run_fixture(self, directory, kinds):
        fixture = successful_replay()

        class FixtureRecorder(TraceRecorder):
            def begin(self, observation):
                # The public reset packet supplies this attempt's generated ID.
                # Build all fixtures before any supervisor decision is requested.
                source = bundle(observation.episode_id, kinds)
                path = directory / 'supervisor.json'
                path.write_text(json.dumps(source), encoding='utf-8')
                self.supervisor = RecordedSupervisor(json.loads(path.read_text(encoding='utf-8')))
                super().begin(observation)

        trace = FixtureRecorder(directory / 'trace', fixture.config, fixture.recorder)
        return fixture, trace

    def test_complete_episode_pass_abstain_seals_and_replays(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture, trace = self.run_fixture(directory, ('pass', 'abstain'))
            outcome = run_episode(fixture.config, fixture.policy, fixture.environment, trace,
                                  supervisor_decider=lambda proposal: trace.supervisor(proposal))
            trace.seal(outcome)
            replay = load_recorded_replay(directory / 'trace')
            self.assertTrue(replay.run().success)
            self.assertEqual(outcome.steps, 2)
            rows = replay.evidence()['decisions']
            self.assertEqual(rows[0]['action_record']['supervisor_pass']['kind'], 'pass')
            self.assertEqual(rows[1]['action_record']['supervisor_abstention']['diagnosis'], 'unknown')
            self.assertEqual(fixture.environment.actions, [('reach', .25), ('place', .75)])

    def test_error_and_missing_fixture_retain_rejection_without_execution(self):
        for kinds, reason in ((('error',), 'provider_error'), ((), 'missing')):
            with self.subTest(kinds=kinds), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                fixture, trace = self.run_fixture(directory, kinds)
                with self.assertRaisesRegex(SupervisorResponseError, reason):
                    run_episode(fixture.config, fixture.policy, fixture.environment, trace,
                                supervisor_decider=lambda proposal: trace.supervisor(proposal))
                self.assertEqual(fixture.environment.actions, [])
                self.assertTrue(fixture.recorder.finalized)
                rows = [json.loads(line) for line in
                        (directory / 'trace' / 'decisions.jsonl').read_text().splitlines()]
                self.assertEqual(rows[-1]['action_record']['disposition'], 'rejected')
                self.assertIn(reason, rows[-1]['action_record']['rejection_reason'])
                self.assertFalse((directory / 'trace' / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
