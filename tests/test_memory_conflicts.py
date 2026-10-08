"""Contradictory historical outcomes remain evidence, never execution authority."""

from copy import deepcopy
import json
import unittest

from decision_memory import DecisionMemory
from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback
from episode_harness import EpisodeConfig, run_episode
from intervention_memory import InterventionMemory
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError
from supervisor_recovery import RecoveryParameter, RecoveryRequestDecoder
from supervisor_response import SupervisorResponseDecoder
from supervisor_vlm import ChronologicalVlmAdapter, SUPERVISOR_PROMPT, VlmSettings
import test_intervention_memory as fixtures
from test_supervisor_vlm import PNG, raw, message


class MemoryConflictTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.InterventionMemoryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.refs, self.records = [], {}
        cases = [('completed', {}),
                 ('aborted', dict(success=True, historical_check='aborted')),
                 ('unknown', dict(success=True, historical_check='continuing'))]
        for name, options in cases:
            fixture.make_episode(name, **options)
            path = fixture.contexts[0]
            context = json.loads(path.read_text())
            context['progress_context'] = {'stage': 'grasp'}
            path.write_text(json.dumps(context))
            ref = fixture.store.append(**(fixture.arguments | {'context': fixtures.pinned(path)}))
            record = fixture.store.read(ref['record_id'], expected_sha256=ref['sha256'])
            self.assertEqual(record['local_outcome']['status'], name)
            replay = load_recorded_replay(fixture.source / 'trace').run()
            self.assertEqual((replay.success, replay.stop_reason),
                             (fixture.outcome.success, fixture.outcome.stop_reason))
            self.refs.append(ref)
            self.records[ref['record_id']] = record
        self.store = InterventionMemory(self.root / 'memory')
        self.query = dict(task=record['task'], robot_capabilities=record['robot_capabilities'],
            compatibility=record['models'] | {'settings': record['configuration']['settings']},
            progress_context={'stage': 'grasp'}, failure_category='suspected_missed_grasp',
            max_entries=3, max_summary_bytes=10000, max_context_bytes=100000)

    def assert_histories(self, summaries):
        self.assertEqual({s['local_outcome']['status'] for s in summaries},
                         {'completed', 'aborted', 'unknown'})
        self.assertEqual({s['record_id'] for s in summaries}, set(self.records))
        for summary in summaries:
            record = self.records[summary['record_id']]
            self.assertEqual(summary['local_outcome'], record['local_outcome'])
            self.assertEqual(summary['evidence_limitations'], record['evidence_limitations'])
            self.assertIn('causal_benefit_unverified', summary['evidence_limitations'])
            self.assertEqual(summary['task_outcome'],
                             {'status': 'withheld', 'reason': 'evaluator_only'})

    def provider(self, respond, sent, decoder=None):
        def transport(payload, deadline, cancel):
            sent.append(deepcopy(payload))
            identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
            self.assert_histories(json.loads(payload['messages'][0]['content'][1]['text']))
            return message(respond(identity))

        selection = DecisionMemory(self.store, self.refs, lambda proposal: self.query)
        adapter = ChronologicalVlmAdapter(VlmSettings('fixture'), lambda frame: PNG,
            transport, decision_memory=selection)
        return BoundedSupervisorProvider(adapter, 5, decoder=decoder)

    def observation(self, index):
        return raw(index) | {'task': self.query['task']['instruction']}

    def test_conflicting_local_outcomes_survive_reopening_and_budget_selection(self):
        result = self.store.retrieve_context(self.refs, **self.query)
        self.assert_histories(result['selected'])
        self.assertEqual(result['ranking']['outcome_preference'], 'none')
        self.assertEqual(self.store.retrieve_context(reversed(self.refs), **self.query), result)
        for cap in (0, 1, 2):
            limited = self.store.retrieve_context(self.refs, **(self.query | {'max_entries': cap}))
            self.assertEqual(limited['selected'], result['selected'][:cap])
            self.assertEqual({s['record_id'] for s in limited['omitted']},
                             {s['record_id'] for s in result['selected'][cap:]})

    def test_conflicting_history_allows_abstention_in_complete_success_and_failure_replays(self):
        def abstain(identity):
            return dict(kind='abstain', **identity, diagnosis='unknown',
                reason='Historical local outcomes disagree; current benefit is uncertain',
                evidence_availability={'main': 'unknown'})

        for success in (False, True):
            config = EpisodeConfig(17, 2)
            observations = [self.observation(i) for i in range(3)]
            policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
            environment = MeasuredReplayEnvironment(17, observations[0], [
                ([0], ReplayStep(observations[1], 0, False, False, False)),
                ([1], ReplayStep(observations[2], 0, success, True, False))])
            trace = TraceRecorder(self.root / f'current-{success}', config, ReplayRecorder())
            sent = []
            outcome = run_episode(config, policy, environment, trace,
                supervisor_decider=self.provider(abstain, sent),
                baseline_fallback=healthy_fallback())
            trace.seal(outcome)
            replay = load_recorded_replay(trace.directory)
            replayed = replay.run()
            self.assertEqual((replayed.success, replayed.stop_reason, replayed.steps),
                             (success, outcome.stop_reason, 2))
            self.assertEqual(environment.actions, [[0], [1]])
            rows = replay.evidence()['decisions']
            self.assertEqual(len(rows), len(sent))
            for row, payload in zip(rows, sent):
                memory = row['action_record']['supervisor_abstention']['memory_context']
                self.assert_histories(memory['selected'])
                self.assertEqual(memory['context_json'], payload['messages'][0]['content'][1]['text'])
                self.assertEqual(payload['system'], SUPERVISOR_PROMPT)

    def assert_rejected(self, name, respond, decoder, reason):
        config = EpisodeConfig(17, 1)
        observation = self.observation(0)
        policy = ReplayPolicy([(observation, [0])])
        environment = ReplayEnvironment(17, observation, [])
        recorder = ReplayRecorder()
        trace = TraceRecorder(self.root / name, config, recorder)
        sent = []
        with self.assertRaises(ProviderRequestError) as caught:
            run_episode(config, policy, environment, trace,
                        supervisor_decider=self.provider(respond, sent, decoder))
        self.assertEqual(caught.exception.result.status, 'rejected')
        self.assertEqual(len(sent), 1)
        self.assertEqual(environment.actions, [])
        failure = recorder.failures[0][3]
        self.assertEqual(failure.action_record.disposition, 'rejected')
        self.assertIn(reason, failure.action_record.rejection_reason)
        self.assertTrue(recorder.finalized)

    def test_historical_success_cannot_bypass_abstention_fallback_guard(self):
        config = EpisodeConfig(17, 1)
        observation = self.observation(0)
        environment = ReplayEnvironment(17, observation, [])
        trace = TraceRecorder(self.root / 'no-fallback', config, ReplayRecorder())
        sent = []
        outcome = run_episode(config, ReplayPolicy([(observation, [0])]), environment, trace,
            supervisor_decider=self.provider(lambda identity: dict(kind='abstain', **identity,
                diagnosis='unknown', reason='Conflicting history',
                evidence_availability={'main': 'unknown'}), sent))
        trace.seal(outcome)
        replay = load_recorded_replay(trace.directory)
        self.assertEqual((outcome.steps, outcome.stop_reason), (0, 'proposal_rejected'))
        self.assertEqual(replay.run().stop_reason, 'proposal_rejected')
        self.assertEqual(environment.actions, [])
        self.assertEqual(len(environment.interruptions), 1)
        self.assert_histories(replay.evidence()['decisions'][0]['action_record'][
            'supervisor_abstention']['memory_context']['selected'])

    def test_history_cannot_authorize_either_correction_kind(self):
        decoder = SupervisorResponseDecoder(
            adjustment=AdjustmentRequestDecoder('left_arm', 'world',
                [AdjustmentComponent('translation_x', 'm', -.03, .03)]),
            recovery=RecoveryRequestDecoder('reopen_and_retreat',
                [RecoveryParameter('retreat_m', 0, .03)]))
        requests = [dict(kind='adjustment', decision_id='adjust', scope='single_action',
                        target='left_arm', frame='world', units={'translation_x': 'm'},
                        residual={'translation_x': .01}),
                    dict(kind='recovery', decision_id='recover', tool_name='reopen_and_retreat',
                        parameters={'retreat_m': .02},
                        evidence=[{'observation_sequence': 0, 'source': 'main'}])]
        for request in requests:
            self.assert_rejected(request['kind'], lambda identity: request | identity, decoder,
                                 'memory-informed corrections require current-scene integration')

    def test_history_cannot_replace_unavailable_current_evidence(self):
        def respond(identity):
            return dict(kind='pass', **identity, temporal_diagnosis={
                'category': 'stall', 'summary': 'Historical success cannot validate this source',
                'evidence': [{'observation_sequence': 1, 'source': 'main',
                              'description': 'Observation outside current episode window'}]})
        self.assert_rejected('unavailable-evidence', respond, SupervisorResponseDecoder(),
                             'observation outside active window')


if __name__ == '__main__':
    unittest.main()
