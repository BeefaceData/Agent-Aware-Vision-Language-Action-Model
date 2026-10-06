"""No-memory isolation through captured provider inputs and complete replay."""

from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from decision_memory import DecisionMemory, NoMemory, disabled_memory
from episode_harness import EpisodeConfig, run_episode
from fixed_memory import FixedMemory
from intervention_memory import InterventionMemory
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings, SUPERVISOR_PROMPT
import test_memory_compatibility as fixtures
from test_supervisor_vlm import PNG, raw, pass_message


class NoMemoryTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain

    def test_two_episodes_keep_current_history_without_any_persistent_access(self):
        # Real retained experience exists, but even corrupt evidence must be
        # irrelevant to this condition rather than causing a retrieval failure.
        prior = self.retain('prior', success=True)
        (self.root / 'prior/context.json').unlink()
        for mode_name, mode in (('explicit', NoMemory()), ('default', None)):
            sent = []

            def transport(payload, deadline, cancellation):
                sent.append(deepcopy(payload))
                return pass_message(payload, deadline, cancellation)

            with ExitStack() as stack:
                guards = [stack.enter_context(patch.object(cls, method,
                    side_effect=AssertionError('persistent memory accessed')))
                    for cls, methods in (
                        (InterventionMemory, ('__init__', 'append', 'read', 'candidates',
                            'filter_candidates', 'rank_candidates', 'retrieve_context')),
                        (MemorySnapshot, ('__init__', 'read', 'freeze')),
                        (DecisionMemory, ('prepare',)), (FixedMemory, ('prepare',)))
                    for method in methods]
                adapter = ChronologicalVlmAdapter(VlmSettings('fixture'),
                    lambda frame: PNG, transport, decision_memory=mode)
                provider = BoundedSupervisorProvider(adapter, 5)
                episode_ids = []
                for episode_index, success in enumerate((False, True)):
                    observations = [raw(episode_index * 10 + i) for i in range(3)]
                    config = EpisodeConfig(17, 3)
                    environment = ReplayEnvironment(17, observations[0], [
                        ([0], ReplayStep(observations[1], 0, False, False, False)),
                        ([1], ReplayStep(observations[2], 0, success, True, False))])
                    policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
                    directory = self.root / f'{mode_name}-{episode_index}'
                    trace = TraceRecorder(directory, config, ReplayRecorder())
                    outcome = run_episode(config, policy, environment, trace,
                        supervisor_decider=provider)
                    self.assertEqual((outcome.steps, outcome.success), (2, success))
                    episode_ids.append(outcome.episode_id)
                    trace.seal(outcome)
                    replay = load_recorded_replay(directory)
                    replayed = replay.run()
                    self.assertEqual((replayed.steps, replayed.success, replayed.stop_reason),
                                     (outcome.steps, success, outcome.stop_reason))
                    decisions = replay.evidence()['decisions']
                    self.assertEqual(len(decisions), 2)
                    for decision in decisions:
                        self.assertEqual(decision['action_record']['supervisor_pass'][
                            'memory_context'], disabled_memory())
                for guard in guards:
                    guard.assert_not_called()

            self.assertNotEqual(*episode_ids)
            self.assertEqual(len(sent), 4)
            for index, payload in enumerate(sent):
                self.assertEqual(payload['system'], SUPERVISOR_PROMPT)
                content = payload['messages'][0]['content']
                rows = [json.loads(block['text']) for block in content if block['type'] == 'text']
                # An enabled empty retrieval would add a JSON list to the wire.
                self.assertTrue(all(type(row) is dict for row in rows))
                history = [row for row in rows if 'robot_state' in row]
                self.assertEqual([row['observation_sequence'] for row in history],
                                 list(range(index % 2 + 1)))
                self.assertEqual([row['robot_state']['position'] for row in history],
                                 [[index // 2 * 10 + i] for i in range(index % 2 + 1)])
                self.assertEqual(rows[0]['request']['episode_id'], episode_ids[index // 2])
                self.assertNotIn(prior['record_id'], json.dumps(payload))
            self.assertNotIn(episode_ids[0], json.dumps(sent[2:]))

    def test_no_memory_returns_detached_disabled_evidence(self):
        mode = NoMemory()
        record = mode.prepare(None)
        record['selected'].append({'record_id': 'foreign'})
        record['retrieval'] = 'enabled'
        self.assertEqual(mode.prepare(None), disabled_memory())

    def test_no_memory_rejects_store_or_query_configuration(self):
        for kwargs in ({'store': object()}, {'references': []},
                       {'query': lambda proposal: {}}, {'snapshot': {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                NoMemory(**kwargs)


if __name__ == '__main__':
    unittest.main()
