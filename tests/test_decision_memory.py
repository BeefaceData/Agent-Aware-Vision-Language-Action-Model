"""Memory provenance verified against captured wire input and sealed replay."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from threading import Event
from time import monotonic
import unittest

from decision_memory import DecisionMemory, disabled_memory, validate_memory
from episode_harness import EpisodeConfig, run_episode
from recorded_replay import TraceRecorder, TraceError, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider
from supervisor_response import SupervisorResponseDecoder
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings, SUPERVISOR_PROMPT
import test_memory_compatibility as fixtures
from test_supervisor_vlm import PNG, raw, proposal, message, pass_message


class DecisionMemoryTests(unittest.TestCase):
    setUp = fixtures.MemoryCompatibilityTests.setUp
    retain = fixtures.MemoryCompatibilityTests.retain

    def selection(self, refs, **changes):
        query = dict(task=fixtures.TASK, robot_capabilities=asdict(self.single),
            progress_context=fixtures.PROGRESS, failure_category='stall',
            compatibility=fixtures.COMPATIBILITY,
            max_entries=4, max_summary_bytes=10000, max_context_bytes=100000)
        query.update(changes)
        return DecisionMemory(self.store, refs, lambda proposal: query,
                              snapshot={'snapshot_id': 'declared-development', 'sha256': 'a' * 64})

    def episode(self, selection, name, abstain=False):
        observations = [raw(i) for i in range(3)]
        config = EpisodeConfig(17, 3)
        policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
        environment = ReplayEnvironment(17, observations[0], [
            ([0], ReplayStep(observations[1], 0, False, False, False)),
            ([1], ReplayStep(observations[2], 1, True, True, False))])
        directory = self.root / name
        trace = TraceRecorder(directory, config, ReplayRecorder())
        sent = []

        def transport(payload, deadline, cancel):
            sent.append(deepcopy(payload))
            if abstain:
                identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
                return message(dict(kind='abstain', **identity, diagnosis='unknown',
                    reason='uncertain', evidence_availability={'main': 'unknown'}))
            return pass_message(payload, deadline, cancel)

        adapter = ChronologicalVlmAdapter(VlmSettings('fixture'), lambda frame: PNG,
            transport, decision_memory=selection)
        outcome = run_episode(config, policy, environment, trace,
            supervisor_decider=BoundedSupervisorProvider(adapter, 5))
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        replayed = replay.run()
        self.assertEqual((replayed.success, replayed.stop_reason),
                         (outcome.success, outcome.stop_reason))
        return directory, replay.evidence()['decisions'], sent

    def test_selected_memories_match_wire_in_complete_pass_and_abstention_replay(self):
        refs = [self.retain('failed'), self.retain('unknown', truncated=True)]
        for abstain in (False, True):
            _, rows, sent = self.episode(self.selection(refs), str(abstain), abstain)
            for row, payload in zip(rows, sent):
                key = 'supervisor_abstention' if abstain else 'supervisor_pass'
                memory = row['action_record'][key]['memory_context']
                wire = payload['messages'][0]['content'][1]['text']
                self.assertEqual(memory['context_json'], wire)
                self.assertEqual(memory['selected'], json.loads(wire))
                self.assertEqual({s['record_id'] for s in memory['selected']},
                                 {r['record_id'] for r in refs})
                self.assertEqual(memory['snapshot']['snapshot_id'], 'declared-development')
                self.assertEqual(memory['settings']['max_entries'], 4)
                self.assertEqual(payload['system'], SUPERVISOR_PROMPT)
                self.assertNotIn('references', wire)
                self.assertNotIn('episode_outcome', wire)

    def test_incompatible_versions_are_excluded_from_complete_episode_replay(self):
        compatible = self.retain('compatible')
        outdated = self.retain('outdated', supervisor_model='older')
        _, rows, sent = self.episode(self.selection([compatible, outdated]), 'versions')
        self.assertTrue(rows)
        for row, payload in zip(rows, sent):
            memory = row['action_record']['supervisor_pass']['memory_context']
            self.assertEqual([item['record_id'] for item in memory['selected']],
                             [compatible['record_id']])
            self.assertEqual(memory['excluded'],
                             [dict(outdated, reasons=['supervisor_mismatch'])])
            self.assertEqual(json.loads(payload['messages'][0]['content'][1]['text']),
                             memory['selected'])

    def test_disabled_is_distinct_from_enabled_empty(self):
        _, rows, sent = self.episode(None, 'disabled')
        self.assertEqual(rows[0]['action_record']['supervisor_pass']['memory_context'], disabled_memory())
        _, rows, sent = self.episode(self.selection([]), 'empty')
        memory = rows[0]['action_record']['supervisor_pass']['memory_context']
        self.assertEqual(memory['retrieval'], 'enabled')
        self.assertEqual(sent[0]['messages'][0]['content'][1]['text'], '[]')

    def test_corrupt_source_fails_before_transport_even_with_zero_budget(self):
        ref = self.retain('source')
        (self.root / 'source/context.json').unlink()
        sent = []
        adapter = ChronologicalVlmAdapter(VlmSettings('fixture'), lambda frame: PNG,
            lambda *args: sent.append(args), decision_memory=self.selection([ref], max_entries=0))
        with self.assertRaises(TraceError):
            adapter(proposal(), monotonic() + 5, Event())
        self.assertEqual(sent, [])

    def test_replay_rejects_inconsistent_memory_even_after_trace_rehash(self):
        ref = self.retain('source')
        directory, _, _ = self.episode(self.selection([ref]), 'tamper')
        path = directory / 'decisions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['action_record']['supervisor_pass']['memory_context']['selected'] = []
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['decisions.jsonl'] = sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TraceError, 'invalid decision memory'):
            load_recorded_replay(directory)

    def test_model_cannot_supply_provenance_and_preparation_is_detached(self):
        ref = self.retain('source')
        selection = self.selection([ref])
        record = selection.prepare(proposal())
        record['selected'].clear()
        self.assertEqual(len(selection.prepare(proposal())['selected']), 1)
        with self.assertRaises(ValueError):
            validate_memory(record)
        with self.assertRaises(ValueError):
            SupervisorResponseDecoder().decode(dict(kind='pass', episode_id='episode',
                observation_sequence=0, proposal_id='episode:0', memory_context=disabled_memory()), proposal())


if __name__ == '__main__':
    unittest.main()
