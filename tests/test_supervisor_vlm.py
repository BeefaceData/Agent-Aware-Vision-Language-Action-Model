"""Mocked provider wire contract and complete sealed episode verification."""

from base64 import b64decode
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from episode_harness import (ActionProposal, EpisodeConfig, ObservationPacket,
                             frame_references_for_observation, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from robot_state_evidence import StateSpec, state_fields_for_observation
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError
from supervisor_vlm import (AnthropicMessagesTransport, ChronologicalVlmAdapter,
                            VlmSettings)


PNG = b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def raw(index, wrist=True):
    return {'task': 'place item', 'pixels': {'image': [[[index, 0, 0]]],
            **({'image2': [[[index, 1, 0]]]} if wrist else {})},
            'robot_state': {'position': [index], 'private': 'HIDDEN'},
            'private_evaluator': 'HIDDEN'}


def proposal(index=0, episode='episode', wrist=True):
    wall = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
    observation = raw(index, wrist)
    return ActionProposal(f'{episode}:{index}', ObservationPacket(
        episode, index, wall, observation, float(index),
        frame_references_for_observation(observation, index, wall, float(index))), [index])


def message(decision):
    return {'type': 'message', 'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': json.dumps(decision)}]}


def pass_message(payload, deadline, cancellation):
    identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
    return message(dict(kind='pass', **identity))


class VlmTests(unittest.TestCase):
    def test_declared_arm_state_reaches_provider_with_frame_and_time(self):
        current = proposal()
        packet = current.observation
        fields = state_fields_for_observation(
            packet.observation,
            (StateSpec('left', 'joint_position', 'robot_state/position', 'joint', 'rad', 1),),
            packet.captured_at, packet.captured_monotonic)
        current = replace(current, observation=replace(packet, state_fields=fields))
        requests = []

        def send(payload, deadline, cancellation):
            requests.append(payload)
            return pass_message(payload, deadline, cancellation)

        provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
            VlmSettings('fixture'), lambda frame: PNG, send), 1)
        self.assertEqual(provider.request(current).status, 'response')
        rows = [json.loads(block['text']) for block in requests[0]['messages'][0]['content']
                if block['type'] == 'text']
        state = next(row['state_fields'] for row in rows if 'robot_state' in row)[0]
        self.assertEqual((state['arm'], state['frame'], state['units'], state['value']),
                         ('left', 'joint', 'rad', [0.0]))
        self.assertEqual(state['captured_at'], packet.captured_at.isoformat())
        self.assertEqual(state['time_basis'], 'observation_return')
        self.assertNotIn('HIDDEN', json.dumps(requests))

    def test_order_bound_missing_camera_and_episode_reset(self):
        requests, images = [], []

        def encode(frame):
            images.append(frame)
            return PNG

        def send(payload, deadline, cancellation):
            self.assertGreater(deadline, monotonic())
            self.assertFalse(cancellation.is_set())
            requests.append(payload)
            return pass_message(payload, deadline, cancellation)

        provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
            VlmSettings('fixture-model', max_tokens=123, max_observations=2), encode, send), 1)
        for index in (0, 2, 3):
            self.assertEqual(provider.request(proposal(index, wrist=index != 2)).status, 'response')
        payload = requests[-1]
        self.assertEqual((payload['model'], payload['max_tokens']), ('fixture-model', 123))
        content = payload['messages'][0]['content']
        rows = [json.loads(block['text']) for block in content if block['type'] == 'text']
        self.assertEqual(rows[0]['task'], 'place item')
        self.assertEqual(rows[0]['omitted_prefix'], {'first': 0, 'last': 1})
        self.assertEqual([row['observation_sequence'] for row in rows if 'robot_state' in row], [2, 3])
        cameras = [row for row in rows if 'camera' in row]
        self.assertEqual([row['camera'] for row in cameras], ['main', 'wrist'] * 2)
        self.assertEqual([row['captured_monotonic'] for row in cameras], [2., None, 3., 3.])
        self.assertEqual(cameras[1]['availability'], 'missing')
        self.assertEqual(cameras[0]['time_basis'], 'observation_return')
        self.assertEqual(images[-3:], [[[[2, 0, 0]]], [[[3, 0, 0]]], [[[3, 1, 0]]]])
        self.assertEqual(sum(block['type'] == 'image' for block in content), 3)
        self.assertNotIn('HIDDEN', json.dumps(requests))
        self.assertEqual(json.loads(requests[1]['messages'][0]['content'][0]['text'])[
            'missing_intervals'], [{'first': 1, 'last': 1}])
        self.assertEqual(provider.request(proposal(0, episode='new')).status, 'response')
        self.assertNotIn('episode:', json.dumps(requests[-1]))
        # The same proposal cannot be appended twice or reorder retained history.
        self.assertEqual(provider.request(proposal(0, episode='new')).status, 'error')
        self.assertEqual(len(requests), 4)

    def test_structured_abstention_and_rejected_stale_or_malformed_decisions(self):
        current = proposal()
        identity = dict(episode_id='episode', observation_sequence=0, proposal_id='episode:0')
        cases = [
            (message(dict(kind='abstain', **identity, diagnosis='unknown', reason='occluded',
                          evidence_availability={'main': 'unknown'})), 'response'),
            (message(dict(kind='pass', **{**identity, 'proposal_id': 'old'})), 'rejected'),
            (message(dict(kind='pass', **identity, code='HIDDEN')), 'rejected'),
            ({'type': 'error', 'error': {'message': 'SECRET'}}, 'error'),
            ({**message(dict(kind='pass', **identity)), 'stop_reason': 'max_tokens'}, 'error'),
            ({'type': 'message', 'stop_reason': 'end_turn',
              'content': [{'type': 'text', 'text': 'SECRET invalid json'}]}, 'error'),
        ]
        for reply, status in cases:
            with self.subTest(status=status):
                provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                    VlmSettings('fixture'), lambda frame: PNG, lambda *args: reply), 1)
                result = provider.request(current)
                self.assertEqual(result.status, status)
                self.assertNotIn('SECRET', repr(result))

    def test_invalid_inputs_and_cancel_do_not_send(self):
        sent = []
        for invalid in (replace(proposal(), observation=replace(
                proposal().observation, frame_references=())),
                replace(proposal(), observation=replace(proposal().observation,
                    observation={'task': 'place item', 'pixels': {}}))):
            provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                VlmSettings('fixture'), lambda frame: PNG, lambda *args: sent.append(args)), 1)
            self.assertEqual(provider.request(invalid).status, 'error')
        cancel = Event()
        cancel.set()
        provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
            VlmSettings('fixture'), lambda frame: PNG, lambda *args: sent.append(args)), 1)
        self.assertEqual(provider.request(proposal(), cancel).status, 'cancelled')
        self.assertEqual(sent, [])
        for kwargs in ({'model': ''}, {'model': 'x', 'max_tokens': True},
                       {'model': 'x', 'max_observations': 0}):
            with self.assertRaises(ValueError):
                VlmSettings(**kwargs)

    def test_auth_only_in_header_and_provider_http_failure(self):
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'SECRET'}), patch(
                'supervisor_vlm.http.client.HTTPSConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = json.dumps(message({'kind': 'pass'})).encode()
            transport = AnthropicMessagesTransport()
            self.assertEqual(transport({'model': 'fixture'}, monotonic() + 1, Event())['type'], 'message')
            request = connection.return_value.request.call_args
            self.assertEqual(request.args, ('POST', '/v1/messages'))
            self.assertEqual(request.kwargs['headers']['x-api-key'], 'SECRET')
            self.assertNotIn(b'SECRET', request.kwargs['body'])
            self.assertNotIn('SECRET', repr(transport))
            response.status = 401
            with self.assertRaisesRegex(RuntimeError, '^provider request failed$'):
                transport({}, monotonic() + 1, Event())
            self.assertEqual(connection.return_value.close.call_count, 2)

    def test_complete_episode_seals_replays_and_failure_preserves_safe_evidence(self):
        for broken in (False, True):
            with self.subTest(broken=broken), TemporaryDirectory() as temporary:
                observations = [raw(i) for i in range(3)]
                config = EpisodeConfig(17, 3)
                policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
                environment = ReplayEnvironment(17, observations[0], [
                    ([0], ReplayStep(observations[1], 0, False, False, False)),
                    ([1], ReplayStep(observations[2], 1, True, True, False))])
                recorder = ReplayRecorder()
                directory = Path(temporary) / 'trace'
                trace = TraceRecorder(directory, config, recorder)
                requests = []

                def transport(payload, deadline, cancel):
                    requests.append(payload)
                    if broken:
                        raise OSError('SECRET credential')
                    return pass_message(payload, deadline, cancel)

                adapter = ChronologicalVlmAdapter(VlmSettings('fixture'), lambda frame: PNG, transport)
                provider = BoundedSupervisorProvider(adapter, 1)
                if broken:
                    with self.assertRaises(ProviderRequestError):
                        run_episode(config, policy, environment, trace, supervisor_decider=provider)
                    self.assertEqual(environment.actions, [])
                    rows = (directory / 'decisions.jsonl').read_text()
                    self.assertIn('rejected', rows)
                    self.assertNotIn('SECRET', rows)
                    self.assertTrue(recorder.finalized)
                else:
                    outcome = run_episode(config, policy, environment, trace, supervisor_decider=provider)
                    trace.seal(outcome)
                    self.assertTrue(load_recorded_replay(directory).run().success)
                    self.assertEqual(environment.actions, [[0], [1]])
                    self.assertEqual(len(requests), 2)
                    self.assertEqual(sum(block['type'] == 'image' for block in
                        requests[-1]['messages'][0]['content']), 4)
                    self.assertNotIn('SECRET', (directory / 'manifest.json').read_text())


if __name__ == '__main__':
    unittest.main()
