"""Conflicting deployable evidence cannot yield a correction request."""
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
from supervisor_response import SupervisorResponseDecoder
from supervisor_provider import BoundedSupervisorProvider
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings
from test_supervisor_vlm import PNG, message, proposal, raw
import test_supervisor_response


def assessment(case):
    descriptions = ({'grasp_state_conflict': (
        'Earlier wrist view suggests object held.',
        'Later robot state indicates reopened gripper; continued grasp is uncertain.'),
        'ambiguous_object_motion': (
        'Main view suggests object translation.',
        'Wrist view moves with camera; object motion cannot be distinguished.')})[case]
    return dict(category='progress', summary='Apparent progress', evidence=[
        dict(observation_sequence=0, source='main', description=descriptions[0]),
        dict(observation_sequence=1, source='robot_state' if case == 'grasp_state_conflict'
             else 'wrist', description=descriptions[1])],
        conflicts=[dict(case=case, evidence_indices=[0, 1])])


class TemporalConflictTests(unittest.TestCase):
    def test_conflicts_override_pass_and_both_correction_kinds(self):
        contracts = test_supervisor_response.SupervisorResponseTests()
        contracts.setUp()
        current = proposal(1)
        history = ObservationWindowBuilder('episode', 'place item')
        for index in (0, 1):
            history.append(proposal(index).observation)
        for case in ('grasp_state_conflict', 'ambiguous_object_motion'):
            for base in (contracts.responses[0], *contracts.responses[2:]):
                response = dict(base, episode_id='episode', observation_sequence=1,
                                proposal_id=current.proposal_id, temporal_diagnosis=assessment(case))
                if response['kind'] == 'recovery':
                    response['evidence'] = [dict(observation_sequence=1, source='robot_state')]
                result = contracts.decoder.decode(response, current, window=history.snapshot())
                self.assertEqual(result.kind, 'abstain')
                self.assertEqual(result.temporal_diagnosis['category'], 'unknown')
                self.assertEqual(result.temporal_diagnosis['evidence'], response['temporal_diagnosis']['evidence'])
                self.assertEqual(result.temporal_diagnosis['conflicts'], response['temporal_diagnosis']['conflicts'])
                with self.assertRaises(SupervisorResponseError):
                    contracts.decoder.decode(dict(response, confidence=1.0), current, window=history.snapshot())
                response['temporal_diagnosis']['evidence'].clear()
                self.assertEqual(len(result.temporal_diagnosis['evidence']), 2)

    def test_invalid_or_unresolved_conflicts_fail_closed(self):
        current = proposal(1)
        base = dict(kind='pass', episode_id='episode', observation_sequence=1,
                    proposal_id=current.proposal_id, temporal_diagnosis=assessment('grasp_state_conflict'))
        for indices in ([0], [0, 0], [0, 2], [True, 1], '0,1'):
            row = deepcopy(base)
            row['temporal_diagnosis']['conflicts'][0]['evidence_indices'] = indices
            with self.assertRaises(SupervisorResponseError):
                SupervisorResponseDecoder().decode(row, current)
        with self.assertRaisesRegex(SupervisorResponseError, 'outside active window'):
            SupervisorResponseDecoder().decode(base, current)

    def test_conflicts_survive_complete_sealed_episode(self):
        for case in ('grasp_state_conflict', 'ambiguous_object_motion'):
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                observations = [raw(i) for i in range(3)]
                config = EpisodeConfig(17, 3)
                policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
                environment = MeasuredReplayEnvironment(17, observations[0], [
                    ([0], ReplayStep(observations[1], 0, False, False, False)),
                    ([1], ReplayStep(observations[2], 1, True, True, False))])
                def transport(payload, deadline, cancel):
                    identity = json.loads(payload['messages'][0]['content'][0]['text'])['request']
                    row = dict(kind='pass', **identity)
                    if identity['observation_sequence'] == 1:
                        row['temporal_diagnosis'] = assessment(case)
                    return message(row)
                provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                    VlmSettings('offline'), lambda image: PNG, transport), 2)
                trace = TraceRecorder(Path(tmp) / 'trace', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace, supervisor_decider=provider,
                                      baseline_fallback=healthy_fallback())
                trace.seal(outcome)
                replay = load_recorded_replay(trace.directory)
                self.assertTrue(replay.run().success)
                decision = replay.evidence()['decisions'][-1]['action_record']['supervisor_abstention']
                self.assertEqual(decision['temporal_diagnosis']['category'], 'unknown')
                self.assertEqual(decision['temporal_diagnosis']['evidence'], assessment(case)['evidence'])
                self.assertEqual(environment.actions, [[0], [1]])
