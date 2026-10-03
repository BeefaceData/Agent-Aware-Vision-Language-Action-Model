"""Public persistence thresholds, reset behavior and sealed episode evidence."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, SupervisorResponseError, run_episode
from observation_window import ObservationWindowBuilder, WindowSettings
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from temporal_persistence import PersistenceSettings, TemporalPersistenceGate
from test_supervisor_vlm import proposal, raw


def response(p, category='stall', cited=None):
    row = dict(kind='abstain' if category == 'unknown' else 'pass',
        episode_id=p.observation.episode_id, proposal_id=p.proposal_id,
        observation_sequence=p.observation.sequence, temporal_diagnosis=dict(
            category=category, summary='Synthetic temporal evidence', evidence=[dict(
                observation_sequence=p.observation.sequence if cited is None else cited,
                source='robot_state', description='Observed position remains unchanged')]))
    if category == 'unknown':
        row.update(diagnosis='unknown', reason='Ambiguous', evidence_availability={
            name: 'unknown' for name in ('main', 'wrist', 'robot_state')})
    return row


def window(p):
    builder = ObservationWindowBuilder(p.observation.episode_id, 'place item')
    builder.append(p.observation)
    return builder.snapshot()


class PersistenceTests(unittest.TestCase):
    def gate(self, count=3, stride=1):
        return TemporalPersistenceGate(PersistenceSettings(count, stride, 'synthetic-v1'))

    def assess(self, gate, index, category='stall', episode='episode'):
        p = proposal(index, episode)
        return gate.assess(response(p, category), p, window=window(p))

    def test_threshold_and_current_proposal_binding(self):
        gate = self.gate()
        considered = []
        def candidate(p):
            considered.append(p.observation.sequence)
            return 'unvalidated candidate'
        for index in range(4):
            result = self.assess(gate, index)
            self.assertEqual(gate.eligible_for(proposal(index)), index >= 2)
            self.assertEqual(result.kind, 'pass' if index >= 2 else 'abstain')
            self.assertIn('synthetic-v1', result.temporal_diagnosis['summary'])
            self.assertLessEqual(len(gate.verdict.supporting_windows), 3)
            self.assertEqual(gate.consider(proposal(index), candidate),
                             'unvalidated candidate' if index >= 2 else None)
        self.assertFalse(gate.eligible_for(proposal(4)))
        self.assertFalse(gate.eligible_for(proposal(3, 'other')))
        self.assertIsNone(gate.consider(proposal(4), candidate))
        self.assertEqual(considered, [2, 3])

    def test_pauses_unknown_family_change_gap_and_episode_reset(self):
        for category in ('progress', 'unknown', 'suspected_missed_grasp'):
            gate = self.gate()
            for i, family in enumerate(('stall', 'stall', category, 'stall', 'stall')):
                self.assess(gate, i, family)
                self.assertFalse(gate.verdict.eligible)
            self.assess(gate, 5)
            self.assertTrue(gate.verdict.eligible)
        for index, episode in ((4, 'episode'), (1, 'episode'), (2, 'new')):
            gate = self.gate()
            self.assess(gate, 0)
            self.assess(gate, 1)
            self.assess(gate, index, episode=episode)
            self.assertFalse(gate.verdict.eligible)
            self.assertEqual(len(gate.verdict.supporting_windows), 1)

    def test_stride_boundary_and_nonincreasing_time(self):
        gate = self.gate(2, 3)
        self.assess(gate, 0)
        self.assess(gate, 3)
        self.assertTrue(gate.verdict.eligible)
        p = proposal(6)
        p = replace(p, observation=replace(p.observation, captured_monotonic=3.))
        gate.assess(response(p), p, window=window(p))
        self.assertFalse(gate.verdict.eligible)

    def test_old_missing_conflicting_and_invalid_evidence_clear_eligibility(self):
        gate = self.gate(2)
        for i in (0, 1):
            self.assess(gate, i)
        p = proposal(2)
        history = ObservationWindowBuilder('episode', 'place item', WindowSettings(3, 0))
        for i in (0, 1, 2):
            history.append(proposal(i).observation)
        gate.assess(response(p, cited=1), p, window=history.snapshot())
        self.assertEqual(gate.verdict.reason, 'incomplete_or_reused_evidence')
        row = response(p)
        row['temporal_diagnosis']['evidence'].append(dict(observation_sequence=1,
            source='main', description='Conflicting view'))
        row['temporal_diagnosis']['conflicts'] = [dict(
            case='ambiguous_object_motion', evidence_indices=[0, 1])]
        result = gate.assess(row, p, window=history.snapshot())
        self.assertEqual(result.kind, 'abstain')
        self.assertFalse(gate.verdict.eligible)
        self.assertEqual(len(result.temporal_diagnosis['conflicts']), 1)
        with self.assertRaises(SupervisorResponseError):
            gate.assess(response(p, cited=99), p, window=history.snapshot())
        self.assertFalse(gate.eligible_for(p))
        history = ObservationWindowBuilder('episode', 'place item')
        for i in (0, 2):
            history.append(proposal(i).observation)
        gate.assess(response(p), p, window=history.snapshot())
        self.assertEqual(gate.verdict.supporting_windows, ())

    def test_settings_reject_missing_calibration_and_invalid_limits(self):
        for count, stride, identity in ((1, 1, 'x'), (33, 1, 'x'), (True, 1, 'x'),
                (2, 0, 'x'), (2, True, 'x'), (2, 1, ''), (2, 1, None)):
            with self.assertRaises(ValueError):
                PersistenceSettings(count, stride, identity)

    def test_complete_sealed_replay_of_persistent_and_isolated_signals(self):
        for families, expected in ((['stall'] * 4, [False, False, True, True]),
                (['stall', 'progress', 'stall', 'unknown'], [False] * 4)):
            with self.subTest(families=families), TemporaryDirectory() as tmp:
                observations = [raw(0) for _ in range(5)]
                actions = [[i] for i in range(4)]
                config = EpisodeConfig(17, 4)
                policy = ReplayPolicy(list(zip(observations, actions)))
                env = ReplayEnvironment(17, observations[0], [
                    (action, ReplayStep(observations[i + 1], 0, i == 3, i == 3, False))
                    for i, action in enumerate(actions)])
                gate = self.gate()
                eligible = []
                def decider(p):
                    result = gate.assess(response(p, families[p.observation.sequence]),
                                         p, window=window(p))
                    eligible.append(gate.eligible_for(p))
                    return result
                trace = TraceRecorder(Path(tmp) / 'trace', config, ReplayRecorder())
                outcome = run_episode(config, policy, env, trace, supervisor_decider=decider)
                trace.seal(outcome)
                replay = load_recorded_replay(trace.directory)
                self.assertTrue(replay.run().success)
                self.assertEqual(eligible, expected)
                self.assertEqual(env.actions, actions)
                for row, allowed in zip(replay.evidence()['decisions'], expected):
                    record = row['action_record']
                    decision = record['supervisor_pass'] or record['supervisor_abstention']
                    self.assertIn('"eligible":' + str(allowed).lower(),
                                  decision['temporal_diagnosis']['summary'])
