"""Dispatch-time correction expiry with a controllable monotonic clock."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import EpisodeConfig, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayRecorder
import test_intervention_limits as fixtures


class CorrectionExpiryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InterventionLimitTests()
        self.fixture.setUp()

    def episode(self, path, kind, dispatch, *, age=5., continuation=None):
        config, policy, env, selector, calls = self.fixture.episode([kind])
        config = replace(config, correction_timeout_seconds=1., correction_max_age_seconds=age)
        now = [10.]
        arrivals = []
        def select(proposal):
            resolution = selector(proposal)
            if resolution.recovery is not None:
                resolution = replace(resolution, recovery=replace(resolution.recovery,
                    monitor=resolution.recovery.monitor | {'max_age_seconds': 5.}))
            arrivals.append(now[0])  # Valid on arrival; delay after resolution.
            now[0] = dispatch
            return resolution
        def after(step, result):
            if continuation is not None:
                now[0] = continuation
        recorder = ReplayRecorder()
        trace = TraceRecorder(path, config, recorder)
        outcome = run_episode(config, policy, env, trace, action_selector=select,
            recovery_observer=self.fixture.recovery.observe, on_step=after, clock=lambda: now[0])
        trace.seal(outcome)
        replay = load_recorded_replay(path)
        replayed = replay.run()
        self.assertEqual((replayed.steps, replayed.stop_reason), (outcome.steps, outcome.stop_reason))
        self.assertEqual(arrivals, [10.])
        return outcome, replay.evidence(), env, recorder

    def test_queued_corrections_just_before_at_and_after_deadline(self):
        for kind in ('override', fixtures.TOOL):
            for dispatch in (10.999, 11., 11.001):
                with self.subTest(kind=kind, dispatch=dispatch), TemporaryDirectory() as tmp:
                    outcome, evidence, env, _ = self.episode(Path(tmp)/'trace', kind, dispatch)
                    record = evidence['decisions'][0]['action_record']
                    self.assertEqual(record['correction_expiry']['deadline'], 11.)
                    if dispatch < 11.:
                        self.assertEqual(outcome.stop_reason, 'step_limit')
                        self.assertEqual(len(env.actions), 1 if kind == 'override' else 2)
                    else:
                        self.assertEqual((outcome.steps, outcome.stop_reason), (0, 'proposal_rejected'))
                        self.assertEqual(env.actions, [])
                        self.assertEqual(record['rejection_reason'], 'correction request expired')
                        self.assertIsNone(record['intervention_budget'])
                        self.assertTrue(record['interruption']['confirmed'])

    def test_observation_age_is_independent_and_inclusive(self):
        for kind in ('override', fixtures.TOOL):
            for dispatch in (10.499, 10.5, 10.501):
                with self.subTest(kind=kind, dispatch=dispatch), TemporaryDirectory() as tmp:
                    outcome, evidence, _, _ = self.episode(Path(tmp)/'trace', kind, dispatch, age=.5)
                    record = evidence['decisions'][0]['action_record']
                    self.assertEqual(record['correction_expiry']['valid'], dispatch <= 10.5)
                    if dispatch > 10.5:
                        self.assertEqual(outcome.steps, 0)
                        self.assertEqual(record['rejection_reason'], 'correction observation too old')

    def test_recovery_does_not_renew_deadline_between_commands(self):
        with TemporaryDirectory() as tmp:
            outcome, evidence, env, _ = self.episode(Path(tmp)/'trace', fixtures.TOOL,
                                                    10.5, continuation=11.)
            self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'proposal_rejected'))
            self.assertEqual(len(env.actions), 1)
            first, last = [row['action_record'] for row in evidence['decisions']]
            self.assertEqual(last['correction_expiry']['deadline'], first['correction_expiry']['deadline'])
            self.assertEqual(last['intervention_budget'], first['intervention_budget'])
            self.assertTrue(last['interruption']['confirmed'])

    def test_dispatch_check_occurs_after_detaching_transport_action(self):
        config, policy, env, selector, _ = self.fixture.episode(['override'])
        config = replace(config, correction_timeout_seconds=1., correction_max_age_seconds=5.)
        now = [10.]
        class DelayedAction(list):
            def __deepcopy__(self, memo):
                now[0] = 11.
                return list(self)
        def select(proposal):
            return replace(selector(proposal), action=DelayedAction([.3]*7))
        recorder = ReplayRecorder()
        outcome = run_episode(config, policy, env, recorder, action_selector=select, clock=lambda: now[0])
        self.assertEqual((outcome.steps, env.actions), (0, []))
        self.assertEqual(recorder.failures[0][-1].action_record.rejection_reason,
                         'correction request expired')

    def test_invalid_limit_pairs_are_rejected(self):
        for timeout, age in ((None, 1), (1, None), (0, 1), (1, -1),
                             (True, 1), (1, float('nan')), (float('inf'), 1)):
            with self.subTest(timeout=timeout, age=age), self.assertRaises(ValueError):
                EpisodeConfig(17, 1, correction_timeout_seconds=timeout, correction_max_age_seconds=age)

    def test_resealed_evidence_cannot_extend_or_remove_validity(self):
        for damage in ('deadline', 'checked_at', 'missing', 'age', 'config', 'reason'):
            with self.subTest(damage=damage), TemporaryDirectory() as tmp:
                path = Path(tmp)/'trace'
                self.episode(path, 'override', 11. if damage == 'reason' else 10.5)
                manifest = json.loads((path/'manifest.json').read_text())
                rows = [json.loads(line) for line in (path/'decisions.jsonl').read_text().splitlines()]
                record = rows[0]['action_record']
                if damage == 'missing':
                    record.pop('correction_expiry')
                elif damage == 'config':
                    manifest['config']['correction_timeout_seconds'] = .4
                elif damage == 'reason':
                    record['correction_expiry']['valid'] = True
                else:
                    key = 'max_age_seconds' if damage == 'age' else damage
                    record['correction_expiry'][key] += 1
                data = ''.join(json.dumps(row)+'\n' for row in rows).encode()
                (path/'decisions.jsonl').write_bytes(data)
                manifest['files']['decisions.jsonl'] = sha256(data).hexdigest()
                (path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(TraceError):
                    load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
