"""Operational elapsed-time limits through episodes and sealed replay."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import json
import unittest

from episode_harness import EpisodeConfig, ExecutionBusy, run_episode
from episode_deadline import EpisodeDeadlineExceeded
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
import test_intervention_limits as fixtures


class EpisodeDeadlineTests(unittest.TestCase):
    def fixture(self, now, limit=1.):
        obs = {'task': 'place item'}
        config = EpisodeConfig(17, 3, max_episode_seconds=limit)
        env = ReplayEnvironment(17, obs, [([0], ReplayStep(obs, 0, False, False, False))] * 3,
                                clock=lambda: now[0])
        return config, ReplayPolicy([(obs, [0])] * 3), env

    def seal(self, path, config, policy, env, **kwargs):
        trace = TraceRecorder(path, config, ReplayRecorder())
        outcome = run_episode(config, policy, env, trace, **kwargs)
        trace.seal(outcome)
        replay = load_recorded_replay(path)
        actual = replay.run()
        self.assertEqual((actual.steps, actual.stop_reason), (outcome.steps, outcome.stop_reason))
        self.assertEqual(actual.wall_clock_limit, outcome.wall_clock_limit)
        return outcome

    def test_waits_expire_at_exact_deadline_and_cannot_fall_back(self):
        for finish in (10.999, 11., 11.001):
            with self.subTest(finish=finish), TemporaryDirectory() as tmp:
                now = [10.]
                config, policy, env = self.fixture(now)
                def supervise(*args):
                    now[0] = finish
                outcome = self.seal(Path(tmp)/'trace', config, policy, env,
                                    supervisor=supervise, clock=lambda: now[0])
                self.assertEqual(outcome.steps, 3 if finish < 11 else 0)
                self.assertEqual(len(env.actions), outcome.steps)
                if finish >= 11:
                    self.assertEqual(outcome.stop_reason, 'wall_clock_limit')
                    self.assertEqual(outcome.supervisor_call_budget['attempted'], 1)
                    self.assertTrue(outcome.wall_clock_limit['interruption']['confirmed'])
                    self.assertEqual(len(env.interruptions), 1)

    def test_policy_wait_is_bounded_and_late_result_cannot_dispatch_or_reset(self):
        entered, release, done = Event(), Event(), Event()
        now = [10.]
        config, policy, env = self.fixture(now)
        def act(packet):
            entered.set()
            release.wait(3)
            done.set()
            return [0]
        policy.act = act
        # The callback advances the controlled clock while staying blocked.
        def advance():
            self.assertTrue(entered.wait(1))
            now[0] = 11.
        from threading import Thread
        Thread(target=advance, daemon=True).start()
        try:
            with TemporaryDirectory() as tmp:
                started = monotonic()
                outcome = self.seal(Path(tmp)/'trace', config, policy, env, clock=lambda: now[0])
                self.assertLess(monotonic() - started, 1.)
                self.assertEqual(outcome.steps, 0)
                with self.assertRaises(ExecutionBusy):
                    run_episode(config, policy, env, ReplayRecorder(), clock=lambda: now[0])
                self.assertEqual(env.actions, [])
        finally:
            release.set()
            self.assertTrue(done.wait(1))
        self.assertEqual(env.actions, [])

    def test_recovery_wait_cancels_remaining_commands_and_replays(self):
        fixture = fixtures.InterventionLimitTests()
        fixture.setUp()
        config, policy, env, selector, _ = fixture.episode([fixtures.TOOL])
        config = replace(config, max_episode_seconds=1.)
        now = [10.]
        def observe(*args):
            now[0] = 11.
            return fixture.recovery.observe(*args)
        with TemporaryDirectory() as tmp:
            outcome = self.seal(Path(tmp)/'trace', config, policy, env, action_selector=selector,
                                recovery_observer=observe, clock=lambda: now[0])
        self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'wall_clock_limit'))
        self.assertEqual(len(env.actions), 1)

    def test_post_execution_and_on_step_time_count_even_at_action_horizon(self):
        for stage in ('execution', 'on_step'):
            now = [10.]
            config, policy, env = self.fixture(now)
            config = replace(config, max_steps=1)
            original = env.step
            def step(action):
                result = original(action)
                now[0] = 11.
                return result
            if stage == 'execution':
                env.step = step
            def after(*args):
                now[0] = 11.
            with TemporaryDirectory() as tmp:
                outcome = self.seal(Path(tmp)/'trace', config, policy, env, clock=lambda: now[0],
                                    on_step=after if stage == 'on_step' else None)
            self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'wall_clock_limit'))

    def test_unacknowledged_execution_is_incomplete_and_stopped(self):
        entered, release, done = Event(), Event(), Event()
        now = [10.]
        config, policy, env = self.fixture(now)
        def step(action):
            entered.set()
            now[0] = 11.
            release.wait(3)
            done.set()
        env.step = step
        recorder = ReplayRecorder()
        try:
            with self.assertRaises(EpisodeDeadlineExceeded) as caught:
                run_episode(config, policy, env, recorder, clock=lambda: now[0])
            partial = caught.exception.episode_interruption
            self.assertEqual(partial.steps, 0)
            self.assertEqual(partial.failed_action.disposition, 'unconfirmed')
            self.assertTrue(partial.failed_action.interruption['confirmed'])
            self.assertEqual(partial.artifact_status, 'incomplete')
        finally:
            release.set()
            self.assertTrue(done.wait(1))

    def test_blocked_supervisor_expires_without_waiting_for_its_request_timeout(self):
        from supervisor_provider import BoundedSupervisorProvider
        entered, release, done = Event(), Event(), Event()
        now = [10.]
        config, policy, env = self.fixture(now)
        def provider(proposal, deadline, cancellation):
            entered.set()
            now[0] = 11.
            release.wait(3)
            done.set()
            return {'late': 'invalid response'}
        decider = BoundedSupervisorProvider(provider, 2.)
        try:
            with TemporaryDirectory() as tmp:
                started = monotonic()
                outcome = self.seal(Path(tmp)/'trace', config, policy, env,
                                    supervisor_decider=decider, clock=lambda: now[0])
                self.assertLess(monotonic() - started, 1.)
                self.assertTrue(entered.is_set())
                self.assertEqual(outcome.steps, 0)
                self.assertEqual(outcome.supervisor_call_budget['attempted'], 1)
                self.assertEqual(outcome.stop_reason, 'wall_clock_limit')
        finally:
            release.set()
            self.assertTrue(done.wait(1))
        self.assertEqual(env.actions, [])

    def test_selection_delay_never_dispatches_or_charges_intervention(self):
        fixture = fixtures.InterventionLimitTests()
        fixture.setUp()
        for kind in ('override', fixtures.TOOL):
            config, policy, env, selector, _ = fixture.episode([kind])
            config = replace(config, max_episode_seconds=1.)
            now = [10.]
            def select(proposal):
                selected = selector(proposal)
                now[0] = 11.
                return selected
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                outcome = self.seal(Path(tmp)/'trace', config, policy, env,
                    action_selector=select, recovery_observer=fixture.recovery.observe,
                    clock=lambda: now[0])
                self.assertEqual(outcome.steps, 0)
                self.assertEqual(env.actions, [])
                self.assertEqual(outcome.stop_reason, 'wall_clock_limit')

    def test_fresh_episode_renews_cap_after_a_completed_expiry(self):
        now = [10.]
        config, policy, env = self.fixture(now)
        def supervise(*args):
            now[0] += 1.
        for started in (10., 11.):
            with TemporaryDirectory() as tmp:
                outcome = self.seal(Path(tmp)/'trace', config, policy, env,
                                    supervisor=supervise, clock=lambda: now[0])
                self.assertEqual(outcome.episode_clock['started_at'], started)
                self.assertEqual(outcome.episode_clock['deadline'], started + 1.)
                self.assertEqual(outcome.steps, 0)

    def test_unconfirmed_stop_does_not_claim_successful_interruption(self):
        now = [10.]
        config, policy, env = self.fixture(now)
        env.interrupt = lambda request: False
        def supervise(*args):
            now[0] = 11.
        with TemporaryDirectory() as tmp:
            outcome = self.seal(Path(tmp)/'trace', config, policy, env,
                                supervisor=supervise, clock=lambda: now[0])
        self.assertEqual(outcome.stop_reason, 'interruption_failed')

    def test_limit_and_interruption_contract_are_required_for_capped_execution(self):
        for value in (0, -1, True, float('nan'), float('inf'), '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EpisodeConfig(17, 3, max_episode_seconds=value)
        now = [10.]
        config, policy, env = self.fixture(now)
        env.interruption_contract = None
        with self.assertRaisesRegex(ValueError, 'hold/stop'):
            run_episode(config, policy, env, ReplayRecorder(), clock=lambda: now[0])
        self.assertEqual(policy.observations, [])

    def test_resealed_limit_or_expiry_tampering_is_rejected(self):
        now = [10.]
        config, policy, env = self.fixture(now)
        def supervise(*args):
            now[0] = 11.
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace'
            self.seal(path, config, policy, env, supervisor=supervise, clock=lambda: now[0])
            original = (path/'manifest.json').read_text()
            for field in ('deadline', 'expired_at', 'started_at', 'limit_seconds'):
                manifest = json.loads(original)
                manifest['wall_clock_limit'][field] += 1
                (path/'manifest.json').write_text(json.dumps(manifest))
                # A later expired_at alone is consistent; the stop predates it.
                if field == 'expired_at':
                    manifest['wall_clock_limit'][field] = 10.5
                    (path/'manifest.json').write_text(json.dumps(manifest))
                with self.subTest(field=field), self.assertRaises(TraceError):
                    load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
