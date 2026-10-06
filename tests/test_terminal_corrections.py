"""Terminal correction boundaries through public episodes and sealed replay."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest

from episode_harness import ActionResolution, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayRecorder
import test_intervention_limits as fixtures


class TerminalCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InterventionLimitTests()
        self.fixture.setUp()

    def terminal_episode(self, choices, ending, terminal_step=1):
        episode = self.fixture.episode(choices)
        step = episode[2].step

        def terminal(action):
            result = step(action)
            return (replace(result, **{ending: True})
                    if result.observation.sequence == terminal_step else result)

        episode[2].step = terminal
        return episode

    def test_terminal_recovery_cancels_assessment_commands_and_resume(self):
        for ending in ('success', 'terminated', 'truncated'):
            for terminal_step in (1, 2):
                with self.subTest(ending=ending, step=terminal_step), TemporaryDirectory() as tmp:
                    episode = self.terminal_episode([fixtures.TOOL, 'pass'], ending, terminal_step)
                    assessed, resumed = [], []
                    episode[1].resume = lambda packet: resumed.append(packet.sequence)

                    def assess(plan, index, packet):
                        assessed.append(packet.sequence)
                        if packet.sequence == terminal_step:
                            raise KeyboardInterrupt('terminal assessment must not run')
                        return self.fixture.recovery.observe(plan, index, packet)

                    outcome, evidence = self.fixture.run_trace(Path(tmp) / 'trace', episode, assess)
                    self.assertEqual((outcome.steps, outcome.stop_reason), (terminal_step, ending))
                    self.assertEqual(outcome.terminal_observation.sequence, terminal_step)
                    self.assertEqual(assessed, list(range(1, terminal_step)))
                    self.assertEqual(resumed, [])
                    self.assertEqual(episode[4], [0])
                    self.assertEqual(len(episode[2].actions), terminal_step)
                    check = evidence['decisions'][-1]['action_record']['recovery']['check']
                    self.assertEqual(check['reason'], 'episode_terminated')
                    self.assertIsNone(check['assessment'])

    def test_late_threaded_correction_cannot_dispatch_or_cross_episode_boundary(self):
        for ending in ('success', 'terminated', 'truncated'):
            for kind in ('override', 'recovery'):
                with self.subTest(ending=ending, kind=kind), TemporaryDirectory() as tmp:
                    config, policy, env, _, _ = self.terminal_episode(['pass', 'pass'], ending)
                    terminal, delivered = Event(), Event()
                    pending, errors, selections, workers = Queue(), [], [], []

                    def select(proposal):
                        selections.append(proposal.proposal_id)
                        if not pending.empty():
                            return pending.get_nowait()
                        source = deepcopy(proposal)

                        def provider():
                            try:
                                if not terminal.wait(3):
                                    raise TimeoutError('terminal result not delivered')
                                resolution = (self.fixture.recovery.resolve(proposal=source, now=10.)
                                    if kind == 'recovery' else ActionResolution('override', [0.3] * 7,
                                        source_identity=(source.observation.episode_id,
                                            source.observation.sequence, source.proposal_id)))
                                pending.put(resolution)
                            except BaseException as exc:
                                errors.append(exc)
                            finally:
                                delivered.set()

                        worker = Thread(target=provider)
                        workers.append(worker)
                        worker.start()
                        return ActionResolution('pass')

                    def after_terminal(step, result):
                        terminal.set()
                        self.assertTrue(delivered.wait(3))

                    trace = TraceRecorder(Path(tmp) / 'first', config, ReplayRecorder())
                    try:
                        outcome = run_episode(config, policy, env, trace, action_selector=select,
                            on_step=after_terminal, clock=lambda: 10.)
                    finally:
                        terminal.set()
                        for worker in workers:
                            worker.join(4)
                            self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(len(selections), 1)
                    self.assertEqual(len(env.actions), 1)
                    self.assertEqual(pending.qsize(), 1)
                    trace.seal(outcome)
                    replay = load_recorded_replay(trace.directory)
                    self.assertEqual(replay.run().stop_reason, ending)
                    self.assertEqual(len(replay.evidence()['decisions']), 1)

                    # Deliberate restart: even the same adapters cannot grant an
                    # old executor result authority over the new attempt.
                    second = TraceRecorder(Path(tmp) / 'second', config, ReplayRecorder())
                    rejected = run_episode(config, policy, env, second, action_selector=select,
                        recovery_observer=self.fixture.recovery.observe, clock=lambda: 10.)
                    self.assertNotEqual(outcome.episode_id, rejected.episode_id)
                    self.assertEqual((rejected.steps, rejected.stop_reason), (0, 'proposal_rejected'))
                    self.assertEqual(env.actions, [])
                    second.seal(rejected)
                    self.assertEqual(load_recorded_replay(second.directory).run().steps, 0)

    def test_terminal_evidence_survives_recording_callback_and_cleanup_faults(self):
        for ending in ('success', 'terminated', 'truncated'):
            for stage in ('recording', 'callback', 'finish'):
                with self.subTest(ending=ending, stage=stage):
                    config, policy, env, select, _ = self.terminal_episode([fixtures.TOOL], ending)

                    class Recorder(ReplayRecorder):
                        def record_step(self, step, source, action, result, ingestion):
                            if stage == 'recording':
                                result.observation.observation.clear()
                                raise KeyboardInterrupt('recording cancelled')
                            super().record_step(step, source, action, result, ingestion)

                        def finish(self):
                            raise KeyboardInterrupt('cleanup cancelled')

                    def callback(step, result):
                        if stage == 'callback':
                            result.observation.observation.clear()
                            raise KeyboardInterrupt('callback cancelled')

                    with self.assertRaises(KeyboardInterrupt) as caught:
                        run_episode(config, policy, env, Recorder(), action_selector=select,
                            on_step=callback, recovery_observer=self.fixture.recovery.observe,
                            clock=lambda: 10.)
                    evidence = caught.exception.episode_interruption
                    self.assertEqual(evidence.terminal_reason, ending)
                    self.assertEqual(evidence.terminal_observation.sequence, 1)
                    self.assertEqual(evidence.last_observation.sequence, 1)
                    self.assertTrue(evidence.last_observation.observation)
                    self.assertEqual(evidence.stop_reason, 'interrupted')
                    self.assertEqual(len(env.actions), 1)

    def test_replay_rejects_terminal_cancellation_without_evaluator_terminal_result(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace'
            self.fixture.run_trace(path,
                self.terminal_episode([fixtures.TOOL, 'pass'], 'success'))
            rows = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
            rows[-1]['result']['success'] = False
            data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            (path / 'decisions.jsonl').write_bytes(data)
            manifest = json.loads((path / 'manifest.json').read_text())
            manifest['files']['decisions.jsonl'] = sha256(data).hexdigest()
            (path / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(TraceError):
                load_recorded_replay(path)

    def test_historical_terminal_local_assessment_remains_replayable(self):
        from episode_harness import RecoverySequence
        from recovery_monitor import check_recovery
        config, policy, env, select, _ = self.terminal_episode([fixtures.TOOL], 'success')
        assess = self.fixture.recovery.observe

        class HistoricalTrace(TraceRecorder):
            def record_step(self, step, source, action, result, ingestion):
                recovery = deepcopy(result.action_record.recovery)
                plan = RecoverySequence(**recovery['sequence'])
                recovery['check'] = check_recovery(plan, 0, result.observation,
                    assess(plan, 0, result.observation), 10.)
                result = replace(result, action_record=replace(result.action_record, recovery=recovery))
                super().record_step(step, source, action, result, ingestion)

        with TemporaryDirectory() as tmp:
            trace = HistoricalTrace(Path(tmp) / 'trace', config, ReplayRecorder())
            outcome = run_episode(config, policy, env, trace, action_selector=select,
                recovery_observer=assess, clock=lambda: 10.)
            trace.seal(outcome)
            replay = load_recorded_replay(trace.directory)
            self.assertEqual(replay.evidence()['decisions'][0]['action_record']
                             ['recovery']['check']['status'], 'continuing')
            self.assertTrue(replay.run().success)


if __name__ == '__main__':
    unittest.main()
