"""Adapter hold/stop via public episodes, including sealed refusal replay."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from baseline_fallback import BaselineFallback
from environment_interruption import InterruptionContract
from episode_harness import ActionResolution, EpisodeConfig, run_episode
from fallback_fixtures import MeasuredReplayEnvironment
from libero_adapter import LiberoEnvironmentAdapter
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from test_baseline_fallback import abstain
from test_episode_timing import ControlledClock


class InterruptionTests(unittest.TestCase):
    def episode(self, directory, operation='stop', acknowledgement=True):
        clock = ControlledClock()
        observations = [{'robot_state': {'position': [i]}} for i in range(2)]

        class Environment(MeasuredReplayEnvironment):
            interruption_contract = InterruptionContract(
                'fixture-' + operation, operation, 'Fixture confirms controller is inactive.')

            def interrupt(self, request):
                self.interruptions.append(request)
                clock.advance(.25)
                if isinstance(acknowledgement, Exception):
                    raise acknowledgement
                return acknowledgement

        env = Environment(17, observations[0], [
            ([.1], ReplayStep(observations[1], 0, False, False, False))], clock)
        policy = ReplayPolicy([(o, [.1]) for o in observations])
        recorder = ReplayRecorder()
        config = EpisodeConfig(17, 3)
        trace = TraceRecorder(directory, config, recorder)
        outcome = run_episode(config, policy, env, trace, clock=clock,
            supervisor_decider=abstain,
            baseline_fallback=BaselineFallback({'robot_state': 1}, lambda: True,
                                               lambda p: p.observation.sequence == 0))
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        self.assertEqual(replay.run().stop_reason, outcome.stop_reason)
        return outcome, env, policy, recorder, replay

    def test_hold_and_stop_preserve_partial_execution_and_end_the_episode(self):
        for operation in ('hold', 'stop'):
            with self.subTest(operation=operation), TemporaryDirectory() as tmp:
                outcome, env, policy, recorder, replay = self.episode(Path(tmp) / 'trace', operation)
                self.assertEqual((outcome.steps, outcome.stop_reason, outcome.task_status),
                                 (1, 'proposal_rejected', 'unknown'))
                self.assertEqual(env.actions, [[.1]])
                self.assertEqual(len(policy.observations), 2)
                self.assertEqual(len(env.interruptions), 1)
                request = env.interruptions[0]
                evidence = replay.evidence()['decisions'][-1]['action_record']
                self.assertEqual(evidence['interruption']['request'], asdict(request))
                self.assertEqual(request.operation, operation)
                self.assertEqual(request.episode_id, outcome.episode_id)
                self.assertEqual(request.observation_sequence, 1)
                self.assertEqual(evidence['interruption']['finished_at'] -
                                 evidence['interruption']['requested_at'], .25)
                self.assertIsNone(evidence['selected_action'])
                self.assertIsNone(evidence['executed_action'])
                self.assertIsNone(evidence['execution_acknowledgement'])
                self.assertTrue(recorder.finalized)

    def test_unconfirmed_or_failed_interruption_is_not_a_successful_stop(self):
        for acknowledgement in (False, None, 1, TimeoutError('PRIVATE transport data')):
            with self.subTest(acknowledgement=acknowledgement), TemporaryDirectory() as tmp:
                outcome, env, _, _, replay = self.episode(
                    Path(tmp) / 'trace', acknowledgement=acknowledgement)
                self.assertEqual((outcome.steps, outcome.stop_reason, outcome.task_status),
                                 (1, 'interruption_failed', 'unknown'))
                self.assertEqual(env.actions, [[.1]])
                self.assertEqual(len(env.interruptions), 1)
                record = replay.evidence()['decisions'][-1]['action_record']['interruption']
                self.assertFalse(record['confirmed'])
                self.assertNotIn('PRIVATE', json.dumps(record))

    def test_missing_capability_blocks_active_configuration_before_reset(self):
        for capability in ('contract', 'method'):
            for mode in ('selector', 'decider', 'fallback'):
                with self.subTest(capability=capability, mode=mode):
                    env = ReplayEnvironment(17, 'initial', [])
                    if capability == 'contract':
                        env.interruption_contract = None
                    else:
                        env.interrupt = None
                    class Policy:
                        def reset(self):
                            raise AssertionError('preflight must precede reset')
                    kwargs = {'selector': dict(action_selector=lambda p: ActionResolution('pass')),
                              'decider': dict(supervisor_decider=abstain),
                              'fallback': dict(baseline_fallback=BaselineFallback(
                                  {'robot_state': 1}, lambda: True, lambda p: True))}[mode]
                    with self.assertRaisesRegex(ValueError, 'hold/stop contract'):
                        run_episode(EpisodeConfig(17, 1), Policy(), env, ReplayRecorder(), **kwargs)

    def test_baseline_does_not_require_interruption_capability(self):
        env = ReplayEnvironment(17, 'initial', [
            ('go', ReplayStep('done', 1, True, True, False))])
        env.interruption_contract = None
        outcome = run_episode(EpisodeConfig(17, 1), ReplayPolicy([('initial', 'go')]),
                              env, ReplayRecorder())
        self.assertTrue(outcome.success)

    def test_libero_stop_disables_steps_until_reset_without_a_zero_command(self):
        class Vector:
            num_envs = 1
            def reset(self, seed):
                return {'robot_state': {'position': [0]}}, {}
            def step(self, action):
                raise AssertionError('unsafe action dispatched')
        env = LiberoEnvironmentAdapter(Vector())
        recorder = ReplayRecorder()
        class Policy:
            def reset(self):
                pass
            def act(self, packet):
                return [0] * 7
        outcome = run_episode(EpisodeConfig(17, 1), Policy(), env, recorder,
                              supervisor_decider=abstain)
        self.assertEqual(outcome.stop_reason, 'proposal_rejected')
        self.assertTrue(recorder.failures[0][3].action_record.interruption['confirmed'])
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            env.step([0] * 7)
        env.reset(17, 'new-episode')
        with self.assertRaisesRegex(AssertionError, 'unsafe action dispatched'):
            env.step([0] * 7)

    def test_resealed_interruption_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory)
            original = (directory / 'decisions.jsonl').read_text()
            for change in ('missing', 'foreign', 'operation', 'acknowledgement', 'time'):
                rows = [json.loads(line) for line in original.splitlines()]
                record = rows[-1]['action_record']
                evidence = record['interruption']
                if change == 'missing':
                    record['interruption'] = None
                elif change == 'foreign':
                    evidence['request']['episode_id'] = 'foreign'
                elif change == 'operation':
                    evidence['request']['operation'] = 'hold'
                elif change == 'acknowledgement':
                    evidence['confirmed'] = False
                else:
                    evidence['finished_at'] = evidence['requested_at'] - 1
                raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                (directory / 'decisions.jsonl').write_bytes(raw)
                manifest_path = directory / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(change=change), self.assertRaises(TraceError):
                    load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
