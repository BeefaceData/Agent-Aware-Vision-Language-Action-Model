"""Recorded files replay through the public harness, without inference."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionResolution, EpisodeConfig, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class RecordedReplayTests(unittest.TestCase):
    def test_supplied_failed_fixture_and_cli(self):
        path = Path(__file__).parent / 'fixtures' / 'recorded_episode'
        recorder = ReplayRecorder()
        outcome = load_recorded_replay(path).run(recorder)
        self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason),
                         (False, 2, 'terminated'))
        self.assertEqual([step[2] for step in recorder.steps], [[0.5], [0.2]])
        self.assertEqual([step[3].action_record.proposed_action for step in recorder.steps],
                         [[0.1], [0.2]])
        result = subprocess.run([sys.executable, str(path.parents[2] / 'recorded_replay.py'),
                                 str(path)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['replay_outcome']['stop_reason'], 'terminated')

    def record(self, root, ending='success'):
        observations = [
            {'task': 'place object', 'pixels': {'image': [[[i]]],
                                             'image2': None if i == 1 else [[[i + 1]]]},
             'robot_state': {'position': [i, 0, 0]}, 'private_evaluator': i}
            for i in range(3)]
        config = EpisodeConfig(17, 2)
        policy = ReplayPolicy(((observations[0], [0.1]), (observations[1], [0.2])))
        environment = ReplayEnvironment(17, observations[0], (
            ([0.5], ReplayStep(observations[1], 0.25, False, False, False)),
            ([0.2], ReplayStep(observations[2], 1.0, ending == 'success',
                               ending in ('success', 'terminated'), ending == 'truncated')),
        ))
        recorder = ReplayRecorder()
        trace = TraceRecorder(root / 'replay', config, recorder)

        def select(proposal):
            if proposal.observation.sequence == 0:
                return ActionResolution('override', [0.5])
            if ending == 'rejected':
                return ActionResolution('reject', reason='stale request')
            return ActionResolution('pass')

        outcome = run_episode(config, policy, environment, trace, action_selector=select)
        trace.seal(outcome)
        return root / 'replay', recorder, outcome

    def rewrite(self, path, name, transform):
        rows = [json.loads(line) for line in (path / name).read_text().splitlines()]
        transform(rows)
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (path / name).write_bytes(raw)
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][name] = sha256(raw).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    def test_complete_roundtrip_preserves_observations_decisions_and_outcomes(self):
        for ending in ('success', 'terminated', 'truncated', 'step_limit', 'rejected'):
            with self.subTest(ending=ending), TemporaryDirectory() as directory:
                path, original, expected = self.record(Path(directory), ending)
                replay = load_recorded_replay(path)
                self.assertEqual(replay.source_episode_id, expected.episode_id)
                for _ in range(2):
                    actual = ReplayRecorder()
                    outcome = replay.run(actual)
                    self.assertNotEqual(outcome.episode_id, expected.episode_id)
                    self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason,
                                      outcome.sum_rewards),
                                     (expected.success, expected.steps, expected.stop_reason,
                                      expected.sum_rewards))
                    self.assertEqual([p.observation for p in actual.observations],
                                     [p.observation for p in original.observations])
                    self.assertEqual([p.frame_references for p in actual.observations],
                                     [p.frame_references for p in original.observations])
                    for before, after in zip(original.steps, actual.steps):
                        self.assertEqual(before[2], after[2])
                        self.assertEqual(before[3].action_record.proposed_action,
                                         after[3].action_record.proposed_action)
                        self.assertEqual(before[3].action_record.disposition,
                                         after[3].action_record.disposition)
                        self.assertEqual((before[3].reward, before[3].success,
                                          before[3].terminated, before[3].truncated),
                                         (after[3].reward, after[3].success,
                                          after[3].terminated, after[3].truncated))
                    self.assertEqual(len(actual.failures), len(original.failures))
                    if ending == 'rejected':
                        self.assertEqual(actual.failures[0][3].action_record.rejection_reason,
                                         'stale request')

    def test_missing_corrupt_and_unsealed_inputs_fail_before_replay(self):
        for name in ('manifest.json', 'observations.jsonl', 'decisions.jsonl'):
            for damage in ('missing', 'corrupt'):
                with self.subTest(name=name, damage=damage), TemporaryDirectory() as directory:
                    path, _, _ = self.record(Path(directory))
                    if damage == 'missing':
                        (path / name).unlink()
                    else:
                        (path / name).write_text('{broken')
                    with self.assertRaises(TraceError):
                        load_recorded_replay(path)

    def test_semantic_corruption_fails_even_with_updated_checksum(self):
        mutations = [
            ('observations.jsonl', lambda rows: rows.reverse()),
            ('observations.jsonl', lambda rows: rows.pop()),
            ('observations.jsonl', lambda rows: rows[1].update(episode_id='foreign')),
            ('observations.jsonl', lambda rows: rows[1].update(sequence=0)),
            ('observations.jsonl', lambda rows: rows[1].update(captured_monotonic=-1)),
            ('observations.jsonl', lambda rows: rows[1].pop('observation')),
            ('observations.jsonl', lambda rows: rows[0]['observation']['pixels'].pop('image')),
            ('decisions.jsonl', lambda rows: rows.reverse()),
            ('decisions.jsonl', lambda rows: rows[0].update(source_episode_id='foreign')),
            ('decisions.jsonl', lambda rows: rows[0].update(source_sequence=1)),
            ('decisions.jsonl', lambda rows: rows[0]['action_record'].update(proposal_id='other')),
            ('decisions.jsonl', lambda rows: rows[0]['action_record'].update(executed_action=[99])),
            ('decisions.jsonl', lambda rows: rows[0]['action_record'].update(execution_acknowledgement=None)),
            ('decisions.jsonl', lambda rows: rows[1]['action_record'].update(proposed_action=[99])),
            ('decisions.jsonl', lambda rows: rows[0]['result'].update(success=True)),
            ('decisions.jsonl', lambda rows: rows[1]['result'].update(reward=99)),
            ('decisions.jsonl', lambda rows: rows.pop()),
        ]
        for name, mutate in mutations:
            with self.subTest(mutation=mutate), TemporaryDirectory() as directory:
                path, _, _ = self.record(Path(directory))
                self.rewrite(path, name, mutate)
                with self.assertRaises(TraceError):
                    load_recorded_replay(path)

    def test_interrupted_episode_is_not_sealed(self):
        with TemporaryDirectory() as directory:
            config = EpisodeConfig(17, 2)
            recorder = TraceRecorder(Path(directory) / 'replay', config, ReplayRecorder())
            environment = ReplayEnvironment(17, 'start', (([1], ReplayStep('next', 0, False, False, False)),))
            policy = ReplayPolicy((('start', [1]),))

            def cancel(step, result):
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                run_episode(config, policy, environment, recorder, on_step=cancel)
            with self.assertRaises(TraceError):
                load_recorded_replay(Path(directory) / 'replay')

    def test_sealing_rejects_wrong_or_incomplete_outcome(self):
        with TemporaryDirectory() as directory:
            config = EpisodeConfig(17, 1)
            trace = TraceRecorder(Path(directory) / 'replay', config, ReplayRecorder())
            outcome = run_episode(config, ReplayPolicy((('start', [1]),)),
                                  ReplayEnvironment(17, 'start', (([1], ReplayStep('end', 1, True, True, False)),)),
                                  trace)
            for invalid in (replace(outcome, artifact_status='incomplete'),
                            replace(outcome, episode_id='foreign'),
                            replace(outcome, sum_rewards=99)):
                with self.assertRaises(TraceError):
                    trace.seal(invalid)
            trace.seal(outcome)
            self.assertTrue(load_recorded_replay(Path(directory) / 'replay').run().success)

    def test_failed_finalization_cannot_publish_replay(self):
        class BrokenRecorder(ReplayRecorder):
            def finish(self):
                raise OSError('sink close failed')

        with TemporaryDirectory() as directory:
            config = EpisodeConfig(17, 1)
            path = Path(directory) / 'replay'
            trace = TraceRecorder(path, config, BrokenRecorder())
            outcome = run_episode(config, ReplayPolicy((('start', [1]),)),
                                  ReplayEnvironment(17, 'start', (([1], ReplayStep('end', 1, True, True, False)),)),
                                  trace)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.artifact_status, 'incomplete')
            with self.assertRaises(TraceError):
                trace.seal(outcome)
            with self.assertRaises(TraceError):
                load_recorded_replay(path)

    def test_finalization_cancellation_keeps_original_exception(self):
        cancellation = KeyboardInterrupt()

        class CancelRecorder(ReplayRecorder):
            def finish(self):
                raise cancellation

        with TemporaryDirectory() as directory:
            config = EpisodeConfig(17, 1)
            trace = TraceRecorder(Path(directory) / 'replay', config, CancelRecorder())
            with self.assertRaises(KeyboardInterrupt) as caught:
                run_episode(config, ReplayPolicy((('start', [1]),)),
                            ReplayEnvironment(17, 'start', (([1], ReplayStep('end', 1, True, True, False)),)),
                            trace)
            self.assertIs(caught.exception, cancellation)
            with self.assertRaises(TraceError):
                load_recorded_replay(Path(directory) / 'replay')


if __name__ == '__main__':
    unittest.main()
