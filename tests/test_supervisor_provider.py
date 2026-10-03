"""Public provider deadline, cancellation and retained episode evidence."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
import unittest

from episode_harness import ActionProposal, ObservationPacket, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError


def response(proposal, deadline, cancel):
    return dict(kind='pass', episode_id=proposal.observation.episode_id,
                observation_sequence=proposal.observation.sequence,
                proposal_id=proposal.proposal_id)


class ProviderTests(unittest.TestCase):
    def proposal(self, name='current'):
        return ActionProposal(name, ObservationPacket(
            'episode', 0, datetime.now(timezone.utc),
            {'task': 'place', 'private_evaluator': 'SECRET'}), [.1])

    def test_configuration(self):
        for timeout in (0, -1, True, '1', float('nan'), float('inf'), 10 ** 400):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                BoundedSupervisorProvider(response, timeout)

    def test_complete_episode_seals_and_replays_provider_passes(self):
        fixture = successful_replay()
        with TemporaryDirectory() as temporary:
            trace = TraceRecorder(Path(temporary) / 'trace', fixture.config, fixture.recorder)
            outcome = run_episode(fixture.config, fixture.policy, fixture.environment, trace,
                                  supervisor_decider=BoundedSupervisorProvider(response, 1))
            trace.seal(outcome)
            replay = load_recorded_replay(Path(temporary) / 'trace')
            self.assertTrue(replay.run().success)
            self.assertEqual(len(replay.evidence()['decisions']), 2)

    def test_timeout_is_bounded_and_late_reply_cannot_be_reused(self):
        release, entered, done = Event(), Event(), Event()
        calls = []

        def blocked(proposal, deadline, cancel):
            calls.append(proposal.proposal_id)
            self.assertNotIn('SECRET', repr(proposal))
            entered.set()
            release.wait(2)
            done.set()
            return response(proposal, deadline, cancel)

        adapter = BoundedSupervisorProvider(blocked, .05)
        try:
            start = monotonic()
            result = adapter.request(self.proposal())
            self.assertLess(monotonic() - start, .75)
            self.assertTrue(entered.is_set())
            self.assertEqual(result.status, 'timeout')
            self.assertIsNone(result.decision)
            self.assertGreaterEqual(result.finished_at, result.deadline)
            self.assertEqual(adapter.request(self.proposal('next')).status, 'busy')
            self.assertEqual(calls, ['current'])
        finally:
            release.set()
            self.assertTrue(done.wait(1))
        # Wait through the public busy result until the old transport is drained.
        limit = monotonic() + 1
        while monotonic() < limit:
            next_result = adapter.request(self.proposal('next'))
            if next_result.status != 'busy':
                break
        self.assertEqual(next_result.status, 'response')
        self.assertEqual(next_result.decision.proposal_id, 'next')
        self.assertIsNone(result.decision)

    def test_cancellation_before_and_during_request(self):
        cancel, entered, release = Event(), Event(), Event()
        cancel.set()
        adapter = BoundedSupervisorProvider(response, 1)
        self.assertEqual(adapter.request(self.proposal(), cancel).status, 'cancelled')
        cancel.clear()

        def blocked(proposal, deadline, signal):
            entered.set()
            release.wait(2)
            return response(proposal, deadline, signal)

        results = []
        adapter = BoundedSupervisorProvider(blocked, 1)
        caller = Thread(target=lambda: results.append(adapter.request(self.proposal(), cancel)))
        caller.start()
        try:
            self.assertTrue(entered.wait(1))
            cancel.set()
            caller.join(.5)
            self.assertFalse(caller.is_alive())
            self.assertEqual(results[0].status, 'cancelled')
            self.assertIsNone(results[0].decision)
        finally:
            release.set()
            caller.join(2)

    def test_transport_and_malformed_results_are_typed(self):
        def broken(*args):
            raise OSError('SECRET')

        for provider, status in ((broken, 'error'), (lambda *args: {}, 'rejected'),
                                 (lambda *args: response(self.proposal('stale'), 0, None),
                                  'rejected')):
            result = BoundedSupervisorProvider(provider, 1).request(self.proposal())
            self.assertEqual(result.status, status)
            self.assertIsNone(result.decision)
            self.assertNotIn('SECRET', repr(result))

    def test_failed_request_is_recorded_and_never_executes(self):
        release = Event()

        def slow(*args):
            release.wait(2)
            return response(*args)

        def broken(*args):
            raise OSError('SECRET')

        try:
            for provider, status in ((slow, 'timeout'), (broken, 'error')):
                fixture = successful_replay()
                with TemporaryDirectory() as temporary:
                    directory = Path(temporary) / 'trace'
                    trace = TraceRecorder(directory, fixture.config, fixture.recorder)
                    with self.assertRaises(ProviderRequestError) as caught:
                        run_episode(fixture.config, fixture.policy, fixture.environment, trace,
                                    supervisor_decider=BoundedSupervisorProvider(provider, .05))
                    self.assertEqual(caught.exception.result.status, status)
                    self.assertEqual(fixture.environment.actions, [])
                    self.assertTrue(fixture.recorder.finalized)
                    rows = [json.loads(row) for row in
                            (directory / 'decisions.jsonl').read_text().splitlines()]
                    self.assertEqual(rows[-1]['action_record']['disposition'], 'rejected')
                    self.assertIn(status, rows[-1]['action_record']['rejection_reason'])
                    self.assertNotIn('SECRET', repr(rows))
                    self.assertFalse((directory / 'manifest.json').exists())
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
