"""Controlled pending decisions through the public simulation episode path."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest

from episode_harness import run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError


class DelayedResponses:
    """Release each scripted response explicitly; never contact a model."""

    def __init__(self):
        self.entered = [Event(), Event()]
        self.release = [Event(), Event()]
        self.returned = [Event(), Event()]
        self.proposals = []

    def __call__(self, proposal, deadline, cancellation):
        index = len(self.proposals)
        self.proposals.append(proposal)
        self.entered[index].set()
        if not self.release[index].wait(5):
            raise TimeoutError('test response was never released')
        self.returned[index].set()
        return dict(kind='pass', episode_id=proposal.observation.episode_id,
                    observation_sequence=proposal.observation.sequence,
                    proposal_id=proposal.proposal_id)


class SynchronousSupervisionTests(unittest.TestCase):
    def test_pending_response_and_timeout_preserve_episode_order(self):
        for timeout_second in (False, True):
            with self.subTest(timeout_second=timeout_second), TemporaryDirectory() as temporary:
                fixture = successful_replay()
                delayed = DelayedResponses()
                adapter = BoundedSupervisorProvider(delayed, 2)
                directory = Path(temporary) / 'trace'
                trace = TraceRecorder(directory, fixture.config, fixture.recorder)
                outcomes, errors = [], []

                def run():
                    try:
                        outcomes.append(run_episode(
                            fixture.config, fixture.policy, fixture.environment,
                            trace, supervisor_decider=adapter))
                    except BaseException as exc:
                        errors.append(exc)

                caller = Thread(target=run)
                caller.start()
                try:
                    for index in range(2):
                        self.assertTrue(delayed.entered[index].wait(1))
                        # While the transport is held, no new action or policy
                        # proposal can be produced by the synchronous harness.
                        self.assertTrue(caller.is_alive())
                        self.assertEqual(len(fixture.environment.actions), index)
                        self.assertEqual(len(fixture.policy.observations), index + 1)
                        self.assertEqual(len(fixture.recorder.steps), index)
                        proposal = delayed.proposals[index]
                        self.assertEqual(proposal.observation.sequence, index)
                        self.assertEqual(adapter.request(proposal).status, 'busy')
                        self.assertEqual(len(delayed.proposals), index + 1)
                        self.assertFalse(fixture.recorder.finalized)
                        if index == 0 or not timeout_second:
                            delayed.release[index].set()

                    caller.join(3)
                    self.assertFalse(caller.is_alive())
                    self.assertTrue(fixture.recorder.finalized)
                    if timeout_second:
                        self.assertEqual(outcomes, [])
                        self.assertEqual(len(errors), 1)
                        self.assertIsInstance(errors[0], ProviderRequestError)
                        self.assertEqual(errors[0].result.status, 'timeout')
                        self.assertEqual(fixture.environment.actions, [('reach', .25)])
                        failure = fixture.recorder.failures[0][3]
                        self.assertIsNone(failure.timing.response_at)
                        self.assertIsNone(failure.timing.execution_started_at)
                        self.assertGreaterEqual(failure.timing.decision_latency_seconds, 2)
                        self.assertAlmostEqual(
                            failure.timing.cumulative_wait_seconds,
                            fixture.recorder.steps[0][3].timing.decision_latency_seconds
                            + failure.timing.decision_latency_seconds)
                        rows = [json.loads(line) for line in
                                (directory / 'decisions.jsonl').read_text().splitlines()]
                        self.assertEqual(rows[-1]['action_record']['disposition'], 'rejected')
                        self.assertFalse((directory / 'manifest.json').exists())
                        delayed.release[1].set()
                        self.assertTrue(delayed.returned[1].wait(1))
                        self.assertEqual(fixture.environment.actions, [('reach', .25)])
                    else:
                        self.assertEqual(errors, [])
                        outcome = outcomes[0]
                        self.assertTrue(outcome.success)
                        self.assertEqual(outcome.steps, 2)
                        self.assertEqual(fixture.environment.actions,
                                         [('reach', .25), ('place', .75)])
                        self.assertGreater(outcome.cumulative_wait_seconds, 0)
                        self.assertAlmostEqual(outcome.cumulative_wait_seconds, sum(
                            timing.decision_latency_seconds for timing in outcome.step_timings))
                        for timing in outcome.step_timings:
                            self.assertGreaterEqual(timing.execution_started_at, timing.response_at)
                        trace.seal(outcome)
                        replay = load_recorded_replay(directory)
                        self.assertTrue(replay.run().success)
                        self.assertEqual(len(replay.evidence()['decisions']), 2)
                finally:
                    for release in delayed.release:
                        release.set()
                    caller.join(6)


if __name__ == '__main__':
    unittest.main()
