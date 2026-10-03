"""Terminal categories survive public replay and JSON report serialization."""
import json
import unittest
from dataclasses import asdict

from episode_harness import ActionResolution, EpisodeConfig, run_episode
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from run_smolvla_episode import evidence_json


class TerminationTests(unittest.TestCase):
    def run_fixture(self, stop, recorder=None, **kwargs):
        environment = ReplayEnvironment(7, 'start', [('move', ReplayStep(
            'end', 0, stop == 'success', stop in ('success', 'terminated'),
            stop == 'truncated'))])
        policy = ReplayPolicy([('start', 'move')])
        result = run_episode(EpisodeConfig(7, 1), policy, environment,
                             recorder or ReplayRecorder(), **kwargs)
        return result, environment

    def test_completed_categories_and_artifact_failures(self):
        for reason, status in [('success', 'success'), ('terminated', 'failure'),
                               ('step_limit', 'failure'), ('truncated', 'unknown'),
                               ('proposal_rejected', 'unknown')]:
            for broken in (False, True):
                with self.subTest(reason=reason, broken=broken):
                    class Recorder(ReplayRecorder):
                        def finish(self):
                            if broken:
                                raise OSError('disk unavailable')
                            return super().finish()
                    options = ({'action_selector': lambda proposal:
                                ActionResolution('reject', reason='bounds')}
                               if reason == 'proposal_rejected' else {})
                    result, environment = self.run_fixture(reason, Recorder(), **options)
                    report = json.loads(json.dumps(asdict(result), default=evidence_json))
                    self.assertEqual(report['stop_reason'], reason)
                    self.assertEqual(report['task_status'], status)
                    self.assertEqual(report['artifact_status'],
                                     'incomplete' if broken else 'completed')
                    self.assertEqual(len(environment.actions),
                                     0 if reason == 'proposal_rejected' else 1)

    def test_faults_keep_observed_task_outcomes(self):
        for error, reason in [(TimeoutError('deadline'), 'timeout'),
                              (RuntimeError('transport'), 'infrastructure_failure'),
                              (KeyboardInterrupt(), 'interrupted')]:
            for stop, status in [('success', 'success'), ('terminated', 'failure'),
                                 ('truncated', 'unknown'), ('step_limit', 'failure')]:
                with self.subTest(reason=reason, stop=stop):
                    def callback(*args):
                        raise error
                    with self.assertRaises(type(error)) as caught:
                        self.run_fixture(stop, on_step=callback)
                    report = json.loads(json.dumps(
                        asdict(caught.exception.episode_interruption), default=evidence_json))
                    self.assertEqual(report['stop_reason'], reason)
                    self.assertEqual(report['task_status'], status)
                    self.assertEqual(report['artifact_status'], 'incomplete')
                    self.assertEqual(report['steps'], 1)

    def test_pre_action_failure_is_unknown_and_finalization_cancel_keeps_success(self):
        def timeout(*args):
            raise TimeoutError('provider deadline')
        with self.assertRaises(TimeoutError) as caught:
            self.run_fixture('success', supervisor=timeout)
        self.assertEqual(caught.exception.episode_interruption.task_status, 'unknown')
        self.assertEqual(caught.exception.episode_interruption.steps, 0)

        class Recorder(ReplayRecorder):
            def finish(self):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_fixture('success', Recorder())
        self.assertEqual(caught.exception.episode_interruption.task_status, 'success')
        self.assertEqual(caught.exception.episode_interruption.stop_reason, 'interrupted')


if __name__ == '__main__':
    unittest.main()
