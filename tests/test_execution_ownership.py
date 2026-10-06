"""Public overlapping-episode behavior, without models or robot actuation."""

from threading import Event, Thread
import unittest

from episode_harness import ExecutionBusy, run_episode
from replay_adapters import successful_replay


class ExecutionOwnershipTests(unittest.TestCase):
    def test_reentrant_calls_reject_each_shared_adapter_but_allow_independent_episodes(self):
        owner = successful_replay()
        rejected = []

        def on_step(*args):
            for shared in ('policy', 'environment', 'recorder'):
                competitor = successful_replay()
                adapters = {name: getattr(owner if name == shared else competitor, name)
                            for name in ('policy', 'environment', 'recorder')}
                with self.assertRaises(ExecutionBusy):
                    run_episode(competitor.config, **adapters)
                rejected.append(shared)
                if shared != 'environment':
                    self.assertEqual(competitor.environment.actions, [])
                if shared != 'policy':
                    self.assertEqual(competitor.policy.observations, [])
                if shared != 'recorder':
                    self.assertFalse(competitor.recorder.finalized)
            independent = successful_replay()
            self.assertTrue(run_episode(independent.config, independent.policy,
                independent.environment, independent.recorder).success)

        outcome = run_episode(owner.config, owner.policy, owner.environment,
                              owner.recorder, on_step=on_step)
        self.assertTrue(outcome.success)
        self.assertEqual(len(rejected), 6)
        self.assertEqual(owner.environment.actions, [('reach', .25), ('place', .75)])
        # Ownership is released on success; a deliberate new attempt can reset.
        self.assertTrue(run_episode(owner.config, owner.policy, owner.environment,
                                    owner.recorder).success)

    def test_ownership_covers_startup_and_finalization(self):
        for phase in ('reset', 'finish'):
            with self.subTest(phase=phase):
                owner = successful_replay()
                entered, release = Event(), Event()
                outcomes, errors = [], []
                adapter = owner.environment if phase == 'reset' else owner.recorder
                original = getattr(adapter, phase)

                def held(*args):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError('ownership test was not released')
                    return original(*args)

                setattr(adapter, phase, held)

                def run():
                    try:
                        outcomes.append(run_episode(owner.config, owner.policy,
                                                     owner.environment, owner.recorder))
                    except BaseException as exc:
                        errors.append(exc)

                worker = Thread(target=run)
                worker.start()
                try:
                    self.assertTrue(entered.wait(3))
                    with self.assertRaises(ExecutionBusy):
                        run_episode(owner.config, owner.policy, owner.environment, owner.recorder)
                    self.assertEqual(len(owner.environment.actions), 0 if phase == 'reset' else 2)
                finally:
                    release.set()
                    worker.join(6)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(outcomes[0].success)

    def test_failures_release_ownership_after_cleanup(self):
        for stage in ('reset', 'step', 'finish'):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(stage=stage, error=error_type):
                    owner = successful_replay()
                    adapter = owner.recorder if stage == 'finish' else owner.environment
                    original = getattr(adapter, stage)

                    def fail(*args):
                        # Even failed-run cleanup must retain ownership.
                        with self.assertRaises(ExecutionBusy):
                            run_episode(owner.config, owner.policy, owner.environment, owner.recorder)
                        raise error_type('injected failure')

                    setattr(adapter, stage, fail)
                    if stage == 'finish' and error_type is RuntimeError:
                        outcome = run_episode(owner.config, owner.policy, owner.environment,
                                              owner.recorder)
                        self.assertEqual(outcome.artifact_status, 'incomplete')
                    else:
                        with self.assertRaises(error_type):
                            run_episode(owner.config, owner.policy, owner.environment, owner.recorder)
                    setattr(adapter, stage, original)
                    self.assertTrue(run_episode(owner.config, owner.policy, owner.environment,
                                                owner.recorder).success)


if __name__ == '__main__':
    unittest.main()
