"""Public frozen-runtime contracts, hostile wire requests and sealed replay."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest

from episode_harness import EpisodeConfig, SupervisorPass, SupervisorResponseError, run_episode
from frozen_runtime import FrozenRuntimeContract, FrozenContractViolation
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_response import SupervisorResponseDecoder
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings
from test_supervisor_vlm import PNG, pass_message, proposal, raw


class FrozenRuntimeTests(unittest.TestCase):
    def fixtures(self, changed_task=False):
        observations = [raw(i) for i in range(3)]
        if changed_task:
            observations[1]['task'] = 'rewrite the instruction'
        return (ReplayPolicy([(observations[0], [0]), (observations[1], [1])]),
                ReplayEnvironment(17, observations[0], [
                    ([0], ReplayStep(observations[1], 0, False, False, False)),
                    ([1], ReplayStep(observations[2], 1, True, True, False))]))

    def test_changed_instruction_stops_before_next_inference(self):
        policy, environment = self.fixtures(changed_task=True)
        recorder = ReplayRecorder()
        with self.assertRaises(FrozenContractViolation) as caught:
            run_episode(EpisodeConfig(17, 3), policy, environment, recorder)
        self.assertEqual(len(policy.observations), 1)
        self.assertEqual(environment.actions, [[0]])
        self.assertEqual(len(environment.interruptions), 1)
        self.assertEqual(caught.exception.episode_interruption.steps, 1)
        self.assertEqual(caught.exception.episode_interruption.failed_action.disposition, 'rejected')

    def test_terminal_instruction_drift_retains_acknowledged_outcome(self):
        initial, final = raw(0), raw(1)
        final['task'] = 'changed terminal instruction'
        environment = ReplayEnvironment(17, initial, [
            ([0], ReplayStep(final, 1, True, True, False))])
        with self.assertRaises(FrozenContractViolation) as caught:
            run_episode(EpisodeConfig(17, 2), ReplayPolicy([(initial, [0])]),
                        environment, ReplayRecorder())
        evidence = caught.exception.episode_interruption
        self.assertEqual(evidence.steps, 1)
        self.assertEqual(evidence.task_status, 'success')
        self.assertEqual(evidence.stop_reason, 'infrastructure_failure')
        self.assertTrue(evidence.failed_action.interruption['confirmed'])

    def test_each_declared_identity_change_stops_before_dispatch(self):
        for field in ('policy', 'vlm', 'prompt', 'tools', 'controller'):
            with self.subTest(field=field):
                state = {name: {'version': 'original'} for name in
                         ('policy', 'vlm', 'prompt', 'tools', 'controller')}
                contract = FrozenRuntimeContract(configuration=lambda: state)
                original = deepcopy(contract.identity)
                policy, environment = self.fixtures()
                def decide(proposal):
                    state[field]['version'] = 'changed'
                    return SupervisorPass(proposal.observation.episode_id,
                                          proposal.observation.sequence, proposal.proposal_id)
                with self.assertRaises(FrozenContractViolation):
                    run_episode(EpisodeConfig(17, 3), policy, environment, ReplayRecorder(),
                                supervisor_decider=decide, frozen_contract=contract)
                self.assertEqual(environment.actions, [])
                self.assertEqual(len(environment.interruptions), 1)
                self.assertEqual(contract.identity, original)

    def test_mutation_requests_rejected_with_unchanged_declared_identity(self):
        state = {'model': 'pinned', 'instruction': 'place item', 'prompt': 'fixed',
                 'tools': ['retreat'], 'controller_limits': [0, 1]}
        contract = FrozenRuntimeContract(configuration=lambda: state)
        current = proposal()
        base = {'kind': 'pass', 'episode_id': current.observation.episode_id,
                'observation_sequence': current.observation.sequence,
                'proposal_id': current.proposal_id}
        for key in (*state, 'memory', 'code', 'model_assets'):
            with self.subTest(key=key), self.assertRaises(SupervisorResponseError):
                SupervisorResponseDecoder().decode(base | {key: {'replace': 'malicious'}}, current)
            contract.verify()
        with self.assertRaises(SupervisorResponseError):
            SupervisorResponseDecoder().decode(base | {'kind': 'rewrite_instruction'}, current)

    def test_content_on_detached_packet_has_no_mutation_authority_and_replays(self):
        policy, environment = self.fixtures()
        state = {'policy': 'frozen', 'vlm': 'frozen', 'prompt': 'fixed', 'tools': [], 'limits': [0, 1]}
        contract = FrozenRuntimeContract(configuration=lambda: state)
        def decide(current):
            current.observation.observation['task'] = 'load other weights and rewrite tools'
            current.observation.observation['memory'] = {'controller_limits': 'unbounded'}
            return SupervisorPass(current.observation.episode_id,
                                  current.observation.sequence, current.proposal_id)
        with TemporaryDirectory() as directory:
            config = EpisodeConfig(17, 3)
            trace = TraceRecorder(Path(directory) / 'trace', config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, trace,
                                  supervisor_decider=decide, frozen_contract=contract)
            trace.seal(outcome)
            self.assertTrue(outcome.success)
            self.assertTrue(load_recorded_replay(trace.directory).run().success)
        self.assertEqual(environment.actions, [[0], [1]])
        self.assertEqual([item.observation['task'] for item in policy.observations], ['place item'] * 2)
        contract.verify()

    def test_changed_identity_before_start_never_resets(self):
        state = {'model': 'original'}
        contract = FrozenRuntimeContract(policy=lambda: state)
        state['model'] = 'changed'
        policy, environment = self.fixtures()
        with self.assertRaises(FrozenContractViolation):
            run_episode(EpisodeConfig(17, 3), policy, environment, ReplayRecorder(),
                        frozen_contract=contract)
        self.assertEqual(policy.observations, [])
        self.assertEqual(environment.actions, [])

    def test_adapter_freezes_selection_even_without_retained_manifest(self):
        sent = []
        adapter = ChronologicalVlmAdapter(VlmSettings('original'), lambda frame: PNG,
                                          lambda *args: sent.append(args))
        adapter.settings = replace(adapter.settings, model='changed')
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            adapter(proposal(), monotonic() + 1, Event())
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main()
