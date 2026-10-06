"""Offline public adapter checks; synthetic states do not establish task performance."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from episode_harness import EpisodeConfig, run_episode
from libero_adapter import LiberoEnvironmentAdapter
from libero_initial_state import select_initial_state, state_digest
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayRecorder
from run_smolvla_episode import parse_args


STATES = [[0., 1., 2.], [0., 3., 4.]]


def select(task=0, state=0, suite='fixture', states=STATES):
    return select_initial_state(suite, task, state,
        suites={'fixture': lambda: SimpleNamespace(tasks=['task'])},
        load_states=lambda suite, task: states)


class Simulator:
    def __init__(self):
        self.corrupt = False
        self.state = np.zeros(3)

    def set_init_state(self, state):
        self.state = np.array(state, dtype=float)
        if self.corrupt:
            self.state[-1] += 1

    def get_sim_state(self):
        return self.state.copy()


class VectorEnvironment:
    num_envs = 1

    def __init__(self):
        wrapper = SimpleNamespace(task_id=0, init_states=True, _init_states=STATES,
                                  _init_state_id=0, _env=Simulator(), num_steps_wait=10)
        wrapper.unwrapped = wrapper
        self.envs = [wrapper]
        self.actions = []
        self.settles = 0
        self.skip_apply = False

    def reset(self, seed):
        wrapper = self.envs[0]
        if not self.skip_apply:
            wrapper._env.set_init_state(wrapper._init_states[wrapper._init_state_id])
        # Deliberately change the state during settling, as the real wrapper can.
        wrapper._env.state[0] += 0.1
        self.settles += 1
        return {'robot_state': {'position': [float(wrapper._env.state[1])]}}, {}

    def step(self, action):
        self.actions.append(action)
        return {'robot_state': {'position': [9.]}}, [1.], [True], [False], {'is_success': [True]}


class Policy:
    def reset(self):
        self.seen = []

    def act(self, packet):
        self.seen.append(packet.observation)
        return [0.] * 7


class InitialStateTests(unittest.TestCase):
    def test_selection_rejects_invalid_ids_and_vectors(self):
        for value in (-1, 2, True, 0.5, '0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select(state=value)
        for value in (-1, 1, True, 0.5, '0'):
            with self.subTest(task=value), self.assertRaises(ValueError):
                select(task=value)
        with self.assertRaises(ValueError):
            select(suite='missing')
        for states in ([], [[]], [[float('nan')]], [[[1.]]]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                select(states=states)

    def test_cli_selects_state_explicitly(self):
        self.assertEqual(parse_args(['--initial-state-id', '1']).initial_state_id, 1)
        self.assertEqual(parse_args([]).initial_state_id, 0)

    def test_two_states_same_seed_have_distinct_readback_and_observations(self):
        identities, observations = [], []
        for state in (0, 1):
            adapter = LiberoEnvironmentAdapter(VectorEnvironment(), initial_state=select(state=state))
            observations.append(adapter.reset(17, 'episode').observation)
            evidence = adapter.initial_state_evidence
            self.assertEqual(evidence['status'], 'verified')
            self.assertEqual(evidence['applied_sha256'], state_digest(STATES[state]))
            self.assertNotEqual(evidence['applied_sha256'], evidence['settled_sha256'])
            self.assertEqual(evidence['settling_steps'], 10)
            identities.append(evidence['applied_sha256'])
        self.assertNotEqual(*identities)
        self.assertNotEqual(*observations)

    def test_bad_readback_prevents_settling_and_policy_control(self):
        env, policy = VectorEnvironment(), Policy()
        env.envs[0]._env.corrupt = True
        adapter = LiberoEnvironmentAdapter(env, initial_state=select())
        with self.assertRaisesRegex(ValueError, 'inconsistent initial-state readback'):
            run_episode(EpisodeConfig(17, 2), policy, adapter, ReplayRecorder())
        self.assertEqual(env.actions, [])
        self.assertEqual(env.settles, 0)
        self.assertEqual(policy.seen, [])
        self.assertEqual(adapter.initial_state_evidence['status'], 'unverified')
        with self.assertRaises(RuntimeError):
            adapter.step([0.] * 7)

    def test_environment_catalog_or_task_mismatch_fails_before_reset(self):
        for field, value in (('task_id', 1), ('init_states', False),
                             ('_init_states', [[0., 9., 2.]])):
            env = VectorEnvironment()
            setattr(env.envs[0], field, value)
            adapter = LiberoEnvironmentAdapter(env, initial_state=select())
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.reset(17, 'episode')
            self.assertEqual(env.settles, 0)

    def test_missing_application_cannot_be_verified(self):
        env = VectorEnvironment()
        env.skip_apply = True
        adapter = LiberoEnvironmentAdapter(env, initial_state=select())
        with self.assertRaisesRegex(ValueError, 'not applied'):
            adapter.reset(17, 'episode')
        self.assertEqual(adapter.initial_state_evidence['status'], 'unverified')

    def test_failed_second_reset_invalidates_prior_evidence_and_can_recover(self):
        env = VectorEnvironment()
        adapter = LiberoEnvironmentAdapter(env, initial_state=select(state=1))
        adapter.reset(17, 'first')
        env.envs[0]._env.corrupt = True
        with self.assertRaises(ValueError):
            adapter.reset(17, 'second')
        self.assertNotIn('settled_sha256', adapter.initial_state_evidence)
        self.assertEqual(adapter.initial_state_evidence['status'], 'unverified')
        env.envs[0]._env.corrupt = False
        adapter.reset(17, 'third')
        self.assertEqual(adapter.initial_state_evidence['status'], 'verified')

    def test_complete_selected_episode_seals_and_replays_without_privileged_state(self):
        for state in (0, 1):
            with self.subTest(state=state), TemporaryDirectory() as directory:
                env, policy = VectorEnvironment(), Policy()
                adapter = LiberoEnvironmentAdapter(env, initial_state=select(state=state))
                config = EpisodeConfig(17, 2)
                trace = TraceRecorder(Path(directory) / 'replay', config, ReplayRecorder())
                outcome = run_episode(config, policy, adapter, trace)
                trace.seal(outcome)
                replayed = load_recorded_replay(trace.directory).run()
                self.assertEqual((outcome.success, outcome.steps), (True, 1))
                self.assertEqual((replayed.success, replayed.steps), (True, 1))
                self.assertEqual(policy.seen, [{'robot_state': {'position': [STATES[state][1]]}}])
                self.assertNotIn('sha256', (trace.directory / 'observations.jsonl').read_text())
                self.assertEqual(json.loads(json.dumps(adapter.initial_state_evidence))['state_id'], state)


if __name__ == '__main__':
    unittest.main()
