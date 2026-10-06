"""Selected and read-back LIBERO state identity, kept outside policy observations.

The integration boundary targets LeRobot 0.4.3's synchronous LiberoEnv. It
intercepts state application before the wrapper's normal settling steps.
"""

from dataclasses import dataclass
from hashlib import sha256
from operator import index

import numpy as np


def _vector(value):
    vector = np.asarray(value, dtype='<f8')
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError('initial state must be a finite nonempty vector')
    return vector


def state_digest(value):
    """SHA-256 of a one-dimensional little-endian float64 state vector."""
    return sha256(_vector(value).tobytes()).hexdigest()


def _selection_index(value, size, name):
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if isinstance(value, (bool, np.bool_)) or not 0 <= result < size:
        raise ValueError(f'{name} out of range [0, {size - 1}]')
    return result


@dataclass(frozen=True)
class InitialStateSelection:
    suite: str
    task_id: int
    state_id: int
    state: tuple[float, ...]

    def identity(self):
        return {'suite': self.suite, 'task_id': self.task_id,
                'state_id': self.state_id, 'selected_sha256': state_digest(self.state),
                'digest_encoding': 'little-endian-float64-vector-v1'}


def select_initial_state(suite_name, task_id, state_id, *, suites=None, load_states=None):
    """Validate the catalog selection before constructing a simulator or policy."""
    if suites is None:
        from libero.libero import benchmark
        suites = benchmark.get_benchmark_dict()
    if load_states is None:
        from lerobot.envs.libero import get_task_init_states
        load_states = get_task_init_states
    if suite_name not in suites:
        raise ValueError('unknown LIBERO suite')
    suite = suites[suite_name]()
    task_id = _selection_index(task_id, len(suite.tasks), 'task ID')
    states = load_states(suite, task_id)
    state_id = _selection_index(state_id, len(states), 'initial-state ID')
    return InitialStateSelection(suite_name, task_id, state_id,
                                 tuple(_vector(states[state_id])))


def reset_selected_state(environment, seed, selection, evidence):
    """Reset a single SyncVectorEnv and verify application before any settling.

    The temporary proxy forwards all other operations unchanged. No raw state
    enters observations, and evidence is invalidated on every reset attempt.
    """
    evidence.clear()
    evidence.update(selection.identity(), status='unverified')
    if environment.num_envs != 1 or len(environment.envs) != 1:
        raise ValueError('initial-state verification requires one synchronous environment')
    wrapper = environment.envs[0].unwrapped
    if wrapper.task_id != selection.task_id or not wrapper.init_states:
        raise ValueError('environment task or initial-state configuration differs')
    state_id = _selection_index(selection.state_id, len(wrapper._init_states), 'initial-state ID')
    if state_digest(wrapper._init_states[state_id]) != evidence['selected_sha256']:
        raise ValueError('environment initial-state catalog differs from selection')
    wrapper._init_state_id = state_id
    simulator = wrapper._env

    class CheckedApplication:
        applied = False

        def __getattr__(self, name):
            return getattr(simulator, name)

        def set_init_state(self, state):
            if self.applied or state_digest(state) != evidence['selected_sha256']:
                raise ValueError('inconsistent initial-state application')
            observation = simulator.set_init_state(state)
            readback = simulator.get_sim_state()
            evidence['applied_sha256'] = state_digest(readback)
            if evidence['applied_sha256'] != evidence['selected_sha256']:
                raise ValueError('inconsistent initial-state readback')
            self.applied = True
            return observation

    proxy = CheckedApplication()
    wrapper._env = proxy
    try:
        observation, info = environment.reset(seed=[seed])
        if not proxy.applied:
            raise ValueError('initial state was not applied during reset')
        evidence.update(settled_sha256=state_digest(simulator.get_sim_state()),
                        settling_steps=wrapper.num_steps_wait, status='verified')
        return observation, info
    finally:
        wrapper._env = simulator
