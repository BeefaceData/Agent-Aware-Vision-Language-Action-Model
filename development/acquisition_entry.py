"""Strict natural-reset acquisition replay and explicit VLA history rebuilding.

Reuses checkpoint frame/tolerance contracts. This is a development entry adapter,
not a full physics snapshot or a general replay of analytic skill journals.
"""

import json
from pathlib import Path

import numpy as np

from harness.evaluation.manifest import runtime_identity
from harness.trace import CAMERAS, copy_images, vector14
from .checkpoint_bank import sha256
from .schemas import RecordedFrame, ReplayOptions


class AcquisitionEntry:
    def __init__(self, episode, identity, *, root):
        self.episode = Path(episode).resolve()
        self.paths = {}
        def read(name):
            path = (self.episode/name).resolve()
            if not path.is_relative_to(self.episode):
                raise ValueError("entry_artifact_outside_episode")
            self.paths[name] = path
            return path
        manifest = json.loads(read('manifest.json').read_text())
        summary = json.loads(read('summary.json').read_text())
        if (summary.get('outcome') != 'acquisition_confirmed' or summary.get('milestone_reached') is not True
                or summary.get('analytic_ticks') != 0 or summary.get('error_type')
                or any(k.endswith('cleanup_error_type') for k in summary)
                or manifest.get('completion_predicate') != 'mug_lifted'):
            raise ValueError('requires_successful_natural_acquisition')
        self.seed = manifest['seed']
        if type(self.seed) is not int or self.seed < 0 or summary.get('seed') != self.seed:
            raise ValueError('entry_seed_mismatch')
        events = [json.loads(line) for line in read('skill_events.jsonl').read_text().splitlines()]
        requests = [e for e in events if e.get('event_type') == 'skill_requested']
        results = [e['result'] for e in events if e.get('event_type') == 'skill_result']
        if (len(requests) != 1 or requests[0]['command']['action'] != 'vla_act'
                or requests[0]['command'].get('stop') != 'mug_lifted' or requests[0]['observation'] != 0
                or len(results) != 1 or not results[0]['completed'] or results[0]['execution_uncertain']
                or results[0]['stop_reason'] != 'predicate_satisfied'):
            raise ValueError('requires_one_confirmed_acquisition_call')
        self.frames = [RecordedFrame(step=0, action=None, command_state=vector14(requests[0]['command_state']))]
        self.chunks = []
        pending = None
        for event in events:
            kind = event.get('event_type', event.get('event'))
            if kind in ('tick_requested', 'analytic_start'):
                raise ValueError('analytic_prefix_not_supported')
            if kind == 'chunk_requested':
                if pending is not None or event['observation'] != len(self.frames)-1:
                    raise ValueError('invalid_chunk_boundary')
                size = event['requested_actions']
                if type(size) is not int or not 1 <= size <= 24:
                    raise ValueError('invalid_chunk_size')
                self.chunks.append((event['observation'], size))
            elif kind == 'action_requested':
                if pending is not None or event['observation'] != len(self.frames)-1:
                    raise ValueError('unconfirmed_or_noncontiguous_action')
                pending = vector14(event['action'])
            elif kind == 'action_confirmed':
                if pending is None or vector14(event['action']) != pending or event['observation'] != len(self.frames):
                    raise ValueError('mismatched_action_confirmation')
                self.frames.append(RecordedFrame(step=len(self.frames), action=pending,
                    command_state=vector14(event['command_state'])))
                pending = None
        count = len(self.frames)-1
        if (pending is not None or not 1 <= count <= 768 or count != summary['policy_actions']
                or count != results[0]['executed_policy_actions'] or count != results[0]['observation_after']
                or len(self.chunks) != results[0]['vla_chunks_requested'] or not 1 <= len(self.chunks) <= 32
                or self.chunks[0][0] != 0):
            raise ValueError('incomplete_entry_prefix')
        for i, (start, size) in enumerate(self.chunks):
            end = self.chunks[i+1][0] if i+1 < len(self.chunks) else count
            if not 0 < end-start <= size:
                raise ValueError('invalid_chunk_span')
        self.images = {}
        for step in (0, count):
            with np.load(read(f'observations/observation_{step:08d}.npz'), allow_pickle=False) as archive:
                if int(archive['sequence']) != step or vector14(archive['command_state']) != self.frames[step].command_state:
                    raise ValueError('entry_snapshot_state_mismatch')
                self.images[step] = copy_images({k:archive[k] for k in CAMERAS})
        oracle = [json.loads(x) for x in read('oracle.jsonl').read_text().splitlines()]
        self.evidence = {e['privileged']['observation_sequence']:e['privileged'] for e in oracle
                        if e.get('event_type') == 'predicate_oracle_evidence'}
        if not {0,count} <= self.evidence.keys():
            raise ValueError('missing_entry_evidence')
        identity = Path(identity).resolve()
        self.paths['source_runtime_identity'] = identity
        captured = json.loads(identity.read_text()); current = runtime_identity(root)
        # Controller changes are allowed; simulator, transport, observations and
        # evidence definitions must match the recorded run. Assets/GPU unverified.
        exact = {'run_inference_chunked.py', 'development/local_simulator.py',
                 'development/skill_oracle.py', 'development/rack_geometry.py',
                 'harness/environment/local.py', 'harness/environment/client.py',
                 'harness/trace.py', 'harness/predicates/hanging_mug.py'}
        critical = exact | {p for p in (*current['files'], *captured['files']) if p.startswith('RoboTwin/')}
        if not exact <= current['files'].keys() or any(captured['files'].get(p) != current['files'].get(p) for p in critical):
            raise ValueError('entry_simulator_provenance_changed')
        if captured['packages'] != current['packages']:
            raise ValueError('entry_package_versions_changed')
        self.hashes = {name:sha256(path) for name,path in self.paths.items()}

    def verify_sources(self):
        if any(sha256(self.paths[k]) != value for k,value in self.hashes.items()):
            raise ValueError('entry_source_changed')


def replay_entry(entry, environment, predicates, *, log, options=None):
    """No policy calls. Return accepted state plus chunk-boundary RGB/state history."""
    options = options or ReplayOptions()
    entry.verify_sources()
    observation = environment.reset(entry.seed)
    predicates.prime(observation, allow_oracle=True)
    boundaries = dict(entry.chunks)
    history = []
    report = dict(accepted=False, replayed_actions=0, full_state_restored=False,
                  command_state_max_abs_error=0., image_errors={}, history_observations=0)

    def check(observation, frame):
        error = float(np.max(np.abs(np.asarray(observation.command_state)-frame.command_state)))
        report['command_state_max_abs_error'] = max(error, report['command_state_max_abs_error'])
        if error > options.state_atol:
            raise ValueError('entry_command_state_drift')
        if frame.step in entry.images:
            errors = {}
            for camera, expected in entry.images[frame.step].items():
                actual = observation.images[camera]
                if actual.shape != expected.shape: raise ValueError('entry_camera_shape_drift')
                errors[camera] = float(np.mean(np.abs(actual.astype(float)-expected.astype(float))))
            report['image_errors'][str(frame.step)] = errors
            if any(e > options.image_mae_tolerance for e in errors.values()):
                raise ValueError('entry_rgb_drift')

    try:
        check(observation, entry.frames[0])
        for frame in entry.frames[1:]:
            if observation.sequence in boundaries:
                history.append((observation.copy(), boundaries[observation.sequence]))
            log(dict(event_type='replay_action_requested', step=frame.step, action=frame.action))
            observation = environment.step(frame.action)
            report['replayed_actions'] += 1
            check(observation, frame)
            predicates.evaluate('mug_lifted', observation, allow_oracle=True)
            log(dict(event_type='replay_action_confirmed', step=frame.step, command_state=observation.command_state))
        report['history_observations'] = len(history)
        lift = predicates.evaluate('mug_lifted', observation, allow_oracle=True)
        report['lift'] = lift.model_dump(mode='json')
        # Compare privileged positions in addition to command-state/RGB checks.
        # Predicate definitions are fixed and evidence remains development-only.
        actual = predicates.hanging_mug.previous
        expected = entry.evidence[entry.frames[-1].step]
        errors = {key:float(np.linalg.norm(np.asarray(getattr(actual,key))-expected[key]))
                  for key in ('mug_position','left_tcp','right_tcp')}
        report['position_errors_m'] = errors
        if lift.value is not True or max(errors.values()) > .01:
            raise ValueError('entry_lift_or_geometry_not_verified')
        entry.verify_sources()
        report['accepted'] = True
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['reason'] = str(exc) if type(exc) is ValueError and str(exc).startswith('entry_') else 'replay_failed'
    return report, history


class HistoryPrimedVLA:
    """Rebuild observation queues after verification; execute none of these predictions."""
    def __init__(self, backend, history, *, log):
        self.backend, self.history, self.log = backend, tuple(history), log
        self.warm_calls = self.discarded_predictions = 0
        self.ready = False
        self.started = False

    def reset_history(self):
        if self.started:
            raise RuntimeError('history_bootstrap_is_single_use')
        self.started = True
        self.backend.reset_history()
        for observation, size in self.history:
            self.warm_calls += 1
            self.log(dict(event_type='history_prediction_requested', observation=observation.sequence, actions=size))
            actions = np.asarray(self.backend.infer(observation, size))
            if actions.shape != (size,14) or not np.isfinite(actions).all():
                raise ValueError('invalid_history_prediction')
            self.discarded_predictions += len(actions)
            self.log(dict(event_type='history_prediction_discarded', observation=observation.sequence, actions=len(actions)))
        self.ready = True

    def infer(self, observation, requested_actions):
        if not self.ready: raise RuntimeError('history_not_ready')
        return self.backend.infer(observation, requested_actions)
