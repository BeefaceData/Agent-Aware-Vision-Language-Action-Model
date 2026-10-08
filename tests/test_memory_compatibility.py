"""Outcome-neutral compatibility over pinned single/two-arm synthetic episodes."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from action_capabilities import libero_native_capabilities
from attempt_identity import AttemptIdentityRecorder
from episode_harness import ActionResolution, EpisodeConfig, RobotStateCapture, run_episode
from intervention_memory import InterventionMemory
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_identity import FrozenSupervisorManifest
from test_intervention_memory import pinned
from test_native_capabilities import two_arm


TASK = {'suite': 'synthetic', 'task_id': 0, 'instruction': 'place object'}
PROGRESS = {'stage': 'grasp', 'object': 'declared-object'}
COMPATIBILITY = {'policy': {'revision': 'synthetic'},
                 'supervisor': {'model': 'synthetic'},
                 'settings': {'action_capabilities': asdict(
                     replace(libero_native_capabilities(), layout='flat'))}}


class MemoryCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InterventionMemory(self.root / 'memory')
        self.single = replace(libero_native_capabilities(), layout='flat')

    def retain(self, name, capabilities=None, *, task=TASK, progress=PROGRESS, success=False,
               category='unknown', truncated=False, policy_revision=None, supervisor_model=None,
               settings_extra=None):
        capabilities = capabilities or self.single
        source = self.root / name
        source.mkdir()
        config = EpisodeConfig(17, 2)
        supervisor = FrozenSupervisorManifest.freeze(source / 'supervisor.json',
                                                     {'model': supervisor_model or 'synthetic'})
        identity = AttemptIdentityRecorder(source, config, lambda: {
            'task': task, 'policy_assets': {'revision': policy_revision or 'synthetic'},
            'settings': {'action_capabilities': asdict(capabilities), **(settings_extra or {})},
            **supervisor.reference}, ReplayRecorder())
        action = [0.0] * len(capabilities.components)
        correction = [0.1] * len(action)
        policy = ReplayPolicy([('start', action), ('middle', action)])
        environment = ReplayEnvironment(17, 'start', [
            (correction, ReplayStep('middle', 0, False, False, False)),
            (action, ReplayStep('done', 0, success, not truncated, truncated))],
            initial_robot_state_capture=RobotStateCapture(
                0, datetime(2026, 1, 1, tzinfo=timezone.utc), 10.))
        policy.action_capabilities = environment.action_capabilities = capabilities
        context_path = source / 'context.json'

        def select(proposal):
            if proposal.observation.sequence:
                return ActionResolution('pass')
            component = capabilities.components[0]
            context = {'episode_id': proposal.observation.episode_id,
                'proposal_id': proposal.proposal_id, 'observation_sequence': 0,
                'diagnosis': {'category': category, 'summary': 'synthetic',
                    'evidence': [] if category == 'unknown' else [
                        {'observation_sequence': 0, 'source': 'robot_state',
                         'description': 'synthetic declared state assessment'}]},
                'request': {'episode_id': proposal.observation.episode_id,
                    'proposal_id': proposal.proposal_id, 'observation_sequence': 0,
                    'decision_id': 'synthetic', 'kind': 'adjustment', 'scope': 'single_action',
                    'target': component.arm, 'frame': component.frame,
                    'units': {component.name: component.unit},
                    'residual': {component.name: 0.1}}}
            if progress is not None:
                context['progress_context'] = progress
            context_path.write_text(json.dumps(context))
            # A scripted host override, not a production robot converter.
            return ActionResolution('override', correction)

        trace = TraceRecorder(source / 'trace', config, identity)
        outcome = run_episode(config, policy, environment, trace, action_selector=select)
        trace.seal(outcome)
        replayed = load_recorded_replay(trace.directory).run()
        self.assertEqual((replayed.steps, replayed.success, replayed.stop_reason),
                         (2, outcome.success, outcome.stop_reason))
        return self.store.append(trace=pinned(trace.directory / 'manifest.json'),
            attempt=pinned(identity.path), supervisor=pinned(supervisor.path),
            context=pinned(context_path))

    def query(self, references, **changes):
        query = dict(task=TASK, robot_capabilities=asdict(self.single),
                     progress_context=PROGRESS, compatibility=COMPATIBILITY)
        return self.store.filter_candidates(references, **(query | changes))

    def test_mixed_single_and_two_arm_records_have_stable_inspectable_exclusions(self):
        failed = self.retain('failed')
        paired = self.retain('paired', two_arm())
        success = self.retain('success', success=True)
        wrong_task = self.retain('task', task=TASK | {'instruction': 'fold towel'})
        wrong_progress = self.retain('progress', progress=PROGRESS | {'stage': 'release'})
        missing = self.retain('missing', progress=None)
        refs = [failed, paired, wrong_task, success, wrong_progress, missing]
        self.store = InterventionMemory(self.root / 'memory')
        result = self.query(refs)
        self.assertEqual([r['record_id'] for r in result['candidates']],
                         [failed['record_id'], success['record_id']])
        self.assertEqual([r['task_outcome']['status'] for r in result['candidates']],
                         ['failure', 'success'])
        reasons = {r['record_id']: r['reasons'] for r in result['excluded']}
        self.assertIn('component_count_mismatch', reasons[paired['record_id']])
        self.assertEqual(reasons[wrong_task['record_id']], ['task_mismatch'])
        self.assertEqual(reasons[wrong_progress['record_id']], ['progress_context_mismatch'])
        self.assertEqual(reasons[missing['record_id']], ['progress_context_missing'])
        self.assertEqual(result, self.query(refs))
        paired_result = self.query(refs, robot_capabilities=asdict(two_arm()),
            compatibility=COMPATIBILITY | {'settings': {
                'action_capabilities': asdict(two_arm())}})
        self.assertEqual([r['record_id'] for r in paired_result['candidates']], [paired['record_id']])
        result['candidates'][0]['progress_context']['stage'] = 'invented'
        self.assertEqual(self.query(refs)['candidates'][0]['progress_context'], PROGRESS)

    def test_semantic_mismatches_are_excluded_without_implicit_conversion(self):
        for field, value in (('name', 'other'), ('arm', 'other'), ('group', 'paired'),
                             ('frame', 'camera'), ('representation', 'absolute_position'),
                             ('unit', 'mm'), ('scale', 0.1), ('minimum', -2.), ('maximum', 2.)):
            capabilities = replace(self.single, components=(
                replace(self.single.components[0], **{field: value}),) + self.single.components[1:])
            ref = self.retain(field, capabilities)
            with self.subTest(field=field):
                result = self.query([ref])
                self.assertEqual(result, {'candidates': [], 'excluded': [
                    dict(ref, reasons=['settings_mismatch', field + '_mismatch'])]})
        for field, value in (('control_frequency_hz', 10.), ('layout', 'single-vector-batch'),
                             ('operations', ('policy_action', 'stop'))):
            ref = self.retain(field, replace(self.single, **{field: value}))
            self.assertEqual(self.query([ref])['excluded'],
                             [dict(ref, reasons=['settings_mismatch', field + '_mismatch'])])

    def test_two_arm_order_and_coordination_are_not_interchangeable(self):
        base = two_arm()
        reordered = self.retain('reordered', replace(base, components=base.components[::-1]))
        unpaired = self.retain('unpaired', replace(base, components=tuple(
            replace(c, group=c.arm) for c in base.components)))
        result = self.query([reordered, unpaired], robot_capabilities=asdict(base))
        self.assertEqual(result['candidates'], [])
        self.assertIn('arm_mismatch', result['excluded'][0]['reasons'])
        self.assertEqual(result['excluded'][1]['reasons'], ['settings_mismatch', 'group_mismatch'])
        # Supported operation order carries no control semantics.
        accepted = self.retain('paired', base)
        active = asdict(replace(base, operations=base.operations[::-1]))
        self.assertEqual(len(self.query([accepted], robot_capabilities=active,
            compatibility=COMPATIBILITY | {'settings': {'action_capabilities': asdict(base)}}
            )['candidates']), 1)

    def test_malformed_query_and_corrupt_incompatible_record_fail_closed(self):
        for changes in ({'task': {}}, {'progress_context': None}, {'progress_context': {}},
                        {'progress_context': {'stage': 1}}, {'robot_capabilities': {}},
                        {'robot_capabilities': asdict(self.single) | {'control_frequency_hz': True}}):
            with self.subTest(changes=changes), self.assertRaises(TraceError):
                self.query([], **changes)
        ref = self.retain('incompatible', two_arm())
        for invalid in (None, {}, dict(ref, sha256='0' * 64)):
            with self.assertRaises(TraceError):
                self.query([invalid])
        (self.root / 'incompatible/context.json').unlink()
        with self.assertRaises(TraceError):
            self.query([ref])

    def test_model_and_settings_versions_require_explicit_exact_identity(self):
        same = self.retain('same')
        policy = self.retain('policy-v2', policy_revision='v2')
        supervisor = self.retain('supervisor-v2', supervisor_model='v2')
        settings = self.retain('settings-v2', settings_extra={'revision': 'v2'})
        refs = [same, policy, supervisor, settings]
        result = self.query(refs)
        self.assertEqual([row['record_id'] for row in result['candidates']], [same['record_id']])
        self.assertEqual({row['record_id']: row['reasons'] for row in result['excluded']}, {
            policy['record_id']: ['policy_mismatch'],
            supervisor['record_id']: ['supervisor_mismatch'],
            settings['record_id']: ['settings_mismatch']})
        for ref, declared in (
                (policy, COMPATIBILITY | {'policy': {'revision': 'v2'}}),
                (supervisor, COMPATIBILITY | {'supervisor': {'model': 'v2'}}),
                (settings, COMPATIBILITY | {'settings': COMPATIBILITY['settings'] |
                                            {'revision': 'v2'}})):
            with self.subTest(ref=ref['record_id']):
                selected = self.query([ref], compatibility=declared)
                self.assertEqual([row['record_id'] for row in selected['candidates']],
                                 [ref['record_id']])
        for malformed in (None, {}, COMPATIBILITY | {'policy': {}},
                          COMPATIBILITY | {'extra': 1}):
            with self.subTest(malformed=malformed), self.assertRaises(TraceError):
                self.query([], compatibility=malformed)

    def test_progress_cannot_be_relabelled_and_legacy_bytes_are_preserved(self):
        ref = self.retain('retained')
        record = self.store.read(ref['record_id'], expected_sha256=ref['sha256'])
        path = self.store.directory / (ref['record_id'] + '.json')
        record['progress_context'] = {'stage': 'invented'}
        path.write_text(json.dumps(record))
        with self.assertRaises(TraceError):
            self.query([dict(ref, sha256=pinned(path)[1])])
        legacy = self.retain('legacy', progress=None)
        path = self.store.directory / (legacy['record_id'] + '.json')
        record = json.loads(path.read_bytes())
        record['version'] = 3
        del record['progress_context']
        path.write_text(json.dumps(record))
        original = path.read_bytes()
        legacy['sha256'] = pinned(path)[1]
        self.assertEqual(self.query([legacy])['excluded'],
                         [dict(legacy, reasons=['progress_context_missing'])])
        self.assertEqual(path.read_bytes(), original)

    def test_malformed_retained_progress_is_rejected_at_admission(self):
        for index, progress in enumerate(({}, {'stage': ''}, {'stage': 1},
                                          {'stage': {'outcome': 'success'}}, ['grasp'])):
            with self.subTest(progress=progress), self.assertRaises(TraceError):
                self.retain(f'invalid-{index}', progress=progress)
        self.assertFalse(self.store.directory.exists())


if __name__ == '__main__':
    unittest.main()
