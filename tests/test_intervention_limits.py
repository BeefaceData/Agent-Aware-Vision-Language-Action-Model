"""Public episode/replay limits with synthetic persistent task failure."""

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionResolution, EpisodeConfig, RobotStateCapture,
                             ViewCapture, run_episode)
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
import test_reopen_retreat as recovery_fixture
from test_supervisor_vlm import raw


TOOL = 'reopen_and_retreat'


class InterventionLimitTests(unittest.TestCase):
    def setUp(self):
        self.recovery = recovery_fixture.ReopenRetreatTests()
        self.recovery.setUp()

    def episode(self, choices, *, maximum=10, limits=((TOOL, 2),)):
        """Locally completed recovery never solves the scripted task failure."""
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        turns, policy_turns, selectors = [], [], {}
        for choice in choices:
            sequence = len(turns)
            selectors[sequence] = choice
            policy_turns.append((raw(sequence), [0.1] * 7))
            actions = ([self.recovery.opening, self.recovery.retreat]
                       if choice not in ('override', 'pass') else
                       [[0.3 if choice == 'override' else 0.1] * 7])
            for action in actions:
                sequence = len(turns) + 1
                turns.append((action, ReplayStep(raw(sequence), 0, False, False, False,
                    {'main': ViewCapture(sequence, wall, 10.)},
                    RobotStateCapture(sequence, wall, 10.))))
        config = EpisodeConfig(17, len(turns), max_interventions=maximum,
                               recovery_attempt_limits=limits)
        environment = ReplayEnvironment(17, raw(0), turns, clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=0.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        policy, calls = ReplayPolicy(policy_turns), []

        def select(proposal):
            calls.append(proposal.observation.sequence)
            choice = selectors[proposal.observation.sequence]
            if choice == 'pass':
                return ActionResolution('pass')
            if choice == 'override':
                return ActionResolution('override', [0.3] * 7)
            # Deliberately rebuild the executor on every attempt: limits must
            # belong to the episode, not a replaceable selector/tool instance.
            selected = self.recovery.resolve(proposal=proposal, now=10.)
            self.assertEqual(selected.kind, 'recovery')
            if choice != TOOL:
                # Trusted synthetic second tool with the same command contract.
                selected = replace(selected, recovery=replace(selected.recovery,
                    request=selected.recovery.request | {'tool_name': choice}))
            return selected

        return config, policy, environment, select, calls

    def run_trace(self, directory, fixture, observer=None):
        config, policy, environment, select, calls = fixture
        trace = TraceRecorder(directory, config, ReplayRecorder())
        outcome = run_episode(config, policy, environment, trace, action_selector=select,
            recovery_observer=observer or self.recovery.observe, clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        actual = replay.run()
        self.assertEqual((actual.steps, actual.stop_reason, actual.success),
                         (outcome.steps, outcome.stop_reason, outcome.success))
        return outcome, replay.evidence()

    def test_persistent_failure_stops_at_exact_tool_or_episode_limit(self):
        for maximum, tool_limit, count, reason in (
            (10, 2, 2, 'recovery attempt'), (2, 10, 2, 'episode intervention'),
            (10, 0, 0, 'recovery attempt'), (0, 10, 0, 'episode intervention')):
            with self.subTest(maximum=maximum, tool_limit=tool_limit), TemporaryDirectory() as tmp:
                fixture = self.episode([TOOL] * 4, maximum=maximum, limits=((TOOL, tool_limit),))
                outcome, evidence = self.run_trace(Path(tmp) / 'trace', fixture)
                self.assertEqual((outcome.steps, outcome.stop_reason, outcome.success),
                                 (2 * count, 'proposal_rejected', False))
                self.assertEqual(len(fixture[2].actions), 2 * count)
                self.assertEqual(fixture[4], list(range(0, 2 * count + 1, 2)))
                records = [row['action_record'] for row in evidence['decisions']]
                self.assertIn(reason, records[-1]['rejection_reason'])
                self.assertIsNone(records[-1]['executed_action'])
                self.assertEqual(records[-1]['intervention_budget']['interventions'], count)
                for index, record in enumerate(records[:-1]):
                    self.assertEqual(record['intervention_budget']['recovery_attempts'][TOOL],
                                     index // 2 + 1)
                    self.assertEqual(record['recovery']['check']['status'],
                                     'continuing' if index % 2 == 0 else 'completed')

    def test_distinct_tools_keep_separate_counts_but_share_episode_limit(self):
        for maximum, choices, expected_steps, reason in (
            (3, [TOOL, 'other_tool', TOOL], 4, 'recovery attempt'),
            (2, [TOOL, 'other_tool', 'third_tool'], 4, 'episode intervention'),
            (3, ['unconfigured_tool'], 0, 'recovery attempt')):
            with self.subTest(choices=choices), TemporaryDirectory() as tmp:
                fixture = self.episode(choices, maximum=maximum,
                    limits=((TOOL, 1), ('other_tool', 1), ('third_tool', 1)))
                outcome, evidence = self.run_trace(Path(tmp) / 'trace', fixture)
                self.assertEqual(outcome.steps, expected_steps)
                rejected = evidence['decisions'][-1]['action_record']
                self.assertIn(reason, rejected['rejection_reason'])
                if expected_steps:
                    self.assertEqual(rejected['intervention_budget']['recovery_attempts'],
                                     {TOOL: 1, 'other_tool': 1})

    def test_adjustments_and_recovery_share_budget_passes_do_not_consume_it(self):
        for choices, steps in ((['override', TOOL, 'override'], 3),
                               ([TOOL, 'pass', 'override', TOOL], 4)):
            with self.subTest(choices=choices), TemporaryDirectory() as tmp:
                outcome, evidence = self.run_trace(Path(tmp) / 'trace',
                    self.episode(choices, maximum=2))
                self.assertEqual((outcome.steps, outcome.stop_reason), (steps, 'proposal_rejected'))
                rejected = evidence['decisions'][-1]['action_record']
                self.assertEqual(rejected['intervention_budget']['recovery_attempts'], {TOOL: 1})
                self.assertIn('episode intervention', rejected['rejection_reason'])

    def test_abort_keeps_charged_attempt_and_never_calls_selector_again(self):
        for abort_index in (0, 1):
            with self.subTest(abort_index=abort_index), TemporaryDirectory() as tmp:
                fixture = self.episode([TOOL] * 3, maximum=2)
                def observe(plan, index, packet):
                    return None if index == abort_index else self.recovery.observe(plan, index, packet)
                outcome, evidence = self.run_trace(Path(tmp) / 'trace', fixture, observe)
                self.assertEqual((outcome.steps, outcome.stop_reason),
                                 (abort_index + 1, 'recovery_aborted'))
                self.assertEqual(fixture[4], [0])
                record = evidence['decisions'][-1]['action_record']
                self.assertEqual(record['intervention_budget']['recovery_attempts'], {TOOL: 1})
                self.assertEqual(record['intervention_budget']['interventions'], 1)
                self.assertEqual(record['recovery']['check']['status'], 'aborted')

    def test_controller_failure_retains_attempt_without_execution_acknowledgement(self):
        config, policy, env, select, calls = self.episode([TOOL])
        def fail(action):
            raise RuntimeError('synthetic controller failure')
        env.step = fail
        recorder = ReplayRecorder()
        with self.assertRaisesRegex(RuntimeError, 'synthetic controller'):
            run_episode(config, policy, env, recorder, action_selector=select,
                recovery_observer=self.recovery.observe, clock=lambda: 10.)
        record = recorder.failures[-1][3].action_record
        self.assertEqual(record.intervention_budget['recovery_attempts'], {TOOL: 1})
        self.assertIsNone(record.executed_action)
        self.assertEqual(calls, [0])

    def test_pass_through_remains_available_when_correction_allowance_is_exhausted(self):
        for choices, maximum, steps in ((['pass', 'pass'], 0, 2),
                                         ([TOOL, 'pass', 'pass'], 1, 4)):
            with self.subTest(choices=choices), TemporaryDirectory() as tmp:
                outcome, evidence = self.run_trace(Path(tmp) / 'trace',
                    self.episode(choices, maximum=maximum))
                self.assertEqual((outcome.steps, outcome.stop_reason), (steps, 'step_limit'))
                self.assertIsNone(evidence['decisions'][-1]['action_record']['intervention_budget'])

    def test_new_episode_gets_new_budget_with_reused_adapters(self):
        fixture = self.episode([TOOL, TOOL], maximum=1)
        with TemporaryDirectory() as tmp:
            first, _ = self.run_trace(Path(tmp) / 'first', fixture)
            second, evidence = self.run_trace(Path(tmp) / 'second', fixture)
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual((first.steps, second.steps), (2, 2))
        self.assertEqual(evidence['decisions'][-1]['action_record']['intervention_budget']
                         ['recovery_attempts'], {TOOL: 1})

    def test_invalid_limits_rejected_and_config_is_detached(self):
        for value in (-1, True, 1.5, float('inf'), '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EpisodeConfig(17, 10, max_interventions=value)
        for limits in (None, 'tool', ((TOOL, -1),), ((TOOL, True),),
                       ((TOOL, 1.5),), (('', 1),), ((TOOL, 1), (TOOL, 2)), ((TOOL,),)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                EpisodeConfig(17, 10, recovery_attempt_limits=limits)
        limits = [[TOOL, 2]]
        config = EpisodeConfig(17, 10, recovery_attempt_limits=limits)
        limits[0][1] = 99
        self.assertEqual(config.recovery_attempt_limits, ((TOOL, 2),))
        self.assertEqual(config.max_interventions, 10)

    def test_replay_rejects_altered_counts_and_limits_even_with_valid_checksums(self):
        for damage in ('count', 'missing', 'missing_rejection', 'continuation', 'admission', 'limit', 'reason'):
            with self.subTest(damage=damage), TemporaryDirectory() as tmp:
                path = Path(tmp) / 'trace'
                self.run_trace(path, self.episode([TOOL] * 3))
                rows = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
                manifest = json.loads((path / 'manifest.json').read_text())
                if damage == 'count':
                    rows[-1]['action_record']['intervention_budget']['recovery_attempts'][TOOL] = 1
                elif damage == 'missing':
                    rows[0]['action_record']['intervention_budget'] = None
                elif damage == 'missing_rejection':
                    rows[-1]['action_record']['intervention_budget'] = None
                elif damage == 'continuation':
                    rows[1]['action_record']['intervention_budget']['interventions'] = 2
                elif damage == 'admission':
                    rows[-1]['action_record']['intervention_budget']['admitted'] = True
                elif damage == 'limit':
                    manifest['config']['recovery_attempt_limits'] = [[TOOL, 1]]
                else:
                    rows[-1]['action_record']['rejection_reason'] = 'unrelated reason'
                data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                (path / 'decisions.jsonl').write_bytes(data)
                manifest['files']['decisions.jsonl'] = sha256(data).hexdigest()
                (path / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(TraceError):
                    load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
