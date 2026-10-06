"""Fresh correction identities across cooldown, executor replacement and replay."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionResolution
from single_action_adjustment import SingleActionAdjustment
from test_recovery_eligibility import fresh_proposal, request_for, scene_for
import test_intervention_limits as fixtures
import test_single_action_adjustment as adjustments


class SuccessiveInterventionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InterventionLimitTests()
        self.fixture.setUp()
        self.adjustment = adjustments.SingleActionAdjustmentTests()
        self.adjustment.setUp()

    def episode(self, choices, cooldown):
        config, policy, environment, select, calls = self.fixture.episode(
            choices, limits=((fixtures.TOOL, 10),))
        config = replace(config, recovery_cooldown_actions=cooldown)

        def bound_select(proposal):
            resolution = select(proposal)
            if resolution.kind == 'override':
                return replace(resolution, source_identity=(proposal.observation.episode_id,
                    proposal.observation.sequence, proposal.proposal_id))
            return resolution

        return config, policy, environment, bound_select, calls

    def test_cached_corrections_reject_after_cooldown_has_elapsed(self):
        for kind in ('override', fixtures.TOOL):
            for cooldown in (0, 2):
                with self.subTest(kind=kind, cooldown=cooldown), TemporaryDirectory() as tmp:
                    config, policy, environment, select, calls = self.episode(
                        [kind] + ['pass'] * cooldown + [kind], cooldown)
                    cached = []

                    def stale(proposal):
                        if not cached:
                            cached.append(select(proposal))
                            return cached[0]
                        if proposal.observation.sequence < config.max_steps - (2 if kind == fixtures.TOOL else 1):
                            return ActionResolution('pass')
                        return cached[0]

                    outcome, evidence = self.fixture.run_trace(Path(tmp) / 'episode',
                        (config, policy, environment, stale, calls))
                    expected = (2 if kind == fixtures.TOOL else 1) + cooldown
                    self.assertEqual((outcome.steps, outcome.stop_reason),
                                     (expected, 'proposal_rejected'))
                    self.assertEqual(len(environment.actions), expected)
                    rejected = evidence['decisions'][-1]['action_record']
                    self.assertEqual(rejected['rejection_reason'],
                                     'resolution must reference the current proposal')
                    self.assertIsNone(rejected['executed_action'])
                    self.assertIsNone(rejected['intervention_budget'])
                    self.assertTrue(rejected['interruption']['confirmed'])

    def test_fresh_corrections_of_either_mode_execute_after_cooldown(self):
        for first in ('override', fixtures.TOOL):
            for second in ('override', fixtures.TOOL):
                for cooldown in (0, 2):
                    with self.subTest(first=first, second=second, cooldown=cooldown), TemporaryDirectory() as tmp:
                        fixture = self.episode([first] + ['pass'] * cooldown + [second], cooldown)
                        outcome, evidence = self.fixture.run_trace(Path(tmp) / 'episode', fixture)
                        self.assertEqual((outcome.steps, outcome.stop_reason),
                                         (fixture[0].max_steps, 'step_limit'))
                        starts = [row for row in evidence['decisions']
                                  if row['action_record']['disposition'] == 'overridden' and
                                  (row['action_record']['recovery'] is None or
                                   row['action_record']['recovery']['action_index'] == 0)]
                        self.assertEqual(len(starts), 2)
                        self.assertGreater(starts[1]['source_sequence'], starts[0]['source_sequence'])
                        self.assertNotEqual(starts[0]['action_record']['proposal_id'],
                                            starts[1]['action_record']['proposal_id'])

    def test_old_adjustment_request_rejects_with_replacement_executor(self):
        fixture = self.adjustment
        old = fixture.proposal
        request = fixture.request(old)
        self.assertEqual(fixture.executor.resolve(old, request).kind, 'override')
        fresh = replace(old, proposal_id='next', observation=replace(old.observation, sequence=1))
        for changes in ({}, {'proposal_id': fresh.proposal_id}, {'observation_sequence': 1}):
            executor = SingleActionAdjustment(fixture.converter)
            self.assertEqual(executor.resolve(fresh, request | changes).kind, 'reject')
        self.assertEqual(SingleActionAdjustment(fixture.converter).resolve(
            fresh, fixture.request(fresh)).kind, 'override')

    def test_old_recovery_request_or_scene_rejects_with_replacement_executor(self):
        fixture = self.fixture.recovery
        old = fixture.current
        request, scene = request_for(old), scene_for(old)
        self.assertEqual(fixture.resolve(proposal=old, request=request, scene=scene).kind, 'recovery')
        fresh = replace(fresh_proposal(3), action=old.action)
        for changes in ({}, {'proposal_id': fresh.proposal_id}, {'observation_sequence': 3}):
            self.assertEqual(fixture.resolve(proposal=fresh, request=request | changes,
                                            scene=scene_for(fresh), now=3.1).kind, 'reject')
        self.assertEqual(fixture.resolve(proposal=fresh, scene=scene, now=3.1).kind, 'reject')
        self.assertEqual(fixture.resolve(proposal=fresh, now=3.1).kind, 'recovery')

    def test_renumbering_a_packet_does_not_refresh_recovery_sensor_evidence(self):
        fixture = self.fixture.recovery
        old = fixture.current
        relabeled = replace(old, proposal_id='next',
                            observation=replace(old.observation, sequence=3))
        result = fixture.resolve(proposal=relabeled)
        self.assertEqual(result.kind, 'reject')
        self.assertIsNone(result.recovery)


if __name__ == '__main__':
    unittest.main()
