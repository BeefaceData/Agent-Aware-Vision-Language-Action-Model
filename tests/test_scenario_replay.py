"""Scenario fixtures stay portable and contain no privileged simulator data."""

import unittest

from tests.scenario_replay import _PATH, scenario


class ScenarioReplayFixtureTests(unittest.TestCase):
    def test_fixture_is_versioned_and_covers_recovery_and_handoff_edges(self):
        import json
        data = json.loads(_PATH.read_text())
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue({
            "empty_grasp_then_successful_retry", "left_transfer_then_hang",
            "left_transfer_unconfirmed", "ambiguous_action_response", "failed_acquisition",
        } <= set(data["scenarios"]))
        self.assertEqual(scenario("left_transfer_then_hang")["expected_locks"],
                         [None, "right", "left"])

    def test_fixture_has_no_oracle_geometry_or_contact_fields(self):
        import json
        data = json.loads(_PATH.read_text())
        forbidden = {"privileged", "oracle", "mug_position", "left_tcp", "right_tcp",
                     "left_contact", "right_contact", "contacts", "depth", "world_pose"}
        def keys(value):
            if isinstance(value, dict):
                yield from value.keys()
                for child in value.values(): yield from keys(child)
            elif isinstance(value, list):
                for child in value: yield from keys(child)
        self.assertFalse(forbidden & set(keys(data)))

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario("not_a_scenario")
