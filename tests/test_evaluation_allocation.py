"""Public allocation planner scenarios; no live campaign or held-out data."""

import unittest

from evaluation_allocation import plan_trial_allocation


def pilot():
    return {task: {'split': 'development', 'pairs': 20, 'c_only': 5,
                   'a_only': 3, 'seconds_per_triplet': 12,
                   'calls_per_triplet': 2, 'cost_per_triplet': 0.5}
            for task in range(10)}


class AllocationTests(unittest.TestCase):
    def plan(self, **kwargs):
        return plan_trial_allocation(
            pilot(), max_states_per_task=200, max_repetitions_per_state=2,
            limits={'seconds': 100000, 'model_calls': 100000, 'cost': 100000},
            **kwargs)

    def test_feasible_equal_task_integer_plan_and_demand(self):
        report = self.plan()
        self.assertEqual(report['status'], 'feasible')
        allocation = report['allocation']
        self.assertEqual(set(allocation), {f'task_{task}' for task in range(10)})
        self.assertEqual(len({tuple(row.values()) for row in allocation.values()}), 1)
        estimate = report['estimate']
        self.assertEqual(estimate['attempts'], estimate['pairs_per_task'] * 30)
        self.assertEqual(estimate['demand']['seconds'],
                         estimate['pairs_per_task'] * 120)
        self.assertTrue(estimate['precision_ok'] and estimate['power_ok'])
        self.assertEqual(report['pilot'][0]['observed_gain_pp'], 10)

    def test_resource_and_statistical_infeasibility_are_visible(self):
        constrained = plan_trial_allocation(
            pilot(), max_states_per_task=200, max_repetitions_per_state=2,
            limits={'seconds': 1, 'model_calls': 1, 'cost': 1})
        self.assertEqual(constrained['status'], 'infeasible')
        self.assertFalse(constrained['estimate']['resource_ok'])
        sparse = plan_trial_allocation(
            pilot(), max_states_per_task=2, max_repetitions_per_state=1,
            limits={'seconds': 100000, 'model_calls': 100000, 'cost': 100000})
        self.assertEqual(sparse['status'], 'infeasible')
        self.assertIsNone(sparse['allocation'])

    def test_rejects_held_out_or_impossible_pilot(self):
        data = pilot()
        data[0]['split'] = 'held_out'
        with self.assertRaisesRegex(ValueError, 'development-only'):
            plan_trial_allocation(data, max_states_per_task=10,
                                  max_repetitions_per_state=1,
                                  limits={'seconds': 1, 'model_calls': 1, 'cost': 1})
        data[0]['split'] = 'development'
        data[0]['c_only'] = 30
        with self.assertRaisesRegex(ValueError, 'discordance'):
            plan_trial_allocation(data, max_states_per_task=10,
                                  max_repetitions_per_state=1,
                                  limits={'seconds': 1, 'model_calls': 1, 'cost': 1})


if __name__ == '__main__':
    unittest.main()
