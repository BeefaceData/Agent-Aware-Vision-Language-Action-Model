"""Public executor selection fixtures; synthetic bounds, no robot execution."""

from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from episode_harness import ActionProposal, ObservationPacket
from recovery_registry import RecoveryRegistry, RecoveryTool
from supervisor_recovery import RecoveryParameter


def fixture_tool():
    return RecoveryTool(
        'reopen_and_retreat', (RecoveryParameter('retreat_m', 0, 0.03),),
        ('single_arm_translation_metres', 'gripper_open'),
        ('main', 'robot_state'), 3,
        ('gripper_open_confirmed', 'retreat_target_reached'),
        ('clearance_unverified', 'stale_observation', 'controller_failure',
         'action_limit_reached'))


class ExecutorFixture:
    """The executor's public selection boundary, without an actuation path."""

    def __init__(self, registry, capabilities):
        self.registry = registry
        self.capabilities = capabilities

    def resolve(self, response, proposal):
        return self.registry.resolve(response, proposal,
                                     capabilities=self.capabilities)


class RecoveryRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tool = fixture_tool()
        self.registry = RecoveryRegistry((self.tool,))
        self.executor = ExecutorFixture(self.registry, self.tool.required_capabilities)
        self.response = json.loads((Path(__file__).parent / 'fixtures' /
                                    'recovery_response.json').read_text())
        self.proposal = ActionProposal('fixture-proposal', ObservationPacket(
            'fixture-episode', 2, datetime.now(timezone.utc), {'task': 'place item'}), [0.1])

    def test_executor_resolves_declared_contract_and_bounded_request(self):
        resolved = self.executor.resolve(self.response, self.proposal)
        self.assertEqual(resolved.tool, self.tool)
        self.assertEqual(resolved.request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(resolved.request.decision_id, 'fixture-decision')
        declaration = json.loads(json.dumps(asdict(resolved)))['tool']
        self.assertEqual(declaration['required_observations'], ['main', 'robot_state'])
        self.assertEqual(declaration['action_limit'], 3)
        self.assertIn('retreat_target_reached', declaration['completion_conditions'])
        self.assertIn('controller_failure', declaration['abort_conditions'])
        self.assertEqual(self.proposal.action, [0.1])

    def test_executor_rejects_incompatible_robot_and_accepts_capability_superset(self):
        for capabilities in ((), ('gripper_open',),
                             ('bimanual_translation_metres', 'gripper_open'),
                             ('single_arm_translation_normalized', 'gripper_open')):
            with self.subTest(capabilities=capabilities):
                executor = ExecutorFixture(self.registry, capabilities)
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    executor.resolve(self.response, self.proposal)
        executor = ExecutorFixture(self.registry,
                                   self.tool.required_capabilities + ('hold',))
        self.assertEqual(executor.resolve(self.response, self.proposal).tool, self.tool)

    def test_unknown_tools_and_all_implementation_overrides_are_rejected(self):
        for change in ({'tool_name': 'exec'}, {'tool_name': []},
                       {'implementation': 'module.function'}, {'code': 'print(1)'},
                       {'tool_definition': asdict(self.tool)}, {'action_limit': 100},
                       {'capabilities': list(self.tool.required_capabilities)},
                       {'parameters': {'retreat_m': 0.02, 'code': 'print(1)'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.executor.resolve(self.response | change, self.proposal)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            RecoveryRegistry(()).resolve(self.response, self.proposal, capabilities=())

    def test_each_tool_uses_its_own_contract_and_current_request_identity(self):
        other = replace(self.tool, name='short_retreat', action_limit=2,
                        parameter_bounds=(RecoveryParameter('retreat_m', 0, 0.01),))
        executor = ExecutorFixture(RecoveryRegistry((self.tool, other)),
                                   self.tool.required_capabilities)
        with self.assertRaisesRegex(ValueError, 'bounds'):
            executor.resolve(self.response | {'tool_name': 'short_retreat'}, self.proposal)
        selected = executor.resolve(self.response | {
            'tool_name': 'short_retreat', 'parameters': {'retreat_m': 0.01}}, self.proposal)
        self.assertEqual(selected.tool.action_limit, 2)
        for change in ({'proposal_id': 'old'}, {'episode_id': 'foreign'},
                       {'observation_sequence': 1}, {'parameters': {'retreat_m': True}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                executor.resolve(self.response | change, self.proposal)

    def test_declarations_and_resolutions_are_detached_and_immutable(self):
        capabilities = list(self.tool.required_capabilities)
        conditions = list(self.tool.abort_conditions)
        tool = replace(self.tool, required_capabilities=capabilities,
                       abort_conditions=conditions)
        declarations = [tool]
        registry = RecoveryRegistry(declarations)
        declarations.clear()
        capabilities.clear()
        conditions.clear()
        resolved = registry.resolve(self.response, self.proposal,
                                    capabilities=self.tool.required_capabilities)
        self.response['parameters']['retreat_m'] = 99
        self.assertEqual(resolved.request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(resolved.tool, self.tool)
        with self.assertRaises(FrozenInstanceError):
            resolved.tool.action_limit = 100
        with self.assertRaises(FrozenInstanceError):
            registry.tools = ()

    def test_incomplete_or_ambiguous_declarations_fail_before_requests(self):
        for change in ({'action_limit': True}, {'action_limit': 0}, {'action_limit': 1.5},
                       {'parameter_bounds': ()}, {'required_capabilities': ()},
                       {'required_observations': ()}, {'completion_conditions': ()},
                       {'abort_conditions': ()}, {'required_capabilities': 'gripper_open'},
                       {'completion_conditions': ('done', 'done')},
                       {'abort_conditions': ('arbitrary code()',)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.tool, **change)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            RecoveryRegistry((self.tool, self.tool))
        with self.assertRaises(ValueError):
            RecoveryRegistry((asdict(self.tool),))


if __name__ == '__main__':
    unittest.main()
