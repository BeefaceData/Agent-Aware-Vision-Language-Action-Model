"""CPU-only diagnostics contracts for the local simulator adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from development.simulator.local_simulator import LocalSimulator
from harness.environment.local import LocalEnvironmentClient


class Task:
    eval_success = False

    def __init__(self, error=None):
        self.error = error
        self.actions = []

    def take_action(self, action):
        if self.error:
            raise self.error
        self.actions.append(action)

    def check_success(self):
        return False


def simulator(task, diagnostics):
    instance = LocalSimulator.__new__(LocalSimulator)
    instance.task = task
    instance.setup = {}
    instance.blank = np.zeros((2, 2, 3), dtype=np.uint8)
    instance.initialized = True
    instance.diagnostics = diagnostics
    instance._step_count = 0
    instance.perception_cameras = {}
    return instance


class LocalSimulatorDiagnosticsTests(unittest.TestCase):
    def test_local_environment_adds_measured_tcp_poses_not_object_state(self):
        from tests.helpers import images

        def arm_pose(x):
            return lambda: np.asarray((x, 0., 0., 0., 0., 0., 1.))

        robot = SimpleNamespace()
        for side, x in (("left", 0.), ("right", 1.)):
            setattr(robot, f"get_{side}_arm_real_jointState", lambda: [0.] * 6)
            setattr(robot, f"get_{side}_gripper_val", lambda: 1.)
            setattr(robot, f"get_{side}_ee_pose", arm_pose(x))
        environment = LocalEnvironmentClient(SimpleNamespace(task=SimpleNamespace(robot=robot)))
        raw = {**images(), "state": np.zeros(14, dtype=np.float32),
               "success": np.asarray(False)}

        observation = environment._accept(raw, 3)

        self.assertEqual(observation.measured_tcp_poses["left"][0], 0.)
        self.assertEqual(observation.measured_tcp_poses["right"][0], 1.)
        self.assertFalse(hasattr(observation, "object_poses"))

    def test_rgbd_flag_enables_only_local_task_depth(self):
        setup = {"data_type": {"rgb": True, "depth": False}}
        with patch("harness.environment.robotwin.build_task",
                          return_value=(SimpleNamespace(), setup)):
            enabled = LocalSimulator("RoboTwin", rgbd_enabled=True)
        self.assertTrue(enabled.setup["data_type"]["depth"])

        setup = {"data_type": {"rgb": True, "depth": False}}
        with patch("harness.environment.robotwin.build_task",
                          return_value=(SimpleNamespace(), setup)):
            baseline = LocalSimulator("RoboTwin", rgbd_enabled=False)
        self.assertFalse(baseline.setup["data_type"]["depth"])

    def test_local_environment_detects_only_direct_left_right_arm_contacts(self):
        def body(name):
            return SimpleNamespace(entity=SimpleNamespace(name=name))

        def contact(a, b):
            return SimpleNamespace(bodies=(body(a), body(b)))

        task = SimpleNamespace(scene=SimpleNamespace(get_contacts=lambda: [
            contact("fl_link7", "mug"), contact("fr_link8", "rack")]))
        environment = LocalEnvironmentClient(SimpleNamespace(task=task))
        self.assertFalse(environment.arm_overlap_detected())

        task.scene.get_contacts = lambda: [contact("left_link4", "fr_link5")]
        self.assertTrue(environment.arm_overlap_detected())

    def test_step_logs_action_and_observation_boundaries_with_durations(self):
        events = []
        instance = simulator(Task(), events.append)
        with patch("harness.environment.robotwin.observation_of", return_value={"rgb": True}):
            observation, timings = instance("local", 0, "/step", {"action": [0.0] * 14})

        self.assertEqual(observation, {"rgb": True})
        self.assertEqual(len(instance.task.actions), 1)
        self.assertGreaterEqual(timings["sim_http_ms"], 0)
        self.assertEqual([(e["event_type"], e.get("phase")) for e in events], [
            ("simulator_request_started", None),
            ("simulator_phase_started", "task_action"),
            ("simulator_phase_completed", "task_action"),
            ("simulator_phase_started", "observation"),
            ("simulator_phase_completed", "observation"),
            ("simulator_request_completed", None),
        ])
        self.assertEqual(events[1]["step_id"], 1)
        self.assertIn("elapsed_ms", events[2])
        self.assertIn("elapsed_ms", events[4])

    def test_action_exception_is_logged_without_sensitive_exception_text(self):
        events = []
        instance = simulator(Task(RuntimeError("private diagnostic detail")), events.append)
        with patch("harness.environment.robotwin.observation_of") as observe:
            with self.assertRaisesRegex(RuntimeError, "private diagnostic detail"):
                instance("local", 0, "/step", {"action": [0.0] * 14})
        observe.assert_not_called()
        failed = events[-1]
        self.assertEqual(failed["event_type"], "simulator_phase_failed")
        self.assertEqual(failed["phase"], "task_action")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertNotIn("private diagnostic detail", str(events))

    def test_diagnostic_sink_failure_does_not_prevent_action(self):
        def broken_sink(_event):
            raise OSError("disk detail")

        instance = simulator(Task(), broken_sink)
        with patch("harness.environment.robotwin.observation_of", return_value={"rgb": True}):
            instance("local", 0, "/step", {"action": [0.0] * 14})
        self.assertEqual(len(instance.task.actions), 1)


if __name__ == "__main__":
    unittest.main()
