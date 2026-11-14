import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/robotwin-grasp-pose-selection.patch"
ROBOTWIN = ROOT / "RoboTwin"


def patched_choose_best_pose():
    source = subprocess.run(
        ["git", "-C", str(ROBOTWIN), "show", "HEAD:envs/_base_task.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    with tempfile.TemporaryDirectory() as directory:
        build_root = Path(directory)
        (build_root / "envs").mkdir()
        (build_root / "envs/_base_task.py").write_text(source)
        subprocess.run(["git", "init", "-q"], cwd=build_root, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=build_root, check=True)
        patched_source = (build_root / "envs/_base_task.py").read_text()

    module = ast.parse(patched_source)
    task_class = next(node for node in module.body
                      if isinstance(node, ast.ClassDef) and node.name == "Base_Task")
    method = next(node for node in task_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == "choose_best_pose")
    namespace = {"ArmTag": str}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace["choose_best_pose"]


class RobotwinPoseSelectionPatchTests(unittest.TestCase):
    def test_selects_shortest_successful_grasp_orientation(self):
        choose_best_pose = patched_choose_best_pose()
        candidates = ["long", "short", "medium", "failed"]
        trajectories = {
            "status": ["Success", "Success", "Success", "Failed"],
            "position": [list(range(20)), list(range(5)), list(range(10)), []],
        }
        robot = SimpleNamespace(
            left_plan_multi_path=lambda poses: trajectories,
            create_target_pose_list=lambda *_args: candidates,
        )
        task = SimpleNamespace(plan_success=True, robot=robot)

        selected = choose_best_pose(task, "mug", "contact", "left")

        self.assertEqual(selected, "short")


if __name__ == "__main__":
    unittest.main()
