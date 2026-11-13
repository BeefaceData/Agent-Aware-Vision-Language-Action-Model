"""CPU tests using synthetic meshes and source files; no Modal calls."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from modal_apps.inspect_planner_and_assets import inspect_local, inspect_sources, scaled_envelope, inspect_mugs
from development.hanging.public_provider import MUG_CLASS_SCALES


class Graph(dict):
    @property
    def nodes_geometry(self):
        return list(self)


def mesh_scene(transform=None):
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    return SimpleNamespace(graph=Graph(node=(np.eye(4) if transform is None else transform, "cube")),
                           geometry={"cube": SimpleNamespace(vertices=vertices)})


class InspectionTests(unittest.TestCase):
    def test_remote_wiring_is_cpu_read_only_bounded_and_network_blocked(self):
        import ast
        tree = ast.parse(Path("modal_apps/inspect_planner_and_assets.py").read_text())
        remote = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=="inspect_remote")
        keywords = {item.arg: item.value for item in remote.decorator_list[0].keywords}
        self.assertNotIn("gpu", keywords)
        self.assertTrue(ast.literal_eval(keywords["block_network"]))
        self.assertEqual(ast.literal_eval(keywords["retries"]), 0)
        self.assertEqual(ast.literal_eval(keywords["timeout"]), 300)
        self.assertEqual(ast.literal_eval(keywords["max_containers"]), 1)
        volume = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id=="assets" for target in node.targets))
        self.assertEqual(volume.value.func.attr, "read_only")
        self.assertEqual(ast.literal_eval(volume.value.func.value.keywords[0].value), False)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.robot = self.root/"RoboTwin"
        self.assets = self.root/"assets"

    def populate_mugs(self):
        mug = self.assets/"objects"/"039_mug"
        for kind in ("visual", "collision"):
            (mug/kind).mkdir(parents=True)
            for i in range(10):
                (mug/kind/f"base{i}.glb").write_bytes(b"synthetic fixture, not a real GLB")
        for i, scale in enumerate(MUG_CLASS_SCALES):
            (mug/f"model_data{i}.json").write_text(json.dumps({"scale": [scale]*3}))
        return mug

    def test_node_transforms_and_outer_scale_define_conservative_sphere(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        transform[:3, 3] = (3., 4., 5.)
        scene = mesh_scene(transform)
        scene.graph["second_instance"] = (np.eye(4), "cube")
        result = scaled_envelope(scene, (.1, .2, .3))
        np.testing.assert_allclose(result["bounds_min_max_m"], [[-.1,-.2,-.3], [.4,1.,1.8]])
        self.assertEqual(result["transformed_nodes"], 2)
        center = np.asarray(result["sphere_center_m"])
        bounds = np.asarray(result["bounds_min_max_m"])
        self.assertAlmostEqual(np.linalg.norm(bounds[1]-center), result["sphere_radius_m"])

    def test_all_ten_visual_and_collision_envelopes_without_claiming_identity(self):
        self.populate_mugs()
        result = inspect_mugs(self.assets, mesh_loader=lambda path: mesh_scene())
        self.assertTrue(result["geometry_complete"])
        self.assertFalse(result["verified_asset_class"])  # Fake bytes do not match the audited hashes.
        self.assertEqual(len(result["variants"]), 10)
        for i, row in enumerate(result["variants"]):
            self.assertEqual(set(row["meshes"]), {"visual", "collision"})
            np.testing.assert_allclose(row["meshes"]["visual"]["extents_m"], [2*MUG_CLASS_SCALES[i]]*3)
        self.assertIsNotNone(result["whole_class_union"])
        self.assertEqual(result["tcp_registration"], "unknown")

    def test_missing_collision_mesh_prevents_complete_class_envelope(self):
        mug = self.populate_mugs()
        (mug/"collision"/"base7.glb").unlink()
        result = inspect_mugs(self.assets, mesh_loader=lambda path: mesh_scene())
        self.assertFalse(result["geometry_complete"])
        self.assertIsNone(result["whole_class_union"])
        self.assertEqual(result["variants"][7]["status"], "incomplete")
        self.assertIn("base7.glb", result["variants"][7]["errors"][0]["path"])

    def test_invalid_scale_mesh_and_budget_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "invalid_asset_scale"):
            scaled_envelope(mesh_scene(), (.1, 0., .1))
        with patch("modal_apps.inspect_planner_and_assets.MAX_VERTICES", 2):
            with self.assertRaisesRegex(ValueError, "mesh_vertex_limit"):
                scaled_envelope(mesh_scene(), (.1, .1, .1))
        bad = mesh_scene()
        bad.geometry["cube"].vertices[0,0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid_mesh"):
            scaled_envelope(bad, (.1, .1, .1))

    def test_source_evidence_has_line_numbers_but_never_executes_or_certifies(self):
        envs = self.robot/"envs"
        envs.mkdir(parents=True)
        (envs/"robot.py").write_text("raise RuntimeError('must not import')\n"
            "def right_plan_path(pose):\n    return planner.plan(pose) # attached mug unsupported?\n")
        embodiment = self.assets/"embodiments"/"aloha"
        embodiment.mkdir(parents=True)
        (embodiment/"config.yml").write_text("urdf_path: robot.urdf\n")
        mplib = self.root/"planner.py"
        mplib.write_text("def IK():\n    if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:\n        break\n")
        result = inspect_sources(self.robot, self.assets, mplib)
        evidence = next(row for row in result["files"] if row["path"].endswith("robot.py"))
        self.assertIn({"line": 2, "text": "def right_plan_path(pose):"}, evidence["lines"])
        self.assertTrue(result["mplib_patch"]["patched_condition_present"])
        self.assertTrue(any(line["line"]==2 for line in result["mplib_patch"]["lines"]))
        self.assertFalse(result["mplib_patch"]["original_condition_present"])
        self.assertTrue(all(value.startswith("unknown") for value in result["coverage"].values()))
        self.assertTrue(any(row["path"].endswith("config.yml") for row in result["files"]))

    def test_audit_reports_missing_evidence_and_never_authorizes_motion(self):
        report = inspect_local(self.robot, self.assets, self.root/"missing.py", mesh_loader=lambda path: mesh_scene())
        self.assertFalse(report["simulator_started"])
        self.assertFalse(report["active_viewing_authorized"])
        self.assertFalse(report["mug_mesh_envelopes"]["geometry_complete"])
        self.assertEqual(report["native_planner"]["mplib_patch"]["status"], "unavailable")
        self.assertTrue(report["native_planner"]["missing_roots"])
        json.dumps(report, allow_nan=False)

