"""Default run folders say when and what ran, and never collide."""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from harness.memory.base import MemoryRecord
from run_skills import build_parser, default_output

MORNING = datetime(2026, 10, 28, 7, 55)


class RunFolderNameTests(unittest.TestCase):
    def test_name_leads_with_sortable_date_then_weekday_time_and_what_ran(self):
        folder = default_output("/repo", "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml", 3,
                                where="modal-T4", now=MORNING)

        self.assertEqual(folder, Path("/repo/data/runs/"
                                      "2026-10-28_Wed_07-55am_opus_full_rgbd_table_lift_seed3_modal-T4"))

    def test_short_enough_names_keep_the_full_config(self):
        folder = default_output("/repo", "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml", 3,
                                where="local", now=MORNING)

        self.assertEqual(folder.name, "2026-10-28_Wed_07-55am_opus_full_task_rgbd_table_lift_seed3_local")

    def test_worst_case_name_still_fits_memory_candidate_ids(self):
        with tempfile.TemporaryDirectory() as root:
            first = default_output(root, "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml", 999,
                                   where="modal-A100-80GB", now=MORNING)
            for _ in range(11):
                first.parent.mkdir(parents=True, exist_ok=True)
                default_output(root, "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml", 999,
                               where="modal-A100-80GB", now=MORNING).mkdir()
            folder = default_output(root, "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml", 999,
                                    where="modal-A100-80GB", now=MORNING)

        self.assertTrue(folder.name.endswith("-12"))
        MemoryRecord(id=f"{folder.name}.observation", phases=("grasp",), source="public")

    def test_base_config_keeps_its_full_name(self):
        folder = default_output("/repo", "configs/skill_runtime.yaml", 0, where="local", now=MORNING)

        self.assertEqual(folder.name, "2026-10-28_Wed_07-55am_skill_runtime_seed0_local")

    def test_afternoon_times_use_pm(self):
        folder = default_output("/repo", "c.yaml", 3, where="local", now=datetime(2026, 10, 28, 15, 5))

        self.assertTrue(folder.name.startswith("2026-10-28_Wed_03-05pm_"))

    def test_runs_started_in_the_same_minute_get_a_suffix(self):
        with tempfile.TemporaryDirectory() as root:
            first = default_output(root, "c.yaml", 3, where="local", now=MORNING)
            first.mkdir(parents=True)

            second = default_output(root, "c.yaml", 3, where="local", now=MORNING)

        self.assertEqual(second.name, f"{first.name}-2")

    def test_output_option_is_optional(self):
        args = build_parser(Path("/repo")).parse_args(["--development-oracle"])

        self.assertIsNone(args.output)


if __name__ == "__main__":
    unittest.main()
