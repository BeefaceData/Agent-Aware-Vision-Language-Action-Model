"""Queue lifecycle and exact-commit merge behavior with simulated GitHub."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_runner as queue
import runner
from runtime import Stop, read_json, write_json
import test_loop


class QueueHub:
    repository = "example/test"
    def __init__(self):
        self.items = {}
        for number in (2, 3, 4):
            self.items[number] = {**test_loop.issue(), "number": number, "assignees": []}
        self.items[3]["body"] = "## Acceptance criteria\n- [ ] answer\n## Blocked by\n- #2"
        self.claimed = []
        self.runs = [{"name": "checks", "app": {"slug": "github-actions"},
                      "status": "completed", "conclusion": "success"}]
    def verify_origin(self): pass
    def api(self, endpoint):
        if endpoint == "user": return {"login": "teammate"}
        raise AssertionError(endpoint)
    def issue(self, number): return copy.deepcopy(self.items[number])
    def blockers(self, number): return []
    def pages(self, endpoint): return []
    def call(self, *args):
        if args[0] == "api": return json.dumps([{"check_runs": self.runs}])
        assert args[:2] == ("issue", "edit"), args
        number = int(args[2])
        self.items[number]["assignees"] = [{"login": args[-1]}]
        self.claimed.append(number)


class QueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = {"name": "test", "repository": "example/test", "issues": [3, 2, 4],
                     "ci_timeout_seconds": 30, "required_checks": ["checks"],
                     "validation": {"image": "python@sha256:" + "0" * 64,
                                    "commands": [["python", "-m", "unittest"]]}}
        path = self.root / "queue.json"
        write_json(path, self.plan)
        self.args = argparse.Namespace(queue=path, max_issues=2, queue_hours=2,
            backend="codex", auto_merge=True, claim_unassigned=True, include_optional=False,
            resume=False, dry_run=False)
        self.hub = QueueHub()
        self.executed = []
        outer = self
        class Worker:
            def __init__(self, root, args, policy, hub): self.args = args
            def execute(self): outer.executed.append(self.args.issue)
        self.worker = Worker
        self.image = patch.object(queue, "check_image")
        self.image.start()
        self.addCleanup(self.image.stop)
        def merged(session, plan, deadline):
            self.hub.items[session.args.issue]["state"] = "closed"
        self.merge = patch.object(queue, "merge_reviewed", side_effect=merged)
        self.mock_merge = self.merge.start()
        self.addCleanup(self.merge.stop)

    def controller(self):
        return queue.Queue(self.root, self.args, {"repository": "example/test", "base_branch": "main"}, self.hub, self.worker)

    def test_dependency_order_sole_assignment_and_limit_survive_resume(self):
        controller = self.controller()
        controller.execute()
        self.assertEqual(self.executed, [2, 3])
        self.assertEqual(self.hub.claimed, [2, 3])
        self.assertEqual(controller.state["completed"], [2, 3])
        self.args.resume = True
        self.controller().execute()
        self.assertEqual(self.executed, [2, 3])
        self.args.max_issues = 3
        with self.assertRaisesRegex(Stop, "scope/options/owner"):
            self.controller().execute()

    def test_dry_run_does_not_assign_invoke_or_merge(self):
        self.args.dry_run = True
        self.controller().execute()
        self.assertFalse(self.hub.claimed)
        self.assertFalse(self.executed)
        self.mock_merge.assert_not_called()

    def test_other_owners_and_open_native_dependencies_are_skipped(self):
        self.hub.items[2]["assignees"] = [{"login": "other"}]
        self.hub.blockers = lambda number: [{"state": "open"}] if number == 4 else []
        controller = self.controller()
        controller.execute()
        self.assertFalse(self.executed)
        self.assertEqual(controller.state["phase"], "no_eligible_issues")

    def test_one_failed_issue_stops_before_claiming_another(self):
        self.worker.execute = lambda worker: (_ for _ in ()).throw(Stop("worker failed"))
        controller = self.controller()
        with self.assertRaisesRegex(Stop, "worker failed"):
            controller.execute()
        self.assertEqual(self.hub.claimed, [2])
        self.assertEqual(controller.state["active"], 2)
        self.assertEqual(controller.state["started"], [2])
        self.mock_merge.assert_not_called()

    def test_manual_mode_stops_after_reviewed_pr(self):
        self.args.auto_merge = False
        controller = self.controller()
        controller.execute()
        self.assertEqual(self.executed, [2])
        self.assertEqual(controller.state["phase"], "waiting_for_merge")
        self.mock_merge.assert_not_called()

    def test_expired_queue_cannot_reset_deadline(self):
        self.controller().execute()
        path = self.root / ".ralph/queues/test/state.json"
        state = read_json(path)
        state["deadline"] = time.time() - 1
        write_json(path, state)
        self.args.resume = True
        with self.assertRaisesRegex(Stop, "time limit"):
            self.controller().execute()

    def test_interrupted_merge_reconciles_without_rerunning_worker(self):
        controller = self.controller()
        # Stop after the active issue is journaled, simulating lost push response.
        self.mock_merge.side_effect = Stop("lost response")
        with self.assertRaises(Stop): controller.execute()
        saved = {"phase": "merging", "issue": 2, "actor": "teammate", "branch": "ralph/issue-2",
                 "pr": "https://github.com/example/test/pull/20", "head": "abc", "base": "def"}
        write_json(self.root / ".ralph/runs/issue-2/state.json", saved)
        self.hub.items[2]["state"] = "closed"
        self.hub.items[3]["assignees"] = [{"login": "someone-else"}]
        self.hub.items[4]["assignees"] = [{"login": "someone-else"}]
        original = self.hub.api
        self.hub.api = lambda endpoint: original(endpoint) if endpoint == "user" else {
            "merged": True, "head": {"sha": "abc", "ref": "ralph/issue-2", "repo": {"full_name": "example/test"}},
            "base": {"ref": "main", "repo": {"full_name": "example/test"}}}
        self.args.resume = True
        resumed = self.controller()
        with patch.object(queue, "git") as git:
            resumed.execute()
        self.assertEqual(resumed.state["completed"], [2])
        self.assertEqual(self.executed, [2])
        self.assertTrue(any("--is-ancestor" in call.args for call in git.call_args_list))


class MergeTests(unittest.TestCase):
    def fixture(self):
        fixture = test_loop.LifecycleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.hub.repository = fixture.policy["repository"]
        fixture.session.execute()
        original = fixture.hub.api
        def api(endpoint):
            result = original(endpoint)
            if "/pulls/" in endpoint:
                result.update(mergeable_state="clean", merged=fixture.hub.ref("main") == fixture.session.state["head"])
                if result["merged"]:
                    fixture.hub.selected["state"] = "closed"
            return result
        fixture.hub.api = api
        return fixture

    def test_only_reviewed_head_lands_on_main_and_issue_closes(self):
        f = self.fixture()
        with patch.object(queue, "ci_ready", return_value=True):
            queue.merge_reviewed(f.session, {"ci_timeout_seconds": 10, "required_checks": ["checks"]}, time.time()+10)
        self.assertEqual(f.hub.ref("main"), f.session.state["head"])
        self.assertEqual(f.session.state["phase"], "merged")
        self.assertEqual(f.hub.selected["state"], "closed")
        self.assertEqual(len(f.calls), 2)

    def test_concurrent_main_update_is_not_overwritten(self):
        f = self.fixture()
        original = f.session.save
        # Race after all freshness checks but before push: another merge lands.
        def save(**changes):
            original(**changes)
            if changes.get("phase") == "merging":
                (f.root / "other.txt").write_text("another contributor")
                runner.git(f.root, "add", "other.txt")
                runner.git(f.root, "commit", "-qm", "test: :white_check_mark: race fixture")
                runner.git(f.root, "push", "origin", "main")
        f.session.save = save
        with patch.object(queue, "ci_ready", return_value=True), self.assertRaises(Stop):
            queue.merge_reviewed(f.session, {"ci_timeout_seconds": 10, "required_checks": ["checks"]}, time.time()+10)
        self.assertEqual(f.hub.ref("main"), runner.git(f.root, "rev-parse", "HEAD"))
        self.assertNotEqual(f.hub.ref("main"), f.session.state["head"])

    def test_blocked_pr_or_failed_ci_never_updates_main(self):
        f = self.fixture()
        original = f.hub.api
        f.hub.api = lambda endpoint: {**original(endpoint), "mergeable_state": "blocked"}
        with patch.object(queue, "ci_ready", return_value=True), self.assertRaisesRegex(Stop, "blocked"):
            queue.merge_reviewed(f.session, {"ci_timeout_seconds": 10, "required_checks": ["checks"]}, time.time()+10)
        self.assertEqual(f.hub.ref("main"), f.base)


class ChecksTests(unittest.TestCase):
    def test_required_check_missing_pending_failed_or_wrong_app_cannot_pass(self):
        hub = QueueHub()
        self.assertTrue(queue.ci_ready(hub, "head", ["checks"]))
        self.assertFalse(queue.ci_ready(hub, "head", ["missing"]))
        hub.runs[0]["status"] = "in_progress"
        self.assertFalse(queue.ci_ready(hub, "head", ["checks"]))
        hub.runs[0].update(status="completed", conclusion="failure")
        with self.assertRaises(Stop): queue.ci_ready(hub, "head", ["checks"])
        hub.runs[0].update(conclusion="success", app={"slug": "untrusted"})
        self.assertFalse(queue.ci_ready(hub, "head", ["checks"]))

    def test_other_failed_or_pending_status_also_prevents_merge(self):
        hub = QueueHub()
        hub.pages = lambda endpoint: [{"context": "external", "state": "pending"}]
        self.assertFalse(queue.ci_ready(hub, "head", ["checks"]))
        hub.pages = lambda endpoint: [{"context": "external", "state": "failure"}]
        with self.assertRaises(Stop): queue.ci_ready(hub, "head", ["checks"])
