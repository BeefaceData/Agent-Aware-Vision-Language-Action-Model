"""Public lifecycle tests with real temporary Git repos and simulated providers."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner
import runtime


def issue():
    return {"number": 2, "title": "Expose answer", "state": "open",
            "body": "## Acceptance criteria\n- [ ] Public value is 42.\n\n## Blocked by\nNone.\n",
            "labels": [{"name": "ready-for-agent"}, {"name": "capstone-v1"}],
            "assignees": [{"login": "teammate"}]}


class FakeGitHub:
    def __init__(self, root, remote, policy):
        self.root, self.remote, self.policy = root, remote, policy
        self.selected = issue()
        self.dependencies = []
        self.ready = False
        self.pr = None
        self.body = None
        self.claims = 0

    def verify_origin(self):
        pass

    def pages(self, endpoint):
        return []

    def issue(self, number):
        return copy.deepcopy(self.selected if number == 2 else {"number": number, "body": "Frozen policy PRD", "state": "closed"})

    def blockers(self, number):
        return self.dependencies

    def api(self, endpoint):
        if "/pulls/" in endpoint:
            return {"state": "open", "draft": not self.ready,
                    "head": {"sha": self.ref("ralph/issue-2"), "ref": "ralph/issue-2", "repo": {"full_name": self.policy["repository"]}},
                    "base": {"sha": self.ref("main"), "ref": "main", "repo": {"full_name": self.policy["repository"]}}}
        assert endpoint == "user"
        return {"login": "teammate"}

    def ref(self, branch):
        return runner.git(self.remote, "rev-parse", f"refs/heads/{branch}")

    def claim(self, branch, base):
        self.claims += 1
        # update-ref's expected all-zero old value gives atomic create semantics.
        runner.git(self.remote, "update-ref", f"refs/heads/{branch}", base, "0" * 40)

    def call(self, *args):
        if args[:2] == ("repo", "view"):
            return json.dumps({"nameWithOwner": self.policy["repository"], "defaultBranchRef": {"name": "main"}})
        if args[:2] == ("pr", "list"):
            return "[]" if not self.pr else json.dumps([{"url": self.pr, "state": "OPEN"}])
        if args[:2] == ("pr", "create"):
            assert "--draft" in args
            self.pr = "https://github.com/example/test/pull/3"
            return self.pr
        if args[:2] == ("pr", "edit"):
            self.body = Path(args[args.index("--body-file") + 1]).read_text(encoding="utf-8")
            return ""
        if args[:2] == ("pr", "view"):
            return json.dumps({"headRefOid": self.ref("ralph/issue-2"), "state": "OPEN", "isDraft": not self.ready})
        if args[:2] == ("pr", "ready"):
            self.ready = "--undo" not in args
            return ""
        raise AssertionError(f"Unexpected GitHub mutation: {args}")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ralph-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.remote = Path(self.temporary.name) / "origin.git"
        self.root.mkdir()
        runner.git(self.root, "init", "-q", "-b", "main")
        runner.git(self.root, "config", "user.name", "Loop test")
        runner.git(self.root, "config", "user.email", "loop@example.invalid")
        self.policy = runtime.read_json(runner.HERE / "policy.json")
        for name in runner.REQUIRED + ("ralph/runner.py", "ralph/runtime.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((runner.ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
        (self.root / "docs/adr").mkdir(parents=True)
        (self.root / ".gitignore").write_text(".ralph/\n.ralph-run/\n__pycache__/\n", encoding="utf-8")
        runner.git(self.root, "add", ".")
        runner.git(self.root, "diff", "--cached")
        runner.git(self.root, "commit", "-qm", "test: :white_check_mark: prepare fixture", "-m", "Exercise real Git isolation without touching GitHub.")
        runner.git(self.root, "init", "--bare", "-q", str(self.remote))
        runner.git(self.root, "remote", "add", "origin", str(self.remote))
        runner.git(self.root, "push", "-q", "origin", "main")
        self.base = runner.git(self.root, "rev-parse", "HEAD")
        self.hub = FakeGitHub(self.root, self.remote, self.policy)
        plan = self.root / ".ralph/checks.json"
        self.image = "python@sha256:" + "0" * 64
        runtime.write_json(plan, {"issue": 2, "image": self.image, "commands": [[sys.executable, "-c", "from answer import value; assert value == 42"]]})
        self.args = argparse.Namespace(issue=2, checks=plan, backend="codex", dry_run=False,
                                       resume=False, include_optional=False)
        self.calls = []
        self.behavior = "pass"
        self.session = runner.Session(self.root, self.args, self.policy, self.hub)
        self.login = patch.object(runner, "personal_login")
        self.login.start()
        self.addCleanup(self.login.stop)
        self.image_check = patch.object(runner, "check_image")
        self.image_check.start()
        self.addCleanup(self.image_check.stop)
        self.model = patch.object(runner, "invoke_model", side_effect=self.invoke)
        self.model.start()
        self.addCleanup(self.model.stop)
        self.checker = patch.object(runner, "run_checks", side_effect=self.checks)
        self.checker.start()
        self.addCleanup(self.checker.stop)

    def checks(self, candidate, plan, directory, remaining):
        # The integration fixture substitutes a known synthetic checker for
        # Docker. It receives only the clean exported candidate, never the worker tree.
        before = {str(p): p.read_bytes() for p in candidate.rglob("*") if p.is_file()}
        results = []
        for command in plan["commands"]:
            result = runtime.run(command, candidate, env=runtime.private_env(), check=False)
            if result.returncode or "Ran 0 tests" in result.stdout:
                raise runtime.Stop("Approved validation failed or discovered no tests")
            results.append({"argv": command, "exit_code": result.returncode, "output": result.stdout})
        after = {str(p): p.read_bytes() for p in candidate.rglob("*") if p.is_file() and "__pycache__" not in str(p)}
        if before != after:
            raise runtime.Stop("Validation changed candidate files")
        return results

    def invoke(self, backend, model, tree, prompt, directory, timeout, review=False):
        self.calls.append((backend, model["model"], model["effort"], review))
        if not review:
            if self.behavior == "worker_error":
                raise runtime.Stop("provider failed")
            (tree / "answer.py").write_text("value = " + ("0" if self.behavior == "bad_test" else "42") + "\n", encoding="utf-8")
            if self.behavior == "revise":
                with (tree / "answer.py").open("a", encoding="utf-8") as stream:
                    stream.write(f"# Repair attempt {len(self.calls)}\n")
            if self.behavior == "control_edit":
                (tree / "ralph/policy.json").write_text("{}", encoding="utf-8")
            report = {"issue": 999 if self.behavior == "wrong_issue" else 2, "status": "ready_for_review",
                      "summary": "Expose the public answer.", "acceptance": [{"criterion": 1, "evidence": "answer.value validated by approved check"}],
                      "checks": [], "risks": [], "merge_danger": {"door": "two-way", "blast_radius": "fixture", "reason": "Revert the isolated source change."},
                      "commit": {"subject": "feat: :sparkles: expose answer", "why": "Satisfy the public interface contract."}}
            runtime.write_json(tree / ".ralph-run/worker.json", report)
            return "<promise>ISSUE CLOSED #2</promise>"
        head = runner.git(tree, "rev-parse", "HEAD")
        verdict = "revise" if self.behavior == "revise" else "pass"
        if self.behavior == "stale":
            head = "0" * 40
        if self.behavior == "scope_changed":
            self.hub.selected["body"] += "Changed requirement."
        if self.behavior == "review_edit":
            (tree / "answer.py").write_text("value = 0\n", encoding="utf-8")
        return json.dumps({"issue": 2, "base_sha": self.base, "head_sha": head, "verdict": verdict,
                           "summary": "Inspected public contract and tests.",
                           "acceptance": [{"criterion": 1, "verdict": verdict, "evidence": "answer.py and controller check"}],
                           "findings": []})

    def test_success_publishes_reviewed_pr_without_merging_or_closing(self):
        self.session.execute()
        self.assertTrue(self.hub.ready)
        self.assertEqual(self.session.state["phase"], "ready_for_human")
        self.assertEqual(self.hub.ref("main"), self.base)
        self.assertEqual(self.hub.selected["state"], "open")
        self.assertEqual(runner.git(self.root, "rev-parse", "HEAD"), self.base)
        self.assertIn("Closes #2", self.hub.body)
        self.assertEqual(self.calls, [("codex", "gpt-6-sol", "xhigh", False), ("codex", "gpt-6-astra", "medium", True)])

    def test_dry_run_does_not_claim_or_invoke_models(self):
        self.args.dry_run = True
        self.session.execute()
        self.assertEqual(self.hub.claims, 0)
        self.assertFalse(self.calls)

    def test_open_dependency_blocks_before_claim(self):
        self.hub.dependencies = [{"number": 7, "state": "open"}]
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertEqual(self.hub.claims, 0)

    def test_wrong_owner_blocks_before_claim(self):
        self.hub.selected["assignees"] = [{"login": "someone-else"}]
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertEqual(self.hub.claims, 0)

    def test_existing_remote_claim_blocks_other_machine(self):
        self.hub.claim("ralph/issue-2", self.base)
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertFalse(self.calls)

    def test_failed_checks_do_not_push_or_review(self):
        self.behavior = "bad_test"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertEqual(self.hub.ref("ralph/issue-2"), self.base)
        self.assertIsNone(self.hub.pr)
        self.assertEqual(len(self.calls), 1)

    def test_untrusted_worker_claim_cannot_approve_wrong_issue(self):
        self.behavior = "wrong_issue"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertIsNone(self.hub.pr)

    def test_control_edits_require_supervised_work(self):
        self.behavior = "control_edit"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertEqual(self.hub.ref("ralph/issue-2"), self.base)

    def test_stale_sha_does_not_make_draft_ready(self):
        self.behavior = "stale"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertIsNotNone(self.hub.pr)
        self.assertFalse(self.hub.ready)

    def test_changed_scope_invalidates_review(self):
        self.behavior = "scope_changed"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertFalse(self.hub.ready)

    def test_reviewer_edits_invalidate_review(self):
        self.behavior = "review_edit"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertFalse(self.hub.ready)

    def test_provider_error_consumes_attempt_and_preserves_worktree(self):
        self.behavior = "worker_error"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertEqual(self.session.state["attempts"], 1)
        self.assertTrue(self.session.worktree.exists())
        self.assertFalse(self.hub.ready)

    def test_review_repairs_are_bounded_and_each_head_is_reviewed(self):
        self.behavior = "revise"
        with self.assertRaisesRegex(runtime.Stop, "budget exhausted"):
            self.session.execute()
        self.assertEqual(self.session.state["attempts"], 2)
        self.assertEqual(len(self.calls), 4)
        self.assertFalse(self.hub.ready)

    def test_resume_cannot_reset_attempt_budget(self):
        self.behavior = "worker_error"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.args.resume = True
        second = runner.Session(self.root, self.args, self.policy, self.hub)
        with self.assertRaises(runtime.Stop): second.execute()
        third = runner.Session(self.root, self.args, self.policy, self.hub)
        with self.assertRaisesRegex(runtime.Stop, "budget exhausted"): third.execute()
        self.assertEqual(len(self.calls), 2)

    def test_validation_that_mutates_code_does_not_publish(self):
        runtime.write_json(self.args.checks, {"issue": 2, "image": self.image, "commands": [[sys.executable, "-c",
            "from pathlib import Path; Path('answer.py').write_text('value = 99\\n')"]]})
        with self.assertRaisesRegex(runtime.Stop, "Validation changed"):
            self.session.execute()
        self.assertIsNone(self.hub.pr)

    def test_zero_test_run_is_not_accepted(self):
        runtime.write_json(self.args.checks, {"issue": 2, "image": self.image, "commands": [[sys.executable, "-c", "print('Ran 0 tests')"]]})
        with self.assertRaisesRegex(runtime.Stop, "discovered no tests"):
            self.session.execute()
        self.assertIsNone(self.hub.pr)

    def test_changed_checks_cannot_be_smuggled_into_resume(self):
        self.behavior = "worker_error"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.args.resume = True
        runtime.write_json(self.args.checks, {"issue": 2, "image": self.image, "commands": [[sys.executable, "-c", "pass"]]})
        second = runner.Session(self.root, self.args, self.policy, self.hub)
        with self.assertRaisesRegex(runtime.Stop, "checks"):
            second.execute()
        self.assertEqual(len(self.calls), 1)

    def test_interrupted_review_resumes_without_another_worker(self):
        original = self.invoke
        def interrupted(*args, **kwargs):
            if kwargs.get("review"):
                raise runtime.Stop("temporary review outage")
            return original(*args, **kwargs)
        self.model.stop()
        with patch.object(runner, "invoke_model", side_effect=interrupted):
            with self.assertRaises(runtime.Stop): self.session.execute()
        self.args.resume = True
        with patch.object(runner, "invoke_model", side_effect=original):
            resumed = runner.Session(self.root, self.args, self.policy, self.hub)
            resumed.execute()
        self.assertEqual(resumed.state["attempts"], 1)
        self.assertTrue(self.hub.ready)

    def test_ignored_worker_artifacts_do_not_validate_commit(self):
        original = self.invoke
        def with_ignored(*args, **kwargs):
            answer = original(*args, **kwargs)
            if not kwargs.get("review"):
                tree = args[2]
                (tree / ".ralph-run/helper.py").write_text("value = 42\n")
                (tree / "answer.py").write_text("import sys\nsys.path.insert(0,'.ralph-run')\nfrom helper import value\n")
            return answer
        with patch.object(runner, "invoke_model", side_effect=with_ignored):
            with self.assertRaisesRegex(runtime.Stop, "validation failed"):
                self.session.execute()
        self.assertIsNone(self.hub.pr)

    def test_retargeted_pr_is_rejected(self):
        original = self.hub.api
        def retargeted(endpoint):
            value = original(endpoint)
            if "/pulls/" in endpoint:
                value["base"]["ref"] = "other-branch"
            return value
        with patch.object(self.hub, "api", side_effect=retargeted):
            with self.assertRaisesRegex(runtime.Stop, "PR repository"):
                self.session.execute()
        self.assertFalse(self.hub.ready)

    def test_export_ignores_archive_transform_attributes(self):
        (self.root / "kept.txt").write_text("$Format:%H$\n", encoding="utf-8")
        (self.root / ".gitattributes").write_text("kept.txt export-ignore export-subst\n", encoding="utf-8")
        runner.git(self.root, "add", "kept.txt", ".gitattributes")
        runner.git(self.root, "diff", "--cached")
        runner.git(self.root, "commit", "-qm", "test: :white_check_mark: seed archive attributes", "-m", "Verify exact blob export without transformations.")
        destination = self.root / ".ralph/exact-export"
        runner.export_commit(self.root, runner.git(self.root, "rev-parse", "HEAD"), destination)
        self.assertEqual((destination / "kept.txt").read_text(), "$Format:%H$\n")

    def test_wrong_ready_pr_is_returned_to_draft_before_review(self):
        self.hub.pr = "https://github.com/example/test/pull/3"
        self.hub.ready = True
        self.behavior = "stale"
        with self.assertRaises(runtime.Stop): self.session.execute()
        self.assertFalse(self.hub.ready)

    def test_git_drivers_cannot_execute_worker_code_during_inspection(self):
        (self.root / "marker.py").write_text("from pathlib import Path\nPath('HOST_CODE_EXECUTED').write_text('bad')\n")
        runner.git(self.root, "config", "filter.bad.clean", "python marker.py")
        runner.git(self.root, "config", "core.fsmonitor", "python marker.py")
        (self.root / ".gitattributes").write_text("* filter=bad\n")
        with self.assertRaises(runtime.Stop): runner.inspect_changes(self.root, self.policy)
        self.assertFalse((self.root / "HOST_CODE_EXECUTED").exists())

    def test_push_never_publishes_implicit_follow_tags(self):
        runner.git(self.root, "config", "push.followTags", "true")
        runner.git(self.root, "tag", "-a", "unrelated-tag", "-m", "Do not publish this tag")
        self.session.execute()
        result = runtime.run(["git", "show-ref", "--verify", "refs/tags/unrelated-tag"], self.remote, check=False)
        self.assertNotEqual(result.returncode, 0)

    @unittest.skipIf(os.name == "nt", "Pathspec magic filename is invalid on Windows")
    def test_hidden_control_edit_and_magic_filename_cannot_enter_candidate(self):
        original = self.invoke
        def hidden_edit(*args, **kwargs):
            answer = original(*args, **kwargs)
            if not kwargs.get("review"):
                tree = args[2]
                runner.git(tree, "update-index", "--assume-unchanged", "ralph/policy.json")
                (tree / "ralph/policy.json").write_text("{}")
                (tree / ":(glob)*").write_text("unexpected pathspec expansion")
            return answer
        with patch.object(runner, "invoke_model", side_effect=hidden_edit):
            with self.assertRaisesRegex(runtime.Stop, "Control/credential path"):
                self.session.execute()
        self.assertEqual(self.hub.ref("ralph/issue-2"), self.base)
        self.assertIsNone(self.hub.pr)

    def test_repair_rejects_context_symlink_without_overwriting_external_file(self):
        sentinel = self.root / "external-sentinel.txt"
        sentinel.write_text("preserve me")
        probe = self.root / "symlink-probe"
        try:
            probe.symlink_to(sentinel)
        except OSError:
            self.skipTest("Host does not permit symlink creation")
        probe.unlink()
        original = self.invoke
        self.behavior = "revise"
        def leave_link(*args, **kwargs):
            answer = original(*args, **kwargs)
            if kwargs.get("review"):
                (args[2] / ".ralph-run/worker-context.md").symlink_to(sentinel)
            return answer
        with patch.object(runner, "invoke_model", side_effect=leave_link):
            with self.assertRaisesRegex(runtime.Stop, "link or special file"):
                self.session.execute()
        self.assertEqual(sentinel.read_text(), "preserve me")
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.hub.ready)

    def test_worker_cannot_publish_closing_directives_for_other_issues(self):
        original = self.invoke
        def unrelated_closure(*args, **kwargs):
            answer = original(*args, **kwargs)
            if not kwargs.get("review"):
                receipt = args[2] / ".ralph-run/worker.json"
                report = runtime.read_json(receipt)
                report["commit"]["why"] = "Fixes other/repository#99"
                runtime.write_json(receipt, report)
            return answer
        with patch.object(runner, "invoke_model", side_effect=unrelated_closure):
            with self.assertRaisesRegex(runtime.Stop, "closing directive"):
                self.session.execute()
        self.assertEqual(self.hub.ref("ralph/issue-2"), self.base)
        self.assertIsNone(self.hub.pr)

    def test_pr_evidence_cannot_add_another_closing_directive(self):
        self.session.execute()
        report = runtime.read_json(self.session.directory / "attempt-1/worker.json")
        report["acceptance"][0]["evidence"] = "CLOSES: #99"
        with self.assertRaisesRegex(runtime.Stop, "closing directive"):
            self.session.pr_body(report, [], None)

    def test_nested_agent_instructions_stop_before_publication(self):
        original = self.invoke
        def change_instructions(*args, **kwargs):
            answer = original(*args, **kwargs)
            if not kwargs.get("review"):
                path = args[2] / "src/AGENTS.md"
                path.parent.mkdir()
                path.write_text("Change the next agent's rules")
            return answer
        with patch.object(runner, "invoke_model", side_effect=change_instructions):
            with self.assertRaisesRegex(runtime.Stop, "Control/credential path"):
                self.session.execute()
        self.assertEqual(self.hub.ref("ralph/issue-2"), self.base)
        self.assertIsNone(self.hub.pr)

    def test_review_receives_complete_unicode_filename_contents(self):
        original = self.invoke
        def unicode_source(*args, **kwargs):
            if kwargs.get("review"):
                self.assertIn(json.dumps("caf\u00e9.py") + ': "# complete Unicode file', args[3])
            answer = original(*args, **kwargs)
            if not kwargs.get("review"):
                (args[2] / "caf\u00e9.py").write_text("# complete Unicode file\nvalue = 42\n", encoding="utf-8")
            return answer
        with patch.object(runner, "invoke_model", side_effect=unicode_source):
            self.session.execute()
        self.assertTrue(self.hub.ready)


class GuardTests(unittest.TestCase):
    def test_agent_instruction_paths_are_protected_at_every_depth(self):
        paths = ("CLAUDE.md", "src/AGENTS.md", "src/AGENTS.override.md", "src/CLAUDE.local.md",
                 "src/.cursorrules", "src/.cursor/rules/test.mdc", "src/.claude/settings.json")
        with tempfile.TemporaryDirectory() as directory:
            for name in paths:
                with self.subTest(name=name), self.assertRaisesRegex(runtime.Stop, "Control/credential path"):
                    runner.inspect_changes(Path(directory), {}, [name])

    def test_closing_keyword_spellings_are_rejected_but_plain_references_work(self):
        for keyword in ("close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved"):
            for reference in ("#99", "other/repository#99", "https://github.com/other/repository/issues/99"):
                with self.subTest(keyword=keyword, reference=reference), self.assertRaises(runtime.Stop):
                    runner.reject_closing_directives(f"**{keyword.upper()}:**\n{reference}")
        runner.reject_closing_directives("fix(harness): :bug: handle stale input\n\nRefs #2 and other/repository#99")

    def test_context_parent_link_is_rejected_before_prompt_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree, outside = root / "tree", root / "outside"
            tree.mkdir()
            outside.mkdir()
            sentinel = outside / "worker-context.md"
            sentinel.write_text("preserve me")
            try:
                (tree / ".ralph-run").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Host does not permit symlink creation")
            with patch.object(runtime, "model_command", return_value=["cursor"]), patch.object(runtime, "run") as process:
                with self.assertRaisesRegex(runtime.Stop, "link or special file"):
                    runtime.invoke_model("cursor", {"model": "pinned", "effort": "xhigh"}, tree,
                                         "new prompt", root / "logs", 10)
                process.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_file_prompt_backends_receive_explicit_prompt_and_closed_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("cursor", "claude"):
                tree = root / backend
                tree.mkdir()
                with patch.object(runtime, "model_command", return_value=[backend, "--print"]), patch.object(runtime, "run") as process:
                    process.return_value = subprocess.CompletedProcess([], 0, '{"result":"finished"}', '')
                    runtime.invoke_model(backend, {"model": "pinned", "effort": "xhigh"}, tree,
                                         "private prompt sentinel", root / (backend + "-logs"), 10)
                    self.assertIsNone(process.call_args.kwargs["text"])
                    self.assertIn("worker-context.md", process.call_args.args[0][-1])
                    self.assertNotIn("private prompt sentinel", str(process.call_args.args[0]))
                    self.assertEqual((tree / ".ralph-run/worker-context.md").read_text(), "private prompt sentinel")

    def test_default_credentials_are_removed_from_child_environment(self):
        values = {"ANTHROPIC_API_KEY": "client", "OPENAI_API_KEY": "api", "GH_TOKEN": "secret", "AWS_SESSION_TOKEN": "cloud", "GIT_DIR": "wrong", "GIT_INDEX_FILE": "wrong", "GIT_CONFIG_COUNT": "1", "SSH_AUTH_SOCK": "socket", "PATH": "path"}
        with patch.dict(os.environ, values, clear=True):
            env = runtime.private_env()
        self.assertEqual(env["PATH"], "path")
        self.assertFalse(set(values) - {"PATH"} & set(env))

    def test_ambiguous_multi_assignee_issue_is_rejected(self):
        selected = issue()
        selected["assignees"].append({"login": "other"})
        with self.assertRaises(runtime.Stop): runner.eligible(selected, "teammate", [])

    def test_optional_work_is_opt_in(self):
        selected = issue()
        selected["labels"].append({"name": "scope:optional"})
        with self.assertRaises(runtime.Stop): runner.eligible(selected, "teammate", [])
        self.assertEqual(len(runner.eligible(selected, "teammate", [], optional=True)), 1)

    def test_conflicting_triage_labels_are_rejected(self):
        selected = issue()
        selected["labels"].append({"name": "needs-info"})
        with self.assertRaises(runtime.Stop): runner.eligible(selected, "teammate", [])

    def test_local_lock_is_exclusive_across_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with runner.issue_lock(root, 2):
                with self.assertRaises(runtime.Stop):
                    with runner.issue_lock(root, 2): pass
            with runner.issue_lock(root, 2): pass

    def test_malformed_review_and_contradictory_pass_fail_closed(self):
        for text in ("I approve", json.dumps({"issue": 2, "base_sha": "b", "head_sha": "h", "verdict": "pass",
                "summary": "claim", "acceptance": [{"criterion": 1, "verdict": "revise", "evidence": "failed"}], "findings": []})):
            with self.assertRaises(runtime.Stop): runner.load_review(text, 2, "b", "h", 1)

    def test_dependency_api_failure_is_not_an_empty_frontier(self):
        hub = runtime.GitHub(Path.cwd(), "owner/repo")
        with patch.object(hub, "call", side_effect=runtime.Stop("API unavailable")):
            with self.assertRaises(runtime.Stop): hub.blockers(2)

    def test_process_failure_and_timeout_are_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(runtime.Stop):
                runtime.run([sys.executable, "-c", "raise SystemExit(7)"], Path(directory))
            with self.assertRaises(runtime.Stop):
                runtime.run([sys.executable, "-c", "import time; time.sleep(20)"], Path(directory), timeout=0.1)


if __name__ == "__main__":
    unittest.main()
