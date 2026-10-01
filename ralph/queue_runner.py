"""Dependency-aware serial issue queue, with explicitly enabled reviewed merges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from runtime import GitHub, Stop, read_json, write_json
from runner import Session, check_image, check_plan, criteria, digest, git, issue_lock, load_review


def load_queue(path: Path, repository: str) -> dict:
    plan = read_json(path)
    numbers = plan.get("issues", [])
    if plan.get("repository") != repository or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", plan.get("name", "")):
        raise Stop("Queue name/repository is invalid.")
    if not numbers or any(type(n) is not int or n <= 1 for n in numbers) or len(numbers) != len(set(numbers)):
        raise Stop("Queue must contain unique implementation issue numbers.")
    checks = plan.get("required_checks", [])
    if not checks or any(not isinstance(c, str) or not c.strip() for c in checks):
        raise Stop("Queue must explicitly name its required GitHub checks.")
    if type(plan.get("ci_timeout_seconds")) is not int or not 1 <= plan["ci_timeout_seconds"] <= 3600:
        raise Stop("Queue CI wait must be 1..3600 seconds.")
    if not isinstance(plan.get("validation"), dict):
        raise Stop("Queue requires an approved validation plan.")
    return plan


def eligibility(hub: GitHub, number: int, actor: str, claim: bool, optional: bool) -> str | None:
    issue = hub.issue(number)
    if issue.get("state") != "open":
        return "closed or unknown state"
    labels = {v["name"] for v in issue.get("labels", [])}
    if issue.get("pull_request") or not {"capstone-v1", "ready-for-agent"} <= labels:
        return "not ready for an agent"
    if labels & {"needs-info", "needs-triage", "ready-for-human", "wontfix"}:
        return "conflicting triage label"
    if "scope:optional" in labels and not optional:
        return "optional scope excluded"
    owners = [v["login"].lower() for v in issue.get("assignees", [])]
    if owners != [actor.lower()] and not (claim and not owners):
        return "assigned elsewhere or unassigned without --claim-unassigned"
    criteria(issue.get("body") or "")
    if any(v.get("state") != "closed" for v in hub.blockers(number)):
        return "open native blocker"
    section = re.search(r"(?ims)^## Blocked by\s*\n(.*?)(?=^## |\Z)", issue.get("body") or "")
    for dependency in set(map(int, re.findall(r"#(\d+)", section[1] if section else ""))):
        if hub.issue(dependency).get("state") != "closed":
            return f"open textual blocker #{dependency}"
    return None


def ci_ready(hub: GitHub, sha: str, required: list[str]) -> bool:
    pages = json.loads(hub.call("api", "--paginate", "--slurp",
        f"repos/{hub.repository}/commits/{sha}/check-runs?filter=latest&per_page=100"))
    runs = [v for page in pages for v in page["check_runs"]]
    statuses = hub.pages(f"repos/{hub.repository}/commits/{sha}/statuses?per_page=100")
    latest_statuses = {}
    for status in statuses:  # GitHub returns newest first.
        latest_statuses.setdefault(status["context"], status)
    if any(v["state"] in ("error", "failure") for v in latest_statuses.values()):
        raise Stop("A commit status failed; queue will not merge.")
    if any(v.get("conclusion") in ("failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale") for v in runs):
        raise Stop("A GitHub check failed; queue will not merge.")
    by_name = {}
    for run in runs:
        if run.get("app", {}).get("slug") == "github-actions":
            by_name.setdefault(run["name"], []).append(run)
    return (all(name in by_name and all(v.get("conclusion") == "success" and v["status"] == "completed"
                for v in by_name[name]) for name in required)
            and all(v["status"] == "completed" and v.get("conclusion") in ("success", "neutral", "skipped") for v in runs)
            and all(v["state"] == "success" for v in latest_statuses.values()))


def merge_reviewed(session: Session, plan: dict, deadline: float, *, sleep=time.sleep, now=time.time):
    """Fast-forward exactly the checked head, guarded by the reviewed base lease.

    GitHub recognizes the reachable PR as merged. This avoids merging an untested
    new combination if main advances between inspection and the merge request.
    Server protection is never disabled or overridden, and blocked PRs stop here.
    """
    state, hub = session.state, session.hub
    stop = min(deadline, state["deadline"], state.get("merge_deadline", now() + plan["ci_timeout_seconds"]))
    if now() >= stop:
        raise Stop("Original issue/CI/queue deadline expired; merge is not authorized.")
    if "merge_deadline" not in state:
        session.save(merge_deadline=stop)
    evidence = read_json(session.directory / f"attempt-{state['attempts']}" / "validated-head.json")
    if (evidence.get("head") != state["head"] or evidence.get("tree") != state["candidate_tree"]
            or not evidence.get("checks") or any(c.get("exit_code") != 0 for c in evidence["checks"])):
        raise Stop("Passing validation evidence must match the reviewed commit.")
    review = load_review(json.dumps(state.get("review")), session.args.issue,
                         state["base"], state["head"], len(session.ac))
    if review["verdict"] != "pass":
        raise Stop("Queue requires an independent passing review for this exact commit.")
    number = int(state["pr"].rstrip("/").split("/")[-1])
    while now() < stop:
        session.fresh()
        pr = session.verified_pr()
        if pr.get("draft") or pr["base"]["sha"] != state["base"]:
            raise Stop("PR draft/base changed before merge.")
        if ci_ready(hub, state["head"], plan["required_checks"]):
            # Includes GitHub review/protection requirements, beyond local AI review.
            if pr.get("mergeable_state") not in ("clean", "unknown", None):
                raise Stop("GitHub reports an unmergeable/blocked PR; no rule bypass is allowed.")
            if pr.get("mergeable_state") == "clean":
                break
        print(f"WAIT CI: issue #{session.args.issue}", flush=True)
        sleep(min(15, max(0, stop - now())))
    else:
        raise Stop("CI/mergeability wait exhausted; resume retains this reviewed PR.")
    session.fresh()
    pr = session.verified_pr()
    if pr.get("mergeable_state") != "clean" or pr.get("draft") or pr["base"]["sha"] != state["base"]:
        raise Stop("PR merge requirements changed.")
    if not ci_ready(hub, state["head"], plan["required_checks"]):
        raise Stop("Checks changed before merge.")
    git(session.worktree, "merge-base", "--is-ancestor", state["base"], state["head"])
    session.save(phase="merging")  # Recovery checks reachability before doing more work.
    if now() >= stop:
        raise Stop("Original issue/CI/queue deadline expired before push.")
    git(session.worktree, "push", "--no-follow-tags",
        f"--force-with-lease=refs/heads/{session.policy['base_branch']}:{state['base']}",
        "origin", f"{state['head']}:refs/heads/{session.policy['base_branch']}")
    while now() < stop:
        pr = hub.api(f"repos/{hub.repository}/pulls/{number}")
        if pr.get("merged") and pr["head"]["sha"] == state["head"] and hub.issue(session.args.issue)["state"] == "closed":
            session.save(phase="merged")
            return
        sleep(min(5, max(0, stop - now())))
    raise Stop("Main updated; waiting for GitHub PR/issue reconciliation. Resume, do not rerun the worker.")


class Queue:
    def __init__(self, root, args, policy, hub=None, session_factory=Session):
        self.root, self.args, self.policy = root, args, policy
        self.plan = load_queue(args.queue, policy["repository"])
        self.hub = hub or GitHub(root, policy["repository"])
        self.session_factory = session_factory
        self.directory = root / ".ralph/queues" / self.plan["name"]
        self.path = self.directory / "state.json"
        self.state = {}

    def save(self, **changes):
        self.state.update(changes)
        write_json(self.path, self.state)

    def execute(self):
        args, hub = self.args, self.hub
        if type(args.max_issues) is not int or args.max_issues < 1:
            raise Stop("--max-issues must be positive; there is no fixed 50-issue ceiling.")
        if not 1 <= args.queue_hours <= 168:
            raise Stop("--queue-hours must be 1..168; each issue keeps its own limits.")
        hub.verify_origin()
        actor = hub.api("user")["login"]
        identity = digest({"plan": self.plan, "policy": self.policy, "actor": actor, "backend": args.backend,
                           "limit": args.max_issues, "hours": args.queue_hours,
                           "merge": args.auto_merge, "claim": args.claim_unassigned, "optional": args.include_optional})
        # Validate without changing a remote issue or consuming model calls.
        self.directory.mkdir(parents=True, exist_ok=True)
        sample = self.directory / "preflight-checks.json"
        write_json(sample, {**self.plan["validation"], "issue": self.plan["issues"][0]})
        validation = check_plan(sample, self.plan["issues"][0])
        check_image(self.root, validation)
        if args.dry_run:
            choices = {n: eligibility(hub, n, actor, args.claim_unassigned, args.include_optional)
                       for n in self.plan["issues"]}
            print(json.dumps({"queue": self.plan["name"], "max_issues": args.max_issues,
                              "auto_merge": args.auto_merge, "issues": choices}, indent=2))
            return
        if self.path.exists():
            if not args.resume:
                raise Stop("Queue has saved state; use --resume with the same options.")
            self.state = read_json(self.path)
            if self.state.get("identity") != identity:
                raise Stop("Queue scope/options/owner changed; saved limits cannot be reset on resume.")
        else:
            if args.resume:
                raise Stop("No queue state exists to resume.")
            self.save(identity=identity, deadline=time.time() + args.queue_hours * 3600,
                      completed=[], started=[], active=None, phase="running")
        while time.time() < self.state["deadline"]:
            active = self.state["active"]
            if active is None:
                if len(self.state["started"]) >= args.max_issues:
                    self.save(phase="limit_reached")
                    print(f"QUEUE LIMIT: {len(self.state['completed'])} completed; {len(self.state['started'])} started")
                    return
                unavailable = {}
                for number in self.plan["issues"]:
                    if number in self.state["completed"]:
                        continue
                    reason = eligibility(hub, number, actor, args.claim_unassigned, args.include_optional)
                    if reason is None:
                        active = number
                        break
                    unavailable[number] = reason
                if active is None:
                    self.save(phase="no_eligible_issues", unavailable=unavailable)
                    print("QUEUE STOP: no eligible issues; see queue state for dependencies/ownership.")
                    return
                # Journal before assignment or model work. One failure stops the queue.
                self.save(active=active, started=self.state["started"] + [active], phase="running")
            issue = hub.issue(active)
            state_path = self.root / ".ralph/runs" / f"issue-{active}/state.json"
            saved = read_json(state_path) if state_path.exists() else None
            if saved and saved.get("phase") in ("ready_for_human", "merging", "merged"):
                pr = hub.api(f"repos/{hub.repository}/pulls/{saved['pr'].rstrip('/').split('/')[-1]}")
                git(self.root, "fetch", "origin", self.policy["base_branch"])
                if saved.get("actor") != actor or saved.get("issue") != active or saved.get("branch") != f"ralph/issue-{active}":
                    raise Stop("Saved merge belongs to a different owner or issue.")
                if (pr.get("merged") and pr["head"]["sha"] == saved["head"] and
                        pr["head"]["ref"] == saved["branch"] and
                        pr["head"]["repo"]["full_name"] == self.policy["repository"] and
                        pr["base"]["repo"]["full_name"] == self.policy["repository"] and
                        pr["base"]["ref"] == self.policy["base_branch"] and issue["state"] == "closed"):
                    git(self.root, "merge-base", "--is-ancestor", saved["head"], f"origin/{self.policy['base_branch']}")
                    self.save(completed=self.state["completed"] + [active], active=None)
                    continue
                if hub.ref(self.policy["base_branch"]) != saved["base"]:
                    raise Stop("Interrupted merge needs GitHub reconciliation; worker will not be rerun.")
            reason = eligibility(hub, active, actor, args.claim_unassigned, args.include_optional)
            if reason is not None:
                raise Stop(f"Active issue #{active} is no longer eligible: {reason}")
            if not issue.get("assignees") and args.claim_unassigned:
                hub.call("issue", "edit", str(active), "--repo", self.policy["repository"], "--add-assignee", actor)
            checks = self.directory / f"checks-{active}.json"
            expected = {**self.plan["validation"], "issue": active}
            if checks.exists() and read_json(checks) != expected:
                raise Stop("Saved queue checks changed.")
            write_json(checks, expected)
            single = argparse.Namespace(issue=active, backend=args.backend, checks=checks, dry_run=False,
                                        resume=bool(saved), include_optional=args.include_optional,
                                        queue_deadline=self.state["deadline"], auto_merge=args.auto_merge,
                                        campaign=self.plan.get("objective", ""))
            with issue_lock(self.root, active):
                session = self.session_factory(self.root, single, self.policy, hub)
                if saved and saved.get("phase") in ("ready_for_human", "merging"):
                    session.preflight()
                    session.state = saved
                    if saved["actor"] != actor or saved["signature"] != session.identity():
                        raise Stop("Reviewed issue identity changed before queue resume.")
                else:
                    session.execute()
                if not args.auto_merge:
                    self.save(phase="waiting_for_merge")
                    print("QUEUE PAUSED: reviewed PR awaits merge. Resume this queue afterwards.")
                    return
                merge_reviewed(session, self.plan, self.state["deadline"])
            self.save(completed=self.state["completed"] + [active], active=None)
            print(f"QUEUE PROGRESS: {len(self.state['completed'])}/{args.max_issues} merged", flush=True)
        raise Stop("Queue time limit reached; saved deadline remains in force.")


def run_queue(root, args, policy):
    queue = Queue(root, args, policy)
    with issue_lock(root, f"queue-{queue.plan['name']}"):
        try:
            queue.execute()
        except (Stop, OSError, ValueError, KeyError, TypeError) as exc:
            if queue.state:
                queue.save(last_stop=str(exc))
            raise
