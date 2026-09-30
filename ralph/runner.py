"""One assigned issue -> bounded worker/review cycles -> PR for human merge.

Run through one of the three *-afk.sh launchers, or pass --backend on Windows.
The GitHub issue and native dependency graph are authoritative, not local state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import subprocess
import tarfile
import time

from runtime import (GitHub, Stop, invoke_model, personal_login, private_env, git_argv,
                     read_json, run, write_json, worker_area)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTROL_PATHS = ("ralph/", ".github/", ".agents/", ".claude/", ".codex/", ".cursor/", ".ralph-run/",
                 "AGENTS.md", "CONTEXT.md", "docs/agents/", "docs/adr/", ".gitignore", ".gitattributes")
REQUIRED = ("AGENTS.md", "CONTEXT.md", "docs/agents/issue-tracker.md",
            "docs/agents/triage-labels.md", ".agents/skills/commit-convention/SKILL.md",
            "ralph/prompt.md", "ralph/worker-prompt.md", "ralph/reviewer-prompt.md", "ralph/policy.json")


def git(root: Path, *args: str) -> str:
    return run(git_argv(root, *args), root, env=private_env()).stdout.strip()


def assert_unfiltered(root: Path, paths: list[str], source: str | None = None):
    if not paths:
        return
    options = [f"--source={source}"] if source else []
    result = run(git_argv(root, "check-attr", *options, "-z", "--stdin", "filter"), root,
                 env=private_env(), text="\0".join(paths) + "\0")
    values = result.stdout.split("\0")[2::3]
    if any(value not in ("unspecified", "unset") for value in values):
        raise Stop("Git filters active on candidate paths require supervised work.")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def criteria(body: str) -> list[str]:
    section = re.search(r"(?ims)^## Acceptance criteria\s*\n(.*?)(?=^## |\Z)", body)
    items = re.findall(r"(?m)^\s*- \[[ xX]\] (.+)$", section[1]) if section else []
    if not items:
        raise Stop("Issue needs a nonempty Acceptance criteria checklist.")
    return items


def eligible(issue: dict, actor: str, blockers: list, *, optional: bool = False) -> list[str]:
    if issue.get("pull_request") or issue.get("state") != "open" or issue.get("number") == 1:
        raise Stop("Select an open implementation issue, not the parent PRD or a PR.")
    labels = {label["name"] for label in issue.get("labels", [])}
    if not {"ready-for-agent", "capstone-v1"} <= labels:
        raise Stop("Issue is not marked ready-for-agent and capstone-v1.")
    if labels & {"needs-info", "needs-triage", "ready-for-human", "wontfix"}:
        raise Stop("Issue has a conflicting triage label.")
    if "scope:optional" in labels and not optional:
        raise Stop("Optional scope requires explicit --include-optional.")
    if [a["login"].lower() for a in issue.get("assignees", [])] != [actor.lower()]:
        raise Stop("Assign the issue to exactly your GitHub account during scrum before running it.")
    if any(b.get("state") != "closed" for b in blockers):
        raise Stop("At least one native blocking dependency is still open or has unknown state.")
    return criteria(issue.get("body") or "")


def issue_snapshot(issue: dict) -> dict:
    return {key: issue.get(key) for key in ("number", "title", "body", "state", "assignees", "labels")}


def check_plan(path: Path, issue: int) -> dict:
    plan = read_json(path)
    commands = plan.get("commands", [])
    if plan.get("issue") != issue or not commands:
        raise Stop("Checks plan must name this issue and contain approved commands.")
    if not re.fullmatch(r"[a-zA-Z0-9._:/-]+@sha256:[0-9a-f]{64}", plan.get("image", "")):
        raise Stop("Checks need a prebuilt Docker image pinned by @sha256 digest. Preload it before AFK use.")
    if any(not isinstance(cmd, list) or not cmd or any(not isinstance(v, str) or not v or "\x00" in v for v in cmd)
           for cmd in commands):
        raise Stop("Check commands must be nonempty argument arrays, never shell command strings.")
    timeout = plan.get("timeout_seconds", 300)
    if type(timeout) is not int or not 1 <= timeout <= 1800:
        raise Stop("Check timeout must be 1..1800 seconds.")
    plan["timeout_seconds"] = timeout
    return plan


def check_image(root: Path, plan: dict):
    try:
        run(["docker", "image", "inspect", plan["image"]], root, env=private_env())
    except Stop as exc:
        raise Stop("Docker or the pinned validation image is unavailable. Start Docker and preload the approved image before AFK use.") from exc


@contextmanager
def issue_lock(root: Path, issue: int):
    directory = root / ".ralph/locks" / f"issue-{issue}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise Stop(f"Issue has a local active/stale lock: {directory}. Inspect its owner; never steal it automatically.") from exc
    try:
        write_json(directory / "owner.json", {"pid": os.getpid(), "started": time.time()})
        yield
    finally:
        (directory / "owner.json").unlink(missing_ok=True)
        directory.rmdir()


def load_worker(path: Path, issue: int, count: int) -> dict:
    report = read_json(path)
    if report.get("issue") != issue or report.get("status") not in ("ready_for_review", "blocked"):
        raise Stop("Worker receipt identifies the wrong issue or an invalid status.")
    if report["status"] == "blocked":
        raise Stop("Worker reported blocked; read its receipt locally. No completion inferred.")
    entries = report.get("acceptance", [])
    if sorted(e.get("criterion", 0) for e in entries) != list(range(1, count + 1)):
        raise Stop("Worker receipt must cover each acceptance criterion exactly once.")
    if any(not isinstance(e.get("evidence"), str) or not e["evidence"].strip() for e in entries):
        raise Stop("Worker receipt lacks acceptance evidence.")
    if not isinstance(report.get("summary"), str) or not report["summary"].strip() or not isinstance(report.get("risks"), list):
        raise Stop("Worker receipt lacks a summary or risks list.")
    risk = report.get("merge_danger", {})
    if risk.get("door") not in ("one-way", "two-way") or not risk.get("blast_radius") or not risk.get("reason"):
        raise Stop("Worker must explain merge reversibility and blast radius.")
    commit = report.get("commit", {})
    if not re.fullmatch(r"(feat|fix|perf|style|refactor|test|docs|build|ci|chore|revert)(\([a-zA-Z0-9_/-]+\))?!?: :[a-z_]+: [^\r\n]+", commit.get("subject", "")):
        raise Stop("Commit subject must use the project Conventional Commit and Gitmoji convention.")
    if not isinstance(commit.get("why"), str) or not commit["why"].strip():
        raise Stop("Commit body must explain why.")
    return report


def reject_closing_directives(text: str) -> None:
    # Only the controller may link automatic closure to the selected issue.
    # Accept plain references; reject GitHub closing keywords, also in markup.
    plain = re.sub(r"[*_`\\]", "", text)
    keyword = r"\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)"
    reference = r"(?:#\d+|[\w.-]+/[\w.-]+#\d+|https?://github\.com/[^\s]+/(?:issues|pull)/\d+)"
    if re.search(keyword + r"[\s:]*" + reference, plain, re.I):
        raise Stop("Agent-derived publication text contains a closing directive; only the controller may close the selected issue.")


def load_review(text: str, issue: int, base: str, head: str, count: int) -> dict:
    try:
        report = json.loads(text)
    except ValueError as exc:
        raise Stop("Reviewer did not return valid JSON; no pass inferred.") from exc
    if (report.get("issue"), report.get("base_sha"), report.get("head_sha")) != (issue, base, head):
        raise Stop("Review is for another issue, base or head; it cannot approve this diff.")
    if report.get("verdict") not in ("pass", "revise", "blocked"):
        raise Stop("Invalid review verdict.")
    entries = report.get("acceptance", [])
    if sorted(e.get("criterion", 0) for e in entries) != list(range(1, count + 1)):
        raise Stop("Review did not assess every acceptance criterion exactly once.")
    if not report.get("summary") or any(not e.get("evidence") or e.get("verdict") not in ("pass", "revise", "blocked") for e in entries):
        raise Stop("Review is missing evidence.")
    findings = report.get("findings")
    if not isinstance(findings, list) or any(f.get("severity") not in ("blocking", "advisory") or
            not all(f.get(k) for k in ("location", "problem", "required_change")) for f in findings):
        raise Stop("Review findings are malformed.")
    if report["verdict"] == "pass" and (any(e["verdict"] != "pass" for e in entries) or
                                          any(f["severity"] == "blocking" for f in findings)):
        raise Stop("Review contradicts its own passing verdict.")
    return report


def changed_paths(worktree: Path, environment: dict | None = None) -> list[str]:
    environment = private_env() if environment is None else environment
    tracked = run(git_argv(worktree, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-only", "-z", "HEAD"), worktree, env=environment).stdout.split("\0")
    untracked = run(git_argv(worktree, "ls-files", "--others", "--exclude-standard", "-z"), worktree, env=environment).stdout.split("\0")
    return sorted(set(p for p in tracked + untracked if p))


def tree_snapshot(worktree: Path) -> str:
    tracked = run(git_argv(worktree, "ls-files", "-z"), worktree, env=private_env()).stdout.split("\0")
    paths = set(filter(None, tracked)) | set(changed_paths(worktree))
    return digest({name: hashlib.sha256(source_bytes(worktree / name)).hexdigest()
                   if (worktree / name).is_file() else "missing" for name in sorted(paths)})


def source_bytes(path: Path, limit: int | None = None) -> bytes:
    """Never let a worker-created pipe/device turn a controller read into a wait."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise Stop(f"Source must be a regular file: {path.name}")
    if limit is not None and info.st_size > limit:
        raise Stop(f"Large artifact needs supervised publication: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise Stop(f"Source became a non-regular file: {path.name}")
        data = stream.read() if limit is None else stream.read(limit + 1)
    if limit is not None and len(data) > limit:
        raise Stop(f"Large artifact needs supervised publication: {path.name}")
    return data


def export_commit(worktree: Path, head: str, target: Path):
    target.mkdir()
    result = subprocess.run(git_argv(worktree, "ls-tree", "-rz", head), cwd=worktree,
                            env=private_env(), capture_output=True, timeout=60, check=True)
    manifest = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = raw_name.decode("utf-8")
        path = target / name
        if mode not in ("100644", "100755") or kind != "blob" or not path.resolve().is_relative_to(target.resolve()):
            raise Stop("Candidate has an unsupported link, submodule or path.")
        blob = subprocess.run(git_argv(worktree, "cat-file", "blob", oid), cwd=worktree,
                              env=private_env(), capture_output=True, timeout=60, check=True).stdout
        algorithm = "sha1" if len(oid) == 40 else "sha256"
        if hashlib.new(algorithm, f"blob {len(blob)}\0".encode() + blob).hexdigest() != oid:
            raise Stop("Exported blob does not match its Git object ID.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        path.chmod(int(mode[-3:], 8))
        manifest[name] = {"mode": mode, "oid": oid, "sha256": hashlib.sha256(blob).hexdigest()}
    write_json(target.parent / "candidate-manifest.json", {"commit": head, "files": manifest})


def run_checks(candidate: Path, plan: dict, directory: Path, remaining) -> list:
    """Copy exact blobs/modes into a Docker volume; tests see it read-only."""
    environment = private_env()
    manifest = read_json(candidate.parent / "candidate-manifest.json")["files"]
    archive_path = directory / "candidate.tar"
    with tarfile.open(archive_path, "w") as archive:
        metadata = json.dumps(manifest).encode()
        header = tarfile.TarInfo("__ralph_manifest__.json")
        header.size, header.mode = len(metadata), 0o600
        archive.addfile(header, io.BytesIO(metadata))
        for name, expected in manifest.items():
            if name == "__ralph_manifest__.json":
                raise Stop("Reserved validation metadata name in candidate.")
            data = (candidate / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise Stop("Candidate changed after exact Git export.")
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), int(expected["mode"][-3:], 8)
            archive.addfile(member, io.BytesIO(data))
    volume = f"neotix-source-{os.getpid()}-{time.time_ns()}"
    helper = volume + "-populate"
    image = plan["image"]
    # The helper is controller code, not source from the candidate. It only
    # copies verified bytes/modes into the new volume; it never imports project code.
    populate = """import hashlib,json,pathlib,stat,tarfile
with tarfile.open('/candidate.tar') as archive:
    manifest=json.load(archive.extractfile('__ralph_manifest__.json'))
    for name, expected in manifest.items():
        path=pathlib.Path('/workspace')/name
        assert path.resolve().is_relative_to(pathlib.Path('/workspace'))
        data=archive.extractfile(name).read()
        assert hashlib.sha256(data).hexdigest()==expected['sha256']
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        path.chmod(int(expected['mode'][-3:],8))
        assert stat.S_IMODE(path.stat().st_mode)==int(expected['mode'][-3:],8)
"""
    results = []
    run(["docker", "image", "inspect", image], candidate, env=environment)
    run(["docker", "volume", "create", "--label", "neotix.ralph=validation", volume], candidate, env=environment)
    restrictions = ["--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=2g", "--cpus=2"]
    try:
        run(["docker", "run", "--rm", "--name", helper, *restrictions,
             "--user=0:0", "--mount", f"type=volume,source={volume},target=/workspace",
             "--mount", f"type=bind,source={archive_path},target=/candidate.tar,readonly",
             "--entrypoint=python", image, "-c", populate], candidate, env=environment,
            timeout=remaining(120), log=directory / "populate.log")
        for index, command in enumerate(plan["commands"], 1):
            print(f"CHECK {index}/{len(plan['commands'])} in isolated container", flush=True)
            name = volume + f"-check-{index}"
            argv = ["docker", "run", "--rm", "--name", name, *restrictions,
                    "--user=65534:65534", "--tmpfs=/tmp:rw,nosuid,nodev,size=256m",
                    "--mount", f"type=volume,source={volume},target=/workspace,readonly",
                    "--workdir=/workspace", "--env=HOME=/tmp", "--env=PYTHONDONTWRITEBYTECODE=1",
                    "--entrypoint", command[0], image, *command[1:]]
            try:
                result = run(argv, candidate, env=environment, timeout=remaining(plan["timeout_seconds"]),
                             log=directory / f"check-{index}.log", check=False)
            finally:
                run(["docker", "rm", "--force", name], candidate, env=environment, check=False)
            results.append({"argv": command, "image": image, "exit_code": result.returncode, "output": result.stdout})
            write_json(directory / "validation.json", results)
            if result.returncode or re.search(r"Ran 0 tests|no tests ran|collected 0 items", result.stdout):
                raise Stop("Approved validation failed or discovered no tests; no publication.")
    finally:
        run(["docker", "rm", "--force", helper], candidate, env=environment, check=False)
        run(["docker", "volume", "rm", volume], candidate, env=environment, check=False)
    return results


def inspect_changes(worktree: Path, policy: dict, paths: list[str] | None = None) -> list[str]:
    paths = changed_paths(worktree) if paths is None else paths
    if not paths:
        raise Stop("No changes; a worker claim or commit count is not implementation evidence.")
    secret = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-ant-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}")
    for name in paths:
        filename = Path(name).name.lower()
        instructions = {"agents.md", "agents.override.md", "claude.md", "claude.local.md", ".cursorrules"}
        control_parts = {".env", ".git", ".ralph", ".agents", ".claude", ".codex", ".cursor"}
        if (name.lower().startswith(tuple(p.lower() for p in CONTROL_PATHS)) or
                filename in instructions | {".gitattributes", ".gitmodules", ".gitconfig"} or
                any(part.lower() in control_parts for part in Path(name).parts)):
            raise Stop(f"Control/credential path requires supervised work: {name}")
        path = worktree / name
        if path.is_symlink() or not path.resolve().is_relative_to(worktree.resolve()):
            raise Stop(f"Changed path escapes the worktree or is a symlink: {name}")
        if not path.exists():
            continue
        data = source_bytes(path, policy["max_changed_file_bytes"])
        if len(data) > policy["max_changed_file_bytes"] or b"\0" in data:
            raise Stop(f"Large or binary artifact needs supervised publication: {name}")
        if path.name.startswith(".env") or path.suffix.lower() in (".pem", ".key", ".p12") or secret.search(data.decode("utf-8", errors="replace")):
            raise Stop(f"Potential credential in {name}; remove it and inspect locally.")
    git(worktree, "diff", "--no-ext-diff", "--no-textconv", "--check")
    return paths


class Session:
    def __init__(self, root: Path, args, policy: dict, hub: GitHub):
        self.root, self.args, self.policy, self.hub = root, args, policy, hub
        self.directory = root / ".ralph/runs" / f"issue-{args.issue}"
        self.worktree = root / ".ralph/worktrees" / f"issue-{args.issue}"
        self.state_path = self.directory / "state.json"
        self.state = {}

    def save(self, **updates):
        self.state.update(updates)
        write_json(self.state_path, self.state)

    def remaining(self, maximum: int) -> float:
        value = min(maximum, self.state["deadline"] - time.time())
        if value <= 0:
            raise Stop("Total run time budget exhausted; retained work requires a supervised handoff.")
        return value

    def live_issue(self):
        issue = self.hub.issue(self.args.issue)
        blockers = self.hub.blockers(self.args.issue)
        ac = eligible(issue, self.actor, blockers, optional=self.args.include_optional)
        # The human-readable dependency section must not introduce an unchecked
        # blocker when someone forgets to update GitHub's native graph.
        section = re.search(r"(?ims)^## Blocked by\s*\n(.*?)(?=^## |\Z)", issue.get("body") or "")
        for number in set(map(int, re.findall(r"#(\d+)", section[1] if section else ""))):
            if self.hub.issue(number).get("state") != "closed":
                raise Stop(f"Textual blocker #{number} is not closed.")
        return issue, ac

    def preflight(self):
        if self.args.backend == "cursor" and os.name == "nt":
            raise Stop("Cursor AFK requires Linux/WSL sandbox support. No issue has been claimed.")
        self.hub.verify_origin()
        repo = json.loads(self.hub.call("repo", "view", "--json", "nameWithOwner,defaultBranchRef"))
        if repo["nameWithOwner"] != self.policy["repository"] or repo["defaultBranchRef"]["name"] != self.policy["base_branch"]:
            raise Stop("Repository or default branch does not match the reviewed loop policy.")
        self.actor = self.hub.api("user")["login"]
        self.issue, self.ac = self.live_issue()
        self.parent = self.hub.issue(self.policy["parent_issue"])
        self.comments = self.hub.pages(f"repos/{self.policy['repository']}/issues/{self.args.issue}/comments?per_page=100")
        self.plan = check_plan(self.args.checks, self.args.issue)
        check_image(self.root, self.plan)
        # Repo-configured filters can execute host code while Git stages/checks
        # out files. This workflow uses unfiltered source and disables hooks.
        assert_unfiltered(self.root, git(self.root, "ls-files").splitlines())
        personal_login(self.args.backend, self.root, self.policy["workers"][self.args.backend])
        reviewer = self.policy["final_review"]
        if reviewer != {"backend": "codex", "model": "gpt-6-astra", "effort": "medium"}:
            raise Stop("Final review must use the approved GPT-6 Astra medium policy; no fallback.")
        personal_login("codex", self.root, reviewer)

    def setup(self):
        git(self.root, "fetch", "origin", self.policy["base_branch"])
        base = git(self.root, "rev-parse", f"origin/{self.policy['base_branch']}")
        # New worktrees must receive the same reviewed instructions as the team.
        # Untracked setup files must first land through a supervised setup PR.
        for path in REQUIRED + ("ralph/runner.py", "ralph/runtime.py"):
            try:
                content = git(self.root, "show", f"{base}:{path}")
            except Stop as exc:
                raise Stop(f"Loop setup must first be reviewed and merged to main: {path}") from exc
            if not (self.root / path).is_file() or content != (self.root / path).read_text(encoding="utf-8").strip():
                raise Stop(f"Loop setup must first be reviewed and merged to main: {path}")
        assert_unfiltered(self.root, git(self.root, "ls-tree", "-r", "--name-only", base).splitlines(), source=base)
        signature = digest({"policy": self.policy, "plan": self.plan, "issue": issue_snapshot(self.issue),
                            "prd": self.parent.get("body"), "comments": self.comments, "backend": self.args.backend})
        if self.state_path.exists():
            if not self.args.resume:
                raise Stop("A preserved run exists. Inspect it, then explicitly use --resume for the same issue.")
            self.state = read_json(self.state_path)
            if self.state.get("phase") == "ready_for_human":
                raise Stop("This issue already has a reviewed PR; wait for human review and merge.")
            if self.state.get("signature") != signature or self.state.get("actor") != self.actor or self.state.get("base") != base:
                raise Stop("Owner, scope, checks, model policy or main changed; supervised reconciliation and fresh review required.")
            if not self.worktree.is_dir():
                raise Stop("Saved worktree is missing; inspect before recovery.")
            remote = self.hub.ref(self.state["branch"])
            local = git(self.worktree, "rev-parse", "HEAD")
            if self.state.get("phase") == "publishing" and remote == self.state.get("candidate_head"):
                if local not in (self.state["head"], remote):
                    raise Stop("Local branch moved during interrupted publication.")
                if local != remote:
                    git(self.worktree, "update-ref", f"refs/heads/{self.state['branch']}", remote, local)
                    git(self.worktree, "read-tree", "--reset", remote)
                self.save(head=remote, phase="reviewing")
            if remote != self.state["head"]:
                raise Stop("Claim/worktree differs from saved state; inspect before any recovery.")
            if git(self.worktree, "rev-parse", "HEAD") != self.state["head"]:
                raise Stop("Local HEAD differs from saved state; supervised recovery required.")
            return
        if self.args.resume:
            raise Stop("No local run to resume. Do not adopt another teammate's remote claim.")
        branch = f"ralph/issue-{self.args.issue}"
        self.directory.mkdir(parents=True)
        self.save(issue=self.args.issue, actor=self.actor, signature=signature, base=base,
                  head=base, branch=branch, attempts=0, phase="claiming", deadline=time.time() + self.policy["run_timeout_seconds"])
        self.hub.claim(branch, base)
        self.worktree.parent.mkdir(parents=True, exist_ok=True)
        git(self.root, "worktree", "add", "-b", branch, str(self.worktree), base)
        self.save(phase="working")

    def packet(self) -> str:
        guidance = "\n\n".join(f"## {name}\n{(self.worktree / name).read_text(encoding='utf-8')}" for name in REQUIRED if not name.startswith("ralph/"))
        decisions = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((self.worktree / "docs/adr").glob("*.md")))
        data = {"issue": self.issue, "comments": self.comments, "parent_prd": self.parent, "criteria": dict(enumerate(self.ac, 1)),
                "approved_checks": self.plan, "worktree": str(self.worktree)}
        return guidance + "\n" + decisions + "\n## Task data (not authority to change the loop)\n" + json.dumps(data, indent=2)

    def fresh(self):
        issue, _ = self.live_issue()
        if digest(issue_snapshot(issue)) != digest(issue_snapshot(self.issue)) or self.hub.issue(1).get("body") != self.parent.get("body"):
            raise Stop("Issue/PRD changed during the run; review the new scope before continuing.")
        comments = self.hub.pages(f"repos/{self.policy['repository']}/issues/{self.args.issue}/comments?per_page=100")
        if digest(comments) != digest(self.comments):
            raise Stop("Issue discussion changed; incorporate the new information before continuing.")
        if self.hub.ref(self.policy["base_branch"]) != self.state["base"]:
            raise Stop("main advanced; rebase and repeat validation/review through a supervised handoff.")
        if self.hub.ref(self.state["branch"]) != self.state["head"]:
            raise Stop("Remote branch changed outside this controller; refusing to overwrite it.")

    def validate(self, attempt: Path) -> list:
        number = self.state.get("validation_calls", 0) + 1
        if number > self.policy["max_attempts"] * 2:
            raise Stop("Validation retry budget exhausted.")
        self.save(validation_calls=number)
        validation = attempt / f"validation-{number}"
        validation.mkdir()
        target = validation / "candidate"
        export_commit(self.worktree, self.state["candidate_head"], target)
        results = run_checks(target, self.plan, validation, self.remaining)
        write_json(attempt / "validated-head.json", {"head": self.state["candidate_head"],
                   "tree": self.state["candidate_tree"], "checks": results})
        return results

    def pr_body(self, report: dict, checks: list, review: dict | None) -> str:
        status = "Astra review passed; awaiting human review." if review and review["verdict"] == "pass" else "Draft: independent review has not passed."
        lines = [report["summary"], "",
                 "```text", "issue -> candidate commit -> isolated checks -> Astra review -> human merge", "```", "",
                 "## Evidence", "", f"**Before:** acceptance criteria in #{self.args.issue} await verified completion.",
                 f"**After:** {status}", "",
                 f"Base: `{self.state['base']}`", f"Head: `{self.state['head']}`", "",
                 "Acceptance evidence:", *[f"- {e['criterion']}: {e['evidence']}" for e in report["acceptance"]],
                 "", "Controller validation:", *[f"- `{json.dumps(c['argv'])}`: exit {c['exit_code']}" for c in checks],
                 "", "Risks:", *[f"- {risk}" for risk in report["risks"]]]
        if review:
            model = self.policy["final_review"]
            lines += ["", f"Independent review: `{model['model']}` / `{model['effort']}` via personal Codex login.",
                      review["summary"], *[f"- {f['severity']}: {f['location']}: {f['problem']} — {f['required_change']}" for f in review["findings"]]]
        risk = report["merge_danger"]
        lines += ["", "## Merge Danger", "", f"**Door:** {risk['door']}", "", risk["reason"], "",
                  f"**Blast Radius:** {risk['blast_radius']}"]
        lines += ["", "Human merge checklist:", "- [ ] Inspect the diff and acceptance evidence; all required CI checks pass.",
                  "- [ ] Confirm the reviewed head and base still match; new commits need a fresh review.",
                  "- [ ] Confirm resource/client gates for any empirical claims; merge only after approval."]
        body = "\n".join(lines) + "\n"
        reject_closing_directives(body)
        return f"## Summary\n\nCloses #{self.args.issue}\n\n" + body

    def verified_pr(self):
        number = int(self.state["pr"].rstrip("/").split("/")[-1])
        pr = self.hub.api(f"repos/{self.policy['repository']}/pulls/{number}")
        if (pr["state"] != "open" or pr["head"]["repo"]["full_name"] != self.policy["repository"] or
                pr["base"]["repo"]["full_name"] != self.policy["repository"] or
                pr["head"]["ref"] != self.state["branch"] or pr["base"]["ref"] != self.policy["base_branch"] or
                pr["head"]["sha"] != self.state["head"]):
            raise Stop("PR repository, branches, head or state differ from the reviewed run.")
        return pr

    def publish(self, report: dict, checks: list, review: dict | None):
        body = self.directory / "pr-body.md"
        body.write_text(self.pr_body(report, checks, review), encoding="utf-8")
        if not self.state.get("pr"):
            existing = json.loads(self.hub.call("pr", "list", "--repo", self.policy["repository"],
                "--head", self.state["branch"], "--state", "all", "--json", "url,state"))
            if existing:
                if len(existing) != 1 or existing[0]["state"] != "OPEN":
                    raise Stop("Unexpected PR state; inspect before resuming.")
                self.save(pr=existing[0]["url"])
            else:
                url = self.hub.call("pr", "create", "--repo", self.policy["repository"], "--draft",
                    "--base", self.policy["base_branch"], "--head", self.state["branch"],
                    "--title", f"#{self.args.issue}: {self.issue['title']}", "--body-file", str(body))
                self.save(pr=url)
        pr = self.verified_pr()
        if not pr["draft"] and (not review or review["verdict"] != "pass"):
            self.hub.call("pr", "ready", self.state["pr"], "--undo")
        self.hub.call("pr", "edit", self.state["pr"], "--body-file", str(body))

    def prepare_candidate(self, attempt: Path, report: dict):
        # Check PR-bound claims before publishing a commit or branch as well.
        self.pr_body(report, [], None)
        environment = private_env()
        environment["GIT_INDEX_FILE"] = str(attempt / "candidate.index")
        def indexed(*args):
            return run(git_argv(self.worktree, *args),
                       self.worktree, env=environment).stdout.strip()
        # The worker's index is untrusted. Rebuild from the saved commit in a
        # private index before discovery, ignoring worker assume-unchanged flags.
        indexed("read-tree", self.state["head"])
        paths = inspect_changes(self.worktree, self.policy, changed_paths(self.worktree, environment))
        assert_unfiltered(self.worktree, paths)
        indexed("--literal-pathspecs", "add", "--", *paths)
        staged = indexed("diff", "--cached", "--no-ext-diff", "--no-textconv")
        (attempt / "staged.diff").write_text(staged, encoding="utf-8")
        tree = indexed("write-tree")
        actual = indexed("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-only", "-z",
                         self.state["head"], tree).split("\0")
        actual = sorted(path for path in actual if path)
        if actual != paths:
            raise Stop("Candidate-tree changes differ from inspected paths.")
        inspect_changes(self.worktree, self.policy, actual)
        message = attempt / "commit-message.txt"
        commit_text = report["commit"]["subject"] + "\n\n" + report["commit"]["why"]
        reject_closing_directives(commit_text)
        message.write_text(commit_text + f"\n\nRefs #{self.args.issue}\n", encoding="utf-8")
        head = indexed("commit-tree", tree, "-p", self.state["head"], "-F", str(message))
        if git(self.worktree, "rev-parse", head + "^{tree}") != tree:
            raise Stop("Candidate commit does not match inspected tree.")
        self.save(candidate_head=head, candidate_tree=tree, source_snapshot=tree_snapshot(self.worktree),
                  phase="validating")

    def push_candidate(self):
        self.fresh()
        if tree_snapshot(self.worktree) != self.state["source_snapshot"]:
            raise Stop("Source changed since candidate creation; validation is stale.")
        head, old = self.state["candidate_head"], self.state["head"]
        git(self.worktree, "merge-base", "--is-ancestor", old, head)
        self.save(phase="publishing")  # Journal before remote mutation.
        # Exact expected-old SHA closes the check/push race. Ancestry above
        # forbids using this lease to rewrite history.
        git(self.worktree, "push", "--no-follow-tags", f"--force-with-lease=refs/heads/{self.state['branch']}:{old}",
            "origin", f"{head}:refs/heads/{self.state['branch']}")
        git(self.worktree, "update-ref", f"refs/heads/{self.state['branch']}", head, old)
        git(self.worktree, "read-tree", "--reset", head)
        self.save(head=head, phase="reviewing")

    def review_head(self, attempt: Path, report: dict, checks: list) -> dict:
        self.fresh()
        head = self.state["head"]
        self.publish(report, checks, None)
        diff = git(self.worktree, "diff", "--no-ext-diff", "--no-textconv", self.state["base"], head)
        entries = run(git_argv(self.worktree, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                              "--name-status", "-z", self.state["base"], head),
                      self.worktree, env=private_env()).stdout.split("\0")[:-1]
        if len(entries) % 2:
            raise Stop("Unexpected review file discovery response.")
        files = {}
        for status, name in zip(entries[::2], entries[1::2]):
            files[name] = "(deleted)" if status == "D" else run(
                git_argv(self.worktree, "show", f"{head}:{name}"), self.worktree, env=private_env()).stdout
        data = {"issue": self.args.issue, "base_sha": self.state["base"], "head_sha": head,
                "worker_claims": report, "controller_checks": checks, "diff": diff, "changed_files": files}
        prompt = (HERE / "prompt.md").read_text(encoding="utf-8") + "\n" + (HERE / "reviewer-prompt.md").read_text(encoding="utf-8") + "\n" + self.packet() + "\n" + json.dumps(data, indent=2)
        if len(prompt.encode()) > self.policy["max_review_bytes"]:
            raise Stop("Review packet exceeds its bound; use a supervised review.")
        call = self.state.get("review_calls", 0) + 1
        if call > self.policy["max_attempts"] + 1:
            raise Stop("Review call budget exhausted; human handoff required.")
        self.save(review_calls=call)
        response = invoke_model("codex", self.policy["final_review"], self.worktree, prompt,
                                attempt / f"review-{call}", self.remaining(self.policy["review_timeout_seconds"]), review=True)
        review = load_review(response, self.args.issue, self.state["base"], head, len(self.ac))
        write_json(attempt / f"review-{call}.json", review)
        self.fresh()
        if git(self.worktree, "rev-parse", "HEAD") != head or changed_paths(self.worktree):
            raise Stop("Tree changed during review; review is stale.")
        self.publish(report, checks, review)
        return review

    def execute(self):
        self.preflight()
        if self.args.dry_run:
            print(json.dumps({"issue": self.args.issue, "eligible": True, "actor": self.actor,
                              "worker": self.policy["workers"][self.args.backend],
                              "reviewer": self.policy["final_review"], "checks": self.plan,
                              "next": "A real run creates one remote claim and an isolated worktree."}, indent=2))
            return
        self.setup()
        while True:
            self.remaining(1)
            phase = self.state["phase"]
            if phase not in ("validating", "publishing", "reviewing"):
                if self.state["attempts"] >= self.policy["max_attempts"]:
                    raise Stop("Repair attempt budget exhausted; draft/worktree retained for handoff.")
                self.fresh()
                number = self.state["attempts"] + 1
                attempt = self.directory / f"attempt-{number}"
                attempt.mkdir()
                self.save(attempts=number, phase="working")
                receipt = worker_area(self.worktree) / "worker.json"
                receipt.unlink(missing_ok=True)
                prompt = (HERE / "prompt.md").read_text(encoding="utf-8") + "\n" + (HERE / "worker-prompt.md").read_text(encoding="utf-8") + "\n" + self.packet()
                if self.state.get("feedback"):
                    prompt += "\n## Findings to resolve\n" + json.dumps(self.state["feedback"], indent=2)
                if len(prompt.encode()) > self.policy["max_review_bytes"]:
                    raise Stop("Worker context exceeds its bound; supervised scope reduction required.")
                invoke_model(self.args.backend, self.policy["workers"][self.args.backend], self.worktree,
                             prompt, attempt / "worker", self.remaining(self.policy["worker_timeout_seconds"]))
                if git(self.worktree, "rev-parse", "HEAD") != self.state["head"]:
                    raise Stop("Worker committed or moved HEAD; inspect manually.")
                worker_area(self.worktree)
                report = load_worker(receipt, self.args.issue, len(self.ac))
                write_json(attempt / "worker.json", report)
                self.prepare_candidate(attempt, report)
            attempt = self.directory / f"attempt-{self.state['attempts']}"
            report = read_json(attempt / "worker.json")
            if self.state["phase"] == "validating":
                self.validate(attempt)
                self.save(phase="publishing")
            evidence = read_json(attempt / "validated-head.json")
            if evidence["head"] != self.state["candidate_head"] or evidence["tree"] != self.state["candidate_tree"]:
                raise Stop("Validation evidence does not match candidate commit.")
            checks = evidence["checks"]
            if self.state["phase"] == "publishing":
                self.push_candidate()
            review = self.review_head(attempt, report, checks)
            if review["verdict"] == "pass":
                self.fresh()
                pr = self.verified_pr()
                if pr["head"]["sha"] != self.state["head"] or pr["base"]["sha"] != self.state["base"]:
                    raise Stop("PR head/base changed before readiness; review is stale.")
                if pr["draft"]:
                    self.hub.call("pr", "ready", self.state["pr"])
                self.save(phase="ready_for_human", review=review)
                print(f"READY FOR HUMAN: {self.state['pr']}\nIssue remains open; this loop never merges.")
                return
            self.save(feedback=review, phase="changes_requested")
            if review["verdict"] == "blocked":
                raise Stop("Reviewer needs missing evidence or a human decision. Draft PR preserved.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("codex", "cursor", "claude"))
    parser.add_argument("--issue", type=int)
    parser.add_argument("--checks", type=Path, help="Human-approved JSON argument arrays for this issue")
    parser.add_argument("--dry-run", action="store_true", help="Read-only GitHub/auth preflight; no model or claim")
    parser.add_argument("--doctor", action="store_true", help="Read-only model/login checks; no issue needed")
    parser.add_argument("--resume", action="store_true", help="Continue this machine's saved run within its original budgets")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args(argv)
    session = None
    try:
        policy = read_json(HERE / "policy.json")
        if args.doctor:
            personal_login(args.backend, ROOT, policy["workers"][args.backend])
            personal_login("codex", ROOT, policy["final_review"])
            print(json.dumps({"worker": policy["workers"][args.backend], "reviewer": policy["final_review"],
                              "authentication": "personal login checks passed; model execution not tested"}, indent=2))
            return 0
        if not args.issue or args.issue <= 1 or not args.checks:
            parser.error("--issue NUMBER (>1) and --checks PATH are required")
        session = Session(ROOT, args, policy, GitHub(ROOT, policy["repository"]))
        with issue_lock(ROOT, args.issue):
            session.execute()
        return 0
    except (Stop, OSError, ValueError, KeyError, TypeError) as exc:
        if session and session.state:
            session.save(last_stop=str(exc))
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
