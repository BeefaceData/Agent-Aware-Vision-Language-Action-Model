"""CLI adapters. No shell evaluation, model fallback, or credential logging."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time


class Stop(RuntimeError):
    """A failed gate; preserve work and require an explicit next action."""


def worker_area(worktree: Path) -> Path:
    """Reject links/special files before controller access to worker-owned context."""
    area = worktree / ".ralph-run"
    def inspect(path):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or
                getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0) or
                not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)) or
                (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)):
            raise Stop("Worker context contains a link or special file; supervised handoff required.")
    inspect(worktree)
    if area.exists() or area.is_symlink():
        inspect(area)
    else:
        area.mkdir()
    if not area.is_dir() or area.resolve().parent != worktree.resolve():
        raise Stop("Worker context escapes its worktree.")
    for current, directories, files in os.walk(area, followlinks=False):
        for name in directories + files:
            inspect(Path(current) / name)
    return area


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise Stop(f"Missing or invalid JSON: {path}") from exc


def private_env() -> dict[str, str]:
    """Use personal CLI logins, never inherited API/cloud/GitHub credentials.

    This is environment hygiene, not isolation from files in the user's home.
    """
    environment = dict(os.environ)
    for key in list(environment):
        upper = key.upper()
        if any(word in upper for word in ("TOKEN", "SECRET", "API_KEY", "PASSWORD")) or upper.startswith(
            ("ANTHROPIC_", "CLAUDE_CODE_USE_", "OPENAI_", "AWS_", "AZURE_", "GOOGLE_APPLICATION_", "GH_", "GITHUB_")
        ):
            environment.pop(key)
        if upper in ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "GIT_ASKPASS", "SSH_ASKPASS"):
            environment.pop(key, None)
        if upper.startswith("GIT_"):
            environment.pop(key, None)
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GH_PROMPT_DISABLED": "1"})
    return environment


def git_argv(root: Path, *args: str) -> list[str]:
    """Disable host-code hooks/drivers before touching any worker-controlled file."""
    prefix = ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
              "-c", f"core.attributesFile={os.devnull}"]
    keys = subprocess.run([*prefix, "config", "--name-only", "--get-regexp",
                           r"^filter\..*\.(clean|smudge|process|required)$"],
                          cwd=root, env=private_env(), capture_output=True, encoding="utf-8", timeout=60)
    if keys.returncode not in (0, 1):
        raise Stop("Cannot inspect configured Git drivers safely.")
    drivers = {key.rsplit(".", 1)[0] for key in keys.stdout.splitlines()}
    for driver in sorted(drivers):
        for setting in ("clean=", "smudge=", "process=", "required=false"):
            prefix += ["-c", f"{driver}.{setting}"]
    return [*prefix, *args]


def executable(name: str) -> list[str]:
    # Node entry points avoid .cmd shell quoting on Windows, including paths
    # containing spaces. Executable overrides are paths, never shell fragments.
    override = os.environ.get("RALPH_" + name.upper() + "_BIN")
    candidates = [override] if override else []
    aliases = ("agent", "cursor-agent") if name == "cursor" else (name,)
    if os.name != "nt":
        candidates += [str(Path.home() / ".local/bin" / alias) for alias in aliases]
    candidates += [shutil.which(alias) for alias in aliases]
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        roaming = Path(os.environ.get("APPDATA", ""))
        if name == "cursor":
            versions = local / "cursor-agent/versions"
            if versions.is_dir() and not override:
                for version in sorted(versions.iterdir(), reverse=True):
                    if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}(?:-\d{2}-\d{2}-\d{2})?-[a-f0-9]+", version.name) and (version / "node.exe").is_file() and (version / "index.js").is_file():
                        return [str(version / "node.exe"), str(version / "index.js")]
            candidates += [str(local / "cursor-agent" / "agent.cmd")]
        if name == "codex":
            js = roaming / "npm/node_modules/@openai/codex/bin/codex.js"
            if js.exists() and not override:
                return [shutil.which("node") or "node", str(js)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            path = Path(candidate)
            if path.suffix.lower() == ".ps1":
                path = path.with_suffix(".cmd")
            if path.suffix.lower() in (".cmd", ".bat"):
                # Cursor ships a native executable next to its wrappers in some
                # releases; otherwise invoke the wrapper through a fixed PS file.
                return ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
                        str(Path(__file__).with_name("invoke-cli.ps1")), str(path)]
            return [str(path)]
    raise Stop(f"Missing {name} CLI. Install it in the same OS environment as this launcher.")


def run(argv: list[str], cwd: Path, *, timeout: float = 60, text: str | None = None,
        env: dict | None = None, log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if timeout <= 0:
        raise Stop("Run time budget exhausted.")
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", **kwargs)
    try:
        output, _ = proc.communicate(text, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # Stop descendants as well as the CLI wrapper. A stopped controller must
        # not leave an unattended model editing the worktree.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        output, _ = proc.communicate()
        if log:
            log.write_text(output, encoding="utf-8")
        raise Stop("Command interrupted or timed out; process tree stopped. See local logs.")
    if log:
        log.write_text(output, encoding="utf-8")
    if check and proc.returncode:
        # Do not echo arbitrary subprocess output: it can contain credentials.
        raise Stop(f"{Path(argv[0]).name} failed (exit {proc.returncode})." +
                   (f" Inspect {log} locally." if log else " Inspect authentication and command prerequisites."))
    return subprocess.CompletedProcess(argv, proc.returncode, output, "")


class GitHub:
    def __init__(self, root: Path, repository: str):
        self.root, self.repository = root, repository

    def call(self, *args: str) -> str:
        return run(executable("gh") + list(args), self.root, env=private_env()).stdout.strip()

    def verify_origin(self) -> None:
        allowed = {f"https://github.com/{self.repository}", f"git@github.com:{self.repository}",
                   f"ssh://git@github.com/{self.repository}"}
        for options in (("--all",), ("--all", "--push")):
            urls = run(git_argv(self.root, "remote", "get-url", *options, "origin"), self.root, env=private_env()).stdout.splitlines()
            if not urls or any(url.removesuffix(".git").rstrip("/") not in allowed for url in urls):
                raise Stop("Origin fetch/push URLs must both identify the exact repository in policy.json.")

    def api(self, endpoint: str, *args: str):
        output = self.call("api", endpoint, *args)
        return json.loads(output) if output else None

    def issue(self, number: int):
        return self.api(f"repos/{self.repository}/issues/{number}")

    def pages(self, endpoint: str) -> list:
        # Pagination errors remain errors. They are never interpreted as no blockers.
        pages = json.loads(self.call("api", "--paginate", "--slurp", endpoint))
        if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
            raise Stop("Unexpected GitHub paginated response.")
        return [item for page in pages for item in page]

    def blockers(self, number: int) -> list:
        return self.pages(f"repos/{self.repository}/issues/{number}/dependencies/blocked_by?per_page=100")

    def ref(self, branch: str) -> str:
        return self.api(f"repos/{self.repository}/git/ref/heads/{branch}")["object"]["sha"]

    def claim(self, branch: str, base: str) -> None:
        # GitHub ref creation is atomic across machines; existing claim = failure.
        self.api(f"repos/{self.repository}/git/refs", "-X", "POST",
                 "-f", f"ref=refs/heads/{branch}", "-f", f"sha={base}")


def model_command(backend: str, model: dict, worktree: Path, last: Path,
                  *, review: bool = False) -> list[str]:
    if backend == "codex":
        windows = ["-c", 'windows.sandbox="elevated"'] if os.name == "nt" else []
        return executable("codex") + windows + ["--ask-for-approval", "never", "exec",
            "--ignore-user-config", "--ephemeral", "--sandbox", "read-only" if review else "workspace-write",
            "--model", model["model"], "-c", f'model_reasoning_effort="{model["effort"]}"',
            "--cd", str(worktree), "--json", "--output-last-message", str(last), "-"]
    if backend == "cursor":
        if review:
            raise Stop("Cursor final review is not configured; use the approved Astra reviewer.")
        if os.name == "nt":
            raise Stop("This Cursor CLI cannot sandbox native Windows. Run cursor-afk.sh in Linux/WSL with a Linux Cursor login; do not disable sandboxing.")
        return executable("cursor") + ["--print", "--output-format", "json", "--trust",
            "--force", "--sandbox", "enabled", "--workspace", str(worktree), "--model", model["cli_model"]]
    if backend == "claude" and not review:
        return executable("claude") + ["--print", "--bare", "--no-session-persistence",
            "--output-format", "json", "--model", model["model"], "--effort", model["effort"],
            "--permission-mode", "dontAsk", "--tools", "Read,Edit,Write,Glob,Grep",
            "--allowedTools", "Read,Edit,Write,Glob,Grep"]
    raise Stop(f"Unapproved backend or review path: {backend}")


def personal_login(backend: str, root: Path, model: dict) -> None:
    environment = private_env()
    if backend == "codex":
        result = run(executable("codex") + ["login", "status"], root, env=environment)
        if "ChatGPT" not in result.stdout:
            raise Stop("Codex must use a personal ChatGPT login, not an API key.")
    elif backend == "cursor":
        result = run(executable("cursor") + ["--list-models"], root, env=environment)
        if not any(line.split(" - ")[0].strip() == model["cli_model"] for line in result.stdout.splitlines()):
            raise Stop(f"Cursor login does not advertise {model['cli_model']}; no fallback allowed.")
    elif backend == "claude":
        status = json.loads(run(executable("claude") + ["auth", "status"], root, env=environment).stdout)
        if not status.get("loggedIn") or status.get("authMethod") != "claude.ai":
            raise Stop("Claude coding requires a personal claude.ai login; client API keys are reserved for VLA supervision.")


def invoke_model(backend: str, model: dict, worktree: Path, prompt: str,
                 directory: Path, timeout: float, *, review: bool = False) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.md").write_text(prompt, encoding="utf-8")
    last = directory / "last-message.txt"
    if last.exists():
        raise Stop("Refusing to overwrite an earlier invocation's evidence.")
    argv = model_command(backend, model, worktree, last, review=review)
    stdin = prompt
    if backend in ("cursor", "claude"):
        # Cursor documents a positional prompt. A short file pointer also avoids
        # Windows command-line limits and keeps task text out of process listings.
        context = worker_area(worktree) / "worker-context.md"
        context.unlink(missing_ok=True)
        # Exclusive creation never truncates an existing symlink or hardlink.
        with context.open("x", encoding="utf-8") as stream:
            stream.write(prompt)
        argv += ["Read .ralph-run/worker-context.md in full and carry out that task. It contains your scope, instructions and completion contract."]
        stdin = None
    write_json(directory / "invocation.json", {"backend": backend, "requested_model": model,
               "review": review, "started_at": time.time(), "timeout_seconds": timeout})
    print(f"{'REVIEW' if review else 'WORK'} {backend} {model['model']} {model['effort']}", flush=True)
    result = run(argv, worktree, timeout=timeout, text=stdin, env=private_env(),
                 log=directory / "output.log")
    if backend == "codex":
        if not last.is_file():
            raise Stop("Codex returned without a final response.")
        return last.read_text(encoding="utf-8").strip()
    envelope = json.loads(result.stdout)
    if envelope.get("is_error") or envelope.get("type") == "error":
        raise Stop("Agent returned an error result; see local logs.")
    answer = envelope.get("result", "")
    last.write_text(answer if isinstance(answer, str) else json.dumps(answer), encoding="utf-8")
    return answer
