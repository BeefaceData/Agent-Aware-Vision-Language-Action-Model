"""Isolate each native simulator episode in a new process on Linux.

CUDA/native state and planner, predicate and anchor instances die between attempts.
The existing executor resets remote policy history on its first VLA invocation.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys


@dataclass(frozen=True)
class AttemptExit:
    returncode: int | None
    timed_out: bool = False


class ProcessAttemptRunner:
    execution_kind = "live"

    def __init__(self, *, robotwin_root, planner="scripted", model=None, memory=None, evaluation=None):
        self.root = Path(__file__).resolve().parents[2]
        self.robotwin_root = Path(robotwin_root).resolve()
        self.planner, self.model, self.memory = planner, model, memory
        self.evaluation = evaluation

    def __call__(self, *, seed, config, output, timeout):
        command = [sys.executable, str(self.root / "run_skills.py"),
            "--seed", str(seed), "--config", str(config), "--output", str(output),
            "--robotwin-root", str(self.robotwin_root), "--planner", self.planner,
            "--development-oracle"]
        if self.model:
            command.extend(["--anthropic-model", self.model])
        if self.memory:
            path, digest = self.memory
            command.extend(["--memory-snapshot", str(path), "--memory-sha256", digest])
        if self.evaluation:
            path, digest = self.evaluation
            command.extend(["--evaluation-manifest", str(path), "--evaluation-sha256", digest])
        # Inherit credentials normally; never source bashrc, invoke a shell, copy
        # environment values into artifacts, or capture unsanitized SDK stderr.
        process = subprocess.Popen(command, cwd=self.root, start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            return AttemptExit(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            return AttemptExit(None, timed_out=True)
        finally:
            # Also reap orphaned native helper processes after a normal exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
