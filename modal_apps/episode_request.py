"""Translate ``run_skills.py`` arguments into one remote Modal episode request.

Argument and run-folder handling with no Modal import, so it is testable
without network or GPU. ``modal_apps/simulator.py`` owns the uploads, the remote
call and the download back into ``--output``.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Local files the container cannot see; each is uploaded under this base name.
UPLOADED_OPTIONS = {"--config": "config", "--memory-snapshot": "memory_snapshot"}
# Frozen evaluation hashes the local checkout, so it stays local for now.
UNSUPPORTED_OPTIONS = ("--evaluation-manifest", "--evaluation-sha256", "--robotwin-root")

# Launcher-only options; never forwarded to run_skills.py.
TIME_LIMIT_OPTION = "--time-limit-minutes"
SKIP_OBSERVATIONS_OPTION = "--skip-observations"
DEFAULT_TIME_LIMIT_MINUTES = 90
# Leaves headroom under the remote function's fixed timeout for shutdown and upload.
MAX_TIME_LIMIT_MINUTES = 210


@dataclass(frozen=True)
class EpisodeRequest:
    run_id: str
    local_output: Path
    local_argv: tuple[str, ...]      # run_skills.py arguments as given, for local validation
    remote_argv: tuple[str, ...]     # the same arguments rewritten for the container
    uploads: dict[str, Path]         # file name in the container's input directory -> local file
    time_limit_minutes: int
    skip_observations: bool


def _split_assignments(argv):
    """Turn ``--option=value`` into ``--option value`` so every option is one token."""
    tokens = []
    for token in argv:
        if token.startswith("--") and "=" in token:
            tokens.extend(token.split("=", 1))
        else:
            tokens.append(token)
    return tokens


def build_request(argv, *, stamp, input_dir, runs_dir, robotwin_root):
    """Return the ``EpisodeRequest`` for ``argv``; raise ``ValueError`` on unusable options.

    ``stamp`` makes the remote run directory unique, so reusing a local name
    never collides with an earlier run on the Volume.
    """
    tokens = _split_assignments(argv)
    local_argv, remote_argv, uploads = [], [], {}
    output = None
    time_limit, skip_observations = DEFAULT_TIME_LIMIT_MINUTES, False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in UNSUPPORTED_OPTIONS:
            raise ValueError(f"{token} is not supported for Modal episodes")
        if token == SKIP_OBSERVATIONS_OPTION:
            skip_observations = True
            index += 1
            continue
        if token not in (*UPLOADED_OPTIONS, "--output", TIME_LIMIT_OPTION):
            local_argv.append(token)
            remote_argv.append(token)
            index += 1
            continue
        if index + 1 >= len(tokens):
            raise ValueError(f"{token} needs a value")
        value = tokens[index + 1]
        index += 2
        if token == TIME_LIMIT_OPTION:
            time_limit = int(value)
        elif token == "--output":
            output = Path(value)
            local_argv.extend((token, value))
        else:
            local = Path(value)
            name = UPLOADED_OPTIONS[token] + local.suffix
            uploads[name] = local
            local_argv.extend((token, value))
            remote_argv.extend((token, str(PurePosixPath(input_dir) / name)))

    if output is None:
        raise ValueError("--output is required")
    if not 1 <= time_limit <= MAX_TIME_LIMIT_MINUTES:
        raise ValueError(f"{TIME_LIMIT_OPTION} must be between 1 and {MAX_TIME_LIMIT_MINUTES}")
    run_id = f"{output.name}-{stamp}"
    remote_argv.extend(("--output", str(PurePosixPath(runs_dir) / run_id),
                        "--robotwin-root", robotwin_root))
    return EpisodeRequest(run_id=run_id, local_output=output, local_argv=tuple(local_argv),
                          remote_argv=tuple(remote_argv), uploads=uploads,
                          time_limit_minutes=time_limit, skip_observations=skip_observations)


def set_aside_previous_attempt(output):
    """Rename a run folder left by an earlier attempt; return its new path, or None.

    Modal restarts a preempted function with the same input, so the retry finds
    the interrupted attempt's folder. Keeping it (as ``<name>.preempted-N``) lets
    the retry start a clean episode without losing the partial artifacts.
    """
    output = Path(output)
    if not output.exists():
        return None
    attempt = 1
    while output.with_name(f"{output.name}.preempted-{attempt}").exists():
        attempt += 1
    target = output.with_name(f"{output.name}.preempted-{attempt}")
    output.rename(target)
    return target
