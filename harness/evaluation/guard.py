"""Episode-level evaluation checks; no planner can reset or write memory."""

from harness.memory import MemorySnapshot


class SingleResetEnvironment:
    def __init__(self, environment):
        self._environment = environment
        self._reset_used = False

    def reset(self, seed):
        if self._reset_used:
            raise RuntimeError("evaluation_reset_forbidden")
        self._reset_used = True  # An ambiguous reset must not be retried either.
        return self._environment.reset(seed)

    def __getattr__(self, name):
        return getattr(self._environment, name)


def episode_guard(frozen, *, root, seed, settings, planner, model_override, memory_path,
                  manifest_path, robotwin_root):
    from pathlib import Path
    from .manifest import FrozenEvaluation
    if (seed not in frozen.settings.seeds or settings != frozen.settings.episode
        or planner != frozen.settings.planner or model_override is not None
        or Path(robotwin_root).resolve() != Path(root).resolve() / "RoboTwin"):
        raise ValueError("episode_differs_from_frozen_evaluation")
    expected_memory = MemorySnapshot(frozen.memory).sha256

    def check():
        FrozenEvaluation.load(manifest_path, expected_sha256=frozen.digest)
        frozen.verify(root)
        MemorySnapshot.load(memory_path, expected_sha256=expected_memory)
    check()
    return check
