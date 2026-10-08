# Baseline startup and preflight

This check verifies the installed baseline interfaces and a selected LIBERO task
and initial state. It does not download model weights, create a simulator, execute
an action, or measure task success. Live inference requires the separate resource
and data-use allowance in PRD #1 and issue #55.

## Declared setup

Use Linux, Python 3.10 and the repository root. The [README setup](../README.md#setup)
records the original Conda and FFmpeg commands. The following isolated CPU setup
was checked on 2026-10-08 in Ubuntu under WSL; it can validate imports without a
CUDA installation. Use a fresh virtual environment, outside the repository:

```bash
python3.10 -m venv /path/to/neotix-baseline-preflight
/path/to/neotix-baseline-preflight/bin/python -m pip install uv
PREFLIGHT=/path/to/neotix-baseline-preflight
"$PREFLIGHT/bin/uv" pip install --python "$PREFLIGHT/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.7.1' 'torchvision==0.22.1'
"$PREFLIGHT/bin/uv" pip install --python "$PREFLIGHT/bin/python" \
  'lerobot[smolvla,libero]==0.4.3'
"$PREFLIGHT/bin/python" -m pip check
```

The CPU wheel is for preflight only. A CUDA episode needs a separately checked
CUDA-capable installation and the applicable run allowance. Keep the exact
resolved versions from the environment you actually use; the requirements above
do not lock all transitive packages.

On first import, `hf-libero` asks where to put datasets. For a noninteractive
check, initialize its config explicitly. This accepts the package defaults; it
does not download a dataset. Keep `LIBERO_CONFIG_PATH` set for subsequent checks:

```bash
export LIBERO_CONFIG_PATH="$PREFLIGHT/libero-config"
printf 'n\n' | "$PREFLIGHT/bin/python" -c 'import libero.libero'
```

From the repository root, check the same imports and task/state selector used by
`run_smolvla_episode.py`:

```bash
"$PREFLIGHT/bin/python" - <<'PY'
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium
import imageio
import numpy
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import create_libero_envs
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.random_utils import set_seed

from baseline_environment import capture_environment
from libero_initial_state import select_initial_state
from pinned_policy import POLICY_REPOSITORY, POLICY_REVISION

with TemporaryDirectory() as directory:
    runtime = capture_environment(Path(directory) / 'environment.json', 'cpu')
selection = select_initial_state('libero_10', 0, 0)
print('preflight:', runtime['preflight']['status'])
print('imports: passed')
print('versions:', {name: version(name) for name in
    ('lerobot', 'torch', 'gymnasium', 'hf-libero', 'robosuite', 'mujoco')})
print('policy:', POLICY_REPOSITORY, POLICY_REVISION)
print('selection:', selection.identity())
PY
```

Expected: `preflight: passed`, `imports: passed`, the pinned policy revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, and a selection with
`suite: libero_10`, `task_id: 0`, `state_id: 0`, and a nonempty SHA-256 digest.
The digest identifies the selected catalog state; simulator application and
readback are verified only during a permitted episode reset.

The matching **episode invocation**, after its resource and data-use gates are
established, is:

```bash
python run_smolvla_episode.py --suite libero_10 --task-id 0 \
  --initial-state-id 0 --seed 0 --device cuda
```

The runner verifies pinned policy and backbone files before loading the model.
Its `environment.json`, `policy-assets.json`, `attempt.json`, `result.json`, and
sealed `replay/` have distinct roles; see [asset resolution](pinned-baseline-assets.md)
and [attempt identity](baseline-attempt-identity.md). A preflight pass does not
prove that model loading, rendering, reset, or inference will succeed.

## Observed clean-environment result

On 2026-10-08, a fresh WSL Python 3.10.22 virtual environment completed the
commands above. `pip check` reported `No broken requirements found.` The resolved
versions included LeRobot 0.4.3, PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu,
Gymnasium 1.4.0, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, Transformers
4.57.6, NumPy 2.2.6 and imageio 2.38.0. The imports and selection passed.
`capture_environment` reported `preflight: passed`, resolved device `cpu`, and
11 installed robosuite controller Python source hashes. Task 0, state 0 selected
SHA-256 `747b653b2ca1feeae8f3bd994880ce743730993ea5ce4bd003623e877f26b3bd`.
No policy assets, inference, or simulator episode were run in this check.

## Failure diagnosis

| Observation | Meaning and next check |
|---|---|
| `Unsupported LeRobot version None` or import error | The command used another interpreter or an incomplete environment. Check `which python`, `python -m pip check`, and `python -m pip show lerobot`. |
| `Do you want to specify a custom path...` followed by `EOFError` | `hf-libero` has no config for this interpreter's `LIBERO_CONFIG_PATH`. Run the one-time config command above. |
| `CUDA unavailable in this interpreter` | CPU preflight must request `cpu`; a CUDA episode requires a CUDA-capable environment. |
| `Cannot identify installed robosuite controller sources` | The installed controller package cannot supply source identity; repair that installation before execution. |
| Task or initial-state ID out of range | Inspect the selected suite and catalog. Changing the seed does not select another initial-state index. |
| Missing or mismatched policy asset | Asset resolution failed before model execution. See the pinned-asset guide; do not treat this as a policy task failure. |

A failed setup check has no task outcome or denominator. A later `result.json`
with `status: completed` and `success: false` is an unsuccessful episode, not a
setup failure. The preflight record alone supports neither a baseline success
rate nor any supervisor improvement claim.
