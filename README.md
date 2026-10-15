# Agent-Aware Vision-Language-Action Model

A runtime harness for a **frozen Neotix policy** on RoboTwin's `hanging_mug` task.
RoboTwin and CuRobo run locally; policy inference runs on Modal. An Anthropic
planner selects one bounded skill per turn: `MOVE_TO`, `ROTATE_WRIST`,
`SET_GRIPPER`, `RELEASE`, `VLA_ACT`, or `ABORT`. It never supplies joint actions.
There is no policy retraining.

Implemented: baseline-compatible supervision, checkpoint/replay tooling,
measured-start analytic skills, stop predicates, structured logs, task/global
memory, bounded exploration, frozen evaluation, and saved-turn planner diagnostics.
Legacy recovery prototypes remain separate from the active skill registry.

**Status:** live staging, retreat and Modal calls work. The current planner runner
targets grasp/lift only; acquisition and full hanging are not validated. Two live
attempts aborted on invalid planner responses. Development grounding still uses
explicit simulator oracles, withheld from planner input.

## Setup

Use Linux, Python 3.10, an NVIDIA GPU with Vulkan support, and a CUDA toolkit/C++
compiler compatible with PyTorch 2.8. Allow space for RoboTwin's roughly 16 GB of
assets. Run commands from this repository after cloning it.

```bash
conda create -n robotwin python=3.10 -y
conda activate robotwin
python -m pip install -r requirements-sim.txt -r requirements-planner.txt 'modal>=1,<2'
git clone https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
git -C RoboTwin checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
git -C RoboTwin apply ../patches/robotwin-renderer.patch
(cd RoboTwin && bash scripts/_download_assets.sh)
git clone --branch v0.7.8 https://github.com/NVlabs/curobo.git RoboTwin/envs/curobo
git -C RoboTwin/envs/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
python -m pip install -e RoboTwin/envs/curobo --no-build-isolation
python -m pip install warp-lang==1.12.0
```

Follow [RoboTwin's installation guidance](https://github.com/RoboTwin-Platform/RoboTwin)
for platform dependencies and its SAPIEN/mplib compatibility patches. The included
renderer patch reproduces our disabled OIDN denoiser. Keep local NumPy at 1.26.4;
the policy's separate Python 3.12/NumPy 2 environment is built on Modal.

```bash
modal setup
modal deploy modal_policy_chunked.py
```

The worker downloads `Neotix-Robotics/robotwin-hanging-mug-dit` into its Modal
volume. Set `ANTHROPIC_API_KEY` securely in your environment; never commit it.
Select an exact model ID available to your account (`claude-opus-5` was used locally).

## Run

```bash
# One development episode; use a new output directory each time.
python run_skills.py --seed 3 --planner anthropic --anthropic-model claude-opus-5 \
  --development-oracle --output data/runs/acquisition_01

# Bounded scripted exploration; add planner/model flags to use Anthropic.
python run_exploration.py --development-oracle --output data/exploration/run_01

# CPU tests; no API calls or simulator startup.
python -m unittest discover -s tests -q
```

For the frozen-policy baseline, start `python run_inference_chunked.py sim` in
one terminal and `python run_inference_chunked.py policy --episodes 1` in another.
Use `python -m development.planner_replay --help` for diagnostics without motion.
For evaluation, declare development seeds in `configs/evaluation.yaml`, freeze
with `run_evaluation.py freeze`, then run the exact returned digest; `--help`
lists required arguments. Local frozen manifests must be regenerated after changes.

Code is organized under `harness/` (runtime), `development/` (offline tools),
`configs/` and `tests/`. Weights, videos, simulator assets, run data and development
notes stay local. `harness/prompts/*.md` remain tracked runtime inputs.
