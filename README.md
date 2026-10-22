# Agent-Aware Vision-Language-Action Model

A runtime supervision harness for the frozen Neotix policy on RoboTwin's
`hanging_mug` task. RoboTwin and CuRobo run locally; policy inference runs on
Modal. The optional Anthropic planner selects one bounded skill per turn and
never generates joint actions. This project does not retrain the policy.

The baseline inference path remains available independently. The supervised
runtime adds typed skills, public RGB-D perception, task predicates, structured
logs, and offline replay tools. Live exploration is still in progress; success
of individual phases is not evidence of reliable full-task hanging.

For the component map, data flow, perception boundary, and contributor
workflow, see [CONTRIBUTE_README.md](CONTRIBUTE_README.md).

## Setup

Use Linux, Python 3.10, and an NVIDIA GPU with Vulkan support. Allow space for
RoboTwin's simulator assets. Run these commands from the repository:

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
for platform dependencies and SAPIEN/mplib compatibility patches. Keep local
NumPy at 1.26.4; the Modal policy worker uses its own Python 3.12 environment.

Deploy the frozen policy worker:

```bash
modal setup
modal deploy modal_policy_chunked.py
```

For Anthropic planning, export `ANTHROPIC_API_KEY` in your environment. Never
commit it. For optional SAM 3 perception, request gated Hugging Face model
access, configure a Modal secret named `huggingface` with `HF_TOKEN`, then run
`modal deploy modal_sam3.py`. SAM 3 dependencies and weights stay on the Modal
T4 worker, separate from the local simulator.

## Run and test

Run CPU-only unit and scenario checks before starting the simulator:

```bash
python -m unittest discover -s tests -q
```

One bounded scripted exploration episode:

```bash
python run_exploration.py --config configs/explore_natural_acquisition.yaml \
  --output data/runs/acquisition_01
```

One Anthropic development episode with public RGB-D perception:

```bash
python run_skills.py \
  --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml \
  --planner anthropic \
  --perception-factory harness.perception.modal_sam3:create_detector \
  --output data/runs/full_task_01
```

Development-oracle mode is an alternative for internal debugging. Do not pass
it together with a public perception factory. Use a unique output directory for
each run. Run RoboTwin jobs serially on machines with limited GPU VRAM.

For the frozen-policy baseline, start `python run_inference_chunked.py sim` in
one terminal and `python run_inference_chunked.py policy --episodes 1` in
another. Run outputs, checkpoints, videos, observations, and validation reports
are local under ignored `data/`; implementation and validation notes under
ignored `docs/` are not part of the remote repository.
