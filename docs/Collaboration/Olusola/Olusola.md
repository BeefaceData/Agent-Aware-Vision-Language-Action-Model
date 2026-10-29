# Olusola — perception and object grounding

## Goal

Evaluate and improve public perception of the mug, rack, table support, and relevant poses. Focus on whether RGB-D evidence can ground useful 3D targets and support phase predicates across reset, grasp, handoff, and hanging views. Do not use simulator object poses as planner input or as a substitute for perception evidence.

## Where to work

- `harness/perception/observations.py`, `rgbd.py`, and `geometry.py`: public observations, depth/calibration contracts, and camera/world transforms.
- `grounding.py`, `pose_grounding.py`, `opus_frame.py`, `opus_rack.py`, `opus_mug_state.py`: semantic detections, image-point reports, and RGB-D grounding.
- `table_plane.py`, `hang_targets.py`, and `anchors.py`: table-relative lift evidence, cached rack references, and symbolic target resolution.
- `harness/runtime/assembly.py` and `run_skills.py`: provider setup and episode-start grounding.
- Replays and tests: `development/replays/`, `tests/test_perception_grounding.py`, `test_rgbd_perception.py`, `test_hang_targets.py`, `test_opus_frame_grounding.py`, and `test_table_plane_lift.py`.

## Camera context

The local RoboTwin checkout on Thabo’s workstation has an additional fixed camera named `side_camera`, using the standard 37° D435 profile at 320×240, aimed diagonally down across the workspace. It was added under `RoboTwin/assets/embodiments/aloha-agilex/config.yml`, but `RoboTwin/` is git-ignored; this camera is **not currently guaranteed to exist in a clean clone or on the remote branch**. Do not assume it appears in `Observation` or the planner’s camera map. First verify `get_obs()` output and `harness/environment/layout.py`; then propose a tracked adapter/config or a patch under `patches/` so teammates can reproduce it.

Use the head and wrist cameras already in the public observation contract as the reproducible baseline. For any new camera, record its identifier, resolution, calibration hash, depth alignment, and view coverage. Keep RGB sent to semantic models separate from same-frame depth/calibration used locally.

## Evaluation approach

1. Start with saved public RGB-D frames or synthetic scenarios; inspect RGB, depth validity, calibration, and projected points together.
2. Evaluate mug/rack support points and 3D errors against development labels only in an offline evaluator. Keep oracle labels out of runtime planner context.
3. Include hard cases: hollow mug opening, partial occlusion, grippers crossing, rack wire/background depth boundaries, missing depth, and camera changes.
4. Preserve `unknown` when a view, mask, or depth sample is ambiguous. Do not lower confidence thresholds solely to make predicates pass; report precision/recall and false-positive cases.
5. Run the focused replay/unit tests, then the CPU suite. Coordinate before any live simulator run; GPU jobs must be serial.

## Change boundaries

Do not add SAM 3 or other heavy vision models to the local simulator environment. The SAM 3 client is `harness/perception/modal_sam3.py`; deployment is under `modal_apps/sam3.py`. Do not change policy weights, Modal policy behavior, action dimensions, or unrelated prompts. Update camera/evidence contracts and regression cases together, and state what the planner can and cannot infer from the resulting evidence.

## Branch and PR

Work from a clean clone or worktree. The shared workstation currently has uncommitted changes; do not reset, stash, or branch from its dirty `main`.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c perception/grounding-<short-topic>
# make the focused change
python -m unittest tests.test_perception_grounding tests.test_rgbd_perception -v
python -m unittest discover -s tests -q
git diff --check
git status --short
git add <only-your-files>
git commit -m "Improve public grounding"
git push -u origin perception/grounding-<short-topic>
gh pr create --base main --head perception/grounding-<short-topic>
```

If your checkout is not clean, stop before switching branches and use a separate clone/worktree. In the PR, list camera/evidence assumptions, tests, unresolved cases, and any required local RoboTwin patch. Never commit `data/`, videos, weights, credentials, or archived implementation reports.
