# Entry 660 reviewed demo: feasibility decision

> **Superseded.** This records the decision at the time of the one-image check.
> The commit of 31 October 2026 ("Implement guarded Entry 660 aperture hanging
> recovery") later wired a reviewed, explicitly overridden Entry 660 aperture to
> guarded recovery motion and recorded a native-checker success on that replay
> (support after release unverified). Current status and live results are in
> `Boniface_Close_Camera_Evaluation.md`.

Decision at the time: **do not wire these annotations to recovery motion**. The one-image
annotation and offline check have been completed; the conditional motion step
is blocked by the existing grounding guard, not by a missing annotation tool.

Saved evidence is in ignored `data/runs/entry660_reviewed_demo_feasibility/`:

- `handle_annotation.json`: nine visible-opening contour vertices and twelve
  visible material samples. Reviewed visually by the assistant on the saved
  rendered image; not independently reviewed by a person. No hidden geometry.
- `peg_annotation.json`: visible peg junction, point toward tip, and post point.
- `annotation_overlay.png`: cyan opening contour, yellow material pixels, red
  peg landmarks. `handle_review.png` enlarges the handle without adding detail.
- `handle_assessment.json`, `peg_assessment.json`, `depth_sample_review.json`:
  exact results and sample measurements, tied to observation 660.

The source is
`data/runs/entry660_20261029T202054Z_1793305254857158600/observations/observation_00000660.npz`.
Its SHA256 is
`d510a6fd877d00bae179d51f0c6a94b76304a20c0cccea2313f2ac0deac8e25a`.

## Result

Peg surface-frame geometry passes numerical checks. It does not establish
physical support, clearance or the complete mating relationship.

The handle estimate returns `material_depth_discontinuity` before plane fitting.
Samples `(124,38)`, `(126,40)` and `(128,42)` have foreground gaps of approximately
64.6, 77.0 and 87.6 mm respectively, exceeding the existing 60 mm limit. Their
3x3 spreads are only 4.1, 4.5 and 4.7 mm. Thus the rejection concerns a nearer
surface in the surrounding 8-pixel neighborhood, not mixed background in those
3x3 patches. This conservative guard alone does not prove erroneous depth or
an impossible physical hang. It does prevent authorizing this target under the
current unchanged checks.

No points were discarded or thresholds changed after this rejection. No
aperture plane or target pose was promoted to motion. Acquisition, handoff,
tracking, release, replay and video code remain unchanged. No live run, API,
commit or push occurred for this assessment.

## Reproduce offline

Use a new output filename (existing files are never overwritten):

```powershell
.\.venv\Scripts\python.exe -m development.hanging.aperture_frame `
  --archive data/runs/entry660_20261029T202054Z_1793305254857158600/observations/observation_00000660.npz `
  --annotation data/runs/entry660_reviewed_demo_feasibility/handle_annotation.json `
  --output "data/runs/entry660_reviewed_demo_feasibility/recheck_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').json"
```

This particular image/estimator route is stopped. It is not a demo-ready
recovery, and another identical Modal recovery command is not recommended.
