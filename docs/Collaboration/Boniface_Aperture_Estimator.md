# Offline visible-aperture estimator

`development.hanging.aperture_frame` is development-only. Nothing in recovery,
target preparation, release authorization or the Modal launcher imports it.
The recovery preflight rejection remains in place.

## Inputs and provenance

Supply an exact saved public observation archive and a JSON annotation:

```json
{
  "archive_sha256": "SHA256 of the complete observation NPZ",
  "sequence": 660,
  "camera": "perception_camera",
  "reviewed": true,
  "complete_visible_contour": true,
  "contour_px": [],
  "material_px": []
}
```

Empty arrays above are schema placeholders, not usable annotations. Do not
label proposed or automatically segmented points as reviewed. The estimator
checks archive identity and explicit review/completeness declarations; it
cannot independently establish that those human declarations are true.

`contour_px` requires at least eight ordered vertices of a complete, strictly
convex visible opening polygon. Do not close an occluded opening by guessing.
`material_px` requires at least twelve distinct visible handle-material pixels
surrounding that opening. Their 3x3 depth neighborhoods must be positive and
continuous. Mug body, gripper, background and empty-opening pixels are excluded.
The three existing model proposal points do not meet this input contract.

## Method and frame convention

Fit a least-squares plane to calibrated world points from visible material.
Check spatial rank, maximum residual and surrounding angular coverage. Intersect
the opening-contour camera rays with that plane and compute its polygon area
centroid. Depth through the empty opening is never sampled.

The candidate's X axis is the plane normal facing the camera, Y is projected
image-up, and Z is X cross Y. This fixes diagnostic sign/roll ambiguity, but
does **not** determine a physical insertion direction. The origin lies on a
fitted visible-surface plane, not an inferred hidden handle midplane.

This assumes the sampled visible handle surface can approximate a plane. Curved
rim geometry can bias the fit even if residuals are small. Occlusion, thickness,
contact, support and peg clearance remain unmeasured. A diagnostic candidate
is not a validated physical mating frame.

## Uncertainty and rejection

Report maximum plane residual, leave-one-sample-out and four rigid material-pixel
shift normal changes, plus center changes from those fits and four rigid contour
shifts. These are deterministic sensitivity probes, not confidence intervals
or exhaustive independent vertex/depth perturbations. Systematic depth and
calibration error remain unknown.

Default exploratory rejection limits: 3 mm plane residual, 5 degrees normal
sensitivity, 5 mm center sensitivity, one-pixel shifts. They are hypotheses for
offline assessment, not validated manipulation tolerances. The existing 60 mm
depth-gap limit is retained for sample continuity. Samples require an 8 mm
minimum second principal RMS span and cannot leave an angular gap of 120 degrees
or more around the opening.

Reject missing/incomplete annotations, wrong archive identity, invalid/mixed
depth, insufficient or duplicate support, degeneracy, nonconvex/self-crossing
contours, near-tangent views, nonplanarity and excessive measured sensitivity.
All outputs set `motion_authorized` to false, including accepted candidates.

## Local command

```powershell
$assessment = "data/runs/aperture_review_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').json"
.\.venv\Scripts\python.exe -m development.hanging.aperture_frame `
  --archive data/runs/entry660_20261029T202054Z_1793305254857158600/observations/observation_00000660.npz `
  --output $assessment
```

Without annotations this deliberately writes a missing-reviewed-geometry
rejection. When an actual reviewed JSON is available, add `--annotation <path>`.
Existing output files are never overwritten. This command has no API or
simulator dependency and does not execute motion.

## Validation

`python -m unittest tests.test_aperture_frame -v` exercises synthetic plane
geometry, ray/calibration use, right-handed frame construction, ignored hole
background depth, depth discontinuities, missing/duplicate samples, incomplete
support, invalid contours and rejection thresholds. Synthetic success does not
establish accuracy or repeatability on the saved mug. No reviewed contour or
material annotation has been invented for observation 660.
