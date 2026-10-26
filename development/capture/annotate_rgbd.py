"""Create reviewed semantic instance masks for recorded public RGB-D frames.

This optional OpenCV desktop tool uses CPU only and imports no simulator or
model code. It accepts archives created with the opt-in RGB-D capture setting.

Keys: 1 mug, 2 rack, 3 table; left click adds polygon vertices; Enter commits
the current instance; n saves this frame and advances; q saves and quits.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from harness.perception.replay import load_public_observation


LABEL_KEYS = {ord("1"): "mug", ord("2"): "rack", ord("3"): "table"}
LABEL_COLORS = {"mug": (40, 220, 40), "rack": (0, 180, 255), "table": (255, 120, 30)}


def polygon_mask(shape, points):
    """Rasterize a validated polygon without requiring GUI calls in CPU tests."""
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("invalid_annotation_image_shape")
    if len(points) < 3:
        raise ValueError("polygon_requires_three_points")
    vertices = np.asarray(points)
    if (vertices.ndim != 2 or vertices.shape[1] != 2
            or not np.issubdtype(vertices.dtype, np.number)
            or not np.isfinite(vertices).all()):
        raise ValueError("invalid_annotation_polygon")
    height, width = shape
    vertices = np.rint(vertices).astype(np.int64)
    if ((vertices[:, 0] < 0).any() or (vertices[:, 0] >= width).any()
            or (vertices[:, 1] < 0).any() or (vertices[:, 1] >= height).any()):
        raise ValueError("annotation_polygon_out_of_bounds")
    from PIL import Image, ImageDraw
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon([tuple(map(int, point)) for point in vertices], fill=1)
    mask = np.asarray(image, dtype=bool)
    if not mask.any():
        raise ValueError("empty_annotation_polygon")
    return mask


def encode_mask_rle(mask):
    """Encode a bool mask as the replay adapter's flat ``[start,length]`` runs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size or mask.size > 8_000_000 or not mask.any():
        raise ValueError("invalid_annotation_mask")
    indices = np.flatnonzero(mask)
    runs = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous+1:
            runs.append([start, previous-start+1])
            start = value
        previous = value
    runs.append([start, previous-start+1])
    return {"shape": [int(mask.shape[0]), int(mask.shape[1])], "runs": runs}


def _frame_row(observation, camera, detections):
    if not observation.camera_frames or camera not in observation.camera_frames:
        raise ValueError("rgbd_required_for_annotation")
    return {"sequence": observation.sequence, "camera": camera,
            "detections": detections}


def annotate_directory(observations_dir, camera, output):
    import cv2

    observations_dir, output = Path(observations_dir).resolve(), Path(output).resolve()
    if camera not in ("head_camera", "left_camera", "right_camera"):
        raise ValueError("invalid_annotation_camera")
    paths = sorted(observations_dir.glob("observation_*.npz"))
    if not paths or len(paths) > 10000:
        raise ValueError("invalid_observation_archive_count")
    observations = [load_public_observation(path) for path in paths]
    for observation in observations:
        if not observation.camera_frames or camera not in observation.camera_frames:
            raise ValueError("rgbd_required_for_annotation")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "camera": camera, "frames": [], "source_sha256": {}}
    window = "RGB-D semantic annotation"
    selected_label = "mug"
    try:
        with output.open("x") as stream:
            for path, observation in zip(paths, observations):
                rgb = observation.images[camera]
                height, width = rgb.shape[:2]
                scale = min(1.0, 1280/width, 800/height)
                display_size = (max(1, int(width*scale)), max(1, int(height*scale)))
                points, detections = [], []
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

                def mouse(event, x, y, flags, _param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        px = min(width-1, max(0, int(x/scale)))
                        py = min(height-1, max(0, int(y/scale)))
                        points.append((px, py))

                cv2.setMouseCallback(window, mouse)
                frame_saved = False
                while True:
                    canvas = cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), display_size)
                    overlay = canvas.copy()
                    for item in detections:
                        mask = np.zeros((height, width), dtype=np.uint8)
                        flat = mask.reshape(-1)
                        for start, length in item["runs"]:
                            flat[start:start+length] = 1
                        visible = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                        overlay[visible] = LABEL_COLORS[item["label"]]
                    for x, y in points:
                        cv2.circle(overlay, (int(x*scale), int(y*scale)), 3,
                                   LABEL_COLORS[selected_label], -1)
                    canvas = cv2.addWeighted(canvas, .55, overlay, .45, 0)
                    cv2.putText(canvas,
                        f"seq={observation.sequence} label={selected_label} polygon={len(points)} masks={len(detections)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(25) & 0xFF
                    if key in LABEL_KEYS:
                        selected_label = LABEL_KEYS[key]
                    elif key in (10, 13) and len(points) >= 3:
                        mask = polygon_mask((height, width), points)
                        encoded = encode_mask_rle(mask)
                        detections.append({"label": selected_label, "confidence": 1.0, **encoded})
                        points.clear()
                    elif key == ord("c"):
                        points.clear()
                    elif key in (ord("n"), ord("q")):
                        data["frames"].append(_frame_row(observation, camera, detections))
                        data["source_sha256"][path.name] = sha256(path.read_bytes()).hexdigest()
                        stream.seek(0)
                        stream.truncate()
                        json.dump(data, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                        stream.flush()
                        frame_saved = True
                        if key == ord("q"):
                            return len(data["frames"])
                        break
                if not frame_saved:
                    break
            return len(data["frames"])
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--camera", choices=("head_camera", "left_camera", "right_camera"),
                        default="head_camera")
    parser.add_argument("--output", type=Path, required=True,
                        help="new annotation JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        count = annotate_directory(args.observations, args.camera, args.output)
    except Exception as exc:
        parser.error(f"annotation_failed: {type(exc).__name__}: {exc}")
    print(json.dumps({"annotated_frames": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
