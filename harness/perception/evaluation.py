"""CPU mask-quality metrics for evaluating an injected semantic detector."""

from collections import defaultdict

import numpy as np

from .grounding import DetectionBatch, SEMANTIC_LABELS


def mask_iou(left, right):
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("mask_shape_mismatch")
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def evaluate_mask_batches(predicted, reference, *, confidence_threshold=0.60,
                          iou_threshold=0.50):
    """Greedy per-label instance match for one observation and reviewed masks."""
    if (not isinstance(predicted, DetectionBatch) or not isinstance(reference, DetectionBatch)
            or predicted.observation_sequence != reference.observation_sequence
            or predicted.camera != reference.camera):
        raise ValueError("evaluation_batch_mismatch")
    values = (confidence_threshold, iou_threshold)
    if (not np.isfinite(values).all() or not 0 <= confidence_threshold <= 1
            or not 0 < iou_threshold <= 1):
        raise ValueError("invalid_detector_evaluation_limits")
    metrics = {}
    for label in sorted(SEMANTIC_LABELS):
        predictions = [item for item in predicted.detections
                       if item.label == label and item.confidence >= confidence_threshold]
        truths = [item for item in reference.detections if item.label == label]
        candidates = []
        for pi, prediction in enumerate(predictions):
            for ti, truth in enumerate(truths):
                candidates.append((mask_iou(prediction.mask, truth.mask), pi, ti))
        candidates.sort(reverse=True)
        matched_predictions, matched_truths, matched_ious = set(), set(), []
        for score, pi, ti in candidates:
            if (score < iou_threshold or pi in matched_predictions or ti in matched_truths):
                continue
            matched_predictions.add(pi)
            matched_truths.add(ti)
            matched_ious.append(score)
        tp = len(matched_predictions)
        fp, fn = len(predictions)-tp, len(truths)-len(matched_truths)
        metrics[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
            "predicted_instances": len(predictions),
            "reference_instances": len(truths),
        }
    return {"observation_sequence": predicted.observation_sequence,
            "camera": predicted.camera, "labels": metrics}


def evaluate_detector(detector, frames, *, confidence_threshold=0.60,
                      iou_threshold=0.50):
    """Evaluate an injected detector on ``(Observation, reference batch)`` pairs.

    Detector errors and mismatched sequences fail the evaluation. This routine
    never loads model weights, starts a simulator, or selects a device.
    """
    totals = defaultdict(lambda: {"true_positive": 0, "false_positive": 0,
                                 "false_negative": 0, "matched_ious": [],
                                 "predicted_instances": 0, "reference_instances": 0})
    rows = []
    count = 0
    previous_sequence = -1
    for observation, reference in frames:
        if observation.sequence <= previous_sequence:
            raise ValueError("evaluation_observations_not_increasing")
        previous_sequence = observation.sequence
        images = {}
        for name, image in observation.images.items():
            snapshot = np.asarray(image).copy()
            snapshot.setflags(write=False)
            images[name] = snapshot
        predicted = detector.detect(images, observation_sequence=observation.sequence)
        if not isinstance(predicted, DetectionBatch) or predicted.observation_sequence != observation.sequence:
            raise ValueError("detector_returned_stale_or_invalid_batch")
        if predicted.camera not in images or predicted.camera != reference.camera:
            raise ValueError("evaluation_camera_mismatch")
        image_shape = np.asarray(images[predicted.camera]).shape[:2]
        if any(item.mask.shape != image_shape
               for item in (*predicted.detections, *reference.detections)):
            raise ValueError("evaluation_mask_shape_mismatch")
        row = evaluate_mask_batches(predicted, reference,
            confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
        rows.append({**row, "rgbd_available": bool(observation.camera_frames)})
        for label, result in row["labels"].items():
            target = totals[label]
            for key in ("true_positive", "false_positive", "false_negative",
                        "predicted_instances", "reference_instances"):
                target[key] += result[key]
            if result["mean_matched_iou"] is not None:
                # The value is repeated once per matched instance to preserve
                # micro-averaged IoU across frames with different counts.
                target["matched_ious"].extend([result["mean_matched_iou"]] * result["true_positive"])
        count += 1
    if count == 0:
        raise ValueError("empty_detector_evaluation")
    summary = {}
    for label, data in sorted(totals.items()):
        tp, fp, fn = data["true_positive"], data["false_positive"], data["false_negative"]
        summary[label] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "mean_matched_iou": (float(np.mean(data["matched_ious"]))
                                 if data["matched_ious"] else None),
            "predicted_instances": data["predicted_instances"],
            "reference_instances": data["reference_instances"],
        }
    return {"schema_version": 1, "mode": "offline_detector_evaluation",
            "detector_callback_invoked": True, "robot_motion": False,
            "simulator_started": False, "frame_count": count,
            "rgbd_frame_count": sum(bool(row["rgbd_available"]) for row in rows),
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold, "summary": summary, "frames": rows}
