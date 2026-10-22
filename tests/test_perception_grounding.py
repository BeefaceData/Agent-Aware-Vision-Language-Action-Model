"""CPU contracts for semantic RGB-D grounding and fail-closed evidence."""

import unittest
from types import SimpleNamespace

import numpy as np

from harness.environment.base import Observation
from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import (
    Detection, DetectionBatch, GroundedObject, RGBDPerceptionProvider,
)
from harness.perception.rgbd import RGBDCamera
from harness.skills.schemas import SkillCommand
from tests.helpers import images


def camera(depth=None):
    if depth is None:
        depth = np.full((2, 3), 2.0, dtype=np.float32)
    return RGBDCamera(np.asarray(depth, dtype=np.float32),
        np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))


def observation(sequence=0, depth=None):
    return Observation(sequence, images(), (0.,) * 14,
                       {"head_camera": camera(depth)})


def mask(pixel=(0, 1)):
    result = np.zeros((2, 3), dtype=bool)
    result[pixel] = True
    return result


class FakeDetector:
    def __init__(self, factory):
        self.factory = factory
        self.calls = []
        self.static_rack_reference = False

    def detect(self, image_map, *, observation_sequence):
        self.calls.append((image_map, observation_sequence))
        return self.factory(observation_sequence)

    def set_static_rack_reference(self, initialized):
        self.static_rack_reference = initialized


class PerceptionGroundingTests(unittest.TestCase):
    def test_configured_table_plane_lift_tracker_overrides_sam_centroid_latch(self):
        class Tracker:
            def evaluate(self, obs):
                return {"value": True, "reason": "stable_public_table_relative_lift",
                        "observation_sequence": obs.sequence}

        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", ()))
        provider = RGBDPerceptionProvider(detector)
        obs = observation(4)
        provider.observe(obs)
        provider.set_mug_lift_evidence_tracker(Tracker())
        evidence = provider.predicate_evidence("mug_lifted", obs)
        self.assertTrue(evidence["value"])
        self.assertEqual(evidence["reason"], "stable_public_table_relative_lift")

        provider.set_mug_lift_evidence_tracker(None)
        unavailable = provider.predicate_evidence("mug_lifted", obs)
        self.assertIsNone(unavailable["value"])
        self.assertEqual(unavailable["reason"], "table_plane_initialization_unavailable")

    @staticmethod
    def _large_observation(sequence, *, rack_mask=True, changed_calibration=False):
        from hashlib import sha256
        size = 20
        image_map = {name: np.zeros((size, size, 3), dtype=np.uint8)
                     for name in ("head_camera", "left_camera", "right_camera")}
        depth = np.full((size, size), .9, dtype=np.float32)
        mask_data = np.zeros((size, size), dtype=bool)
        mask_data[8:13, 11:16] = True
        depth[mask_data] = .5
        intrinsic = np.array([[30., 0., 9.5], [0., 30., 9.5], [0., 0., 1.]])
        extrinsic = np.eye(4)
        if changed_calibration:
            extrinsic[0, 3] = .01
        frame = RGBDCamera(depth, intrinsic, extrinsic)
        obs = Observation(sequence, image_map, (0.,) * 14, {"head_camera": frame})
        digest = sha256()
        digest.update(np.asarray(frame.intrinsic, dtype=np.float64).tobytes())
        digest.update(np.asarray(frame.world_to_camera, dtype=np.float64).tobytes())
        reference = {
            "status": "grounded", "source": "opus_rgb", "camera": "head_camera",
            "observation_sequence": 0, "image_shape_px": [size, size],
            "bbox_normalized": [0., 0., 1., 1.],
            "peg_uv_normalized": [10 / 19, 10 / 19], "confidence": .7,
            "camera_calibration_sha256": digest.hexdigest(),
        }
        return obs, reference, DetectionBatch(0, "head_camera", (
            Detection("rack", .5, mask_data),) if rack_mask else ())

    def test_static_sam_depth_point_is_cached_for_measured_start_rack_staging(self):
        from harness.perception.preparation import PerceptionAnchorPreparer
        initial, reference, batch = self._large_observation(0)
        detector = FakeDetector(lambda seq: batch if seq == 0 else
            DetectionBatch(seq, "head_camera", ()))
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(detector, anchors,
            min_confidence_by_label={"rack": .18})

        quality = provider.set_static_rack_reference(reference, observation=initial)

        self.assertTrue(quality["world_anchor_available"])
        self.assertEqual(quality["world_anchor_reason"], "head_sam_mask_local_depth")
        self.assertEqual(quality["world_anchor_valid_depth_samples"], 25)
        self.assertEqual(quality["world_anchor_peg_to_mask_px"], 1.0)
        self.assertTrue(detector.static_rack_reference)
        summary = provider.planner_summary(initial)
        self.assertTrue(summary["rack_static_reference"]["world_anchor_available"])
        self.assertNotIn("world_position_xyz", summary["rack_static_reference"])

        later, _, _ = self._large_observation(1, rack_mask=False)
        environment = SimpleNamespace(arm_state=lambda _arm: {
            "pose": (0., 0., 0., 1., 0., 0., 0.)})
        prepare = PerceptionAnchorPreparer(provider, environment, anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_approach"}})
        anchor = prepare(command, later)
        self.assertEqual(anchor.observation_sequence, 1)
        self.assertEqual(anchor.source, "perception")
        self.assertEqual(anchors.resolve("rack_approach", "right", later), anchor)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(anchor.pose[:3])
            - np.array([0., 0., 0.]))), .383, places=2)

    def test_static_rack_anchor_rejects_weak_mask_depth_and_keeps_detection_enabled(self):
        initial, reference, _ = self._large_observation(0)
        empty = DetectionBatch(0, "head_camera", ())
        detector = FakeDetector(lambda _seq: empty)
        provider = RGBDPerceptionProvider(detector,
            min_confidence_by_label={"rack": .18})

        quality = provider.set_static_rack_reference(reference, observation=initial)

        self.assertFalse(quality["world_anchor_available"])
        self.assertEqual(quality["world_anchor_reason"], "rack_mask_depth_support_not_found")
        self.assertFalse(detector.static_rack_reference)

    def test_static_rack_anchor_invalidates_when_camera_calibration_changes(self):
        from harness.perception.preparation import PerceptionAnchorPreparer
        initial, reference, batch = self._large_observation(0)
        detector = FakeDetector(lambda seq: batch if seq == 0 else
            DetectionBatch(seq, "head_camera", ()))
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(detector, anchors,
            min_confidence_by_label={"rack": .18})
        provider.set_static_rack_reference(reference, observation=initial)
        changed, _, _ = self._large_observation(1, changed_calibration=True,
                                                rack_mask=False)

        self.assertFalse(provider.static_rack_world_anchor_available(changed))
        self.assertFalse(detector.static_rack_reference)

    def test_cached_opus_rack_reference_survives_current_frame_miss(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", ()))
        provider = RGBDPerceptionProvider(detector)
        provider.set_static_rack_reference({
            "status": "grounded", "source": "opus_rgb", "camera": "head_camera",
            "observation_sequence": 0, "image_shape_px": [2, 3],
            "bbox_normalized": [.2, 0., 1., 1.], "peg_uv_normalized": [.8, .25],
            "confidence": .62,
        })

        summary = provider.planner_summary(observation(0))

        self.assertEqual(summary["status"]["rack"], "not_detected")
        self.assertTrue(summary["rack_static_reference"]["initialized"])
        self.assertEqual(summary["rack_static_reference"]["current_visibility"], "not_detected")
        self.assertEqual(summary["rack_static_reference"]["peg_uv_normalized"], [.8, .25])
        self.assertNotIn("world_position", summary["rack_static_reference"])

    def test_cached_rack_reference_invalidates_if_camera_shape_changes(self):
        provider = RGBDPerceptionProvider(FakeDetector(
            lambda seq: DetectionBatch(seq, "head_camera", ())))
        provider.set_static_rack_reference({
            "status": "grounded", "source": "opus_rgb", "camera": "head_camera",
            "observation_sequence": 0, "image_shape_px": [2, 3],
            "bbox_normalized": [.2, 0., 1., 1.], "peg_uv_normalized": [.8, .25],
            "confidence": .62,
        })
        original = observation(1)
        changed_images = {name: np.zeros((3, 4, 3), dtype=np.uint8)
                          for name in original.images}
        changed_frames = {"head_camera": RGBDCamera(np.ones((3, 4), dtype=np.float32),
            np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))}
        changed = Observation(1, changed_images, original.command_state, changed_frames)

        summary = provider.planner_summary(changed)

        self.assertFalse(summary["rack_static_reference"]["initialized"])
        self.assertEqual(summary["rack_static_reference"]["validity_reason"],
                         "camera_shape_or_calibration_changed")

    def test_camera_stamped_rack_fallback_uses_matching_rgbd_calibration(self):
        head = DetectionBatch(5, "head_camera", (
            Detection("mug", .90, mask()), Detection("table", .95, mask()),))
        right = DetectionBatch(5, "right_camera", (Detection("rack", .21, mask()),))
        obs = Observation(5, images(), (0.,) * 14,
            {"head_camera": camera(), "right_camera": camera()})
        provider = RGBDPerceptionProvider(FakeDetector(lambda seq: (head, right)),
            min_confidence_by_label={"rack": .18})

        result = provider.observe(obs)

        self.assertEqual(result.status["rack"], "grounded")
        self.assertEqual(result.objects["rack"].camera, "right_camera")
        self.assertEqual(result.objects["mug"].camera, "head_camera")
        self.assertEqual(result.objects["table"].camera, "head_camera")

    def test_right_camera_can_resolve_ambiguous_head_rack_instances(self):
        head = DetectionBatch(5, "head_camera", (
            Detection("rack", .46, mask()),
            Detection("rack", .40, mask((1, 1))),
        ))
        right = DetectionBatch(5, "right_camera", (Detection("rack", .31, mask()),))
        obs = Observation(5, images(), (0.,) * 14,
            {"head_camera": camera(), "right_camera": camera()})
        provider = RGBDPerceptionProvider(FakeDetector(lambda seq: (head, right)),
            min_confidence_by_label={"rack": .18})

        result = provider.observe(obs)

        self.assertEqual(result.status["rack"], "grounded")
        self.assertEqual(result.objects["rack"].camera, "right_camera")

    def test_multiple_camera_batches_fail_closed_on_stale_secondary_view(self):
        fresh = DetectionBatch(5, "head_camera", ())
        stale = DetectionBatch(4, "right_camera", ())
        obs = Observation(5, images(), (0.,) * 14,
            {"head_camera": camera(), "right_camera": camera()})
        provider = RGBDPerceptionProvider(FakeDetector(lambda seq: (fresh, stale)))

        result = provider.observe(obs)

        self.assertEqual(set(result.status.values()), {"stale_detection"})

    def test_per_label_threshold_can_opt_in_lower_rack_score_only(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("rack", .45, mask()),)))
        obs = observation()
        default = RGBDPerceptionProvider(detector).observe(obs)
        self.assertEqual(default.status["rack"], "low_confidence")

        exploration = RGBDPerceptionProvider(detector,
            min_confidence_by_label={"rack": .40})
        grounded = exploration.observe(obs)
        self.assertEqual(grounded.status["rack"], "grounded")
        self.assertNotIn("mug", grounded.objects)
        with self.assertRaisesRegex(ValueError, "invalid_label_confidence_thresholds"):
            RGBDPerceptionProvider(detector, min_confidence_by_label={"mug": 2.0})

    def test_lower_mug_cutoff_accepts_lowest_observed_score(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .40, mask()),)))
        result = RGBDPerceptionProvider(detector,
            min_confidence_by_label={"mug": .40}).observe(observation())
        self.assertEqual(result.status["mug"], "grounded")

    def test_confident_mask_is_grounded_and_binds_fresh_bounded_anchor(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .95, mask()),)))
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(detector, anchors)
        obs = observation(4)
        result = provider.observe(obs)

        self.assertEqual(result.status["mug"], "grounded")
        self.assertEqual(result.objects["mug"].observation_sequence, 4)
        self.assertAlmostEqual(result.objects["mug"].position[0], 0.0)
        self.assertFalse(detector.calls[0][0]["head_camera"].flags.writeable)
        anchor = provider.bind_approach_anchor("mug_pregrasp", "left", obs,
            measured_tcp_pose=(-.5, -.5, 2., 1., 0., 0., 0.), offset_m=.1)
        self.assertEqual(anchor.observation_sequence, 4)
        self.assertEqual(anchor.source, "perception")
        self.assertAlmostEqual(anchor.pose[0], -.1)
        with self.assertRaisesRegex(ValueError, "approach_offset_out_of_bounds"):
            provider.bind_approach_anchor("mug_pregrasp", "left", obs,
                measured_tcp_pose=(-.5, -.5, 2., 1., 0., 0., 0.), offset_m=.21)

    def test_planner_summary_exposes_descriptors_without_world_pose_or_identity_label(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .95, np.ones((2, 3), dtype=bool)),)))
        provider = RGBDPerceptionProvider(detector)
        obs = observation(4)
        colored = images()
        colored["head_camera"][:] = (37, 88, 141)
        obs = Observation(obs.sequence, colored, obs.command_state, obs.camera_frames)

        summary = provider.planner_summary(obs)
        mug = summary["objects"]["mug"]
        self.assertEqual(summary["source"], "public_rgbd")
        self.assertEqual(mug["normalized_bbox"], (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(mug["median_rgb_0_255"], (37.0, 88.0, 141.0))
        self.assertTrue(all(value > 0 for value in mug["visible_surface_extent_m"][:2]))
        self.assertEqual(summary["interpretation"], "view_dependent_descriptors_not_variant_labels")
        self.assertNotIn("position", mug)
        self.assertNotIn("world_to_camera", str(summary))

    def test_close_scoring_instances_are_ambiguous_and_do_not_bind(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .92, mask()), Detection("mug", .87, mask((1, 1))))))
        provider = RGBDPerceptionProvider(detector)
        obs = observation()
        result = provider.observe(obs)
        self.assertEqual(result.status["mug"], "ambiguous_instances")
        self.assertNotIn("mug", result.objects)
        with self.assertRaisesRegex(ValueError, "grounded_target_unavailable"):
            provider.bind_approach_anchor("mug_pregrasp", "left", obs,
                measured_tcp_pose=(-.5, -.5, 2., 1., 0., 0., 0.))

    def test_occlusion_invalid_depth_and_empty_mask_remain_unknown(self):
        cases = ((Detection("mug", .9, mask()), np.zeros((2, 3), np.float32),
                  "occluded_or_invalid_depth"),
                 (Detection("mug", .9, np.zeros((2, 3), dtype=bool)), None,
                  "occluded_or_empty_mask"))
        for detection, depth, expected in cases:
            with self.subTest(expected=expected):
                detector = FakeDetector(lambda seq, d=detection: DetectionBatch(
                    seq, "head_camera", (d,)))
                provider = RGBDPerceptionProvider(detector)
                obs = observation(depth=depth)
                result = provider.observe(obs)
                self.assertEqual(result.status["mug"], expected)
                evidence = provider.predicate_evidence("mug_near_rack", obs)
                self.assertIsNone(evidence["value"])

    def test_stale_detector_batch_and_stale_observation_are_rejected(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq - 1, "head_camera", (
            Detection("mug", .99, mask()),)))
        provider = RGBDPerceptionProvider(detector)
        obs = observation(5)
        self.assertEqual(provider.observe(obs).status["mug"], "stale_detection")
        duplicate = provider.observe(observation(5))
        self.assertEqual(duplicate.status["mug"], "stale_detection")
        stale = provider.observe(observation(4))
        self.assertEqual(stale.status["mug"], "stale_observation")
        with self.assertRaisesRegex(ValueError, "grounded_target_unavailable"):
            provider.bind_approach_anchor("mug_pregrasp", "left", obs,
                measured_tcp_pose=(-.5, -.5, 2., 1., 0., 0., 0.))

    def test_nonvisual_contact_predicates_are_not_inferred_from_proximity(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, mask()), Detection("rack", .99, mask((1, 1))))))
        provider = RGBDPerceptionProvider(detector)
        obs = observation()
        provider.observe(obs)
        for name in ("mug_grasp_confirmed", "rack_contact_detected", "task_success"):
            with self.subTest(name=name):
                evidence = provider.predicate_evidence(name, obs)
                self.assertIsNone(evidence["value"])
                self.assertEqual(evidence["source"], "perception")
                self.assertEqual(evidence["reason"], "not_provable_from_semantic_rgbd_alone")

    def test_release_ready_requires_mug_lower_surface_on_local_table(self):
        def obj(label, center_z, point_zs):
            points = tuple((0.0, 0.0, float(z)) for z in point_zs)
            return GroundedObject(label, (0.0, 0.0, center_z), .99, 2,
                                  "head_camera", 1.0, points,
                                  points if label == "mug" else ())

        mug = obj("mug", 1.1, (1.00, 1.01, 1.02, 1.10, 1.20))
        table = obj("table", 1.0, (1.0, 1.0, 1.0, 1.0))
        value, reason = RGBDPerceptionProvider._table_support_evidence(
            {"mug": mug, "table": table})
        self.assertTrue(value)
        self.assertEqual(reason, "mug_silhouette_meets_local_table")

        lifted = obj("mug", 1.2, (1.12, 1.13, 1.14, 1.20, 1.25))
        value, reason = RGBDPerceptionProvider._table_support_evidence(
            {"mug": lifted, "table": table})
        self.assertIsNone(value)
        self.assertEqual(reason, "mug_table_support_not_established")

    def test_release_ready_stays_unknown_when_local_table_geometry_is_missing(self):
        mug = GroundedObject("mug", (0., 0., 1.), .99, 1, "head_camera", 1.,
                             ((0., 0., 1.),) * 4, ((0., 0., 1.),) * 4)
        table = GroundedObject("table", (1., 1., 1.), .99, 1, "head_camera", 1.,
                               ((1., 1., 1.),) * 4)
        value, reason = RGBDPerceptionProvider._table_support_evidence(
            {"mug": mug, "table": table})
        self.assertIsNone(value)
        self.assertEqual(reason, "local_table_surface_unavailable")

    def test_hollow_mug_interior_depth_does_not_prove_table_support(self):
        # The full mug mask can see the table through the hollow center. Only
        # the outer silhouette is eligible for the support predicate.
        mug = GroundedObject("mug", (0., 0., 1.1), .99, 1, "head_camera", 1.,
            ((0., 0., 1.0),) * 8 + ((0., 0., 1.1),) * 8,
            ((0., 0., 1.1),) * 8)
        table = GroundedObject("table", (0., 0., 1.0), .99, 1,
            "head_camera", 1., ((0., 0., 1.0),) * 8)
        value, reason = RGBDPerceptionProvider._table_support_evidence(
            {"mug": mug, "table": table})
        self.assertIsNone(value)
        self.assertEqual(reason, "mug_table_support_not_established")

    def test_right_lift_gate_uses_fresh_rgbd_height_and_stays_unknown_without_mug(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, mask()),)))
        provider = RGBDPerceptionProvider(detector)
        baseline = observation(0)
        provider.observe(baseline)
        self.assertFalse(provider.predicate_evidence("right_mug_lifted", baseline)["value"])

        raised = Observation(1, images(), (0.,) * 14, {"head_camera": camera(
            np.full((2, 3), 2.1, dtype=np.float32))})
        provider.observe(raised)
        self.assertTrue(provider.predicate_evidence("right_mug_lifted", raised)["value"])

        missing_detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", ()))
        missing_provider = RGBDPerceptionProvider(missing_detector)
        missing_provider.observe(baseline)
        self.assertIsNone(missing_provider.predicate_evidence(
            "right_mug_lifted", baseline)["value"])

    def test_staged_milestone_requires_public_lift_then_table_support(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, mask()),)))
        provider = RGBDPerceptionProvider(detector)
        support = {"value": True}
        provider._table_support_evidence = lambda _objects: (
            support["value"], "mug_silhouette_meets_local_table"
            if support["value"] is True else "mug_table_support_not_established")

        baseline = observation(0)
        provider.observe(baseline)
        initial = provider.predicate_evidence("mug_staged_on_table", baseline)
        self.assertFalse(initial["value"])
        self.assertEqual(initial["reason"], "mug_lift_not_previously_observed")

        # Even a misleading silhouette/table overlap cannot confirm support
        # while the mug center is above its reset table height.
        support["value"] = True
        lifted = Observation(1, images(), (0.,) * 14,
            {"head_camera": camera(np.full((2, 3), 2.1, dtype=np.float32))})
        provider.observe(lifted)
        self.assertTrue(provider._last_result.mug_lift_seen)
        self.assertIsNone(provider.predicate_evidence("mug_staged_on_table", lifted)["value"])
        self.assertEqual(provider.predicate_evidence("release_ready", lifted)["reason"],
                         "mug_above_initial_table_height")

        support["value"] = True
        returned = observation(2)
        provider.observe(returned)
        staged = provider.predicate_evidence("mug_staged_on_table", returned)
        self.assertTrue(staged["value"])
        self.assertEqual(staged["reason"], "public_lift_then_table_support")

    def test_predicate_registry_and_orchestrator_publish_cached_public_evidence(self):
        from types import SimpleNamespace
        import json
        from harness.perception.grounding import Detection, DetectionBatch
        from harness.planner.base import PlannerContext
        from harness.planner.prompt_builder import user_content
        from harness.predicates.registry import PredicateRegistry
        from harness.skill_orchestrator import SkillOrchestrator

        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, mask()), Detection("rack", .99, mask()))))
        provider = RGBDPerceptionProvider(detector)
        registry = PredicateRegistry(perception=provider)
        obs = observation(9)
        registry.prime(obs)
        fake_executor = SimpleNamespace(predicates=registry,
            environment=SimpleNamespace(native_success=False), allow_oracle=False)
        controller = SkillOrchestrator(fake_executor, object(), log=lambda _: None,
            save_observation=lambda _: None, episode_id="cpu", seed=1)
        results = controller._predicates(obs)
        visual_context = provider.planner_summary(obs)
        context = PlannerContext(1, obs, results, None, visual_context=visual_context)

        self.assertEqual(context.value("mug_lifted"), False)
        self.assertEqual(context.value("mug_near_rack"), True)
        rack_contact = next(item for item in results if item.name == "rack_contact_detected")
        self.assertIsNone(rack_contact.value)
        self.assertEqual(rack_contact.source, "perception")
        self.assertEqual(rack_contact.observation_sequence, 9)
        blocks = user_content(context, limits={}, history=[])
        public_payload = json.loads(blocks[-1]["text"])
        sent = {item["name"]: item for item in public_payload["predicates"]}
        self.assertEqual(sent["mug_near_rack"]["value"], True)
        self.assertIsNone(sent["rack_contact_detected"]["value"])
        self.assertEqual(public_payload["visual_context"]["objects"]["mug"]["camera"], "head_camera")
        self.assertNotIn("position", public_payload["visual_context"]["objects"]["mug"])
        self.assertNotIn("world_to_camera", blocks[-1]["text"])
        # Prime + all planner predicates share one detector result per frame.
        self.assertEqual(len(detector.calls), 1)

    def test_unknown_configured_perception_never_falls_back_to_development_oracle(self):
        from harness.predicates.base import PredicateResult
        from harness.predicates.registry import PredicateRegistry

        class Oracle:
            source = "development_oracle"
            def evaluate(self, name, observation, *, arm="auto"):
                return PredicateResult(name=name, value=True, source=self.source,
                    observation_sequence=observation.sequence, reason="test_oracle")

        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", ()))
        provider = RGBDPerceptionProvider(detector)
        registry = PredicateRegistry(Oracle(), perception=provider)
        result = registry.evaluate("mug_lifted", observation(12), allow_oracle=True)
        self.assertIsNone(result.value)
        self.assertEqual(result.source, "perception")
        self.assertEqual(result.reason, "mug_geometry_unavailable")

    def test_symbolic_anchor_preparer_uses_measured_tcp_and_shared_registry(self):
        from types import SimpleNamespace
        from harness.perception.preparation import perception_anchor_preparer
        from harness.skills.schemas import SkillCommand

        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, mask()),)))
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(detector, anchors)
        measured_pose = (-.5, -.5, 2., 1., 0., 0., 0.)
        environment = SimpleNamespace(arm_state=lambda arm: {"pose": measured_pose})
        prepare = perception_anchor_preparer(provider, environment, anchors,
                                             approach_offset_m=.1)
        obs = observation(21)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp"}})
        bound = prepare(command, obs)
        self.assertEqual(bound.source, "perception")
        self.assertEqual(bound.observation_sequence, 21)
        np.testing.assert_allclose(bound.pose[:3], (-.1, -.5, 2.))
        resolved = anchors.resolve("mug_pregrasp", "left", obs)
        self.assertEqual(resolved, bound)
        self.assertEqual(detector.calls[0][1], 21)

        current_ee = command.model_copy(update={"parameters": command.parameters.model_copy(
            update={"target": "current_ee"})})
        self.assertIsNone(prepare(current_ee, observation(22)))
        self.assertEqual(len(detector.calls), 1)

    def test_anchor_preparer_rejects_registry_mismatch_and_unapproved_symbols(self):
        from types import SimpleNamespace
        from harness.perception.preparation import perception_anchor_preparer
        from harness.skills.schemas import SkillCommand

        provider = RGBDPerceptionProvider(FakeDetector(lambda seq:
            DetectionBatch(seq, "head_camera", ())))
        with self.assertRaisesRegex(ValueError, "perception_anchor_registry_mismatch"):
            perception_anchor_preparer(provider, SimpleNamespace(), AnchorRegistry())
        prepare = perception_anchor_preparer(provider, SimpleNamespace(), provider.anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "left_home"}})
        with self.assertRaisesRegex(ValueError, "unsupported_perception_anchor"):
            prepare(command, observation())

    def test_anchor_preparation_fails_closed_before_motion_without_rgbd(self):
        from types import SimpleNamespace
        from harness.perception.preparation import perception_anchor_preparer
        from harness.skills.schemas import SkillCommand

        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(FakeDetector(lambda seq:
            DetectionBatch(seq, "head_camera", ())), anchors)
        pose_reads = []
        environment = SimpleNamespace(arm_state=lambda arm: pose_reads.append(arm) or
                                      {"pose": (-.5, -.5, 2., 1., 0., 0., 0.)})
        prepare = perception_anchor_preparer(provider, environment, anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp"}})
        with self.assertRaisesRegex(ValueError, "grounded_target_unavailable"):
            prepare(command, Observation(30, images(), (0.,) * 14))
        self.assertEqual(pose_reads, ["left"])
        with self.assertRaisesRegex(ValueError, "anchor_missing_or_stale"):
            anchors.resolve("mug_pregrasp", "left", Observation(30, images(), (0.,) * 14))

    def test_bad_camera_mask_and_unsupported_label_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_detection_label"):
            Detection("robot_joint_action", .9, mask())
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .99, np.ones((3, 3), dtype=bool)),)))
        provider = RGBDPerceptionProvider(detector)
        result = provider.observe(observation())
        self.assertEqual(result.status["mug"], "mask_shape_mismatch")


if __name__ == "__main__":
    unittest.main()
