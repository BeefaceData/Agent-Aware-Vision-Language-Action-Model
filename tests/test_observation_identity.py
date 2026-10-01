"""Observation provenance and ordering at the public ingestion and episode seams."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from episode_harness import (ObservationIngestor, ObservationPacket,
                             ObservationRejected, run_episode)
from replay_adapters import successful_replay


CAPTURE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class ObservationIngestionTests(unittest.TestCase):
    def packet(self, sequence, captured_at=CAPTURE, episode_id='episode-a',
               captured_monotonic=100.0):
        return ObservationPacket(episode_id, sequence, captured_at, 'frame',
                                 captured_monotonic)

    def test_duplicate_and_out_of_order_packets_have_diagnostic_outcomes(self):
        ingestor = ObservationIngestor('episode-a')
        self.assertTrue(ingestor.ingest(self.packet(0)).accepted)

        duplicate = ingestor.ingest(self.packet(0))
        self.assertEqual((duplicate.accepted, duplicate.code,
                          duplicate.expected_sequence, duplicate.received_sequence),
                         (False, 'duplicate', 1, 0))
        ahead = ingestor.ingest(self.packet(2))
        self.assertEqual((ahead.accepted, ahead.code, ahead.expected_sequence),
                         (False, 'out_of_order', 1))
        self.assertTrue(ingestor.ingest(self.packet(1, CAPTURE +
                                                     timedelta(seconds=1))).accepted)
        self.assertEqual(ingestor.ingest(self.packet(0)).code, 'duplicate')

    def test_foreign_episode_or_regressed_capture_time_cannot_advance_stream(self):
        ingestor = ObservationIngestor('episode-a')
        self.assertEqual(ingestor.ingest(self.packet(1)).code, 'out_of_order')
        self.assertEqual(ingestor.ingest(self.packet(0, episode_id='episode-b')).code,
                         'wrong_episode')
        self.assertEqual(ingestor.ingest(self.packet(0, CAPTURE.replace(tzinfo=None))).code,
                         'invalid_capture_time')
        self.assertTrue(ingestor.ingest(self.packet(0)).accepted)
        self.assertEqual(ingestor.ingest(self.packet(1, CAPTURE -
                                               timedelta(seconds=1))).code,
                         'out_of_order_capture_time')
        self.assertTrue(ingestor.ingest(self.packet(1, CAPTURE)).accepted)

    def test_monotonic_capture_must_be_finite_and_ordered(self):
        ingestor = ObservationIngestor('episode-a')
        self.assertEqual(ingestor.ingest(self.packet(0, captured_monotonic=None)).code,
                         'invalid_monotonic_capture_time')
        self.assertTrue(ingestor.ingest(self.packet(0)).accepted)
        self.assertEqual(ingestor.ingest(self.packet(1, captured_monotonic=float('nan'))).code,
                         'invalid_monotonic_capture_time')
        self.assertEqual(ingestor.ingest(self.packet(1, captured_monotonic=99.0)).code,
                         'out_of_order_monotonic_capture_time')
        self.assertTrue(ingestor.ingest(self.packet(1, captured_monotonic=101.0)).accepted)

    def test_future_capture_is_rejected_at_receipt(self):
        ingestor = ObservationIngestor('episode-a')
        outcome = ingestor.ingest(self.packet(0, captured_monotonic=101.0),
                                  received_at=100.0)
        self.assertEqual(outcome.code, 'future_monotonic_capture_time')
        self.assertEqual(outcome.expected_sequence, 0)
        self.assertTrue(ingestor.ingest(self.packet(0), received_at=100.0).accepted)


class ObservationEpisodeTests(unittest.TestCase):
    def test_complete_replay_links_initial_decisions_and_terminal_observation(self):
        fixture = successful_replay()
        outcome = run_episode(fixture.config, fixture.policy,
                              fixture.environment, fixture.recorder)

        packets = fixture.recorder.observations
        self.assertTrue(outcome.success)
        self.assertEqual([packet.sequence for packet in packets], [0, 1, 2])
        self.assertEqual({packet.episode_id for packet in packets},
                         {outcome.episode_id})
        self.assertEqual([packet.captured_at for packet in packets],
                         [CAPTURE + timedelta(seconds=sequence)
                          for sequence in range(3)])
        self.assertEqual(fixture.policy.observations, packets[:2])
        self.assertEqual([(source, result.observation)
                          for _, source, _, result, _ in fixture.recorder.steps],
                         [(packets[0], packets[1]), (packets[1], packets[2])])
        self.assertEqual([ingestion.code for _, _, _, _, ingestion
                          in fixture.recorder.steps], ['accepted', 'accepted'])

    def test_rejected_step_packet_stops_before_next_decision(self):
        for bad_sequence, diagnostic in ((0, 'duplicate'), (2, 'out_of_order')):
            with self.subTest(diagnostic=diagnostic):
                fixture = successful_replay()

                class BadEnvironment:
                    def reset(self, seed, episode_id):
                        return fixture.environment.reset(seed, episode_id)

                    def step(self, action):
                        result = fixture.environment.step(action)
                        bad_packet = replace(result.observation,
                                             sequence=bad_sequence)
                        return replace(result, observation=bad_packet)

                with self.assertRaises(ObservationRejected) as caught:
                    run_episode(fixture.config, fixture.policy,
                                BadEnvironment(), fixture.recorder)

                self.assertEqual(caught.exception.outcome.code, diagnostic)
                self.assertEqual(fixture.environment.actions, [('reach', 0.25)])
                self.assertEqual(len(fixture.policy.observations), 1)
                self.assertEqual(len(fixture.recorder.steps), 1)
                self.assertEqual(fixture.recorder.steps[0][1],
                                 fixture.policy.observations[0])
                self.assertEqual(fixture.recorder.steps[0][4].code, diagnostic)
                self.assertFalse(fixture.recorder.finalized)


if __name__ == '__main__':
    unittest.main()
