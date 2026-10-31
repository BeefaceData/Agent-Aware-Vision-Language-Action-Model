"""Modal policy client deadlines; no Modal account or network."""

import threading
import io
from types import SimpleNamespace
import unittest

import numpy as np

from harness.environment.layout import ACTION_DIM
from harness.vla.modal_client import ModalPolicyClient, call_with_timeout, pack_observation
from tests.helpers import images


def client(act_chunk, timeout_seconds):
    instance = ModalPolicyClient.__new__(ModalPolicyClient)  # skips the Modal lookup
    instance.worker = SimpleNamespace(act_chunk=SimpleNamespace(remote=act_chunk))
    instance.timeout_seconds = timeout_seconds
    return instance


class ModalPolicyClientTimeoutTests(unittest.TestCase):
    def test_perception_camera_is_excluded_from_policy_payload(self):
        original = images()
        packed, _, _ = pack_observation(
            {**original, "perception_camera": original["head_camera"]}, np.zeros(ACTION_DIM))
        with np.load(io.BytesIO(packed)) as archive:
            self.assertEqual(set(archive.files), {*original, "state"})

    def test_a_policy_call_that_never_returns_raises_instead_of_hanging(self):
        released = threading.Event()
        policy = client(lambda *args, **kwargs: released.wait(), timeout_seconds=0.05)
        try:
            with self.assertRaisesRegex(TimeoutError, "modal_policy_call_timed_out"):
                policy.act_chunk(images(), np.zeros(ACTION_DIM), 16)
        finally:
            released.set()

    def test_a_prompt_answer_is_returned_unchanged(self):
        def act_chunk(packed, requested_actions):
            return {"actions": np.zeros((requested_actions, ACTION_DIM)), "num_actions": requested_actions,
                    "chunk_limit": 24}

        actions, timing = client(act_chunk, timeout_seconds=5).act_chunk(images(), np.zeros(ACTION_DIM), 16)

        self.assertEqual(actions.shape, (16, ACTION_DIM))
        self.assertEqual(timing["returned_actions"], 16)

    def test_errors_from_the_call_reach_the_caller(self):
        def fail():
            raise ConnectionError("remote failure")

        with self.assertRaisesRegex(ConnectionError, "remote failure"):
            call_with_timeout(fail, 5)


if __name__ == "__main__":
    unittest.main()
