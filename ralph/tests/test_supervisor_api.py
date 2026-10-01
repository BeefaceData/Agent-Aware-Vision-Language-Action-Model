"""Public budgeted transport behavior; no paid requests or real credentials."""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Stop, read_json
from supervisor_api import SupervisorAPI, Budget
import runner


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = read_json(Path(__file__).resolve().parents[1] / "supervisor-api-policy.json")
        self.policy["pricing_valid_until"] = (date.today() + timedelta(days=1)).isoformat()
        self.calls = []

    def transport(self, payload, key):
        self.assertEqual(key, "synthetic-test-secret")
        self.assertEqual(payload["model"], "claude-sonnet-5-5")
        self.assertEqual(payload["service_tier"], "standard_only")
        self.calls.append(payload)
        return {"model": payload["model"], "usage": {"input_tokens": 120, "output_tokens": 30},
                "content": [{"type": "text", "text": "abstain"}]}

    def client(self, **kwargs):
        return SupervisorAPI(self.root, self.policy, transport=kwargs.get("transport", self.transport),
                             key_loader=lambda root: "synthetic-test-secret")

    def request(self, client, identity="episode:1"):
        return client.message(identity, [{"role": "user", "content": [{"type": "text", "text": "Observe synthetic scene"}]}],
                              system="Diagnose only; abstain when uncertain.", max_tokens=100)

    def test_success_reconciles_actual_usage_and_restart_rejects_duplicate(self):
        self.request(self.client())
        restarted = self.client()
        self.assertEqual(Decimal(restarted.budget.status()["spent_or_reserved_usd"]), Decimal("0.00054"))
        with self.assertRaisesRegex(Stop, "already"):
            self.request(restarted)
        self.assertEqual(len(self.calls), 1)
        ledger = (self.root / ".ralph/supervisor-budget.sqlite3").read_bytes()
        self.assertNotIn(b"synthetic-test-secret", ledger)
        self.assertNotIn(b"Observe synthetic scene", ledger)

    def test_concurrent_uncertain_calls_cannot_spend_past_shared_cap(self):
        path = self.root / "ledger.sqlite3"
        Budget(path, self.policy)
        def reserve(i):
            try:
                Budget(path, self.policy).reserve(str(i), {"max_tokens": 100})
                return True
            except Stop:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reserve, range(40)))
        self.assertEqual(sum(results), 22)
        status = Budget(path, self.policy).status()
        self.assertLessEqual(Decimal(status["spent_or_reserved_usd"]), 50)
        self.assertEqual(status["uncertain_calls"], 22)

    def test_transport_failure_retains_reservation_without_retry(self):
        calls = []
        def fail(payload, key):
            calls.append(payload)
            raise Stop("timeout")
        client = self.client(transport=fail)
        with self.assertRaisesRegex(Stop, "timeout"):
            self.request(client)
        self.assertEqual(client.budget.status()["spent_or_reserved_usd"], "2.201")
        with self.assertRaisesRegex(Stop, "already"):
            self.request(self.client())
        self.assertEqual(len(calls), 1)

    def test_expired_pricing_bad_content_and_output_bound_send_nothing(self):
        client = self.client()
        for content in [None, [{"type": "image", "source": "bad"}], [{"type": "tool_use"}]]:
            with self.assertRaises(Stop):
                client.message("bad", [{"role": "user", "content": content}], system="test")
        with self.assertRaises(Stop):
            client.message("large", [], system="test", max_tokens=2049)
        client.policy = {**self.policy, "pricing_valid_until": "2000-01-01"}
        with self.assertRaisesRegex(Stop, "pricing"):
            self.request(client)
        self.assertFalse(self.calls)
        self.assertEqual(client.budget.status()["spent_or_reserved_usd"], "0")

    def test_bad_usage_blocks_future_requests_and_preserves_reservation(self):
        client = self.client(transport=lambda payload, key: {
            "model": "claude-sonnet-5-5", "usage": {"input_tokens": 1200000, "output_tokens": 1}})
        with self.assertRaisesRegex(Stop, "accounting"):
            self.request(client)
        restarted = self.client()
        with self.assertRaisesRegex(Stop, "reconciliation"):
            self.request(restarted, "episode:2")
        self.assertTrue(restarted.budget.status()["accounting_blocked"])
        self.assertFalse(self.calls)

    def test_malformed_accounting_or_output_overage_blocks_across_restart(self):
        examples = [None, [], {"model": "claude-sonnet-5-5", "usage": None},
                    {"model": "claude-sonnet-5-5", "usage": "invalid"},
                    {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 1100000, "output_tokens": 2048}}]
        for i, response in enumerate(examples):
            with self.subTest(response=response):
                root = self.root / str(i)
                client = SupervisorAPI(root, self.policy, transport=lambda payload, key: response,
                                       key_loader=lambda root: "synthetic-test-secret")
                with self.assertRaises(Stop): self.request(client)
                restarted = SupervisorAPI(root, self.policy, transport=self.transport,
                                          key_loader=lambda root: "synthetic-test-secret")
                with self.assertRaisesRegex(Stop, "reconciliation"):
                    self.request(restarted, "episode:2")
                status = restarted.budget.status()
                self.assertTrue(status["accounting_blocked"])
                if i == len(examples)-1:
                    self.assertEqual(status["spent_or_reserved_usd"], "2.22048")
        self.assertFalse(self.calls)

    def test_changed_budget_or_policy_cannot_reset_spending(self):
        self.request(self.client())
        self.policy = {**self.policy, "budget_usd": "49.00"}
        with self.assertRaisesRegex(Stop, "policy changed"):
            self.client()
        self.policy["budget_usd"] = "51"
        with self.assertRaisesRegex(Stop, "at most"):
            self.client()

    def test_chronological_images_reach_transport_with_no_key_in_payload(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "camera wrist at t=1"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "fixture"}},
            {"type": "text", "text": "camera wrist at t=2"}]}]
        self.client().message("episode:images", copy.deepcopy(messages), system="Observe")
        self.assertEqual(self.calls[0]["messages"], messages)
        self.assertNotIn("synthetic-test-secret", json.dumps(self.calls))

    def test_git_worktrees_resolve_one_production_budget(self):
        runner.git(self.root, "init", "-q", "-b", "main")
        runner.git(self.root, "config", "user.name", "test")
        runner.git(self.root, "config", "user.email", "test@example.invalid")
        runner.git(self.root, "commit", "--allow-empty", "-qm", "test fixture")
        other = self.root / "linked"
        runner.git(self.root, "worktree", "add", "--detach", str(other))
        first = SupervisorAPI.from_worktree(self.root)
        second = SupervisorAPI.from_worktree(other)
        self.assertEqual(first.budget.path, second.budget.path)
        first.budget.reserve("one", {"max_tokens": 100})
        self.assertEqual(second.budget.status()["spent_or_reserved_usd"], "2.201")
