"""Budgeted Anthropic Messages transport for simulation supervision, never coding.

All worktrees share one repository budget. Reserve the full model input ceiling
plus maximum output before a call; uncertain calls retain their full reservation.
Pricing covers standard API token usage, not unrelated account use or taxes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import urllib.error
import urllib.request

try:
    from .runtime import Stop, git_argv, private_env, read_json, run
except ImportError:
    from runtime import Stop, git_argv, private_env, read_json, run


class Budget:
    def __init__(self, path: Path, policy: dict):
        self.path, self.policy = path, policy
        self.limit = int(Decimal(policy["budget_usd"]) * 1_000_000)
        if not 0 < self.limit <= 50_000_000:
            raise Stop("This authorization allows at most USD 50 total.")
        if (policy.get("input_usd_per_million") != "2.00" or policy.get("output_usd_per_million") != "10.00"
                or policy.get("maximum_input_tokens") != 1100000
                or type(policy.get("maximum_output_tokens")) is not int
                or not 1 <= policy["maximum_output_tokens"] <= 2048):
            raise Stop("Pricing or token bounds differ from the verified authorization.")
        self.fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, policy TEXT NOT NULL, cap INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, digest TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS accounting_faults (reason TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS request_bounds (id TEXT PRIMARY KEY, max_output INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?, ?)", (self.fingerprint, self.limit))
            if db.execute("SELECT policy, cap FROM budget WHERE id=1").fetchone() != (self.fingerprint, self.limit):
                raise Stop("Budget policy changed; retain the ledger and reconcile under supervision.")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, request_id: str, request: dict) -> int:
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", request_id):
            raise Stop("A stable episode/observation request ID is required.")
        if type(request.get("max_tokens")) is not int or not 1 <= request["max_tokens"] <= self.policy["maximum_output_tokens"]:
            raise Stop("Invalid reservation output bound.")
        amount = self.cost(self.policy["maximum_input_tokens"], request["max_tokens"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM accounting_faults").fetchone()[0]:
                raise Stop("Provider accounting needs reconciliation; further paid calls are blocked.")
            used = db.execute("SELECT COALESCE(SUM(COALESCE(actual, reserved)),0) FROM calls").fetchone()[0]
            if used + amount > self.limit:
                raise Stop("Supervisor API budget exhausted; no request sent.")
            try:
                db.execute("INSERT INTO calls VALUES (?, ?, ?, NULL, 'reserved')", (request_id,
                    hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(), amount))
                db.execute("INSERT INTO request_bounds VALUES (?, ?)", (request_id, request["max_tokens"]))
            except sqlite3.IntegrityError as exc:
                raise Stop("Request already reserved/completed; automatic paid retries are disabled.") from exc
        return amount

    def cost(self, input_tokens, output_tokens):
        return int((Decimal(self.policy["input_usd_per_million"]) * input_tokens +
                    Decimal(self.policy["output_usd_per_million"]) * output_tokens).to_integral_value(rounding=ROUND_CEILING))

    def fault(self, request_id=None):
        with self.connect() as db:
            db.execute("INSERT INTO accounting_faults VALUES ('unexpected provider accounting')")
            if request_id:
                db.execute("UPDATE calls SET state='accounting_fault' WHERE id=?", (request_id,))

    def settle(self, request_id: str, usage: dict):
        counts = [usage.get("input_tokens"), usage.get("output_tokens")] if isinstance(usage, dict) else []
        valid_counts = len(counts) == 2 and all(type(v) is int and 0 <= v <= 10**12 for v in counts)
        amount = self.cost(*counts) if valid_counts else None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved, state, max_output FROM calls LEFT JOIN request_bounds USING(id) WHERE id=?", (request_id,)).fetchone()
            invalid = (not valid_counts or not row or row[1] != "reserved" or row[2] is None or
                       counts[0] > self.policy["maximum_input_tokens"] or counts[1] > row[2] or
                       amount > row[0] or usage.get("cache_creation_input_tokens", 0) or usage.get("cache_read_input_tokens", 0))
            if invalid:
                db.execute("INSERT INTO accounting_faults VALUES ('unreconcilable provider usage')")
                if row:
                    # A confirmed overage must remain visible even though no more
                    # requests may be sent. Unknown charges keep their reservation.
                    recorded = max(row[0], amount) if amount is not None else row[0]
                    db.execute("UPDATE calls SET actual=MAX(COALESCE(actual,0),?), state='accounting_fault' WHERE id=?", (recorded, request_id))
            else:
                db.execute("UPDATE calls SET actual=?, state='completed' WHERE id=?", (amount, request_id))
        if invalid:
            raise Stop("Unexpected provider accounting; further calls blocked pending reconciliation.")

    def status(self):
        with self.connect() as db:
            used, pending = db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0), SUM(state!='completed') FROM calls").fetchone()
            faults = db.execute("SELECT COUNT(*) FROM accounting_faults").fetchone()[0]
        return {"cap_usd": str(Decimal(self.limit) / 1_000_000),
                "spent_or_reserved_usd": str(Decimal(used) / 1_000_000), "uncertain_calls": pending or 0,
                "accounting_blocked": bool(faults)}


def repository_root(directory: Path) -> Path:
    common = run(git_argv(directory, "rev-parse", "--path-format=absolute", "--git-common-dir"),
                 directory, env=private_env()).stdout.strip()
    return Path(common).resolve().parent


def load_key(root: Path) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and (root / ".env").is_file():
        for line in (root / ".env").read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                key = match[1].strip("\"'")
                break
    if not key:
        raise Stop("ANTHROPIC_API_KEY is missing; keep it in the environment or ignored root .env.")
    return key


def post_message(payload: dict, key: str) -> dict:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = urllib.request.Request("https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(), method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=90) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        # Provider bodies/headers may contain private context. Do not echo them.
        raise Stop("Supervisor provider request failed; reservation retained, no automatic retry.") from None


class SupervisorAPI:
    def __init__(self, root: Path, policy: dict, *, transport=post_message, key_loader=load_key):
        self.root, self.policy = root, policy
        self.transport, self.key_loader = transport, key_loader
        self.budget = Budget(root / ".ralph/supervisor-budget.sqlite3", policy)

    @classmethod
    def from_worktree(cls, directory: Path):
        """Production entry point: all Git worktrees use the canonical ledger."""
        root = repository_root(directory)
        return cls(root, read_json(Path(__file__).with_name("supervisor-api-policy.json")))

    def message(self, request_id: str, messages: list, *, system: str, max_tokens: int = 1024) -> dict:
        if date.today() > date.fromisoformat(self.policy["pricing_valid_until"]):
            raise Stop("Supervisor pricing needs reverification before further paid requests.")
        if self.policy["model"] != "claude-sonnet-5-5" or self.policy["purpose"] != "simulation-supervision":
            raise Stop("This campaign authorizes Sonnet 5.5 for simulation supervision only.")
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.policy["maximum_output_tokens"]:
            raise Stop("Output token limit exceeds the approved allowance.")
        if not isinstance(system, str) or not isinstance(messages, list) or not messages:
            raise Stop("Supervisor request needs its task/system text and observations.")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != "user" or not isinstance(message["content"], list):
                raise Stop("Only user observation messages are allowed; tools and prompt caches are disabled.")
            for block in message["content"]:
                if not isinstance(block, dict):
                    raise Stop("Observation content must be text or an embedded image.")
                if block.get("type") == "text" and set(block) == {"type", "text"} and isinstance(block["text"], str):
                    continue
                if block.get("type") == "image" and set(block) == {"type", "source"}:
                    source = block["source"]
                    if (isinstance(source, dict) and set(source) == {"type", "media_type", "data"} and source["type"] == "base64" and
                            source["media_type"] in ("image/png", "image/jpeg", "image/webp", "image/gif") and isinstance(source["data"], str)):
                        continue
                raise Stop("Unsupported supervisor content; external tools, documents, URLs and caching are excluded.")
        payload = {"model": self.policy["model"], "max_tokens": max_tokens, "messages": messages,
                   "system": system, "service_tier": "standard_only", "output_config": {"effort": "low"}}
        if len(json.dumps(payload).encode()) > 20_000_000:
            raise Stop("Supervisor request exceeds the local payload size limit.")
        key = self.key_loader(self.root)
        self.budget.reserve(request_id, payload)
        response = self.transport(payload, key)
        if not isinstance(response, dict) or response.get("model") != self.policy["model"]:
            self.budget.fault(request_id)
            raise Stop("Unexpected response model; reservation retained.")
        self.budget.settle(request_id, response.get("usage", {}))
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--request", type=Path, help="JSON containing system, messages and optional max_tokens")
    parser.add_argument("--request-id", help="Stable episode:observation identity; paid retries require reconciliation")
    args = parser.parse_args()
    client = SupervisorAPI.from_worktree(Path(__file__).resolve().parent.parent)
    if args.status:
        print(json.dumps(client.budget.status(), indent=2))
        return
    if not args.request or not args.request_id:
        parser.error("Use --status or --request PATH --request-id ID")
    packet = read_json(args.request)
    result = client.message(args.request_id, packet["messages"], system=packet["system"],
                            max_tokens=packet.get("max_tokens", 1024))
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except (Stop, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"STOP: {exc}") from None
