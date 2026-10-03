"""Append-only model-call evidence, independent of decision/execution authority."""

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4


@dataclass(frozen=True)
class ModelReply:
    """A transport response plus reported metadata; never inferred pricing."""

    response: object
    usage: dict | None = None
    cost: float | None = None
    currency: str | None = None


def _metadata(reply):
    usage = reply.usage
    if usage is not None and (type(usage) is not dict or any(
            type(key) is not str or not key or type(value) not in (int, float)
            or not isfinite(value) or value < 0 for key, value in usage.items())):
        raise ValueError('usage must contain nonnegative finite reported units')
    if reply.cost is None:
        if reply.currency is not None:
            raise ValueError('currency requires reported cost')
    elif (type(reply.cost) not in (int, float) or not isfinite(reply.cost)
          or reply.cost < 0 or type(reply.currency) is not str or not reply.currency):
        raise ValueError('reported cost requires a finite amount and currency')
    return dict(usage=usage, cost=reply.cost, currency=reply.currency)


class ModelCallJournal:
    """One new JSONL file per episode; flush starts before calling a transport.

    Each explicit retry is another call, optionally linked by retry_of. No retry
    is initiated here. A start without a finish means pending/unknown, including
    after caller timeout or process interruption. Late transport completion may
    append accounting evidence but cannot change a supervisor decision.
    """

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('x', encoding='utf-8'):
            pass
        self._lock = Lock()
        self._episode_id = None
        self._calls = set()

    def _append(self, row):
        # Opening for each event permits late workers to finish after the harness
        # has finalized its other sinks. This journal is not a sealed replay file.
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
            stream.flush()

    def call(self, proposal, provider, model, operation, *, retry_of=None):
        """Call operation() once, returning its response and retaining accounting.

        operation may return ModelReply or an ordinary response (unknown usage).
        Provider/model and metric names must be caller-approved public metadata.
        Exceptions and response bodies are never serialized.
        """
        if any(type(value) is not str or not value for value in (provider, model)):
            raise ValueError('explicit provider and model identity required')
        episode = proposal.observation.episode_id
        with self._lock:
            if self._episode_id is not None and episode != self._episode_id:
                raise ValueError('model-call journal belongs to another episode')
            if retry_of is not None and retry_of not in self._calls:
                raise ValueError('retry must reference an existing journal call')
            self._episode_id = episode
            call_id = uuid4().hex
            start = monotonic()
            self._append(dict(version=1, event='started', call_id=call_id,
                              episode_id=episode, proposal_id=proposal.proposal_id,
                              observation_sequence=proposal.observation.sequence,
                              provider=provider, model=model, retry_of=retry_of,
                              started_monotonic=start))
            self._calls.add(call_id)
        metadata = dict(usage=None, cost=None, currency=None)
        status = 'error'
        try:
            reply = operation()
            if type(reply) is ModelReply:
                metadata = _metadata(reply)
                response = reply.response
            else:
                response = reply
            status = 'returned'
            return response
        finally:
            with self._lock:
                self._append(dict(version=1, event='finished', call_id=call_id,
                                  status=status, wall_seconds=monotonic() - start,
                                  **metadata))

    def summary(self):
        """Detached per-call and per-episode totals; missing values stay unknown."""
        with self._lock:
            rows = [json.loads(line) for line in self.path.read_text(
                encoding='utf-8').splitlines()]
        calls = {}
        for row in rows:
            if row['event'] == 'started':
                calls[row['call_id']] = {**row, 'status': 'pending',
                                        'wall_seconds': None, 'usage': None,
                                        'cost': None, 'currency': None}
            else:
                calls[row['call_id']].update(row)
        values = list(calls.values())
        known_cost = {}
        known_usage = {}
        for row in values:
            for unit, amount in (row['usage'] or {}).items():
                known_usage[unit] = known_usage.get(unit, 0) + amount
            if row['cost'] is not None:
                currency = row['currency']
                known_cost[currency] = known_cost.get(currency, 0) + row['cost']
        complete_cost = all(row['cost'] is not None for row in values)
        return dict(episode_id=self._episode_id, call_count=len(values), calls=values,
                    pending_calls=sum(row['status'] == 'pending' for row in values),
                    failed_calls=sum(row['status'] == 'error' for row in values),
                    reported_usage_subtotals=known_usage,
                    total_usage={unit: (amount if all(
                        row['usage'] is not None and unit in row['usage']
                        for row in values) else None)
                        for unit, amount in known_usage.items()},
                    unknown_usage_calls=sum(row['usage'] is None for row in values),
                    reported_cost_subtotals=known_cost,
                    total_cost_by_currency=known_cost if complete_cost else None,
                    unknown_cost_calls=sum(row['cost'] is None for row in values),
                    call_wall_seconds=(sum(row['wall_seconds'] for row in values)
                                       if all(row['wall_seconds'] is not None
                                              for row in values) else None))
