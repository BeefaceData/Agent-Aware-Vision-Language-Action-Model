"""Episode-wide monotonic allowance and invalidation of late callback results."""
from math import isfinite
from queue import Empty, Queue
from threading import Lock, Thread


class EpisodeDeadlineExceeded(TimeoutError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__('episode wall-clock limit exhausted')


_lock = Lock()
_workers = {}


def workers_busy(identities):
    with _lock:
        return any(_workers.get(identity, 0) for identity in identities)


class EpisodeDeadline:
    """Bound waiting, not Python thread lifetime or physical stopping latency.

    Late workers retain ownership of their adapters until they actually exit.
    A callback has no authority to dispatch another action through the harness.
    The adapter must make interrupt safe concurrently with an in-flight step.
    """
    def __init__(self, limit, clock, owners):
        self.limit, self.clock, self.owners = limit, clock, owners
        self.started_at = clock()
        self.deadline = None if limit is None else self.started_at + limit
        if self.deadline is not None and not isfinite(self.deadline):
            raise ValueError('episode deadline must be finite')

    def check(self, stage):
        if self.deadline is not None and self.clock() >= self.deadline:
            raise EpisodeDeadlineExceeded(stage)

    def call(self, stage, callback, *args, accept_late_return=False):
        if self.limit is None:
            return callback(*args)
        self.check(stage)
        mailbox = Queue(maxsize=1)
        identities = {id(owner) for owner in self.owners if owner is not None}
        with _lock:
            for identity in identities:
                _workers[identity] = _workers.get(identity, 0) + 1

        def work():
            try:
                self.check(stage)
                try:
                    mailbox.put((True, callback(*args)))
                except BaseException as exc:
                    mailbox.put((False, exc))
            except BaseException as exc:
                mailbox.put((False, exc))
            finally:
                with _lock:
                    for identity in identities:
                        _workers[identity] -= 1
                        if not _workers[identity]:
                            del _workers[identity]
        try:
            Thread(target=work, daemon=True).start()
        except BaseException:
            with _lock:
                for identity in identities:
                    _workers[identity] -= 1
                    if not _workers[identity]:
                        del _workers[identity]
            raise
        while True:
            try:
                ok, value = mailbox.get_nowait()
            except Empty:
                self.check(stage)
                try:
                    ok, value = mailbox.get(timeout=min(.01, max(0, self.deadline - self.clock())))
                except Empty:
                    continue
            if not accept_late_return:
                self.check(stage)
            if not ok:
                raise value
            return value

    def evidence(self, stage, interruption, pending_call=None):
        return dict(limit_seconds=self.limit, started_at=self.started_at,
                    deadline=self.deadline, expired_at=self.clock(), stage=stage,
                    interruption=interruption, pending_supervisor_call=pending_call)
