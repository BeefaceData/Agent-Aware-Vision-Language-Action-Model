"""Deadline-bounded provider boundary; no retries or execution authority."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from queue import Empty, Queue
from sys import float_info
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Literal

from episode_harness import (ActionProposal, SupervisorResponseError,
                             supervisor_observation)
from supervisor_response import SupervisorResponseDecoder
from supervisor_retry import RecoverableProviderError


@dataclass(frozen=True)
class ProviderResult:
    episode_id: str
    observation_sequence: int
    proposal_id: str
    requested_at: float
    deadline: float
    finished_at: float
    status: Literal['response', 'timeout', 'error', 'cancelled', 'busy', 'rejected']
    decision: Any = None
    reason: str | None = None
    error_class: str | None = None


class ProviderRequestError(SupervisorResponseError):
    """A failed request, retaining typed evidence without provider payloads."""

    def __init__(self, result: ProviderResult):
        self.result = result
        super().__init__(f'supervisor provider {result.status}: {result.reason}')


class BoundedSupervisorProvider:
    """Wrap provider(proposal, deadline, cancellation_event) with a deadline.

    The provider receives sanitized detached input and must honor the monotonic
    deadline and cancellation signal in its own transport. A daemon worker
    bounds caller waiting even for an uncooperative transport. Python cannot
    kill that worker: until it exits, further requests return busy. Late output
    is discarded in a request-local mailbox and can never authorize a new turn.
    No retry, fallback action, or remote cancellation guarantee is implied.
    """

    def __init__(self, provider, timeout_seconds: float,
                 decoder: SupervisorResponseDecoder | None = None):
        if (type(timeout_seconds) not in (int, float) or
                not 0 < timeout_seconds <= float_info.max):
            raise ValueError('provider timeout must be finite and positive')
        if not callable(provider):
            raise ValueError('provider must be callable')
        self._provider = provider
        self._timeout = timeout_seconds
        self._decoder = decoder or SupervisorResponseDecoder()
        self._inflight = Lock()

    def request(self, proposal: ActionProposal,
                cancellation: Event | None = None) -> ProviderResult:
        started = monotonic()
        deadline = started + self._timeout
        if not isfinite(deadline):
            raise ValueError('provider deadline must be finite')
        identity = (proposal.observation.episode_id,
                    proposal.observation.sequence, proposal.proposal_id)
        cancel = cancellation if cancellation is not None else Event()

        def result(status, decision=None, reason=None, error_class=None):
            return ProviderResult(*identity, started, deadline, monotonic(),
                                  status, decision, reason, error_class)

        if cancel.is_set():
            return result('cancelled', reason='request cancelled')
        if not self._inflight.acquire(blocking=False):
            return result('busy', reason='previous provider request still running')
        mailbox = Queue(maxsize=1)

        def work():
            try:
                current = ActionProposal(proposal.proposal_id,
                                         supervisor_observation(proposal.observation),
                                         deepcopy(proposal.action))
                try:
                    response = self._provider(deepcopy(current), deadline, cancel)
                except RecoverableProviderError as exc:
                    mailbox.put(('error', None, 'recoverable provider failure', exc.error_class))
                    return
                except BaseException:
                    mailbox.put(('error', None, 'provider transport or processing failed'))
                    return
                decision = self._decoder.decode(response, current)
                mailbox.put(('response', decision, None))
            except SupervisorResponseError as exc:
                mailbox.put(('rejected', None, str(exc)))
            except BaseException:
                # Provider exception text can contain credentials or raw inputs.
                mailbox.put(('error', None, 'provider transport or processing failed'))
            finally:
                self._inflight.release()

        try:
            Thread(target=work, daemon=True).start()
        except BaseException:
            self._inflight.release()
            return result('error', reason='provider worker could not start')
        while True:
            if cancel.is_set():
                return result('cancelled', reason='request cancelled')
            remaining = deadline - monotonic()
            if remaining <= 0:
                cancel.set()
                return result('timeout', reason='configured request deadline exceeded')
            try:
                entry = mailbox.get(timeout=min(remaining, .01))
                status, decision, reason = entry[:3]
                error_class = entry[3] if len(entry) == 4 else None
            except Empty:
                continue
            if cancel.is_set():
                return result('cancelled', reason='request cancelled')
            if monotonic() >= deadline:
                cancel.set()
                return result('timeout', reason='configured request deadline exceeded')
            return result(status, decision, reason, error_class)

    def __call__(self, proposal: ActionProposal):
        """Use as run_episode(supervisor_decider=adapter).

        Failures use the harness's recorded rejection/termination path. Correction
        requests retain their existing lack of execution authority.
        """
        result = self.request(proposal)
        if result.status != 'response':
            raise ProviderRequestError(result)
        return result.decision
