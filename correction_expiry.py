"""Host-owned monotonic validity windows, retained through queued execution."""

from math import isfinite


def check_expiry(config, requested_at, captured_at, source_proposal_id, checked_at):
    """Use an exclusive request deadline and inclusive maximum observation age.

    A recovery retains its original request and capture times for every command;
    neither arrival nor fresh intermediate observations renew its authority.
    """
    times = (requested_at, captured_at, checked_at)
    if (any(type(value) not in (int, float) or not isfinite(value) for value in times)
            or not captured_at <= requested_at <= checked_at):
        raise ValueError('correction expiry requires ordered finite monotonic times')
    deadline = requested_at + config.correction_timeout_seconds
    if not isfinite(deadline):
        raise ValueError('correction deadline must be finite')
    age = checked_at - captured_at
    reason = ('correction request expired' if checked_at >= deadline else
              'correction observation too old' if age > config.correction_max_age_seconds else
              None)
    return dict(source_proposal_id=source_proposal_id, requested_at=requested_at,
                captured_at=captured_at, deadline=deadline, checked_at=checked_at,
                max_age_seconds=config.correction_max_age_seconds,
                observation_age_seconds=age, valid=reason is None, reason=reason)
