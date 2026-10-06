"""Explicit transport error classes; retry authority belongs to the harness."""

RECOVERABLE_ERRORS = ('rate_limited', 'unavailable')


class RecoverableProviderError(RuntimeError):
    """Transport has completed with a declared transient failure, not a response.

    Transport implementations must not classify decoding or control validation
    failures this way. Neither arbitrary exception text nor payloads are retained.
    """
    def __init__(self, error_class):
        if error_class not in RECOVERABLE_ERRORS:
            raise ValueError('unsupported recoverable provider error')
        self.error_class = error_class
        super().__init__(error_class)
