"""Optional adapter evidence when a step has no execution acknowledgement."""


class ExecutionFailure(RuntimeError):
    """A failed/ambiguous step, with transport evidence supplied by the adapter.

    sent=True confirms transmission only, never physical execution. False means
    the adapter can establish no command was sent; None means it cannot tell.
    Adapters must not retry an ambiguous command inside this contract.
    """

    def __init__(self, message='execution acknowledgement unavailable', *, sent=None):
        if sent is not None and type(sent) is not bool:
            raise ValueError('sent must be True, False or None')
        super().__init__(message)
        self.sent = sent
