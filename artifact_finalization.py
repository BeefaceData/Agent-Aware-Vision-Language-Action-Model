"""Close every evidence sink once, retaining all finalization diagnostics."""


class ArtifactCloseError(RuntimeError):
    pass


class ArtifactResources:
    """Own named close callbacks; a failed close cannot become complete on retry."""

    def __init__(self):
        self._callbacks = []
        self._closed = False
        self._diagnostics = ()

    def add(self, name, close):
        if self._closed:
            raise RuntimeError('artifact resources already closed')
        self._callbacks.append((name, close))

    def close(self):
        if not self._closed:
            self._closed = True
            errors = []
            interruption = None
            for name, close in reversed(self._callbacks):
                try:
                    close()
                except BaseException as exc:
                    errors.append(f'{name}: {type(exc).__name__}: {exc}')
                    if not isinstance(exc, Exception) and interruption is None:
                        interruption = exc
            self._diagnostics = tuple(errors)
            if interruption is not None:
                raise interruption
        if self._diagnostics:
            raise ArtifactCloseError('; '.join(self._diagnostics))
