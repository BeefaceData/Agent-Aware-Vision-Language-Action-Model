"""Detect declared runtime drift; never interpret model or memory text as code."""

from collections.abc import Mapping
import json


class FrozenContractViolation(ValueError):
    """The episode can no longer execute under its declared frozen contract."""


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


class FrozenRuntimeContract:
    """Snapshot trusted, credential-free identity/configuration readers.

    Readers belong to host code, not supervisor responses or memory records.
    Include policy/VLM identity, prompt digest, tool declarations and controller
    limits as applicable. This checks declared state, not arbitrary Python code
    or remote provider weights. Persist ``identity`` in the attempt manifest.
    """

    def __init__(self, **readers):
        if not readers or any(not callable(reader) for reader in readers.values()):
            raise ValueError('trusted configuration readers required')
        self._readers = dict(readers)
        self._expected = _encode({name: reader() for name, reader in readers.items()})

    @property
    def identity(self):
        return json.loads(self._expected)

    def verify(self):
        try:
            actual = _encode({name: reader() for name, reader in self._readers.items()})
        except Exception as exc:
            raise FrozenContractViolation('frozen runtime identity unavailable') from exc
        if actual != self._expected:
            raise FrozenContractViolation('frozen runtime identity changed')


class EpisodeInstruction:
    """Freeze the initial task field, including its absence in legacy replay."""

    def __init__(self, packet):
        self._expected = self._read(packet)

    @staticmethod
    def _read(packet):
        raw = packet.observation
        return _encode({'task': raw['task']} if isinstance(raw, Mapping) and 'task' in raw else {})

    def verify(self, packet):
        if self._read(packet) != self._expected:
            raise FrozenContractViolation('VLA task instruction changed during episode')
