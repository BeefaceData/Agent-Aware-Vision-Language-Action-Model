"""Exclusive, portable supervisor selection records for frozen experiments."""

from hashlib import sha256
import json
from pathlib import Path


def _encode(identity):
    if type(identity) is not dict or not identity:
        raise ValueError('nonempty supervisor identity required')
    return (json.dumps(identity, sort_keys=True, indent=2, allow_nan=False)
            + '\n').encode('utf-8')


class FrozenSupervisorManifest:
    """Reject drift against both retained bytes and the expected manifest digest.

    Keep the reference in the attempt identity or sealed experiment manifest.
    Loading requires that reference's digest, not a digest recomputed from an
    untrusted replacement file. Metadata must contain no credentials or images.
    """

    def __init__(self, path, expected_sha256):
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        self._read()

    @classmethod
    def freeze(cls, path, identity):
        raw = _encode(identity)
        path = Path(path)
        with path.open('xb') as stream:
            stream.write(raw)
        return cls(path, sha256(raw).hexdigest())

    @property
    def reference(self):
        return {'supervisor_manifest': self.path.name,
                'supervisor_sha256': self.expected_sha256}

    def _read(self):
        raw = self.path.read_bytes()
        if sha256(raw).hexdigest() != self.expected_sha256:
            raise ValueError('frozen supervisor manifest digest mismatch')
        return json.loads(raw)

    def verify(self, identity):
        if _encode(self._read()) != _encode(identity):
            raise ValueError('supervisor selection differs from frozen manifest')
        return self.reference
