"""Write baseline identity before policy control; retain it with outcome artifacts."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path


def reserve_attempt(directory):
    """Claim a new destination without modifying any existing attempt evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


class AttemptIdentityRecorder:
    """Recorder wrapper whose trusted host supplies resolved identity after reset.

    The callback must return JSON-compatible task, applied-state, model and
    effective runtime settings. These are evaluator evidence, not policy inputs.
    A failed write prevents policy control; an interrupted run keeps its manifest.
    """

    def __init__(self, directory, config, identity, recorder):
        self.path = Path(directory) / 'attempt.json'
        self.config = config
        self.identity = identity
        self.recorder = recorder
        self.reference = {}

    def begin(self, observation):
        manifest = {**self.identity(), 'version': 1,
                    'episode_id': observation.episode_id,
                    'episode_config': asdict(self.config)}
        raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)
               + '\n').encode('utf-8')
        with self.path.open('xb') as stream:
            stream.write(raw)
        self.reference = {'attempt_manifest': self.path.name,
                          'attempt_sha256': sha256(raw).hexdigest()}
        self.recorder.begin(observation)

    def record_step(self, *args):
        self.recorder.record_step(*args)

    def record_failure(self, *args):
        self.recorder.record_failure(*args)

    def finish(self):
        return {**self.recorder.finish(), **self.reference}
