"""Final baseline evidence inventory and offline integrity verification."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from recorded_replay import TraceError, load_recorded_replay, _read


REQUIRED = frozenset(('attempt.json', 'environment.json', 'policy-assets.json',
    'result.json', 'steps.jsonl', 'frames.jsonl', 'replay/manifest.json',
    'replay/observations.jsonl', 'replay/decisions.jsonl', 'replay/execution.jsonl'))
VIDEOS = {'main': 'episode.mp4', 'wrist': 'episode_wrist.mp4'}


def _digest(path):
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(root):
    attempt = _read((root / 'attempt.json').read_bytes())
    result = _read((root / 'result.json').read_bytes())
    if (attempt['version'] != 1 or result['status'] != 'completed' or
            result['artifact_status'] != 'completed' or
            result['episode_id'] != attempt['episode_id'] or
            result['attempt_sha256'] != _digest(root / 'attempt.json')):
        raise TraceError('incomplete or inconsistent attempt identity')
    assets = _read((root / 'policy-assets.json').read_bytes())
    if not assets or assets != attempt['policy_assets'] or assets != result['policy_assets']:
        raise TraceError('inconsistent model identity')
    replay = load_recorded_replay(root / 'replay')
    if (replay.source_episode_id != attempt['episode_id'] or
            replay.evidence()['config'] != attempt['episode_config']):
        raise TraceError('inconsistent replay identity/configuration')
    rows = [_read(line) for line in (root / 'frames.jsonl').read_bytes().splitlines()]
    observations = [_read(line) for line in
                    (root / 'replay/observations.jsonl').read_bytes().splitlines()]
    expected = [(packet, ref) for packet in observations for ref in packet['frame_references']]
    if len(rows) != len(expected):
        raise TraceError('missing or extra frame index entry')
    names = set(REQUIRED)
    counts = dict(main=0, wrist=0)
    for row, (packet, ref) in zip(rows, expected):
        camera = ref['camera']
        available = ref['availability'] == 'available'
        if (row['episode_id'] != attempt['episode_id'] or
                row['observation_sequence'] != packet['sequence'] or
                row['camera'] != camera or row['availability'] != ref['availability'] or
                row['frame_index'] != (counts[camera] if available else None)):
            raise TraceError('inconsistent frame index')
        if available:
            names.add(VIDEOS[camera])
            counts[camera] += 1
    identity = {'episode_id': attempt['episode_id'],
                'episode_config': attempt['episode_config'],
                'policy_assets': assets, 'settings': attempt['settings']}
    return names, identity, replay


def seal_artifact_bundle(directory):
    """Publish once, after result.json and all evidence sinks have closed."""
    root = Path(directory)
    names, identity, _ = _evidence(root)
    manifest = {'version': 1, 'identity': identity,
                'files': {name: _digest(root / name) for name in sorted(names)}}
    path = root / 'bundle.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def verify_artifact_bundle(directory):
    """Return replay only after checking the final inventory and identities.

    Relative inventory names allow moving a bundle. Digests detect accidental
    changes, not forgery by someone who can rewrite the manifest and evidence.
    Video bytes are checked without loading a codec or model.
    """
    try:
        root = Path(directory)
        manifest = _read((root / 'bundle.json').read_bytes())
        if type(manifest['version']) is not int or manifest['version'] != 1:
            raise TraceError('unsupported artifact bundle version')
        files = manifest['files']
        if not REQUIRED <= set(files) or not set(files) <= REQUIRED | set(VIDEOS.values()):
            raise TraceError('invalid required artifact inventory')
        for name, digest in files.items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or _digest(path) != digest:
                raise TraceError(f'artifact checksum mismatch: {name}')
        names, identity, replay = _evidence(root)
        if set(files) != names or manifest['identity'] != identity:
            raise TraceError('inconsistent bundle inventory or identity')
        return replay
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise TraceError(f'invalid artifact bundle: {exc}') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    verified = verify_artifact_bundle(args.directory)
    print(json.dumps({'verified': True, 'episode_id': verified.source_episode_id,
                      'replayed': verified.run().steps if args.replay else None}))
