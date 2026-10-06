"""Export explicitly permitted evidence from a verified final episode bundle."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

from artifact_bundle import verify_artifact_bundle
from recorded_replay import TraceError


def export_shareable_report(directory, destination, *, selected=(),
                            classifications=None, permit_private=()):
    """Export locally, without publishing or copying the raw source manifest.

    Every artifact defaults to private. Selection and private permission are
    separate explicit allowlists. Credential-tagged files can never be exported.
    Classifications are supplied by the caller after content review; this API
    cannot detect secrets hidden inside a file declared public.
    """
    root = Path(directory)
    source_bytes = (root / 'bundle.json').read_bytes()
    replay = verify_artifact_bundle(root)
    source = json.loads(source_bytes)
    files = source['files']
    selected, permitted = set(selected), set(permit_private)
    tags = dict(classifications or {})
    if not (selected | permitted | tags.keys()) <= files.keys():
        raise ValueError('selection, permission and classification must name inventoried artifacts')
    if not permitted <= selected:
        raise ValueError('private permission requires explicit selection')
    if any(tag not in ('public', 'private', 'credentials') for tag in tags.values()):
        raise ValueError('unknown artifact classification')
    payloads, entries = {}, []
    for index, (name, digest) in enumerate(sorted(files.items())):
        tag = tags.get(name, 'private')
        included = name in selected and (tag == 'public' or
                                         (tag == 'private' and name in permitted))
        reason = ('included' if included else 'credentials excluded' if tag == 'credentials'
                  else 'not selected' if name not in selected else 'private permission required')
        # Opaque output names avoid disclosing source paths or evaluating them as links.
        target = f'evidence/{index:03d}.bin' if included else None
        entries.append({'source_artifact': name, 'sha256': digest,
                        'included': included, 'reason': reason, 'export_path': target})
        if included:
            payload = (root / name).read_bytes()
            if sha256(payload).hexdigest() != digest:
                raise TraceError('selected artifact changed during export')
            payloads[target] = payload
    if (root / 'bundle.json').read_bytes() != source_bytes:
        raise TraceError('source manifest changed during export')
    outcome = replay.evidence()['outcome']
    # Deliberately exclude free-form strings, paths, model settings and observation payloads.
    summary = {'success': outcome['success'], 'steps': outcome['steps']}
    manifest = {'version': 1, 'source_bundle_sha256': sha256(source_bytes).hexdigest(),
                'summary': summary, 'artifacts': entries,
                'limitations': 'Partial export; omitted evidence requires the retained source bundle. '
                               'Checksums establish integrity relative to that bundle, not authenticity.'}
    rows = ''.join('<tr><td>' + escape(entry['source_artifact']) + '</td><td>' +
                   escape(entry['sha256']) + '</td><td>' +
                   (f'<a href="{entry["export_path"]}">Included evidence</a>'
                    if entry['included'] else escape(entry['reason'])) + '</td></tr>'
                   for entry in entries)
    report = ('<!doctype html><html lang="en"><meta charset="utf-8">'
              '<title>Shareable episode report</title><h1>Shareable episode report</h1>'
              f'<p>Recorded success: {summary["success"]}; acknowledged actions: {summary["steps"]}.</p>'
              '<p>This is a partial evidence export, not a replay bundle or a performance claim.</p>'
              f'<p>Source bundle SHA-256: {manifest["source_bundle_sha256"]}</p>'
              '<p>Omitted artifacts remain in the retained source bundle. No omitted link is followed.</p>'
              '<table><tr><th>Source artifact</th><th>SHA-256</th><th>Export status</th></tr>'
              + rows + '</table></html>')
    destination = Path(destination)
    # Refuse existing destinations, including the source or a prior export.
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = destination / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
    (destination / 'report.html').write_text(report, encoding='utf-8')
    (destination / 'export.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return destination / 'report.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('destination')
    parser.add_argument('--select', action='append', default=[])
    parser.add_argument('--permit-private', action='append', default=[])
    parser.add_argument('--classifications', type=Path,
                        help='JSON mapping inventoried filenames to public/private/credentials')
    args = parser.parse_args()
    print(export_shareable_report(args.directory, args.destination,
        selected=args.select, permit_private=args.permit_private,
        classifications=json.loads(args.classifications.read_text(encoding='utf-8'))
        if args.classifications else None))
