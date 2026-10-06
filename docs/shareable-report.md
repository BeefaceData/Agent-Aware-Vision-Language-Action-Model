# Shareable episode exports

The local timeline embeds raw observation and evaluator payloads. To prepare a
separate report for sharing, use a verified final baseline bundle:

```powershell
python shareable_report.py outputs/my-attempt outputs/shareable-attempt
```

This writes `report.html` and `export.json` to a new directory. By default it
copies no evidence files, imagery, raw documents, credentials, model settings,
or free-form trace text. The report contains only recorded success and action
count plus the fixed artifact inventory, hashes and omission reasons. Unknown
files beside the sealed inventory are never exported. It does not upload anything.

After reviewing content and obtaining permission for the intended audience,
select exact inventoried files. Every file defaults to private; including it
requires both selection and an explicit private permission:

```powershell
python shareable_report.py outputs/my-attempt outputs/permitted-example --select episode.mp4 --permit-private episode.mp4
```

An optional `--classifications classifications.json` accepts a JSON object
mapping inventoried filenames to `public`, `private`, or `credentials`. A public
file still requires `--select`; a private file requires both flags; credentials
are excluded even if selected and permitted. Do not classify a file as public
without reviewing its complete contents, including embedded observations or
documents. Classification is a caller declaration, not automated secret detection.
Keep files containing credentials excluded. This command does not redact selected
file contents: allowed files are copied byte for byte under opaque output names.

Python callers use `export_shareable_report(source, destination, selected=(),
classifications=None, permit_private=())`. Invalid selection or classification,
changed source evidence, and existing destinations fail rather than silently
overwriting evidence. The source bundle is verified even if nothing is selected.

The export manifest records the source `bundle.json` SHA-256 and every source
artifact's hash, inclusion status and omission reason. Retain the original seal
privately: a reviewer with access can reconcile every reference against it. The
raw seal is not copied because its identity/settings may contain private data.
Included evidence links resolve inside the export; omitted evidence has no link.
This partial report is not a replay bundle and cannot independently establish
omitted evidence, authenticity, or scientific performance. Original files remain
unchanged and retain their complete replay and integrity contracts.

Run the offline export checks with:

```powershell
python -m unittest discover -s tests -p test_shareable_report.py -v
```

These use a complete synthetic camera episode, private document and credential
sentinels, and public/private/credential classifications. No live inference or
client imagery is required.
