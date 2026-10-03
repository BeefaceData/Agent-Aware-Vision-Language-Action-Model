"""Render a sealed episode as a standalone local HTML evidence report."""

import argparse
from html import escape
import json
from pathlib import Path

from recorded_replay import TraceError, load_recorded_replay


def _text(value):
    return escape(str(value))


def _json(value):
    return escape(json.dumps(value, indent=2, ensure_ascii=False))


def render_episode_timeline(directory):
    """Validate a bundle and return HTML with embedded, linked source evidence.

    No models, simulator, network resources or JavaScript are used. Missing or
    corrupt required files raise TraceError rather than imply a valid outcome.
    """
    evidence = load_recorded_replay(directory).evidence()
    packets = {row['sequence']: row for row in evidence['observations']}
    parts = ['''<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Episode intervention timeline</title>
<style>
body { font: 16px/1.5 system-ui, sans-serif; margin: 2rem; color: #182331; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #bbc5ce; padding: .6rem; text-align: left; vertical-align: top; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f0f3f6; padding: .7rem; }
.table { overflow-x: auto; } section { margin-top: 2rem; }
a { color: #164da0; } .missing { color: #8b3500; }
</style><body><h1>Episode intervention timeline</h1>''']
    parts.append(f'<p>Source episode: <code>{_text(evidence["source_episode_id"])}</code></p>')
    parts.append('<p>Validated sealed bundle. Observation and decision references below '
                 'resolve to embedded copies of the recorded evidence. This local report '
                 'includes private evaluator evidence; it is not a supervisor input.</p>')
    outcome = evidence['outcome']
    parts.append(f'<p><strong>Task success: {_text(outcome["success"])}</strong>; '
                 f'acknowledged actions: {outcome["steps"]}; '
                 f'termination reason: <code>{_text(outcome["stop_reason"])}</code>; '
                 f'total reward: {_text(outcome["sum_rewards"])}.</p>')
    parts.append('<p>A failed task outcome does not identify its cause. Model diagnosis '
                 'and exact decision/execution timestamps are unavailable in this trace '
                 'format. Times below are original observation-return/capture evidence, '
                 'not replay latency. An acknowledgement confirms an adapter return, '
                 'not independent physical actuation.</p>')
    parts.append('<p class="missing">Video artifacts: unavailable in this bundle contract; '
                 'recorded pixel payloads remain inspectable below.</p>')
    parts.append('<h2>Decisions and actions</h2><div class="table"><table><thead><tr>'
                 '<th>Decision</th><th>Source observation / time</th><th>Proposed → executed</th>'
                 '<th>Disposition / reason</th><th>Result evidence</th></tr></thead><tbody>')
    for decision in evidence['decisions']:
        record = decision['action_record']
        sequence = decision['source_sequence']
        packet = packets[sequence]
        acknowledgement = record['execution_acknowledgement']
        if acknowledgement is None:
            result = 'No execution acknowledgement; no result observation.'
            difference = 'Not executed'
        else:
            target = acknowledgement['result_sequence']
            result = (f'<a href="#observation-{target}">Observation {target}</a>'
                      f'<pre>{_json(decision["result"])}</pre>')
            difference = ('Changed' if record['proposed_action'] != record['executed_action']
                          else 'Unchanged')
        parts.append(f'<tr><td><a href="#decision-{decision["step"]}">'
                     f'{_text(record["proposal_id"])}</a></td>'
                     f'<td><a href="#observation-{sequence}">Observation {sequence}</a><br>'
                     f'{_text(packet["captured_at"])}<br>Monotonic: '
                     f'{_text(packet["captured_monotonic"])}</td>'
                     f'<td>{difference}<pre>{_json(record["proposed_action"])}</pre> → '
                     f'<pre>{_json(record["executed_action"])}</pre></td>'
                     f'<td>{_text(record["disposition"])}<br>'
                     f'{_text(record["rejection_reason"] or "No rejection reason recorded")}</td>'
                     f'<td>{result}</td></tr>')
    parts.append('</tbody></table></div><h2>Recorded observations</h2>')
    for packet in evidence['observations']:
        sequence = packet['sequence']
        parts.append(f'<section id="observation-{sequence}"><h3>Observation {sequence}</h3>'
                     f'<p>observations.jsonl, line {sequence + 1}; episode '
                     f'<code>{_text(packet["episode_id"])}</code>; '
                     f'{_text(packet["captured_at"])}</p><ul>')
        for ref in packet['frame_references']:
            status = ('available in embedded payload' if ref['availability'] == 'available'
                      else 'unavailable (missing recorded view)')
            parts.append(f'<li>{_text(ref["camera"])} / {_text(ref["image_key"])}: {status}; '
                         f'time: {_text(ref["captured_at"])}; '
                         f'basis: {_text(ref["time_basis"])}; '
                         f'synchronization: {_text(ref["synchronization"])}</li>')
        if not packet['frame_references']:
            parts.append('<li>Camera references: unavailable (not recorded).</li>')
        parts.append('</ul><details><summary>Full recorded observation and metadata</summary>'
                     f'<pre>{_json(packet)}</pre></details></section>')
    parts.append('<h2>Recorded decisions</h2>')
    for decision in evidence['decisions']:
        step = decision['step']
        parts.append(f'<section id="decision-{step}"><h3>Decision {step}</h3>'
                     f'<p>decisions.jsonl, line {step}</p>'
                     f'<pre>{_json(decision)}</pre></section>')
    parts.append(f'<h2>Recorded configuration</h2><pre>{_json(evidence["config"])}</pre>'
                 '</body></html>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='sealed replay bundle')
    parser.add_argument('output', help='new local HTML file (never overwritten)')
    args = parser.parse_args()
    try:
        report = render_episode_timeline(args.directory)
        with Path(args.output).open('x', encoding='utf-8') as stream:
            stream.write(report)
    except (TraceError, OSError) as exc:
        parser.exit(1, f'Report unavailable: {exc}\n')
    print(Path(args.output).resolve())


if __name__ == '__main__':
    main()
