"""Public report checks against complete episodes and retained source evidence."""

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionResolution, run_episode
from episode_timeline import render_episode_timeline
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import successful_replay


FIXTURE = Path(__file__).parent / 'fixtures' / 'recorded_episode'


class ReportParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids, self.links, self.tags, self.words = [], [], [], []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if 'href' in attrs:
            self.links.append(attrs['href'])

    def handle_data(self, data):
        self.words.append(data)


class EpisodeTimelineTests(unittest.TestCase):
    def test_failed_fixture_links_resolve_and_actions_match_replay(self):
        replay = load_recorded_replay(FIXTURE)
        self.assertFalse(replay.run().success)
        report = render_episode_timeline(FIXTURE)
        parsed = ReportParser(report)
        self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
        self.assertTrue(parsed.links)
        for link in parsed.links:
            self.assertTrue(link.startswith('#'))
            self.assertIn(link[1:], parsed.ids)
        self.assertIn('Changed<pre>[\n  0.1\n]</pre> → <pre>[\n  0.5\n]</pre>', report)
        self.assertIn('Unchanged', report)
        self.assertIn('Task success: False', report)
        self.assertIn('termination reason: <code>terminated</code>', report)
        self.assertIn('2026-01-01T00:00:00+00:00', report)
        self.assertIn('unavailable (missing recorded view)', report)
        self.assertIn('observations.jsonl, line 3', report)
        self.assertNotIn('script', parsed.tags)

    def test_success_and_rejection_render_without_inventing_execution(self):
        for reject in (False, True):
            with self.subTest(reject=reject), TemporaryDirectory() as directory:
                fixture = successful_replay()
                path = Path(directory) / 'trace'
                trace = TraceRecorder(path, fixture.config, fixture.recorder)
                reason = '<script>alert("unsafe")</script>'
                selector = (lambda proposal: ActionResolution('reject', reason=reason)) if reject else None
                outcome = run_episode(fixture.config, fixture.policy, fixture.environment,
                                      trace, action_selector=selector)
                trace.seal(outcome)
                report = render_episode_timeline(path)
                parsed = ReportParser(report)
                self.assertIn(f'Task success: {outcome.success}', report)
                self.assertIn(outcome.stop_reason, report)
                self.assertNotIn('script', parsed.tags)
                if reject:
                    self.assertIn(reason, ''.join(parsed.words))
                    self.assertIn('Not executed', report)
                    self.assertIn('No execution acknowledgement', report)
                    self.assertNotIn('observation-1', parsed.ids)
                else:
                    self.assertIn('observation-2', parsed.ids)

    def test_inspection_is_detached_and_preserves_original_timestamps(self):
        replay = load_recorded_replay(FIXTURE)
        evidence = replay.evidence()
        evidence['decisions'][0]['action_record']['executed_action'][0] = 99
        evidence['observations'][0]['observation'].clear()
        self.assertEqual(replay.evidence()['decisions'][0]['action_record']['executed_action'], [0.5])
        self.assertEqual(replay.evidence()['observations'][0]['captured_monotonic'], 100)
        self.assertFalse(replay.run().success)

    def test_required_missing_or_corrupt_evidence_is_refused(self):
        for name in ('manifest.json', 'observations.jsonl', 'decisions.jsonl'):
            for corrupt in (False, True):
                with self.subTest(name=name, corrupt=corrupt), TemporaryDirectory() as directory:
                    path = Path(directory) / 'trace'
                    shutil.copytree(FIXTURE, path)
                    if corrupt:
                        (path / name).write_text('broken', encoding='utf-8')
                    else:
                        (path / name).unlink()
                    with self.assertRaises(TraceError):
                        render_episode_timeline(path)

    def test_cli_writes_standalone_report_and_preserves_existing_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'timeline.html'
            command = [sys.executable, str(FIXTURE.parents[2] / 'episode_timeline.py'),
                       str(FIXTURE), str(path)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = path.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_bytes(), report)
            missing = subprocess.run(command[:-2] + [str(Path(directory) / 'missing'),
                                                     str(Path(directory) / 'absent.html')],
                                     capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn('Report unavailable:', missing.stderr)
            self.assertFalse((Path(directory) / 'absent.html').exists())


if __name__ == '__main__':
    unittest.main()
