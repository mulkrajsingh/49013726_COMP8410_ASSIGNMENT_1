# Build small temporary review fixtures and check their displayed evidence and links.
# Test changed files, unsafe paths and incomplete logs as well as valid cases.
# These software fixtures do not add results to the Assignment 1 experiment.

"""Small software fixtures for the offline review index; no experiment results."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote

from build_review_index import Review, allowed_file, build_index
from a1.io import file_records
from a1.study import planned_jobs


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attributes, self.targets, self.ids = [], [], [], set()

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(key for key, value in attrs)
        for key, value in attrs:
            if key in {'href', 'src'}:
                self.targets.append(value)
            if key == 'id':
                self.ids.add(value)


class ReviewIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.write('data/prepared/input.png', b'fixture image bytes, not a model result')
        self.write('data/retail/catalogue.json', '{}')
        self.write('README.md', 'Fixture only')
        self.write('data/manual_reviews.csv', 'job_id,visible_attributes\n')
        self.write('evidence/self_evaluation/observations.csv', 'pair,observations\n')
        cases, models = [], {}
        for pair, kind in [('P01', 'answer'), ('P02', 'image'), ('P03', 'asset'), ('P04', 'agent')]:
            item = {'id': 'item1', 'source_group': 'item1', 'split': 'final',
                    'image': 'data/prepared/input.png', 'files': ['data/prepared/input.png'],
                    'prompt': 'Fixture question <script>prompt()</script>'}
            if pair == 'P04':
                item['catalogue'] = 'data/retail/catalogue.json'
            annotation = {'item1': {'reviewed': True, 'reviewed_by': 'fixture',
                                  'accepted_answers': ['fixture answer'], 'expected': {'status': 'fixture'},
                                  'visible_identifiers': ['Fixture feature']}}
            self.write('data/annotations/' + pair + '.json', json.dumps(annotation))
            cases.append({'pair': pair, 'seeds': [25, 26], 'items': [item],
                          'scoring': {'checks': []}, 'output': {'kind': kind},
                          'annotations_file': 'data/annotations/' + pair + '.json'})
            for side in ['A', 'B']:
                models[pair + '_' + side] = {'model_id': 'fixture/' + side, 'implementation': 'fixture',
                    'weight_revision': '0' * 40, 'code_revision': None, 'components': [], 'parameters': {}}
        self.write('config/study.json', json.dumps({'protocol_version': 'fixture', 'cases': cases}))
        self.write('config/models.json', json.dumps(models))
        self.jobs = planned_jobs(self.root, phase='final')

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        return path

    def job(self, pair):
        return next(job for job in self.jobs if job['pair'] == pair and job['side'] == 'A' and job['seed'] == 25)

    def success(self, job, payload):
        folder = f"outputs/{job['pair']}/{job['side']}/final/{job['job_id']}/attempt_01"
        name = {'answer': 'answer.json', 'agent': 'agent.json', 'image': 'image.png', 'asset': 'asset.glb'}[job['output']['kind']]
        content = json.dumps(payload) if name.endswith('.json') else payload
        output = self.write(folder + '/' + name, content)
        sources = file_records(self.root, job['item']['files'])
        settings = self.write(folder + '/settings.json', json.dumps({'job': job, 'input_files': sources}))
        artifacts = []
        for path in [output, settings]:
            artifacts.append({'path': str(path.relative_to(self.root)), 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        record = {key: job[key] for key in ['job_id', 'pair', 'side', 'phase', 'seed', 'input_id']}
        record.update(status='success', attempt=1, input_files=sources, artifacts=artifacts,
                      output_path=str(output.relative_to(self.root)), output_validation={'kind': job['output']['kind']},
                      model_load_seconds=1.5, generation_seconds=2, end_to_end_seconds=3.5,
                      model_reused=False, hardware={'gpu': {'name': 'Fixture only'}})
        self.write(f"outputs/{job['pair']}/{job['side']}/final/attempts.jsonl", json.dumps(record) + '\n')
        return output, record

    def test_escape_actual_model_text_and_trace(self):
        attack = '<script>alert(1)</script><img src=x onerror="alert(2)">'
        self.success(self.job('P01'), {'answer': attack, 'raw_text': attack, 'structured_parse_success': False})
        self.success(self.job('P04'), {'final': {'message': attack}, 'turns': [{'raw_text': attack}],
                                     'tool_calls': [{'result': attack}], 'termination': 'final'})
        report = build_index(self.root)
        page = (self.root / report['output']).read_text()
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
        parsed = Page()
        parsed.feed(page)
        self.assertNotIn('script', parsed.tags)
        self.assertNotIn('onerror', parsed.attributes)
        self.assertNotIn('onclick', parsed.attributes)
        self.assertEqual(report['states']['saved'], 2)

    def test_unlogged_file_is_pending_and_all_seeds_stay_present(self):
        job = self.job('P02')
        self.write(f"outputs/P02/A/final/{job['job_id']}/attempt_01/image.png", b'unlogged fixture')
        report = build_index(self.root)
        self.assertEqual(report['planned_final_jobs'], 16)
        self.assertEqual(report['states'], {'pending': 16})
        self.assertEqual({row['seed'] for row in report['jobs']}, {25, 26})
        self.assertFalse(any('/image.png' in path for path in report['local_link_targets']))

    def test_changed_artifact_and_changed_input_are_unverified(self):
        job = self.job('P01')
        output, record = self.success(job, {'answer': 'fixture answer'})
        output.write_text('{"answer":"altered"}')
        self.assertEqual(Review(self.root).inspect(job)['state'], 'unverified')
        self.success(job, {'answer': 'fixture answer'})
        self.write('data/prepared/input.png', b'changed fixture input')
        self.assertEqual(Review(self.root).inspect(job)['state'], 'unverified')

    def test_all_links_are_offline_existing_and_inside_root(self):
        self.success(self.job('P01'), {'answer': 'fixture answer'})
        report = build_index(self.root)
        page = Page()
        page.feed((self.root / report['output']).read_text())
        for url in page.targets:
            if url.startswith('#'):
                self.assertIn(url[1:], page.ids)
                continue
            self.assertNotIn(':', url)
            self.assertFalse(url.startswith('//'))
            target = (self.root / 'results/review' / unquote(url)).resolve()
            self.assertTrue(target.is_relative_to(self.root))
            self.assertTrue(target.is_file())

    def test_historical_source_uses_snapshot_without_accepting_changed_input(self):
        # A readability edit may differ from execution, but the historical source
        # and the scientific input still have to match their recorded checksums.
        source = self.write('a1/assets.py', 'original execution source')
        job = self.job('P03')
        job['item']['files'].append('a1/assets.py')
        output, record = self.success(job, b'fixture GLB')
        original = next(row for row in record['input_files'] if row['path']=='a1/assets.py')
        saved = self.write('evidence/executed_source/assets.py', source.read_bytes())
        self.write('evidence/submission_export.json', json.dumps({'files': [], 'execution_sources': [
            {'path': 'a1/assets.py', 'sha256': original['sha256'], 'saved_path': str(saved.relative_to(self.root))}]}))
        source.write_text('commented submission source')
        self.assertEqual(Review(self.root).inspect(job)['state'], 'saved')
        self.write('data/prepared/input.png', b'changed scientific input')
        self.assertEqual(Review(self.root).inspect(job)['state'], 'unverified')

    def test_outside_paths_private_paths_and_malformed_logs_are_rejected(self):
        with self.assertRaises(ValueError):
            allowed_file(self.root, '../outside.json')
        self.write('evidence/runpod_session.json', '{"private":"fixture"}')
        with self.assertRaises(ValueError):
            allowed_file(self.root, 'evidence/runpod_session.json')
        job = self.job('P01')
        self.write('outputs/P01/A/final/attempts.jsonl', '{incomplete')
        self.assertEqual(Review(self.root).inspect(job)['state'], 'pending')

    def test_low_views_must_match_actual_saved_glb_and_image_hash(self):
        job = self.job('P03')
        output, record = self.success(job, b'fixture GLB bytes, not a model result')
        view = self.write(str(output.parent.relative_to(self.root)) + '/inspection/low_yaw_000.png', b'fixture view')
        metadata = {'status': 'complete', 'source_glb_sha256': '0' * 64,
                    'views': [{'file': view.name, 'yaw_degrees': 0,
                               'sha256': hashlib.sha256(view.read_bytes()).hexdigest()}]}
        path = view.parent / 'metadata.json'
        path.write_text(json.dumps(metadata))
        review = Review(self.root)
        info = review.inspect(job)
        self.assertEqual(info['state'], 'saved')
        self.assertNotIn('<img', review.saved_low_views(info))
        metadata['source_glb_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        path.write_text(json.dumps(metadata))
        self.assertIn('<img', review.saved_low_views(info))
        view.write_bytes(b'changed fixture view')
        self.assertNotIn('<img', review.saved_low_views(info))


if __name__ == '__main__':
    unittest.main()
