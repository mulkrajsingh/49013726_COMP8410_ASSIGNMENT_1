# Read the saved study, attempt records and local outputs; build an offline review page.
# Verify evidence and local links before writing the HTML and build receipt.
# The page displays existing outputs; it does not add ratings or generate media.

"""Build a plain offline index of saved final outputs. No inference or ratings."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a1.io import project_path, read_attempts
from a1.study import planned_jobs
from a1.evidence import verify_inputs, verified_file


def esc(value):
    return html.escape(str(value), quote=True)


def json_text(value):
    return esc(json.dumps(value, indent=2, ensure_ascii=False))


def allowed_file(root, relative):
    """Allow project evidence only, including after resolving symlinks."""
    path = project_path(root, str(relative))
    parts = path.relative_to(root).parts
    if not parts or parts[0] not in {'outputs', 'data', 'evidence', 'results', 'config', 'a1', 'scripts', 'README.md'}:
        raise ValueError('File is outside the review sources')
    blocked = {'cache', 'third_party', '.git', '.aws', '.ssh', 'secrets',
               'credentials', 'runpod_session.json'}
    if any(part.lower() in blocked for part in parts):
        raise ValueError('Private or cached source is excluded')
    return path


class Review:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.folder = self.root / 'results/review'
        self.hashes = {}
        self.links = set()

    def digest(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
            self.hashes[key] = h.hexdigest()
        return self.hashes[key]

    def file(self, relative):
        path = allowed_file(self.root, relative)
        if not path.is_file():
            raise FileNotFoundError('Saved local file is missing')
        return path

    def url(self, path):
        path = self.file(path)
        if path.suffix.lower() in {'.pt', '.bin', '.safetensors', '.pkl', '.html', '.js'}:
            raise ValueError('This file type is not linked by the review page')
        self.links.add(str(path.relative_to(self.root)))
        return quote(os.path.relpath(path, self.folder), safe='/')

    def link(self, path, label, download=False):
        try:
            url = self.url(path)
        except (OSError, ValueError):
            return '<span>Not imported locally</span>'
        return f'<a href="{esc(url)}"' + (' download' if download else '') + f'>{esc(label)}</a>'

    def image(self, path, label):
        try:
            url = self.url(path)
        except (OSError, ValueError):
            return '<p>Image not available in this local snapshot.</p>'
        return f'<img src="{esc(url)}" alt="{esc(label)}" loading="lazy">'

    def inspect(self, job):
        """Check hashes and recorded validation; do not load a GLB or ML library."""
        log = f"outputs/{job['pair']}/{job['side']}/final/attempts.jsonl"
        try:
            rows = read_attempts(self.file(log))
        except FileNotFoundError:
            return {'state': 'pending', 'message': 'Pending: no matching completed local attempt.'}
        except (OSError, ValueError, KeyError):
            return {'state': 'pending', 'message': 'Pending: local log is incomplete or unreadable.'}
        rows = [row for row in rows if row.get('job_id') == job['job_id']]
        if not rows:
            return {'state': 'pending', 'message': 'Pending: no attempt for this current planned job ID.'}
        record = rows[-1]
        if record.get('status') != 'success':
            state = 'pending' if record.get('status') in {'started', 'interrupted'} else 'failed'
            return {'state': state, 'message': 'No completed local output. Recorded status: ' + str(record.get('status')),
                    'record': record, 'log': log}
        try:
            # Historical source comes from the preserved execution snapshot.
            # Images and catalogue inputs must still match the recorded bytes.
            verify_inputs(self.root, job, record)
            for key in ['pair', 'side', 'phase', 'seed', 'input_id']:
                if record.get(key) != job[key]:
                    raise ValueError('Saved attempt identity differs from the planned job.')
            artifacts = record.get('artifacts', [])
            if not artifacts:
                raise ValueError('No recorded artifact checksums.')
            attempt_folder = allowed_file(self.root,
                f"outputs/{job['pair']}/{job['side']}/final/{job['job_id']}/attempt_{int(record['attempt']):02d}")
            verified = {}
            for item in artifacts:
                path = verified_file(self.root, item)
                if not path.is_relative_to(attempt_folder):
                    raise ValueError('Artifact leaves its recorded job attempt folder.')
                verified[str(path.relative_to(self.root))] = item
            output = self.file(record['output_path'])
            if output.parent != attempt_folder:
                raise ValueError('Output leaves its recorded job attempt folder.')
            output_relative = str(output.relative_to(self.root))
            if output_relative not in verified or output.stat().st_size == 0:
                raise ValueError('Required output is not in the verified artifact list.')
            expected_suffix = {'asset': '.glb', 'image': '.png', 'answer': '.json', 'agent': '.json'}[job['output']['kind']]
            if output.suffix != expected_suffix:
                raise ValueError('Unexpected output file type.')
            settings = output.parent / 'settings.json'
            if str(settings.relative_to(self.root)) not in verified:
                raise ValueError('Saved settings are not in the artifact list.')
            saved = json.loads(settings.read_text())
            if saved.get('job') != job or saved.get('input_files') != record['input_files']:
                raise ValueError('Saved scientific settings differ from this planned job.')
            if record.get('output_validation', {}).get('kind') != job['output']['kind']:
                raise ValueError('Recorded worker output validation is missing.')
            if job['output'].get('turntable_required'):
                for name in ['preview.png', 'preview.json', 'turntable.mp4']:
                    path = output.parent / name
                    if str(path.relative_to(self.root)) not in verified or path.stat().st_size == 0:
                        raise ValueError('Required saved preview is not verified.')
            return {'state': 'saved', 'message': 'Saved success; input, artifact and historical-source checks pass.',
                    'record': record, 'log': log, 'output': output, 'verified': verified,
                    'verification': 'SHA256, size, explicit metadata exports, preserved execution source, job/settings and recorded worker validation. No new output decoding.'}
        except (OSError, ValueError, KeyError, TypeError):
            return {'state': 'unverified', 'message': 'Unverified: local evidence is incomplete, changed, or does not match the current job.',
                    'record': record, 'log': log}

    def saved_low_views(self, info):
        folder = info['output'].parent / 'inspection'
        try:
            metadata_path = self.file(folder / 'metadata.json')
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('status') != 'complete' or metadata.get('source_glb_sha256') != self.digest(info['output']):
                return '<p>Additional views are not verified for this saved GLB.</p>'
            parts = ['<details><summary>Saved low views</summary><div class="low-views">']
            views = sorted(metadata.get('views', []), key=lambda row: row.get('yaw_degrees', 0))
            count = 0
            for view in views:
                name = view.get('file', '')
                if Path(name).name != name or not name.startswith('low_yaw_') or not name.endswith('.png'):
                    continue
                image = self.file(folder / name)
                if self.digest(image) != view['sha256']:
                    continue
                label = 'Yaw ' + str(view.get('yaw_degrees')) + ' degrees'
                parts.append('<figure>' + self.image(image, label) + f'<figcaption>{esc(label)}</figcaption></figure>')
                count += 1
            if not count:
                return '<p>Additional low views are not imported locally.</p>'
            parts.append('</div>' + self.link(metadata_path, 'Saved view/camera metadata') + '</details>')
            return ''.join(parts)
        except (OSError, ValueError, KeyError, TypeError):
            return '<p>Additional low views are not imported locally.</p>'

    def timing(self, info):
        record = info.get('record')
        if not record:
            return ''
        def seconds(key):
            try:
                value = float(record[key])
                return f'{value:.3f} s' if math.isfinite(value) and value >= 0 else 'Unavailable'
            except (KeyError, TypeError, ValueError):
                return 'Unavailable'
        fields = [('Model load for this attempt', seconds('model_load_seconds')),
                  ('Generation / export / preview', seconds('generation_seconds')),
                  ('Recorded end-to-end', seconds('end_to_end_seconds')),
                  ('Model reused', record.get('model_reused', 'Unavailable')),
                  ('GPU', record.get('hardware', {}).get('gpu', {}).get('name', 'Unavailable'))]
        parts = ['<details><summary>Timing and saved evidence</summary><table>']
        parts += [f'<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>' for key, value in fields]
        parts.append('</table><p>' + esc(record.get('generation_timing_boundary', 'Timing boundary unavailable.')) + '</p>')
        parts.append(self.link(info['log'], 'Attempt log'))
        if info.get('output'):
            for name in ['settings.json', 'resources/summary.json', 'resources/metadata.json']:
                path = info['output'].parent / name
                if str(path.relative_to(self.root)) in info['verified']:
                    parts.append(' | ' + self.link(path, name))
        return ''.join(parts) + '</details>'

    def card(self, job, info):
        title = f"{job['side']}: {job['model']['model_id']} / seed {job['seed']}"
        parts = ['<article><h4>' + esc(title) + '</h4><p class="status">' + esc(info['message']) + '</p>']
        if info['state'] == 'saved':
            output = info['output']
            kind = job['output']['kind']
            if kind == 'image':
                parts += [self.image(output, title), '<p>' + self.link(output, 'Saved image', True) + '</p>']
            elif kind == 'asset':
                preview = output.parent / 'preview.png'
                if str(preview.relative_to(self.root)) in info['verified']:
                    parts.append(self.image(preview, title + ' standard preview'))
                video = output.parent / 'turntable.mp4'
                if str(video.relative_to(self.root)) in info['verified']:
                    parts.append(f'<video controls preload="none" src="{esc(self.url(video))}"></video>')
                links = [self.link(output, 'Download saved GLB', True)]
                if str(video.relative_to(self.root)) in info['verified']:
                    links.append(self.link(video, 'Turntable MP4', True))
                parts.append('<p>' + ' | '.join(links) + '</p>')
                parts.append(self.saved_low_views(info))
            else:
                try:
                    result = json.loads(output.read_text())
                    if kind == 'answer':
                        parts.append('<p><strong>Actual answer:</strong> ' + esc(result.get('answer', 'No answer field')) + '</p>')
                        flags = {key: result.get(key) for key in ['structured_parse_success', 'possibly_truncated', 'generated_tokens']}
                        parts.append('<pre>' + json_text(flags) + '</pre>')
                        parts.append('<details><summary>Raw saved answer and parse data</summary><pre>' + json_text(result) + '</pre></details>')
                    else:
                        parts.append('<p><strong>Recorded termination:</strong> ' + esc(result.get('termination', 'Unavailable')) + '</p>')
                        parts.append('<p><strong>Actual final answer</strong></p><pre>' + json_text(result.get('final')) + '</pre>')
                        parts.append('<details><summary>Actual tool calls and results</summary><pre>' + json_text(result.get('tool_calls', [])) + '</pre></details>')
                        parts.append('<details><summary>Actual model turns</summary><pre>' + json_text(result.get('turns', [])) + '</pre></details>')
                    parts.append('<p>' + self.link(output, 'Full saved JSON', True) + '</p>')
                except (OSError, ValueError, TypeError):
                    parts.append('<p>Saved JSON could not be displayed; inspect the linked evidence.</p>')
        elif info.get('record', {}).get('error'):
            parts.append('<pre>' + esc(info['record']['error']) + '</pre>')
        parts.append(self.timing(info))
        parts.append('<details><summary>Current planned job ID</summary><code>' + esc(job['job_id']) + '</code></details></article>')
        return ''.join(parts)

    def issues(self):
        parts = ['<section id="issues"><h2>Issues and fixes</h2>']
        try:
            path = self.file('evidence/issues_and_fixes.json')
            data = json.loads(path.read_text())
            parts.append('<p>' + self.link(path, 'Full issue log and evidence references') + '</p>')
            parts.append('<table><thead><tr><th>ID</th><th>Issue</th><th>Fix / status / remaining limits</th></tr></thead><tbody>')
            for row in data.get('issues', []):
                fix = row.get('fix', {})
                description = fix.get('description', '') if isinstance(fix, dict) else str(fix)
                status = fix.get('status', '') if isinstance(fix, dict) else ''
                text = '<p>' + esc(description) + '</p><p>Status at original record: ' + esc(status) + '</p>'
                for event in row.get('events', []):
                    text += '<p><strong>Later event:</strong> ' + esc(event.get('event', '')) + '</p>'
                if row.get('open_limitations'):
                    text += '<details><summary>Recorded limitations</summary><ul>'
                    text += ''.join('<li>' + esc(value) + '</li>' for value in row['open_limitations']) + '</ul></details>'
                parts.append('<tr><td>' + esc(row.get('id', '')) + '</td><td>' + esc(row.get('title', '')) + '</td><td>' + text + '</td></tr>')
            parts.append('</tbody></table>')
        except (OSError, ValueError, TypeError):
            parts.append('<p>Issue log not available in this local snapshot.</p>')
        parts.append('</section>')
        return ''.join(parts)


def build_index(root):
    review = Review(root)
    study = json.loads(review.file('config/study.json').read_text())
    review.file('config/models.json')
    jobs = planned_jobs(review.root, phase='final')
    inspected = {job['job_id']: review.inspect(job) for job in jobs}
    counts = Counter(info['state'] for info in inspected.values())
    now = datetime.now(timezone.utc).isoformat()
    titles = {'P01': 'Product questions', 'P02': 'Photo editing', 'P03': 'Product 3D assets', 'P04': 'Retail tool agents'}
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' file:; media-src \'self\' file:; style-src \'unsafe-inline\'; script-src \'none\'; connect-src \'none\'; object-src \'none\'; base-uri \'none\'">',
             '<title>Assignment 1: saved final outputs</title><style>',
             'body{font:16px/1.5 sans-serif;max-width:1500px;margin:24px auto;padding:0 18px;color:#111;background:white}h1,h2,h3,h4{line-height:1.25}section{border-top:2px solid #888;margin-top:30px;padding-top:16px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}article{border:1px solid #aaa;padding:14px;min-width:0}img,video{display:block;width:100%;max-height:430px;object-fit:contain;background:#fafafa}figure{margin:8px 0}.reference{max-width:360px}.low-views{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.low-views img{height:160px}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#f5f5f5;padding:10px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:8px;text-align:left;vertical-align:top}details{margin:12px 0}.status{font-weight:bold}nav a{margin-right:16px}a{color:#174c85}small{display:block}',
             '</style></head><body><h1>Assignment 1: saved final outputs</h1>',
             '<p>This offline page presents saved final attempts for the current study IDs. It contains no new model output, rankings or student ratings.</p>',
             '<p>Built ' + esc(now) + '. Saved with matching hashes: ' + str(counts['saved']) + ' / ' + str(len(jobs)) + '. Pending: ' + str(counts['pending']) + '. Failed: ' + str(counts['failed']) + '. Unverified: ' + str(counts['unverified']) + '.</p>',
             '<p>Verification uses input/artifact SHA256, file sizes, preserved execution source, explicitly mapped metadata exports, saved settings and the worker\'s recorded output validation. It does not treat later source edits as the code used for old runs. This builder does not decode GLBs again. Timing and technical evidence are expandable below each output. A zero load time can mean the model was already loaded.</p>',
             '<nav>']
    for index, case in enumerate(study['cases']):
        parts.append(f'<a href="#pair-{index}">{esc(case["pair"])}: {esc(titles.get(case["pair"], case["pair"]))}</a>')
    parts.append('<a href="#issues">Issues and fixes</a><a href="#feedback">Feedback scope</a></nav>')
    for index, case in enumerate(study['cases']):
        pair = case['pair']
        parts.append(f'<section id="pair-{index}"><h2>{esc(pair)}: {esc(titles.get(pair, pair))}</h2>')
        annotations = json.loads(review.file(case['annotations_file']).read_text())
        for item in [item for item in case['items'] if item['split'] == 'final']:
            parts.append('<h3>' + esc(item['id']) + '</h3>')
            if item.get('prompt'):
                parts.append('<p><strong>Question / instruction:</strong> ' + esc(item['prompt']) + '</p>')
            if item.get('image'):
                parts.append('<figure class="reference">' + review.image(item['image'], 'Source photograph ' + item['id']) + '<figcaption>Source photograph</figcaption></figure>')
            label = annotations.get(item['id'], {})
            if pair == 'P01':
                parts.append('<p><strong>Reviewed reference:</strong> ' + esc(label.get('accepted_answers', []) if label.get('reviewed') else 'Not reviewed') + '</p>')
                parts.append('<small>Reference reviewer: ' + esc(label.get('reviewed_by', 'Unavailable')) + '. ' + esc(label.get('review_notes', '')) + '</small>')
            elif pair == 'P04':
                parts.append('<details><summary>Reviewed deterministic reference / mock catalogue</summary><pre>' + json_text(label.get('expected') if label.get('reviewed') else 'Reference not reviewed') + '</pre>' + review.link(item['catalogue'], 'Local mock catalogue') + '</details>')
            elif label.get('visible_identifiers'):
                parts.append('<details><summary>Input-visible feature checklist</summary><ul>' + ''.join('<li>' + esc(value) + '</li>' for value in label['visible_identifiers']) + '</ul><p>These are input checklists, not output ratings.</p></details>')
            group = [job for job in jobs if job['pair'] == pair and job['input_id'] == item['id']]
            parts.append('<div class="cards">' + ''.join(review.card(job, inspected[job['job_id']]) for job in group) + '</div>')
        parts.append('</section>')
    parts.append(review.issues())
    parts.append('<section id="feedback"><h2>Feedback scope</h2><p>The review worksheet and observation records distinguish per-output acceptance from the student preference. Reference checks cover input labels, background compliance and software validation where recorded. Student observations cover only the outputs named by their job IDs. This is an unblinded self-evaluation, without external participants.</p><p>' + review.link('evidence/self_evaluation/observations.csv', 'Observation records') + ' | ' + review.link('data/manual_reviews.csv', 'Review worksheet') + '</p></section>')
    parts.append('<details><summary>Current implementation and study sources</summary><p>Current code links are technical context; saved per-attempt settings identify the actual recorded job.</p>' + ' | '.join(review.link(path, path) for path in ['a1/models.py', 'a1/worker.py', 'README.md']) + '</details></body></html>')
    page = ''.join(parts)
    # Validate every local target before writing an index that appears complete.
    for relative in review.links:
        review.file(relative)
    review.folder.mkdir(parents=True, exist_ok=True)
    target = review.folder / 'index.html'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=review.folder, delete=False) as stream:
        stream.write(page)
        temporary = Path(stream.name)
    os.replace(temporary, target)
    report = {'built_at_utc': now, 'planned_final_jobs': len(jobs), 'states': dict(counts),
              'verification': 'Hashes/settings/log metadata; no new output decoding, ML imports, inference or ratings.',
              'output': str(target.relative_to(review.root)), 'html_sha256': hashlib.sha256(page.encode()).hexdigest(),
              'local_link_targets': sorted(review.links),
              'jobs': [{'job_id': job['job_id'], 'pair': job['pair'], 'side': job['side'], 'seed': job['seed'],
                        'state': inspected[job['job_id']]['state'], 'message': inspected[job['job_id']]['message']} for job in jobs]}
    (review.folder / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build_index(args.root)
    print(json.dumps({key: report[key] for key in ['output', 'planned_final_jobs', 'states']}, indent=2))
