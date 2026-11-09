# Read all saved P02 final images and their attempt metadata.
# Write labelled comparison sheets and source records for later visual review.
# Arranging existing images adds neither model samples nor quality judgments.

"""Arrange all saved P02 finals for later review. No model or scoring is run."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.io import file_records, project_path, read_attempts
from a1.study import planned_jobs
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def read_json(path):
    return json.loads(Path(path).read_text())


def collect(root):
    study = read_json(root / 'config/study.json')
    case = next(row for row in study['cases'] if row['pair'] == 'P02')
    assert case['seeds'] == [20261025, 20261026]
    jobs = [row for row in planned_jobs(root, phase='final') if row['pair'] == 'P02']
    assert len(jobs) == case['expected_final_jobs'] == 12
    annotations_path = project_path(root, case['annotations_file'])
    annotations = read_json(annotations_path)
    products = []
    for item in [row for row in case['items'] if row['split'] == 'final']:
        name = item['id']
        note = annotations[name]
        reference = project_path(root, item['image'])
        assert note['reference'] == item['image']
        assert digest(reference) == note['reference_sha256']
        assert digest(project_path(root, item['mask'])) == note['mask_sha256']
        assert note['required_checks'] == case['scoring']['checks']
        row = {'product': name, 'literal_requested_edit': item['prompt'],
               'visible_identifiers': note['visible_identifiers'],
               'required_checks': note['required_checks'],
               'annotation_review_scope': note['review_scope'],
               'annotation_reviewed_by': note['reviewed_by'],
               'annotation_reviewed_at_utc': note['reviewed_at_utc'],
               'mask_limitations': note['review_notes'],
               'source_input': record(reference), 'evaluation_mask': record(project_path(root, item['mask'])),
               'outputs': []}
        for side in ['A', 'B']:
            for seed in case['seeds']:
                selected = [job for job in jobs if (job['input_id'], job['side'], job['seed']) == (name, side, seed)]
                assert len(selected) == 1
                job = selected[0]
                log = root / f'outputs/P02/{side}/final/attempts.jsonl'
                attempts = [entry for entry in read_attempts(log) if entry['job_id'] == job['job_id']]
                saved = attempts[-1]
                assert saved['status'] == 'success', job['job_id']
                assert saved['input_files'] == file_records(root, job['item']['files'])
                output = project_path(root, saved['output_path'])
                settings = output.parent / 'settings.json'
                settings_data = read_json(settings)
                assert settings_data['job'] == job
                assert settings_data['input_files'] == saved['input_files']
                artifacts = {entry['path']: entry for entry in saved['artifacts']}
                for path in [output, settings]:
                    expected = artifacts[str(path.relative_to(root))]
                    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
                with Image.open(output) as image:
                    image.verify()
                with Image.open(output) as image:
                    assert image.size == (case['output']['width'], case['output']['height'])
                row['outputs'].append({'side': side, 'seed': seed, 'job_id': job['job_id'],
                                       'attempt': saved['attempt'], 'model_id': job['model']['model_id'],
                                       'model_short_name': job['model']['model_id'].split('/')[-1],
                                       'model_revision': job['model']['weight_revision'],
                                       'image': record(output), 'settings': record(settings),
                                       'attempt_log': record(log), 'input_files': saved['input_files'],
                                       'recorded_output_validation': saved['output_validation']})
        products.append(row)
    assert len(products) == 3
    assert len({entry['job_id'] for row in products for entry in row['outputs']}) == 12
    return products, annotations_path


def wrap_pixels(draw, text, font, width):
    lines, current = [], ''
    for word in text.split():
        candidate = (current + ' ' + word).strip()
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def draw_row(row, target, font_path, root):
    panel, gap, margin = 480, 18, 28
    width = 5 * panel + 4 * gap + 2 * margin
    fonts = {size: ImageFont.truetype(font_path, size) for size in [18, 21, 23, 25, 34]}
    scratch = ImageDraw.Draw(Image.new('RGB', (width, 20)))
    sections = [('Requested edit (literal): ' + row['literal_requested_edit'], 23),
                ('Pre-run check IDs (unrated): ' + ' | '.join(row['required_checks']), 21),
                ('Pre-run input feature checklist; no output ratings supplied:', 23)]
    sections += [(' - ' + value, 23) for value in row['visible_identifiers']]
    text_lines = []
    for text, size in sections:
        for line in wrap_pixels(scratch, text, fonts[size], width - margin * 2):
            text_lines.append((line, size))
    text_height = sum(size + 8 for _, size in text_lines)
    heading_y = 76 + text_height + 16
    image_y = heading_y + 76
    height = image_y + panel + 122
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    label = {'product01': 'Chairs', 'product02': 'Lamp', 'product03': 'Vase'}[row['product']]
    draw.text((margin, 20), 'P02 / ' + row['product'] + ' / ' + label, font=fonts[34], fill='black')
    y = 76
    for line, size in text_lines:
        draw.text((margin, y), line, font=fonts[size], fill='black')
        y += size + 8
    columns = [{'image': row['source_input'], 'heading': 'Source input',
                'caption': str(Path(row['source_input']['path']).relative_to(root))}]
    columns += [{'image': item['image'], 'heading': item['side'] + ': ' + item['model_short_name'] + '\nSeed ' + str(item['seed']),
                 'caption': item['job_id']} for item in row['outputs']]
    placements = []
    for column, item in enumerate(columns):
        x = margin + column * (panel + gap)
        draw.multiline_text((x, heading_y), item['heading'], font=fonts[25], fill='black', spacing=7)
        with Image.open(item['image']['path']) as original:
            size, mode = list(original.size), original.mode
            assert mode in {'RGB', 'RGBA'}, 'Unexpected mode needs explicit processing review: ' + mode
            alpha_range = list(original.getchannel('A').getextrema()) if mode == 'RGBA' else None
            image = original.copy()
            image.thumbnail((panel, panel), Image.Resampling.LANCZOS)
        position = (x + (panel - image.width) // 2, image_y + (panel - image.height) // 2)
        canvas.paste(image, position, image.getchannel('A') if mode == 'RGBA' else None)
        draw.rectangle((x - 1, image_y - 1, x + panel, image_y + panel), outline='#b0b0b0')
        captions = wrap_pixels(draw, item['caption'], fonts[18], panel)
        assert len(captions) <= 3
        draw.multiline_text((x, image_y + panel + 10), '\n'.join(captions), font=fonts[18], fill='black', spacing=3)
        placements.append({'column': column, 'source': item['image'], 'original_size': size,
                           'original_mode': mode, 'original_alpha_range': alpha_range, 'placed_size': list(image.size),
                           'position': list(position), 'panel_box': [x, image_y, x + panel, image_y + panel]})
    draw.text((margin, height - 38), 'All four finals shown. Full images fit on white (native alpha displayed on white); no crops, retouching, inference or ratings.', font=fonts[21], fill='black')
    canvas.save(target)
    return {'figure': record(target), 'size': list(canvas.size), 'placements': placements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--font')
    args = parser.parse_args()
    root = args.root.resolve()
    destination = (args.destination or root / 'evidence/self_evaluation/P02').resolve()
    allowed = (root / 'evidence/self_evaluation/P02').resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError('Write P02 review figures only under evidence/self_evaluation/P02')
    destination.mkdir(parents=True, exist_ok=True)
    products, annotations_path = collect(root)
    fonts = [args.font] if args.font else ['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font = next((str(Path(path).resolve()) for path in fonts if Path(path).is_file()), None)
    if font is None:
        raise FileNotFoundError('Pass an installed TrueType font with --font')
    common = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'created_by': 'reference_check',
              'script': record(__file__), 'font': record(font), 'study': record(root / 'config/study.json'),
              'model_config': record(root / 'config/models.json'), 'annotations': record(annotations_path),
              'selection_policy': 'Every current planned P02 final: three products, both models, protocol seeds 20261025 and 20261026. Columns input/A25/A26/B25/B26. No quality selection.',
              'processing': 'Pillow Lanczos fit of entire original RGB/RGBA images into 480px square panels on white. Native alpha is retained during resizing and displayed by standard alpha compositing on white; original alpha ranges are recorded per panel. Labels/borders stay outside the images. No crops, retouching or colour correction.',
              'ratings_recorded': False, 'student_feedback_inferred': False, 'inference_rerun': False,
              'writes_limited_to': str(destination), 'original_source_files_modified': False,
              'secondary_metric_display_caveat': 'Frozen unchanged_region_mae uses a1.media.image_array, which converts the saved image to RGB colour channels. It does not use this sheet\'s alpha-composited-on-white appearance. This task does not change or recompute that metric.',
              'secondary_metric_source': record(root / 'a1/media.py'),
              'secondary_metric_definition': 'a1/media.py: image_array converts saved channels to RGB; a1/analysis.py computes unchanged_region_mae from that RGB array.',
              'limitations': ['Pre-run checklists describe input features, not completed output ratings.',
                             'Manual checks remain unrated; no subjective score or preferred output is selected.',
                             'The fixed evaluation masks are approximate; mask-specific limitations are retained per product.'],
              'reproduction_argv': [sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--destination', str(destination), '--font', font]}
    figures = []
    for row in products:
        name = row['product'] + '_review'
        figure = draw_row(row, destination / (name + '.png'), font, root)
        (destination / (name + '.json')).write_text(json.dumps({**common, 'product': row, **figure}, indent=2) + '\n')
        figures.append(figure)
    width = figures[0]['size'][0]
    total_height = 116 + sum(figure['size'][1] for figure in figures) + 28 * (len(figures) - 1) + 54
    all_sheet = Image.new('RGB', (width, total_height), 'white')
    draw = ImageDraw.Draw(all_sheet)
    draw.text((28, 20), 'P02: all 12 saved final outputs / both fixed seeds', font=ImageFont.truetype(font, 34), fill='black')
    draw.text((28, 70), 'Fixed product and column order; source inputs, literal prompt and pre-run checklists. No output ratings.', font=ImageFont.truetype(font, 23), fill='black')
    y, row_positions = 116, []
    for figure in figures:
        with Image.open(figure['figure']['path']) as image:
            all_sheet.paste(image, (0, y))
        row_positions.append({'row_figure': figure['figure'], 'position': [0, y], 'size': figure['size']})
        y += figure['size'][1] + 28
    draw.text((28, total_height - 38), 'All 12 planned finals included; only labelled thumbnail resizing on white. Review deferred until all experiments finish.', font=ImageFont.truetype(font, 21), fill='black')
    all_path = destination / 'final_outputs_review.png'
    all_sheet.save(all_path)
    proof = {**common, 'products': products, 'figure': record(all_path),
             'size': list(all_sheet.size), 'rows': row_positions}
    (destination / 'final_outputs_review.json').write_text(json.dumps(proof, indent=2) + '\n')
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>P02 saved final review sheets</title>',
            '<style>body{font:18px sans-serif;max-width:1600px;margin:24px auto;padding:0 20px}img{width:100%;height:auto}p{line-height:1.5}</style>',
            '<h1>P02 saved final review sheets</h1><p>All 12 outputs; no model quality ratings supplied. Review is deferred until all experiments finish.</p>',
            '<p><a href="final_outputs_review.png">All-output sheet</a> | <a href="final_outputs_review.json">Source/job/hash provenance</a></p>']
    for row in products:
        name = row['product'] + '_review'
        html += ['<h2>' + escape(row['product']) + '</h2>',
                 '<p><a href="' + name + '.png">Full-size row</a> | <a href="' + name + '.json">Provenance</a></p>',
                 '<img src="' + name + '.png" alt="' + escape(row['product']) + ': source input and all four final outputs">',
                 '<p>Existing mask limitation: ' + escape(row['mask_limitations']) + '</p>']
    html.append('</html>')
    index = destination / 'index.html'
    index.write_text('\n'.join(html) + '\n')
    links = re.findall(r'(?:href|src)="([^"]+)"', index.read_text())
    assert all((destination / link).resolve().is_relative_to(destination) and (destination / link).is_file() for link in links)
    sources = [entry for row in products for entry in [row['source_input'], row['evaluation_mask']]]
    sources += [entry for row in products for output in row['outputs'] for entry in [output['image'], output['settings'], output['attempt_log']]]
    for source in sources:
        assert digest(source['path']) == source['sha256']
    for figure in [entry['figure'] for entry in figures] + [proof['figure']]:
        with Image.open(figure['path']) as image:
            image.verify()
    validation = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'planned_final_jobs': 12,
                  'unique_final_jobs': len({output['job_id'] for row in products for output in row['outputs']}),
                  'original_source_hash_checks_passed': len(sources), 'output_images_verified': 12,
                  'output_image_dimensions': [1024, 1024], 'all_sheet_dimensions': proof['size'],
                  'row_dimensions': [figure['size'] for figure in figures], 'relative_links_checked': len(links),
                  'all_relative_links_exist_within_output': True, 'index': record(index),
                  'layout_visual_check': 'Pending internal layout QA; no user review opened.',
                  'ratings_recorded': False, 'writes_limited_to': str(destination)}
    (destination / 'verification.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
