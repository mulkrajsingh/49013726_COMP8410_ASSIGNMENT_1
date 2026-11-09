# Read all saved P03 final previews and optional low-angle inspection views.
# Write labelled sheets that keep source images and generated examples traceable.
# This arranges saved views; it does not run a model or assign review scores.

"""Put every saved P03 final preview into a labelled review sheet.

This only reads saved images and metadata. It does not run a model or score it.
Pass --inspection-root to the unpacked derived-view snapshot for the lamp sheet.
"""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def choose_font(requested):
    candidates = [requested] if requested else [
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for name in candidates:
        if name and Path(name).is_file():
            return str(Path(name).resolve())
    raise FileNotFoundError('Pass an installed TrueType font with --font')


def draw_sheet(rows, target, title, subtitle, font_path):
    panel, gap, margin = 400, 20, 20
    top, row_height = 172, 466
    width = margin * 2 + panel * 5 + gap * 4
    height = top + row_height * len(rows) + 66
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    big = ImageFont.truetype(font_path, 34)
    normal = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 21)
    draw.text((margin, 16), title, font=big, fill='black')
    draw.text((margin, 62), subtitle, font=small, fill='black')
    headings = ['Input photograph', 'A: TRELLIS.2\nSeed 20261025',
                'A: TRELLIS.2\nSeed 20261026', 'B: Hunyuan3D\nSeed 20261025',
                'B: Hunyuan3D\nSeed 20261026']
    for column, text in enumerate(headings):
        draw.multiline_text((margin + column * (panel + gap), 102), text,
                            font=normal, fill='black', spacing=5)
    placements = []
    for index, row in enumerate(rows):
        y = top + index * row_height
        draw.text((margin, y), row['label'], font=normal, fill='black')
        for column, source in enumerate(row['images']):
            x, box_y = margin + column * (panel + gap), y + 42
            with Image.open(source['path']) as original:
                original_size = list(original.size)
                image = original.convert('RGB')
                image.thumbnail((panel, panel), Image.Resampling.LANCZOS)
            position = (x + (panel - image.width) // 2,
                        box_y + (panel - image.height) // 2)
            canvas.paste(image, position)
            draw.rectangle((x, box_y, x + panel, box_y + panel), outline='#b0b0b0')
            placements.append({'product': row['product'], 'column': column,
                               'source': source, 'original_size': original_size,
                               'placed_size': list(image.size), 'position': list(position)})
    scope = 'All four lamp finals shown.' if len(rows) == 1 else 'All 12 planned finals shown.'
    draw.text((margin, height - 52),
              scope + ' Full images resized to equal panels; no crops, retouching or ratings.',
              font=small, fill='black')
    canvas.save(target)
    return {'output': file_record(target), 'size': list(canvas.size), 'placements': placements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inspection-root', type=Path,
                        help='Directory containing outputs/P03/.../inspection/low_yaw_000.png')
    parser.add_argument('--font')
    args = parser.parse_args()
    root = args.root.resolve()
    inspection_root = (args.inspection_root or root).resolve()
    font_path = choose_font(args.font)
    annotations = read_json(root / 'data/annotations/P03.json')
    rows, jobs = [], []
    for product, label in [('product01', 'Chair'), ('product02', 'Lamp'), ('product03', 'Vase')]:
        reference = root / annotations[product]['reference']
        assert digest(reference) == annotations[product]['reference_sha256']
        row = {'product': product, 'label': label + ' / ' + product,
               'images': [file_record(reference)]}
        for side in ['A', 'B']:
            for seed in [20261025, 20261026]:
                pattern = f'outputs/P03/{side}/final/P03_{side}_final_{product}_{seed}_*/attempt_01/preview.png'
                matches = list(root.glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f'Expected one fixed first-attempt final: {pattern}; got {matches}')
                preview = matches[0]
                settings = read_json(preview.parent / 'settings.json')
                job = settings['job']
                assert (job['phase'], job['side'], job['input_id'], job['seed']) == ('final', side, product, seed)
                entry = {'side': side, 'seed': seed, 'product': product, 'job_id': job['job_id'],
                         'attempt': 1, 'preview': file_record(preview),
                         'settings': file_record(preview.parent / 'settings.json'),
                         'asset': file_record(preview.parent / 'asset.glb')}
                jobs.append(entry)
                row['images'].append({**entry['preview'], 'job_id': job['job_id'], 'seed': seed, 'side': side})
        rows.append(row)
    assert len(jobs) == 12 and len({job['job_id'] for job in jobs}) == 12

    # Check every underside against its saved metadata and the actual exported GLB.
    lamp = {'product': 'product02', 'label': 'Lamp / identical low camera, yaw 0 degrees',
            'images': [rows[1]['images'][0]]}
    lamp_sources = []
    for job in [item for item in jobs if item['product'] == 'product02']:
        folder = inspection_root / f"outputs/P03/{job['side']}/final/{job['job_id']}/attempt_01/inspection"
        metadata = read_json(folder / 'metadata.json')
        assert metadata['status'] == 'complete'
        assert metadata['source_glb_sha256'] == job['asset']['sha256']
        camera = metadata['camera']
        assert camera['eye_height'] == -0.8 and camera['radius'] == 2
        assert camera['projection'] == 'orthographic' and camera['xmag'] == camera['ymag'] == 0.8
        view = next(item for item in metadata['views'] if item['file'] == 'low_yaw_000.png')
        assert view['yaw_degrees'] == 0
        image = file_record(folder / view['file'])
        assert image['sha256'] == view['sha256']
        lamp['images'].append({**image, 'job_id': job['job_id'], 'seed': job['seed'], 'side': job['side']})
        lamp_sources.append({'job_id': job['job_id'], 'side': job['side'], 'seed': job['seed'],
                             'image': image, 'metadata': file_record(folder / 'metadata.json'),
                             'source_glb_sha256': job['asset']['sha256'], 'camera': camera,
                             'selected_yaw_degrees': 0})

    destination = root / 'evidence/self_evaluation/P03'
    destination.mkdir(parents=True, exist_ok=True)
    common = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'created_by': 'reference_check',
              'script': file_record(__file__), 'font': file_record(font_path),
              'selection_policy': 'All 12 planned final jobs: all three products, both sides, both protocol seeds, attempt_01. Use saved standard preview.png without quality selection. Lamp uses yaw0 for every output.',
              'processing': 'Pillow RGB conversion and Lanczos fit of the entire image into equal400px panels. No cropping, retouching, model inference or generative editing.',
              'model_pipeline_rerun': False, 'ratings_recorded': False, 'original_files_modified': False,
              'annotations': file_record(root / 'data/annotations/P03.json'),
              'visible_feature_checklists': {product: annotations[product]['visible_identifiers'] for product in ['product01', 'product02', 'product03']},
              'limitations': ['Standard previews show one angle; use saved turntables/GLBs for further inspection.',
                             'Visible-feature review does not establish dimensions, physical materials, unseen geometry or file importability.',
                             'Prior student feedback covered seed20261025 only; no new review or rating is inferred from this sheet.'],
              'reproduction_command': f'python scripts/make_p03_review_sheet.py --inspection-root {inspection_root} --font {font_path}'}
    main_figure = draw_sheet(rows, destination / 'final_outputs_review.png',
                            'P03: all 12 saved final outputs',
                            'Three products; both fixed seeds for each model. Standard preview/frame 0.', font_path)
    (destination / 'final_outputs_review.json').write_text(json.dumps({**common, 'jobs': jobs, 'figure': main_figure}, indent=2) + '\n')
    lamp_figure = draw_sheet([lamp], destination / 'lamp_review.png',
                            'P03 lamp: all four saved underside views',
                            'Same low camera for every output: yaw 0 degrees, eye height -0.8, radius 2.', font_path)
    (destination / 'lamp_review.json').write_text(json.dumps({**common, 'sources': lamp_sources, 'figure': lamp_figure}, indent=2) + '\n')
    # Reopen inputs after composition to check that source bytes were unchanged.
    for image in [source for row in rows for source in row['images']] + lamp['images']:
        assert digest(image['path']) == image['sha256']
    print(json.dumps({'final_jobs': len(jobs), 'lamp_views': len(lamp_sources),
                      'outputs': [str(destination / name) for name in ['final_outputs_review.png', 'final_outputs_review.json', 'lamp_review.png', 'lamp_review.json']],
                      'original_image_hashes_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
