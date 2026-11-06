"""Re-crop a webcam manifest at the margin selected on validation before fine-tuning."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from detectors import align_crop
from webcam_data import load_manifest, read_image, write_image, CROP_VERSION, configure_console


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory.')
    parser.add_argument('--margin', type=float, required=True)
    parser.add_argument('--no-align', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() or not 0 <= args.margin <= 0.5:
        parser.error('output phải là thư mục mới; margin trong [0,0.5].')
    rows = load_manifest(args.manifest)
    if any(not row.get('frame_path') for row in rows):
        parser.error('Cần frame gốc cho mọi ảnh để crop lại.')
    args.output.mkdir(parents=True)
    with (args.output / 'manifest.jsonl').open('w', encoding='utf-8') as log:
        for index, row in enumerate(rows):
            crop = align_crop(read_image(row['frame_path']), row['box'],
                              0 if args.no_align else row['angle'], args.margin)
            path = args.output / 'crops' / f'{index:07d}.png'
            write_image(path, crop)
            row.update({'path': str(path.resolve()), 'frame_path': str(row['frame_path']),
                        'crop_margin': args.margin, 'align': not args.no_align, 'crop_version': CROP_VERSION})
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(args.output / 'manifest.jsonl')


if __name__ == '__main__':
    main()
