"""Offline validation/test, crop-margin comparison and sad/fear confusion reports."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
import numpy as np
from detectors import align_crop
from fusion import Fusion
from predict import predict, DEFAULT_MODEL, MODEL_NAMES, MODEL_DIR
from webcam_data import (LABELS, CROP_VERSION, load_manifest, split_manifest, read_image, file_hash,
                         image_hash, check_content_leakage, configure_console)
from scripts.train_rafdb import scores_from_confusion


def select_splits(manifest, model_path, val_fraction=0.15, test_fraction=0.2, seed=42):
    rows = load_manifest(manifest)
    metadata_path = Path(model_path).with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else None
    if metadata and metadata['sha256'] != file_hash(model_path):
        raise ValueError('Metadata không khớp checkpoint EfficientNet.')
    if metadata and metadata.get('subjects'):
        groups = metadata['subjects']
        known = set().union(*(set(values) for values in groups.values()))
        if {str(row['subject']) for row in rows} != known:
            raise ValueError('Manifest phải chứa đúng các subject đã chia lúc fine-tune. Dùng bộ dữ liệu mới độc lập để đánh giá riêng.')
        splits = {name: [row for row in rows if str(row['subject']) in set(subjects)]
                  for name, subjects in groups.items()}
    else:
        splits = split_manifest(rows, val_fraction, test_fraction, seed)
    # Also catch copies with changed names/subject IDs.
    check_content_leakage({name: [(row['path'], LABELS.index(row['label'])) for row in values]
                           for name, values in splits.items()})
    if metadata:
        used = set(metadata.get('image_hashes', {}).get('train', []))
        used_pixels = set(metadata.get('pixel_hashes', {}).get('train', []))
        for name in ('validation', 'test'):
            if any(file_hash(row['path']) in used or (used_pixels and image_hash(row['path']) in used_pixels)
                   for row in splits[name]):
                raise ValueError('Tập đánh giá chứa ảnh EfficientNet đã fine-tune.')
    return splits


def predict_row(row, model, margin, align=True):
    if row.get('frame_path'):
        crop = align_crop(read_image(row['frame_path']), row['box'], row['angle'] if align else 0, margin)
    else:
        if (row.get('crop_margin') != margin or row.get('align') != align or
                row.get('crop_version') != CROP_VERSION):
            raise ValueError('Thiếu frame gốc để thay đổi crop.')
        crop = read_image(row['path'])
    return predict(crop, model)[2]


def score_predictions(rows, predictions):
    matrix = np.zeros((7, 7), dtype=np.int64)
    for row, probs in zip(rows, predictions):
        matrix[LABELS.index(row['label']), LABELS.index(max(probs, key=probs.get))] += 1
    return scores_from_confusion(matrix)


def write_reports(directory, results, details):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    records = [{'config': config, 'label': label, **values}
               for config, metrics in results.items() for label, values in metrics['per_class'].items()]
    for filename, rows in [('metrics_per_class.csv', records), ('predictions.csv', details)]:
        with (directory / filename).open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for config, metrics in results.items():
        matrix = np.array(metrics['confusion_matrix'])
        fig, axis = plt.subplots(figsize=(7, 6))
        axis.imshow(matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, cmap='Blues')
        axis.set(xticks=range(7), yticks=range(7), xticklabels=LABELS, yticklabels=LABELS,
                 xlabel='Predicted', ylabel='True', title=config)
        plt.setp(axis.get_xticklabels(), rotation=45, ha='right')
        for i in range(7):
            for j in range(7):
                axis.text(j, i, str(matrix[i, j]), ha='center', va='center')
        fig.tight_layout()
        fig.savefig(directory / f'confusion_{config}.png', dpi=150)
        plt.close(fig)


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--margins', type=float, nargs='+', default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument('--no-align', action='store_true')
    parser.add_argument('--fusion', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(not 0 <= margin <= 0.5 for margin in args.margins):
        parser.error('margins phải trong [0,0.5].')
    if args.split == 'test' and len(args.margins) != 1:
        parser.error('Chọn đúng một margin từ validation trước khi đánh giá test.')
    if args.output.exists():
        parser.error('output phải là thư mục mới.')
    model_path = Path(MODEL_DIR) / f'{args.model}.keras'
    rows = select_splits(args.manifest, model_path, args.val_fraction, args.test_fraction, args.seed)[args.split]
    if not rows:
        parser.error('Tập đánh giá rỗng.')
    results, details = {}, []
    for margin in args.margins:
        correction = Fusion(args.fusion, model_path, margin, not args.no_align) if args.fusion else None
        if correction and any((row.get('subject') is not None and str(row['subject']) in correction.data['train_subjects']) or
                              file_hash(row['path']) in correction.data['train_image_hashes'] or
                              image_hash(row['path']) in correction.data.get('train_pixel_hashes', []) for row in rows):
            parser.error('Tập đánh giá trùng dữ liệu train fusion.')
        predictions, fused = [], []
        fused_count = 0
        for row in rows:
            probs = predict_row(row, args.model, margin, not args.no_align)
            predictions.append(probs)
            if correction:
                can_fuse = bool(row.get('blendshapes')) and all(key in row['blendshapes'] for key in correction.names)
                fused.append(correction.predict(probs, row['blendshapes']) if can_fuse else probs)
                fused_count += int(can_fuse)
            top3 = sorted(probs, key=probs.get, reverse=True)[:3]
            details.append({'path': str(row['path']), 'subject': row['subject'], 'true': row['label'],
                            'margin': margin, 'prediction': top3[0], 'top3': json.dumps({k: probs[k] for k in top3}),
                            'fusion_prediction': max(fused[-1], key=fused[-1].get) if correction else ''})
        key = f'{args.split}_margin_{margin:.2f}'
        results[key] = score_predictions(rows, predictions)
        if correction:
            results[key + '_fusion'] = score_predictions(rows, fused)
            results[key + '_fusion']['fusion_coverage'] = {'used': fused_count, 'total': len(rows),
                                                          'fallback': len(rows) - fused_count}
        print(key, json.dumps({label: results[key]['per_class'][label] for label in ('sad', 'fear')}), flush=True)
    write_reports(args.output, results, details)
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
