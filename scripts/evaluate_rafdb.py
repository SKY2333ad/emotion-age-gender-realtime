"""Compare EfficientNet emotion checkpoints on exactly the same held-out RAF-DB images."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
import numpy as np
from predict import LABELS, MODEL_NAMES, MODEL_DIR, _get_inference
from preprocessing import prepare_face
from webcam_data import read_image, file_hash, image_hash, check_content_leakage, configure_console
from scripts.train_rafdb import load_samples, split_validation, check_splits, scores_from_confusion
from scripts.evaluate_webcam import write_reports, select_splits


def read_split_file(path):
    splits = {name: [] for name in ('train', 'validation', 'test')}
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            if row['split'] not in splits or row['label'] not in LABELS:
                raise ValueError('Nhãn/split trong splits.csv không hợp lệ.')
            image = Path(row['path'])
            if not image.is_absolute():
                raise ValueError('splits.csv phải dùng đường dẫn ảnh tuyệt đối.')
            if not image.is_file():
                raise ValueError(f'Thiếu ảnh: {image}')
            splits[row['split']].append((image.resolve(), LABELS.index(row['label'])))
    check_splits(splits)
    return splits


def resolve_splits(data_dir, models, split_file=None, val_fraction=0.15, seed=42):
    original_train, original_val, original_test = load_samples(Path(data_dir))
    reference = read_split_file(split_file) if split_file else None
    for model in models:
        metadata_path = Path(MODEL_DIR) / f'{model}.metadata.json'
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('split_file'):
            candidate = read_split_file(metadata['split_file'])
            if reference is None:
                reference = candidate
            elif any(set(candidate[key]) != set(reference[key]) for key in reference):
                raise ValueError('Các checkpoint có cách chia tập khác nhau. Chỉ so sánh trên cùng các tập đã giữ riêng.')
    if reference is not None:
        if (set(reference['train'] + reference['validation']) != set(original_train + original_val) or
                set(reference['test']) != set(original_test)):
            raise ValueError('splits.csv không khớp dataset RAF-DB được chỉ định.')
        splits = reference
    else:
        if not original_val:
            original_train, original_val = split_validation(original_train, val_fraction, seed, image_hash)
        splits = {'train': original_train, 'validation': original_val, 'test': original_test}
    check_splits(splits)
    check_content_leakage(splits)
    return splits


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-dir', type=Path)
    source.add_argument('--manifest', type=Path)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=['efficientnet'])
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--splits-file', type=Path)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.val_fraction < 1:
        parser.error('batch-size >= 1; val-fraction trong (0,1).')
    if args.output.exists() or len(set(args.models)) != len(args.models):
        parser.error('output phải là thư mục mới; models không được trùng.')
    if args.manifest:
        reference_path = Path(MODEL_DIR) / f'{args.models[-1]}.keras'
        grouped = select_splits(args.manifest, reference_path, args.val_fraction, 0.2, args.seed)
        splits = {key: [(row['path'], LABELS.index(row['label'])) for row in values] for key, values in grouped.items()}
    else:
        splits = resolve_splits(args.data_dir, args.models, args.splits_file, args.val_fraction, args.seed)
    samples = splits[args.split]
    if not samples:
        parser.error(f'Tập {args.split} rỗng.')
    heldout_hashes = {file_hash(path) for path, _ in samples}
    heldout_pixels = {image_hash(path) for path, _ in samples}
    results, details = {}, []
    for name in args.models:
        metadata_path = Path(MODEL_DIR) / f'{name}.metadata.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (heldout_hashes & set(metadata.get('image_hashes', {}).get('train', [])) or
                    heldout_pixels & set(metadata.get('pixel_hashes', {}).get('train', []))):
                parser.error(f'Tập đánh giá chứa ảnh đã dùng để fine-tune {name}.')
        inference = _get_inference(name)
        matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        for offset in range(0, len(samples), args.batch_size):
            batch = samples[offset:offset + args.batch_size]
            raw = [read_image(path) for path, _ in batch]
            probabilities = inference(np.stack([prepare_face(image) for image in raw])).numpy()
            if (probabilities.shape != (len(batch), len(LABELS)) or not np.isfinite(probabilities).all() or
                    np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
                raise ValueError(f'Đầu ra checkpoint không phải xác suất {len(LABELS)} lớp.')
            for (path, truth), probs in zip(batch, probabilities):
                predicted = int(probs.argmax())
                matrix[truth, predicted] += 1
                top3 = np.argsort(probs)[-3:][::-1]
                details.append({'model': name, 'split': args.split, 'path': str(path),
                                'true': LABELS[truth], 'prediction': LABELS[predicted],
                                'correct': int(predicted == truth),
                                'top3': json.dumps({LABELS[i]: float(probs[i]) for i in top3})})
        results[name] = scores_from_confusion(matrix)
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, macro-F1={results[name]['macro_f1']:.4f}, "
              f"sad={results[name]['per_class']['sad']['f1']:.4f}, angry={results[name]['per_class']['angry']['f1']:.4f}")
    write_reports(args.output, results, details)
    comparison = [{'model': name, 'dataset': 'RAF-DB', 'split': args.split,
                   'images': len(samples), 'accuracy': scores['accuracy'], 'macro_f1': scores['macro_f1'],
                   'sad_f1': scores['per_class']['sad']['f1'], 'angry_f1': scores['per_class']['angry']['f1']}
                  for name, scores in results.items()]
    with (args.output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (args.output / 'scope.json').write_text(json.dumps({
        'dataset': str((args.manifest or args.data_dir).resolve()), 'split': args.split,
        'models': {name: file_hash(Path(MODEL_DIR) / f'{name}.keras') for name in args.models},
        'limitation': 'Initial model training data is unknown; held out from the new fine-tune only.',
        'detector': 'MediaPipe features were prepared once; evaluation retains all samples, including fallback.'
                    if args.manifest else 'Not rerun: RAF-DB images are already cropped/aligned.'
    }, indent=2), encoding='utf-8')
    print(f'Báo cáo: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
