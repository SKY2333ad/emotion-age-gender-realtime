"""Learn a small blendshape correction; keep EfficientNet if held-out scores do not improve."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
import numpy as np
from fusion import features, softmax
from predict import DEFAULT_MODEL, MODEL_NAMES, MODEL_DIR
from webcam_data import LABELS, CROP_VERSION, file_hash, image_hash, configure_console
from selection import qualifies
from scripts.evaluate_webcam import select_splits, predict_row, score_predictions


def fit_correction(train_x, train_p, train_y, val_x, val_p, val_y,
                   epochs=500, learning_rate=0.02, regularization=0.1, val_mask=None):
    """Full-batch weighted residual softmax with validation-based selection."""
    counts = np.bincount(train_y, minlength=7)
    if np.any(counts == 0):
        raise ValueError('Train fusion cần đủ 7 lớp.')
    weights_per_sample = len(train_y) / (7 * counts[train_y])
    weights_per_sample /= weights_per_sample.sum()
    mean, scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 0.05)
    x, vx = (train_x - mean) / scale, (val_x - mean) / scale
    offset = np.log(np.clip(train_p, 1e-7, 1))
    val_offset = np.log(np.clip(val_p, 1e-7, 1))
    val_mask = np.ones(len(val_y), dtype=bool) if val_mask is None else np.asarray(val_mask, dtype=bool)
    if val_mask.shape != (len(val_y),):
        raise ValueError('Validation fusion mask không hợp lệ.')
    weight, bias = np.zeros((x.shape[1], 7)), np.zeros(7)
    truth = np.eye(7)[train_y]
    dummy_rows = [{'label': LABELS[int(i)]} for i in val_y]

    def metrics(p):
        return score_predictions(dummy_rows, [dict(zip(LABELS, row)) for row in p])

    baseline = metrics(val_p)
    best_metrics, best, best_epoch, stale = baseline, (weight.copy(), bias.copy()), 0, 0
    history = []
    for epoch in range(epochs):
        p = softmax(offset + x @ weight + bias)
        delta = (p - truth) * weights_per_sample[:, None]
        weight -= learning_rate * (x.T @ delta + regularization * weight)
        bias -= learning_rate * delta.sum(axis=0)
        scores = metrics(softmax(val_offset + (vx @ weight + bias) * val_mask[:, None]))
        improved = qualifies(scores, baseline, best_metrics['macro_f1'])
        if improved:
            best_metrics, best, best_epoch, stale = scores, (weight.copy(), bias.copy()), epoch + 1, 0
        else:
            stale += 1
        history.append({'epoch': epoch + 1, 'macro_f1': scores['macro_f1'],
                        'sad_f1': scores['per_class']['sad']['f1'],
                        'fear_f1': scores['per_class']['fear']['f1']})
        if stale >= 100:
            break
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': best[0].tolist(),
            'bias': best[1].tolist(), 'best_epoch': best_epoch,
            'baseline_validation': baseline, 'best_validation': best_metrics, 'history': history}


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--model', choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--no-align', action='store_true')
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--regularization', type=float, default=0.1)
    parser.add_argument('--output', type=Path, default=ROOT / 'models' / 'efficientnet_fusion.json')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        parser.error('output/report đã tồn tại; chọn tên mới.')
    if (not 0 <= args.margin <= 0.5 or args.epochs < 1 or
            not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not np.isfinite(args.regularization) or args.regularization < 0):
        parser.error('margin: 0..0.5; epochs >= 1; learning-rate > 0; regularization >= 0.')
    model_path = Path(MODEL_DIR) / f'{args.model}.keras'
    splits = select_splits(args.manifest, model_path, args.val_fraction, args.test_fraction, args.seed)
    landmarker_hashes = {row.get('landmarker_sha256') for values in splits.values() for row in values}
    if len(landmarker_hashes) != 1:
        parser.error('Các mẫu phải dùng cùng phiên bản MediaPipe Face Landmarker.')
    valid_train = [row for row in splits['train'] if row.get('blendshapes')]
    names = sorted(valid_train[0]['blendshapes']) if valid_train else []
    if len(names) < 1:
        parser.error('Manifest thiếu blendshape; không thể train fusion.')
    arrays, masks = {}, {}
    coverage = {}
    for name in ('train', 'validation'):
        vectors, probabilities, labels, available = [], [], [], []
        for row_index, row in enumerate(splits[name], start=1):
            can_fuse = bool(row.get('blendshapes')) and all(key in row['blendshapes'] for key in names)
            if name == 'train' and not can_fuse:
                continue
            probs = predict_row(row, args.model, args.margin, not args.no_align)
            vectors.append(features(probs, row['blendshapes'], names) if can_fuse else np.zeros(7 + len(names)))
            probabilities.append([probs[label] for label in LABELS])
            labels.append(LABELS.index(row['label']))
            available.append(can_fuse)
            if row_index % 500 == 0:
                print(f'Extracting {name}: {row_index}/{len(splits[name])}', flush=True)
        arrays[name] = (np.array(vectors), np.array(probabilities), np.array(labels))
        masks[name] = np.array(available)
        coverage[name] = {'used': int(sum(available)), 'total': len(splits[name])}
        if set(labels) != set(range(7)):
            parser.error(f'Tập {name} không đủ 7 lớp có blendshape để train/validation fusion.')
        print(f"Features {name}: {coverage[name]['used']}/{len(splits[name])}", flush=True)
    result = fit_correction(*arrays['train'], *arrays['validation'], args.epochs,
                            args.learning_rate, args.regularization, val_mask=masks['validation'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = args.output.with_suffix('.report.json')
    result['coverage'] = coverage
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['best_epoch'] == 0:
        print(f'Fusion không cải thiện validation sad/fear + macro-F1. Giữ EfficientNet. Báo cáo: {report}')
        return 0
    artifact = {key: result[key] for key in ('mean', 'scale', 'weights', 'bias', 'best_epoch')}
    artifact.update({'version': 1, 'labels': LABELS, 'blendshape_names': names,
                     'base_sha256': file_hash(model_path), 'base_model': args.model,
                     'crop_config': {'margin': args.margin, 'align': not args.no_align, 'version': CROP_VERSION},
                     'train_subjects': sorted({str(row['subject']) for row in splits['train'] if row.get('subject') is not None}),
                     'train_image_hashes': sorted({file_hash(row['path']) for row in splits['train']}),
                     'train_pixel_hashes': sorted({image_hash(row['path']) for row in splits['train']}),
                     'coverage': coverage,
                     'landmarker_sha256': next(iter(landmarker_hashes)),
                     'baseline_validation': result['baseline_validation'],
                     'best_validation': result['best_validation']})
    args.output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(f'Fusion: {args.output}. Test chưa được dùng; chạy evaluate_webcam.py --split test sau khi chốt cấu hình.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
