"""E3/E5/E6: so sánh model cảm xúc trên FER+ (tập độc lập), kèm TTA lật ngang và temperature scaling.

Temperature và lựa chọn chỉ dùng validation; test chỉ báo cáo. FER+ là ảnh xám 48×48 từ web,
không đại diện webcam. Khoảng tin cậy 95% bootstrap theo nhóm ảnh trùng pixel (không có ID người).
python scripts/evaluate_ferplus.py --models efficientnet hsemotion --output results/ferplus
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from predict import LABELS, _get_inference, hsemotion_batch
from preprocessing import prepare_face
from scripts.train_rafdb import scores_from_confusion
from webcam_data import configure_console

BOOTSTRAP_SEED, BOOTSTRAP_ROUNDS = 42, 1000


def efficientnet_batch(images):
    inference = _get_inference('efficientnet')
    return inference(np.stack([prepare_face(image) for image in images])).numpy()


def pick(splits, split, limit):
    """Mẫu ngẫu nhiên cố định seed, tránh thứ tự lớp trong file; limit=None lấy toàn bộ."""
    index = np.flatnonzero(splits == split)
    return index if limit is None else np.sort(np.random.default_rng(42).permutation(index)[:limit])


def run_model(call, pixels, batch_size, flip=False):
    output = []
    for offset in range(0, len(pixels), batch_size):
        images = [cv2.cvtColor(item, cv2.COLOR_GRAY2BGR) for item in pixels[offset:offset + batch_size]]
        probs = call(images)
        if flip:
            probs = (probs + call([cv2.flip(image, 1) for image in images])) / 2
        output.append(probs)
    return np.concatenate(output)


def confusion(truth, predicted):
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def calibration(probs, truth, bins=15):
    confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
    correct = predicted == truth
    edges = np.linspace(0, 1, bins + 1)
    ece = sum(abs(correct[mask].mean() - confidence[mask].mean()) * mask.mean()
              for low, high in zip(edges, edges[1:])
              for mask in [(confidence > low) & (confidence <= high)] if mask.any())
    one_hot = np.eye(7)[truth]
    return {'ece': float(ece), 'brier': float(((probs - one_hot) ** 2).sum(axis=1).mean()),
            'nll': float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1)).mean())}


def fit_temperature(probs, truth):
    """Nghiệm một chiều trên validation: lưới rồi tinh chỉnh, không cần thư viện tối ưu."""
    logits = np.log(np.clip(probs, 1e-12, 1))

    def nll(temperature):
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        return float((np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(len(truth)), truth]).mean())

    grid = np.arange(0.3, 5.01, 0.05)
    best = grid[int(np.argmin([nll(value) for value in grid]))]
    fine = np.arange(best - 0.05, best + 0.0501, 0.005)
    return float(fine[int(np.argmin([nll(value) for value in fine]))])


def apply_temperature(probs, temperature):
    scaled = np.log(np.clip(probs, 1e-12, 1)) / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def bootstrap(truth, predicted, group):
    """Lấy lại mẫu theo nhóm; trả CI 95% cho accuracy, macro-F1 và F1 từng lớp."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique, inverse = np.unique(group, return_inverse=True)
    order = np.argsort(inverse)
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    samples = {'accuracy': [], 'macro_f1': [], **{label: [] for label in LABELS}}
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, len(unique), len(unique))
        counts = np.bincount(picked, minlength=len(unique))
        weights = np.zeros(len(truth))
        lengths = ends - starts
        weights[order] = np.repeat(counts, lengths)
        matrix = np.zeros((7, 7))
        np.add.at(matrix, (truth, predicted), weights)
        true_positive = np.diag(matrix)
        f1 = 2 * true_positive / np.maximum(matrix.sum(axis=0) + matrix.sum(axis=1), 1)
        samples['accuracy'].append(true_positive.sum() / matrix.sum())
        samples['macro_f1'].append(f1.mean())
        for index, label in enumerate(LABELS):
            samples[label].append(f1[index])
    return {key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for key, values in samples.items()}


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'ferplus' / 'ferplus.npz')
    parser.add_argument('--hsemotion', type=Path, default=ROOT / 'models' / 'hsemotion_enet_b0_8_best_afew.onnx')
    parser.add_argument('--models', nargs='+', default=['efficientnet', 'hsemotion'],
                        choices=['efficientnet', 'hsemotion'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, help='Lấy N ảnh validation và 7N/3 ảnh test ngẫu nhiên (CPU chậm).')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'ferplus')
    args = parser.parse_args()
    data = np.load(args.data)
    truth_all = np.array([LABELS.index(label) for label in data['labels']])
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    limits = {'validation': args.limit, 'test': args.limit and args.limit * 7 // 3}
    for name in args.models:
        call = efficientnet_batch if name == 'efficientnet' else (lambda images: hsemotion_batch(images, args.hsemotion))
        for flip in (False, True):
            key = name + ('+flip' if flip else '')
            result, stored = {}, {}
            for split in ('validation', 'test'):
                index = pick(data['split'], split, limits[split])
                start = time.perf_counter()
                stored[split] = run_model(call, data['pixels'][index], args.batch_size, flip)
                result[split + '_ms_per_image'] = (time.perf_counter() - start) / len(index) * 1000
                print(key, split, f"{result[split + '_ms_per_image']:.1f} ms/ảnh", flush=True)
            validation = pick(data['split'], 'validation', limits['validation'])
            test = pick(data['split'], 'test', limits['test'])
            temperature = fit_temperature(stored['validation'], truth_all[validation])
            np.savez_compressed(args.output / f'{key}.npz', **stored)
            truth, probs = truth_all[test], stored['test']
            predicted = probs.argmax(axis=1)
            matrix = confusion(truth, predicted)
            scores = scores_from_confusion(matrix)
            weights = matrix.sum(axis=1) / matrix.sum()
            result.update({
                'validation_macro_f1': scores_from_confusion(confusion(truth_all[validation], stored['validation'].argmax(axis=1)))['macro_f1'],
                'test': scores, 'test_weighted_f1': float(sum(
                    weights[i] * scores['per_class'][label]['f1'] for i, label in enumerate(LABELS))),
                'test_ci95': bootstrap(truth, predicted, data['group'][test]),
                'temperature': temperature, 'test_calibration_before': calibration(probs, truth),
                'test_calibration_after': calibration(apply_temperature(probs, temperature), truth)})
            report[key] = result
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(key, f"test acc {scores['accuracy']:.4f} macro-F1 {scores['macro_f1']:.4f} "
                       f"T={temperature:.3f}", flush=True)
    print('Báo cáo:', report_path)


if __name__ == '__main__':
    main()
