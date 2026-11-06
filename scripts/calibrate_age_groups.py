"""Chọn crop scale và ngưỡng 6 nhóm tuổi trên validation, báo kết quả test riêng."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ['0–12', '13–19', '20–29', '30–39', '40–49', '50+']
DEFAULT_THRESHOLDS = np.array([12.5, 19.5, 29.5, 39.5, 49.5])


def split_is_validation(filename):
    """Split ổn định 20%; không phụ thuộc thứ tự file hay random state."""
    return int(hashlib.sha256(filename.encode('utf-8')).hexdigest()[:8], 16) % 5 == 0


def load_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    return [{'file': row['file'], 'age_pred': float(row['age_pred']),
             'target': GROUPS.index(row['age_group_true'])} for row in rows]


def predict_groups(ages, thresholds):
    return np.searchsorted(np.asarray(thresholds), ages, side='right')


def scores(targets, predictions):
    confusion = np.zeros((len(GROUPS), len(GROUPS)), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    f1, recall = [], []
    for index in range(len(GROUPS)):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
    return {'accuracy': float(np.trace(confusion) / max(confusion.sum(), 1)),
            'macro_f1': float(np.mean(f1)), 'balanced_accuracy': float(np.mean(recall)),
            'per_group_recall': dict(zip(GROUPS, map(float, recall))),
            'confusion_matrix': confusion.tolist()}


def objective(ages, targets, thresholds):
    result = scores(targets, predict_groups(ages, thresholds))
    return result['macro_f1'], result['accuracy']


def tune_thresholds(ages, targets):
    """Coordinate ascent nhiều điểm khởi tạo; ngưỡng đơn điệu, bước 0.5 tuổi thô."""
    candidates = np.arange(0.5, 100.5, 0.5)
    medians = [np.median(ages[targets == index]) for index in range(len(GROUPS))]
    midpoint_start = np.array([(medians[i] + medians[i + 1]) / 2 for i in range(5)])
    quantile_start = np.quantile(ages, np.cumsum(np.bincount(targets, minlength=6))[:-1] / len(targets))
    rng = np.random.default_rng(20261106)
    starts = [DEFAULT_THRESHOLDS, midpoint_start, quantile_start]
    starts += [np.sort(rng.choice(candidates, size=5, replace=False)) for _ in range(12)]
    best_thresholds, best_score = None, (-1.0, -1.0)
    for start in starts:
        thresholds = np.round(np.clip(start, .5, 100) * 2) / 2
        thresholds = np.maximum.accumulate(thresholds + np.arange(5) * 1e-6)
        for _ in range(8):
            changed = False
            for index in range(5):
                low = thresholds[index - 1] + .5 if index else .5
                high = thresholds[index + 1] - .5 if index < 4 else 100
                local_best, local_value = thresholds[index], objective(ages, targets, thresholds)
                for candidate in candidates[(candidates >= low) & (candidates <= high)]:
                    trial = thresholds.copy()
                    trial[index] = candidate
                    value = objective(ages, targets, trial)
                    if value > local_value:
                        local_best, local_value = candidate, value
                if local_best != thresholds[index]:
                    thresholds[index], changed = local_best, True
            if not changed:
                break
        value = objective(ages, targets, thresholds)
        if value > best_score:
            best_thresholds, best_score = thresholds.copy(), value
    return best_thresholds


def evaluate_file(path):
    rows = load_csv(path)
    validation = [row for row in rows if split_is_validation(row['file'])]
    test = [row for row in rows if not split_is_validation(row['file'])]
    val_ages = np.array([row['age_pred'] for row in validation])
    val_targets = np.array([row['target'] for row in validation])
    thresholds = tune_thresholds(val_ages, val_targets)
    result = {'file': str(path), 'validation_count': len(validation), 'test_count': len(test),
              'thresholds': thresholds.tolist()}
    for name, subset in [('validation', validation), ('test', test)]:
        ages = np.array([row['age_pred'] for row in subset])
        targets = np.array([row['target'] for row in subset])
        result[name] = scores(targets, predict_groups(ages, thresholds))
        result[name + '_default_thresholds'] = scores(
            targets, predict_groups(ages, DEFAULT_THRESHOLDS))
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, nargs='+')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'utkface_attributes' / 'calibration.json')
    args = parser.parse_args(argv)
    results = [evaluate_file(path) for path in args.predictions]
    best = max(results, key=lambda item: (item['validation']['macro_f1'],
                                          item['validation']['accuracy']))
    report = {'selection_metric': 'validation macro_f1',
              'split': 'SHA-256(filename) modulo 5: 0=validation, còn lại=test',
              'groups': GROUPS, 'candidates': results, 'selected': best}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in results:
        print(f"{item['file']}: val macro-F1 {item['validation']['macro_f1']:.1%}, "
              f"test macro-F1 {item['test']['macro_f1']:.1%}, test accuracy {item['test']['accuracy']:.1%}, "
              f"ngưỡng {item['thresholds']}")
    print(f"Đã chọn: {best['file']} · test macro-F1 {best['test']['macro_f1']:.1%} · "
          f"test accuracy {best['test']['accuracy']:.1%}")
    print(f'Báo cáo: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
