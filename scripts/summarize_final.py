"""Tổng hợp số cuối vào results/final_summary.json: CI trên test UTKFace. Số cảm xúc RAF-DB nằm ở results/rafdb_test/.

Chỉ tính lại từ file dự đoán đã có; không chọn tham số nào ở đây.
python scripts/summarize_final.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from scripts.calibrate_age_groups import GROUPS, predict_groups
from scripts.calibrate_attributes import load
from ui_helpers import AGE_GROUP_THRESHOLDS

SEED, ROUNDS = 42, 1000


def macro_f1(truth, predicted):
    matrix = np.zeros((6, 6))
    np.add.at(matrix, (truth, predicted), 1)
    return float((2 * np.diag(matrix) / np.maximum(matrix.sum(axis=0) + matrix.sum(axis=1), 1)).mean())


def utkface():
    path = ROOT / 'results' / 'utkface_baseline' / 'predictions.csv'
    data = load(path)
    with path.open(newline='', encoding='utf-8-sig') as file:
        groups = np.array([GROUPS.index(row['age_group_true']) for row in csv.DictReader(file)])
    test = ~data['validation']
    truth, ages, true_age = groups[test], data['predicted_age'][test], data['age'][test]
    predicted = predict_groups(ages, list(AGE_GROUP_THRESHOLDS))
    correct = (data['logit'][test] >= 0) == data['male'][test]
    rng = np.random.default_rng(SEED)
    samples = {key: [] for key in ('age_group_accuracy', 'age_group_macro_f1', 'age_mae', 'gender_accuracy')}
    for _ in range(ROUNDS):
        index = rng.integers(0, len(ages), len(ages))
        samples['age_group_accuracy'].append((predicted[index] == truth[index]).mean())
        samples['age_group_macro_f1'].append(macro_f1(truth[index], predicted[index]))
        samples['age_mae'].append(np.abs(ages[index] - true_age[index]).mean())
        samples['gender_accuracy'].append(correct[index].mean())
    return {'images': len(ages), 'seed': SEED, 'rounds': ROUNDS, 'unit': 'ảnh',
            'age_group_accuracy': float((predicted == truth).mean()), 'age_group_macro_f1': macro_f1(truth, predicted),
            'age_mae': float(np.abs(ages - true_age).mean()), 'gender_accuracy': float(correct.mean()),
            'ci95': {key: [float(value) for value in np.percentile(values, [2.5, 97.5])] for key, values in samples.items()}}


def main():
    report = {'utkface_test': utkface()}
    (ROOT / 'results' / 'final_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['utkface_test'], ensure_ascii=True))


if __name__ == '__main__':
    main()
