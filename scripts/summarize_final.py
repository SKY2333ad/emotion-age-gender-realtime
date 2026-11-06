"""Tổng hợp số cuối vào results/final_summary.json: CI trên test UTKFace và confusion matrix FER+ trước/sau.

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
import keras
from predict import LABELS
from scripts.calibrate_age_groups import GROUPS, predict_groups
from scripts.calibrate_attributes import load
from scripts.evaluate_ferplus import confusion, pick
from scripts.train_rafdb import scores_from_confusion
from ui_helpers import AGE_GROUP_THRESHOLDS

SEED, ROUNDS, TEST_SAMPLE = 42, 1000, 7000


def macro_f1(truth, predicted):
    matrix = np.zeros((6, 6))
    np.add.at(matrix, (truth, predicted), 1)
    return float((2 * np.diag(matrix) / np.maximum(matrix.sum(axis=0) + matrix.sum(axis=1), 1)).mean())


def ferplus():
    data = np.load(ROOT / 'data' / 'ferplus' / 'ferplus.npz')
    truth = np.array([LABELS.index(label) for label in data['labels']])[pick(data['split'], 'test', TEST_SAMPLE)]
    head = keras.models.load_model(ROOT / 'results' / 'emotion_head' / 'best_head.keras', compile=False)
    after = head(np.load(ROOT / 'results' / 'emotion_head' / 'features_test.npy'), training=False).numpy().argmax(axis=1)
    before = np.load(ROOT / 'results' / 'ferplus' / 'efficientnet.npz')['test'].argmax(axis=1)
    scores = scores_from_confusion(confusion(truth, after))
    return {'labels': LABELS, 'efficientnet': confusion(truth, before).tolist(),
            'efficientnet_ferplus': confusion(truth, after).tolist(), 'efficientnet_ferplus_per_class': scores['per_class'],
            'efficientnet_ferplus_weighted_f1': sum(value['f1'] * value['support'] for value in scores['per_class'].values()) / len(truth)}


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
    report = {'utkface_test': utkface(), 'ferplus_test': ferplus()}
    (ROOT / 'results' / 'final_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['utkface_test'], ensure_ascii=True))


if __name__ == '__main__':
    main()
