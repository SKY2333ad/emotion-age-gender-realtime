"""Chọn ngưỡng tuổi và TTA chỉ bằng validation UTKFace, bootstrap theo cặp ảnh.

Quy tắc đặt trước: chỉ đổi khỏi cấu hình hiện tại khi CI 95% của mức chênh nằm hoàn toàn trên 0.
Script này không đọc ảnh test; đánh giá test nằm ở scripts/calibrate_attributes.py.
python scripts/select_attribute_config.py --output results/attribute_selection.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from scripts.calibrate_age_groups import GROUPS, predict_groups, split_is_validation, tune_thresholds
from ui_helpers import AGE_GROUP_THRESHOLDS
from webcam_data import configure_console

ROUNDS, SEED = 1000, 42


def load_validation(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        rows = {row['file']: row for row in csv.DictReader(file) if split_is_validation(row['file'])}
    return rows


def macro_f1(targets, predictions):
    matrix = np.zeros((6, 6))
    np.add.at(matrix, (targets, predictions), 1)
    return float((2 * np.diag(matrix) / np.maximum(matrix.sum(axis=0) + matrix.sum(axis=1), 1)).mean())


def compare(name, values, base):
    """values, base: hàm nhận chỉ số ảnh và trả điểm; lấy lại cùng một bộ ảnh cho hai bên."""
    count = len(base['targets'])
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(ROUNDS):
        index = rng.integers(0, count, count)
        differences.append(values(index) - base['score'](index))
    low, high = np.percentile(differences, [2.5, 97.5])
    point = values(np.arange(count)) - base['score'](np.arange(count))
    return {'config': name, 'difference': float(point), 'ci95': [float(low), float(high)], 'accept': bool(low > 0)}


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results' / 'utkface_baseline' / 'predictions.csv')
    parser.add_argument('--tta', type=Path, default=ROOT / 'results' / 'utkface_tta' / 'predictions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'attribute_selection.json')
    args = parser.parse_args()
    plain, flipped = load_validation(args.baseline), load_validation(args.tta)
    files = sorted(set(plain) & set(flipped))
    targets = np.array([GROUPS.index(plain[name]['age_group_true']) for name in files])
    truth_age = np.array([float(plain[name]['age_true']) for name in files])
    ages = {key: np.array([float(rows[name]['age_pred']) for name in files])
            for key, rows in (('plain', plain), ('tta', flipped))}
    gender = {key: np.array([int(rows[name]['gender_correct']) for name in files], float)
              for key, rows in (('plain', plain), ('tta', flipped))}
    current = np.array(AGE_GROUP_THRESHOLDS)
    refit = {key: tune_thresholds(values, targets) for key, values in ages.items()}

    def age_score(key, thresholds):
        predicted = predict_groups(ages[key], thresholds)
        return lambda index: macro_f1(targets[index], predicted[index])

    base = {'targets': targets, 'score': age_score('plain', current)}
    age_results = [compare(name, score, base) for name, score in (
        ('ngưỡng fit lại, không TTA', age_score('plain', refit['plain'])),
        ('ngưỡng hiện tại, TTA', age_score('tta', current)),
        ('ngưỡng fit lại, TTA', age_score('tta', refit['tta'])))]
    gender_base = {'targets': targets, 'score': lambda index: gender['plain'][index].mean()}
    gender_result = compare('TTA giới tính (accuracy)', lambda index: gender['tta'][index].mean(), gender_base)
    mae_base = {'targets': targets, 'score': lambda index: -np.abs(ages['plain'][index] - truth_age[index]).mean()}
    mae_result = compare('TTA tuổi (giảm MAE)', lambda index: -np.abs(ages['tta'][index] - truth_age[index]).mean(), mae_base)
    report = {'split': 'UTKFace validation', 'images': len(files), 'bootstrap': {'rounds': ROUNDS, 'seed': SEED, 'unit': 'ảnh'},
              'rule': 'Chỉ đổi khi CI 95% của mức chênh theo cặp nằm hoàn toàn trên 0',
              'caveat': 'Ngưỡng fit lại được chọn trên chính validation này nên mức chênh của chúng thiên lạc quan.',
              'thresholds': {'current': current.tolist(), 'refit_plain': refit['plain'].tolist(), 'refit_tta': refit['tta'].tolist()},
              'age_group_macro_f1': age_results, 'gender': gender_result, 'age_mae': mae_result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
