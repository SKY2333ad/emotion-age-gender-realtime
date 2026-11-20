"""Hiệu chuẩn gender_prob (temperature) trên validation UTKFace, báo kết quả test và độ chênh giữa các nhóm.

Split cố định theo hash tên file; test chỉ để báo cáo. Ghi tham số riêng khỏi trọng số model.
Nhóm tuổi trong báo cáo lấy từ nhãn tuổi thật của UTKFace, chỉ để xem giới tính đúng/sai ở nhóm nào.
python scripts/calibrate_attributes.py results/utkface_baseline/predictions.csv --report-dir results/attribute_calibration
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT))
from webcam_data import configure_console

BOOTSTRAP_SEED, BOOTSTRAP_ROUNDS = 42, 1000
TRUE_AGE_GROUPS = ['0–12', '13–19', '20–29', '30–39', '40–49', '50+']


def split_is_validation(filename):
    """Split ổn định 20%; không phụ thuộc thứ tự file hay random state."""
    return int(hashlib.sha256(filename.encode('utf-8')).hexdigest()[:8], 16) % 5 == 0


def load(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    probability = np.clip([float(row['gender_prob']) for row in rows], 1e-6, 1 - 1e-6)
    predicted_male = np.array([row['gender_pred'] == 'male' for row in rows])
    p_male = np.where(predicted_male, probability, 1 - probability)
    return {'validation': np.array([split_is_validation(row['file']) for row in rows]),
            'logit': np.log(p_male / (1 - p_male)),
            'male': np.array([row['gender_true'] == 'male' for row in rows]),
            'group': np.array([row['age_group_true'] for row in rows])}


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def binary_metrics(probability, truth, bins=15):
    confidence = np.maximum(probability, 1 - probability)
    correct = (probability >= .5) == truth
    edges = np.linspace(.5, 1, bins + 1)
    ece, curve = 0.0, []
    for low, high in zip(edges, edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            gap = correct[mask].mean() - confidence[mask].mean()
            ece += abs(gap) * mask.mean()
            curve.append({'confidence': float(confidence[mask].mean()), 'accuracy': float(correct[mask].mean()),
                          'count': int(mask.sum())})
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {'accuracy': float(correct.mean()), 'ece': float(ece),
            'brier': float(((probability - truth) ** 2).mean()),
            'log_loss': float(-(truth * np.log(clipped) + (~truth) * np.log(1 - clipped)).mean()),
            'reliability': curve}


def fit_temperature(logit, truth):
    def loss(temperature):
        probability = np.clip(sigmoid(logit / temperature), 1e-12, 1 - 1e-12)
        return float(-(truth * np.log(probability) + (~truth) * np.log(1 - probability)).mean())

    grid = np.arange(0.5, 30.01, 0.05)
    return float(grid[int(np.argmin([loss(value) for value in grid]))])


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def subgroup(mask, probability, data):
    """Không có ID người nên bootstrap theo ảnh; UTKFace có thể chứa nhiều ảnh của một người."""
    truth, chosen = data['male'][mask], probability[mask]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accuracy = []
    for _ in range(BOOTSTRAP_ROUNDS):
        index = rng.integers(0, len(truth), len(truth))
        accuracy.append(((chosen[index] >= .5) == truth[index]).mean())
    metrics = binary_metrics(chosen, truth)
    predicted_male = chosen >= .5
    tp, fp, fn = (predicted_male & truth).sum(), (predicted_male & ~truth).sum(), (~predicted_male & truth).sum()
    return {'count': int(mask.sum()), 'gender_accuracy': metrics['accuracy'], 'gender_accuracy_ci95': ci(accuracy),
            'male_precision': float(tp / max(tp + fp, 1)), 'male_recall': float(tp / max(tp + fn, 1)),
            'male_f1': float(2 * tp / max(2 * tp + fp + fn, 1)), 'gender_ece': metrics['ece'],
            'male_fnr': float((~predicted_male[truth]).mean()) if truth.any() else None,
            'male_fpr': float(predicted_male[~truth].mean()) if (~truth).any() else None}


def gap(table, names):
    """Khoảng cách độ chính xác giới tính giữa nhóm tốt nhất và kém nhất."""
    ranked = sorted(names, key=lambda name: table[name]['gender_accuracy'])
    return {'gap': table[ranked[-1]]['gender_accuracy'] - table[ranked[0]]['gender_accuracy'],
            'lowest': ranked[0], 'highest': ranked[-1]}


def reliability_plot(before, after, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([.5, 1], [.5, 1], color='gray', linestyle='--', label='Hoàn hảo')
    for name, curve in (('Trước hiệu chuẩn', before), ('Sau hiệu chuẩn', after)):
        axis.plot([item['confidence'] for item in curve], [item['accuracy'] for item in curve], marker='o', label=name)
    axis.set(xlabel='Độ tin cậy của model', ylabel='Tỉ lệ đúng thực tế', title='Reliability diagram: gender_prob (UTKFace test)')
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'results' / 'attribute_calibration')
    parser.add_argument('--output', type=Path, default=ROOT / 'models' / 'genderage_calibration.json')
    args = parser.parse_args()
    data = load(args.predictions)
    validation, test = data['validation'], ~data['validation']
    temperature = fit_temperature(data['logit'][validation], data['male'][validation])
    before, after = sigmoid(data['logit']), sigmoid(data['logit'] / temperature)
    report = {
        'validation_count': int(validation.sum()), 'test_count': int(test.sum()), 'temperature': temperature,
        'gender_test_before': binary_metrics(before[test], data['male'][test]),
        'gender_test_after': binary_metrics(after[test], data['male'][test]),
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'rounds': BOOTSTRAP_ROUNDS, 'unit': 'ảnh; UTKFace không có ID người'},
        'subgroups_test': {}}
    subgroups = {'gender=female': ~data['male'], 'gender=male': data['male']}
    subgroups.update({f'age={name}': data['group'] == name for name in TRUE_AGE_GROUPS})
    subgroups.update({f'gender={sex}, age={name}': mask & (data['group'] == name)
                      for sex, mask in (('female', ~data['male']), ('male', data['male'])) for name in TRUE_AGE_GROUPS})
    for name, mask in subgroups.items():
        report['subgroups_test'][name] = subgroup(mask & test, after, data)
    groups = report['subgroups_test']
    report['gaps_test'] = {
        'theo giới tính': gap(groups, ['gender=female', 'gender=male']),
        'theo nhóm tuổi thật': gap(groups, [f'age={name}' for name in TRUE_AGE_GROUPS]),
        'giới tính × nhóm tuổi thật': gap(groups, [key for key in groups if ', ' in key])}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    reliability_plot(report['gender_test_before']['reliability'], report['gender_test_after']['reliability'],
                     args.report_dir / 'gender_reliability.png')
    (args.report_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.write_text(json.dumps({
        'gender_temperature': temperature,
        'fitted_on': 'UTKFace validation (SHA-256 tên file % 5 == 0), crop_scale 1.5'}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    gender = report['gender_test_before'], report['gender_test_after']
    print(f"T={temperature:.2f}; ECE {gender[0]['ece']:.4f} -> {gender[1]['ece']:.4f}; "
          f"Brier {gender[0]['brier']:.4f} -> {gender[1]['brier']:.4f}; "
          f"log loss {gender[0]['log_loss']:.4f} -> {gender[1]['log_loss']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
