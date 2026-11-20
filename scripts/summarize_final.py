"""Tổng hợp số cuối vào results/final_summary.json: CI giới tính trên test UTKFace. Số cảm xúc RAF-DB nằm ở results/rafdb_test/.

Chỉ tính lại từ file dự đoán đã có; không chọn tham số nào ở đây.
python scripts/summarize_final.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from scripts.calibrate_attributes import load

SEED, ROUNDS = 42, 1000


def utkface():
    data = load(ROOT / 'results' / 'utkface_baseline' / 'predictions.csv')
    test = ~data['validation']
    correct = (data['logit'][test] >= 0) == data['male'][test]
    rng = np.random.default_rng(SEED)
    samples = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(ROUNDS)]
    return {'images': len(correct), 'seed': SEED, 'rounds': ROUNDS, 'unit': 'ảnh',
            'gender_accuracy': float(correct.mean()),
            'ci95': {'gender_accuracy': [float(value) for value in np.percentile(samples, [2.5, 97.5])]}}


def main():
    report = {'utkface_test': utkface()}
    (ROOT / 'results' / 'final_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['utkface_test'], ensure_ascii=True))


if __name__ == '__main__':
    main()
