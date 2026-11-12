"""Điểm chất lượng có ích không: mặt điểm thấp có bị đoán sai nhiều hơn không, ảnh xấu đi có bị hạ điểm không.

Chỉ dùng UTKFace validation (không chạm test). Không dùng được cho cảm xúc vì UTKFace không có nhãn cảm xúc.
python scripts/system/evaluate_quality.py --output results/system/quality_evaluation.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from attributes import predict_attributes
from detectors import detect_image_faces
from quality import face_quality
from scripts.age_gender.calibrate_age_groups import split_is_validation
from webcam_data import configure_console, read_image


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results' / 'age_gender' / 'utkface_baseline' / 'predictions.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data' / 'utkface' / 'UTKFace')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'system' / 'quality_evaluation.json')
    args = parser.parse_args()
    with args.predictions.open(newline='', encoding='utf-8-sig') as file:
        rows = [row for row in csv.DictReader(file) if split_is_validation(row['file'])]
    scores, parts, gender_wrong, age_error = [], [], [], []
    degraded = {'gốc': [], 'mờ σ=3': [], 'mờ σ=6': [], 'tối ×0.3': [], 'sáng ×2.5': []}
    harm = {name: [] for name in degraded}      # (giới tính sai, sai số tuổi) khi ảnh bị làm xấu đi
    for index, row in enumerate(rows):
        image = read_image(args.images / row['file'])
        faces = detect_image_faces(image)
        if not faces:
            continue
        face = max(faces, key=lambda item: item['box'][2] * item['box'][3])
        quality = face_quality(image, face['box'], face['angle'], face['yaw'])
        scores.append(quality['score'])
        parts.append([quality[key] for key in ('size', 'sharpness', 'lighting', 'pose')])
        gender_wrong.append(int(row['gender_correct']) == 0)
        age_error.append(float(row['age_abs_error']))
        if index % 4 == 0:
            degraded['gốc'].append(quality['score'])
            harm['gốc'].append((gender_wrong[-1], age_error[-1]))
            for name, changed in (('mờ σ=3', cv2.GaussianBlur(image, (0, 0), 3)),
                                  ('mờ σ=6', cv2.GaussianBlur(image, (0, 0), 6)),
                                  ('tối ×0.3', (image * .3).astype(np.uint8)),
                                  ('sáng ×2.5', np.clip(image * 2.5, 0, 255).astype(np.uint8))):
                degraded[name].append(face_quality(changed, face['box'], face['angle'], face['yaw'])['score'])
                result = predict_attributes(changed, face['box'])
                if result is None:
                    continue
                harm[name].append((result['gender'] != row['gender_true'], abs(result['age'] - float(row['age_true']))))
    scores, gender_wrong, age_error = np.array(scores), np.array(gender_wrong), np.array(age_error)
    # Xếp theo điểm rồi chia đều, vì nhiều mặt có điểm bằng nhau nên không dùng được biên phân vị.
    quartiles = [{'score_range': [float(scores[group].min()), float(scores[group].max())], 'count': len(group),
                  'gender_error': float(gender_wrong[group].mean()), 'age_mae': float(age_error[group].mean())}
                 for group in np.array_split(np.argsort(scores, kind='stable'), 4)]
    report = {'split': 'UTKFace validation', 'images': len(scores),
              'score_percentiles': dict(zip(('p5', 'p50', 'p95'), map(float, np.percentile(scores, [5, 50, 95])))),
              'part_means': dict(zip(('size', 'sharpness', 'lighting', 'pose'), map(float, np.mean(parts, axis=0)))),
              'quartiles': quartiles,
              'degraded_median_score': {name: float(np.median(values)) for name, values in degraded.items()},
              'degraded_prediction_error': {name: {'images': len(values), 'gender_error': float(np.mean([v[0] for v in values])),
                                                   'age_mae': float(np.mean([v[1] for v in values]))}
                                            for name, values in harm.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
