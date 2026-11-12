"""Đánh giá genderage.onnx trên UTKFace bằng đúng bbox MediaPipe của dự án.

Chỉ đọc ảnh và lưu số liệu dự đoán; không tạo hoặc lưu crop khuôn mặt.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'core'))
from attributes import CROP_SCALE, MODEL_PATH, predict_attributes
from detectors import detect_image_faces
from ui_helpers import AGE_GROUPS, age_group, reference_age_group
from webcam_data import configure_console, file_hash, read_image

FIELDS = ['file', 'age_true', 'gender_true', 'race', 'age_pred', 'gender_pred',
          'gender_prob', 'age_abs_error', 'age_group_true', 'age_group_pred',
          'age_group_correct', 'gender_correct']
GENDERS = {0: 'male', 1: 'female'}
RACES = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other'}


def parse_label(path):
    parts = path.name.split('_')
    try:
        age, gender, race = map(int, parts[:3])
    except (ValueError, IndexError):
        return None
    if not 0 <= age <= 116 or gender not in GENDERS or race not in RACES:
        return None
    return age, GENDERS[gender], RACES[race]


def prediction_row(path, label, crop_scale, tta=False):
    image = read_image(path)
    faces = detect_image_faces(image)
    if not faces:
        return None, 'no_face'
    if len(faces) > 1:
        # UTKFace mô tả một mặt mỗi ảnh; chọn mặt lớn nhất nhưng ghi lại để kiểm toán.
        reason = 'many_faces'
    else:
        reason = 'ok'
    face = max(faces, key=lambda value: value['box'][2] * value['box'][3])
    result = predict_attributes(image, face['box'], face.get('angle', 0.0), crop_scale=crop_scale, tta=tta)
    if result is None:
        return None, 'invalid_prediction'
    age_true, gender_true, race = label
    group_true, group_pred = reference_age_group(age_true), age_group(result['age'])
    return {
        'file': path.name, 'age_true': age_true, 'gender_true': gender_true, 'race': race,
        'age_pred': f"{result['age']:.6f}", 'gender_pred': result['gender'],
        'gender_prob': f"{result['gender_prob']:.6f}",
        'age_abs_error': f"{abs(result['age'] - age_true):.6f}",
        'age_group_true': group_true, 'age_group_pred': group_pred,
        'age_group_correct': int(group_true == group_pred),
        'gender_correct': int(gender_true == result['gender'])}, reason


def evaluate_item(item):
    """Worker độc lập: mỗi process có MediaPipe/ONNX session riêng."""
    path, label, crop_scale, tta = item
    try:
        row, reason = prediction_row(path, label, crop_scale, tta)
        return row, reason, None
    except (OSError, ValueError, RuntimeError) as exc:
        return None, 'error', {'file': path.name, 'error': str(exc)}


def read_predictions(path):
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError('CSV dự đoán có schema khác; hãy xóa hoặc chọn --output-dir khác.')
        return list(reader)


def safe_mean(values):
    return float(np.mean(values)) if values else None


def metrics(rows):
    errors = [float(row['age_abs_error']) for row in rows]
    gender_correct = [int(row['gender_correct']) for row in rows]
    groups_correct = [int(row['age_group_correct']) for row in rows]
    result = {
        'evaluated': len(rows),
        'age': {'mae': safe_mean(errors),
                'median_absolute_error': float(np.median(errors)) if errors else None,
                'within_5_years': safe_mean([error <= 5 for error in errors]),
                'within_10_years': safe_mean([error <= 10 for error in errors]),
                'group_accuracy': safe_mean(groups_correct)},
        'gender': {'accuracy': safe_mean(gender_correct)},
        'by_gender': {}, 'by_race': {}, 'by_age_group': {},
        'gender_confusion': {'female': {'female': 0, 'male': 0},
                             'male': {'female': 0, 'male': 0}}}
    for row in rows:
        result['gender_confusion'][row['gender_true']][row['gender_pred']] += 1
    dimensions = [('by_gender', 'gender_true', ('female', 'male')),
                  ('by_race', 'race', tuple(RACES.values())),
                  ('by_age_group', 'age_group_true', tuple(AGE_GROUPS))]
    for output_key, field, ordered_values in dimensions:
        for value in ordered_values:
            subset = [row for row in rows if row[field] == value]
            subset_errors = [float(row['age_abs_error']) for row in subset]
            result[output_key][value] = {
                'count': len(subset), 'age_mae': safe_mean(subset_errors),
                'age_group_accuracy': safe_mean([int(row['age_group_correct']) for row in subset]),
                'gender_accuracy': safe_mean([int(row['gender_correct']) for row in subset])}
    accuracies = [value['gender_accuracy'] for value in result['by_gender'].values()
                  if value['gender_accuracy'] is not None]
    result['gender']['balanced_accuracy'] = safe_mean(accuracies)
    return result


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'utkface' / 'UTKFace')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'age_gender' / 'utkface_attributes')
    parser.add_argument('--limit', type=int, help='Chỉ chạy N ảnh đầu để smoke test.')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Số tiến trình MediaPipe/ONNX chạy song song (mặc định tối đa 4).')
    parser.add_argument('--crop-scale', type=float, default=CROP_SCALE,
                        help='Hệ số ngữ cảnh crop thuộc tính; mặc định lấy từ core/attributes.py.')
    parser.add_argument('--resume', action='store_true', help='Tiếp tục CSV đã có, bỏ qua ảnh đã xử lý.')
    parser.add_argument('--tta', action='store_true', help='Trung bình ảnh gốc và ảnh lật ngang.')
    args = parser.parse_args(argv)
    if (args.limit is not None and args.limit < 1 or args.workers < 1 or
            not np.isfinite(args.crop_scale) or args.crop_scale <= 0):
        parser.error('--limit, --workers và --crop-scale phải hợp lệ, lớn hơn 0.')
    images = sorted(args.data_dir.glob('*.jpg'))
    if not images:
        parser.error(f'Không tìm thấy ảnh JPG trong {args.data_dir}')
    malformed = [path.name for path in images if parse_label(path) is None]
    valid = [(path, parse_label(path)) for path in images if parse_label(path) is not None]
    if args.limit:
        valid = valid[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'predictions.csv'
    if csv_path.exists() and not args.resume:
        parser.error(f'{csv_path} đã tồn tại; dùng --resume hoặc chọn --output-dir khác.')
    old_rows = read_predictions(csv_path) if args.resume else []
    done = {row['file'] for row in old_rows}
    pending = [(path, label, args.crop_scale, args.tta) for path, label in valid if path.name not in done]
    failures, reason_counts = [], Counter()
    started = time.perf_counter()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open('a', newline='', encoding='utf-8-sig' if write_header else 'utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        iterator = (executor.map(evaluate_item, pending, chunksize=16) if executor
                    else map(evaluate_item, pending))
        try:
            for index, (row, reason, failure) in enumerate(iterator, 1):
                reason_counts[reason] += 1
                if failure:
                    failures.append(failure)
                if row:
                    writer.writerow(row)
                if index % 250 == 0 or index == len(pending):
                    file.flush()
                    elapsed = time.perf_counter() - started
                    print(f'{index}/{len(pending)} ảnh mới · {index / max(elapsed, 1e-9):.1f} ảnh/giây · '
                          f'không phát hiện {reason_counts["no_face"]}', flush=True)
        finally:
            if executor:
                executor.shutdown()
    rows = read_predictions(csv_path)
    requested = len(valid)
    report = {
        'dataset': 'UTKFace Aligned & Cropped Faces',
        'dataset_dir': str(args.data_dir.resolve()), 'images_found': len(images),
        'labels_valid': len(valid), 'malformed_count': len(malformed),
        'model_path': str(MODEL_PATH.resolve()), 'model_sha256': file_hash(MODEL_PATH),
        'crop_scale': args.crop_scale, 'detector': 'MediaPipe Face Landmarker (IMAGE)',
        'largest_face_selected_when_multiple': True,
        'coverage': {'requested': requested, 'predicted': len(rows),
                     'rate': len(rows) / requested if requested else None,
                     'new_result_reasons': dict(reason_counts)},
        'metrics': metrics(rows),
        'limitations': [
            'Bản aligned/cropped khác phân phối ảnh webcam tự nhiên.',
            'UTKFace nói nhãn tuổi/giới tính được DEX ước lượng rồi kiểm tra thủ công; không phải hồ sơ hộ tịch.',
            'Không có ID người ổn định để tính khoảng tin cậy theo người.',
            'Gender là hai lớp male/female theo nhãn UTKFace và model, không phải bản dạng giới.'],
        'malformed_examples': malformed[:20], 'failure_examples': failures[:50],
        'elapsed_seconds_this_run': time.perf_counter() - started}
    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    values = report['metrics']
    print(f"Phủ: {len(rows)}/{requested} ({report['coverage']['rate']:.1%})")
    print(f"Tuổi MAE: {values['age']['mae']:.2f} năm · trong ±5: {values['age']['within_5_years']:.1%} · "
          f"trong ±10: {values['age']['within_10_years']:.1%}")
    print(f"Đúng nhóm tuổi: {values['age']['group_accuracy']:.1%} · "
          f"đúng giới tính: {values['gender']['accuracy']:.1%} · "
          f"balanced accuracy: {values['gender']['balanced_accuracy']:.1%}")
    print(f'Báo cáo: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
