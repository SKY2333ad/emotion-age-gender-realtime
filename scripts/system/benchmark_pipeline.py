"""Đo tốc độ, 3 chế độ:

1. Mặc định: pipeline đầy đủ (tracking, chất lượng, tuổi/giới tính, vẽ) trên ảnh mẫu Grace Hopper ghép vào khung
   640×480, 1 và 3 mặt, kèm thời gian từng bước và từng model cảm xúc. Không cần camera, không gồm Streamlit.
       python scripts/system/benchmark_pipeline.py
2. --video: mục 3 của báo cáo, 9 tổ hợp (3 detector × 3 model) trên CÙNG một video quay sẵn; 30 khung warm-up,
   300 khung đo, lặp 3 lần, thứ tự xáo trộn (seed 42). Quay video bằng scripts/webcam/record_video.py.
       python scripts/system/benchmark_pipeline.py --video data/webcam/bench_input.mp4
3. --camera: đọc webcam thật N giây, đo FPS camera và FPS xử lý.
       python scripts/system/benchmark_pipeline.py --camera 12 --output results/system/webcam_smoke.json
--root đo một bản code khác (ví dụ bản sao lưu) để so trước/sau.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--root', type=Path, default=ROOT)
parser.add_argument('--frames', type=int, default=150)
parser.add_argument('--warmup', type=int, default=15)
parser.add_argument('--model', default='efficientnet')
parser.add_argument('--detector', default='mediapipe', choices=['haar', 'mediapipe', 'yolo'])
parser.add_argument('--camera', type=float, metavar='GIÂY', help='Chỉ đo camera thật trong N giây rồi thoát.')
parser.add_argument('--video', type=Path, help='Đo 9 tổ hợp detector × model trên video này (mục 3 báo cáo).')
parser.add_argument('--repeats', type=int, default=3, help='Số lần lặp khi đo --video.')
parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'system' / 'benchmark_pipeline.json')
args = parser.parse_args()
sys.path.insert(0, str(args.root / 'core'))

import cv2
import matplotlib
import numpy as np
import psutil

from detectors import detect_image_faces
from pipeline import Pipeline
from ui_helpers import draw_results
from webcam_data import read_image

FUSION = next((p for p in (args.root / 'models' / 'emotion' / 'efficientnet_fusion.json',
                           args.root / 'models' / 'efficientnet_fusion.json') if p.exists()), None)   # --root bản cũ: models/ phẳng
image = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')
x, y, width, height = detect_image_faces(image)[0]['box']
pad = height // 2
face = image[max(y - pad, 0):y + height + pad // 2, max(x - pad // 2, 0):x + width + pad // 2]


def scene(count):
    frame = np.full((480, 640, 3), 90, np.uint8)
    cell = 640 // count
    side = min(cell - 10, 440)
    tile = cv2.resize(face, (side, int(side * face.shape[0] / face.shape[1])), interpolation=cv2.INTER_AREA)[:470]
    for index in range(count):
        frame[5:5 + tile.shape[0], 5 + cell * index:5 + cell * index + tile.shape[1]] = tile
    return frame


def run(name, frame, **kwargs):
    pipe = Pipeline(**kwargs)
    process = psutil.Process()
    results = []
    for _ in range(args.warmup):
        results = pipe.process_frame(frame, args.detector, args.model)
    process.cpu_percent()
    values = []
    for _ in range(args.frames):
        start = time.perf_counter()
        results = pipe.process_frame(frame, args.detector, args.model)
        draw_results(frame, results)
        values.append(time.perf_counter() - start)
    values = np.array(values) * 1000
    row = {'config': name, 'faces_found': len(results), 'mean_ms': float(values.mean()),
           'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(values.max()),
           'fps_mean': float(1000 / values.mean()), 'fps_min': float(1000 / values.max()),
           'cpu_percent': process.cpu_percent(), 'rss_mb': process.memory_info().rss / 2 ** 20}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def stage_times(frame):
    """Thời gian từng bước trên một mặt (bản hiện tại), cùng khung với phép đo toàn pipeline."""
    from attributes import predict_attributes
    from detectors import align_crop, detect_faces
    from fusion import Fusion
    from predict import MODEL_DIR, predict
    from preprocessing import prepare_face
    from quality import face_quality
    from tracker import FaceTracker
    found = detect_faces(frame)[0]
    crop = align_crop(frame, found['box'], found['angle'], 0.15)
    correction = Fusion(FUSION, Path(MODEL_DIR) / 'efficientnet.keras', 0.15, True)
    probs = predict(crop)[2]
    tracker = FaceTracker()
    fake = Pipeline(attributes=False)
    fake.process_frame(frame)
    results = fake.process_frame(frame)
    stages = {'phát hiện (MediaPipe)': lambda: detect_faces(frame),
              'crop căn chỉnh': lambda: align_crop(frame, found['box'], found['angle'], 0.15),
              'tiền xử lý': lambda: prepare_face(crop),
              'tracking': lambda: tracker.update([found['box']]),
              'chất lượng': lambda: face_quality(frame, found['box'], found['angle'], found['yaw']),
              'cảm xúc': lambda: predict(crop),
              'fusion': lambda: correction.predict(probs, found['blendshapes']),
              'tuổi và giới tính (1 lần)': lambda: predict_attributes(frame, found['box']),
              'vẽ khung': lambda: draw_results(frame, results)}
    output = {}
    for name, call in stages.items():
        call()
        values = []
        for _ in range(100):
            start = time.perf_counter()
            call()
            values.append(time.perf_counter() - start)
        output[name] = {'mean_ms': float(np.mean(values) * 1000), 'p95_ms': float(np.percentile(values, 95) * 1000)}
    return output


def camera_smoke(seconds):
    """Camera thật: FPS đọc được, FPS xử lý kịp, số khung có mặt; không lưu ảnh."""
    from pipeline import CameraThread
    camera = CameraThread(0)
    if not camera.ok:
        sys.exit('Không mở được webcam.')
    pipe, seen, times, faces, ids, frame = Pipeline(), 0, [], 0, set(), None
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            if camera.count == seen:
                time.sleep(0.002)
                continue
            seen, frame = camera.count, camera.read()
            began = time.perf_counter()
            results = pipe.process_frame(frame, args.detector, args.model)
            times.append(time.perf_counter() - began)
            faces += bool(results)
            ids.update(r['track_id'] for r in results)
        captured = camera.count
    finally:
        camera.stop()
    elapsed = time.perf_counter() - start
    values = np.array(times) * 1000
    return {'seconds': seconds, 'frame': list(frame.shape) if frame is not None else None,
            'camera_frames': captured, 'processed': len(times), 'camera_fps': round(captured / elapsed, 1),
            'processed_fps': round(len(times) / elapsed, 1), 'mean_ms': round(float(values.mean()), 1),
            'p95_ms': round(float(np.percentile(values, 95)), 1), 'frames_with_face': faces, 'track_ids': len(ids)}


def model_speed(frame):
    """ms cho 1 mặt và cho 3 mặt gộp một lần gọi, theo từng cách chạy model cảm xúc."""
    from detectors import align_crop, detect_faces
    from predict import MODEL_NAMES, _get_inference, predict_batch
    from preprocessing import prepare_face
    found = detect_faces(frame)[0]
    crop = align_crop(frame, found['box'], found['angle'], 0.15)

    def keras_call(inference):
        return lambda faces: inference(np.stack([prepare_face(f) for f in faces])).numpy()

    calls = {f'{name} (keras)': keras_call(_get_inference(name)) for name in MODEL_NAMES}
    calls['efficientnet (onnx, app dùng)'] = lambda faces: predict_batch(faces, 'efficientnet')
    output = {}
    for name, call in calls.items():
        output[name] = {}
        for count in (1, 3):
            faces = [crop] * count
            call(faces)
            values = []
            for _ in range(50):
                began = time.perf_counter()
                call(faces)
                values.append(time.perf_counter() - began)
            output[name][f'{count} mặt (ms)'] = float(np.median(values) * 1000)
    return output


def machine_info():
    import os
    import platform
    return (f"{platform.processor() or platform.machine()} | {os.cpu_count()} luồng | "
            f"RAM {psutil.virtual_memory().total / 2 ** 30:.1f} GB | Python {platform.python_version()} | "
            f"cắm sạc: {getattr(psutil.sensors_battery(), 'power_plugged', 'không rõ')} | "
            f"tải CPU trước khi đo: {psutil.cpu_percent(interval=1):.0f}%")


def video_benchmark(path, repeats, warmup=30, measure=300, detect_width=480):
    """Giống benchmark của báo cáo: chỉ phân loại mặt đầu tiên, tìm mặt ở mọi khung; mỗi lần lặp một file CSV."""
    import csv
    from datetime import date
    from detectors import DETECTORS, YOLO_MODEL, align_crop, detect_faces
    from predict import MODEL_NAMES, REPORT_MODELS, predict
    capture, frames = cv2.VideoCapture(str(path)), []
    while len(frames) < warmup + measure:
        ok, frame = capture.read()
        if not ok:
            if not frames:
                sys.exit(f'Không đọc được video: {path}')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)          # video ngắn thì lặp lại
            continue
        frames.append(frame)
    capture.release()
    detectors = [d for d in DETECTORS if d != 'yolo' or Path(YOLO_MODEL).exists()]
    combos = [(d, m) for d in detectors for m in REPORT_MODELS if m in MODEL_NAMES]

    def run(batch, detector, model):
        detect, classify, total, found = [], [], [], 0
        started = time.perf_counter()
        for frame in batch:
            height, width = frame.shape[:2]
            scale = min(1.0, detect_width / width)
            small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
            t0 = time.perf_counter()
            faces = detect_faces(small, detector)
            t1 = time.perf_counter()
            if faces:
                box = tuple(int(v / scale) for v in faces[0]['box'])
                predict(align_crop(frame, box, faces[0]['angle'], 0.15), model)
                found += 1
            t2 = time.perf_counter()
            detect.append(t1 - t0), classify.append(t2 - t1), total.append(t2 - t0)
        return {'detector': detector, 'model': model, 'ms_detect': round(np.mean(detect) * 1000, 2),
                'ms_classify': round(np.mean(classify) * 1000, 2),
                'fps': round(len(batch) / (time.perf_counter() - started), 2),
                'ms_p95': round(float(np.percentile(total, 95)) * 1000, 2),
                'face_pct': round(100 * found / len(batch), 1)}

    folder = args.output if args.output.suffix == '' else ROOT / 'results' / 'system' / 'benchmark_9_to_hop' / str(date.today())
    folder.mkdir(parents=True, exist_ok=True)
    machine, rng, runs = machine_info(), np.random.default_rng(42), []
    print('Máy:', machine)
    for repeat in range(1, repeats + 1):
        rows = []
        for index in rng.permutation(len(combos)):
            detector, model = combos[index]
            run(frames[:warmup], detector, model)
            row = dict(run(frames[warmup:], detector, model), machine=machine)
            rows.append(row)
            print(f"lần {repeat} | {detector} + {model}: {row['fps']} FPS, p95 {row['ms_p95']} ms", flush=True)
        with (folder / f'benchmark_lan{repeat}.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (DETECTORS.index(r['detector']), REPORT_MODELS.index(r['model']))))
        runs.extend(rows)
    import pandas as pd
    table = pd.DataFrame(runs).groupby(['detector', 'model'], sort=False)[['ms_detect', 'ms_classify', 'ms_p95', 'fps']]
    summary = table.mean().round(2).join(table.std().round(2).add_suffix('_sd')).reset_index()
    summary.to_csv(folder / 'tong_hop_mean_sd.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False), '\nĐã lưu', folder)


if args.video:
    video_benchmark(args.video, args.repeats)
    sys.exit(0)

if args.camera:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = camera_smoke(args.camera)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    sys.exit(0)

rows = []
for count in (1, 3):
    frame = scene(count)
    rows.append(run(f'{count} mặt, chỉ cảm xúc', frame, attributes=False))
    rows.append(run(f'{count} mặt, đầy đủ', frame, attributes=True,
                    fusion_path=str(FUSION) if args.model == 'efficientnet' and args.detector == 'mediapipe' else None))
has_stages = (args.root / 'core' / 'tracker.py').is_file() and args.model == 'efficientnet'
batching = has_stages and hasattr(__import__('predict'), 'predict_batch')     # bản cũ qua --root chưa có
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'root': str(args.root), 'model': args.model, 'detector': args.detector, 'frame': '640x480',
                                   'frames': args.frames, 'rows': rows,
                                   'stages': stage_times(scene(1)) if has_stages else None,
                                   'model_speed': model_speed(scene(1)) if batching else None},
                                  ensure_ascii=False, indent=2), encoding='utf-8')
