"""E10: đo pipeline trên khung thật có mặt người, gồm cả vẽ; so được với bản khác qua --root.

Dùng ảnh mẫu Grace Hopper của matplotlib ghép vào khung 640×480, không cần camera và không gồm Streamlit.
python scripts/benchmark_pipeline.py --output results/benchmark_pipeline.json
python scripts/benchmark_pipeline.py --root D:/ban_goc/emotion_realtime --output results/benchmark_baseline.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--root', type=Path, default=ROOT)
parser.add_argument('--frames', type=int, default=150)
parser.add_argument('--warmup', type=int, default=15)
parser.add_argument('--model', default='efficientnet')
parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'benchmark_pipeline.json')
args = parser.parse_args()
sys.path.insert(0, str(args.root / 'core'))

import cv2
import matplotlib
import numpy as np
import psutil

from detectors import detect_image_faces
from pipeline import Pipeline
from ui_helpers import draw_results
from webcam_data import configure_console, read_image

configure_console()

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
        results = pipe.process_frame(frame, model=args.model)
    process.cpu_percent()
    values = []
    for _ in range(args.frames):
        start = time.perf_counter()
        results = pipe.process_frame(frame, model=args.model)
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
    from predict import predict
    from preprocessing import prepare_face
    from quality import face_quality
    from tracker import FaceTracker
    found = detect_faces(frame)[0]
    crop = align_crop(frame, found['box'], found['angle'], 0.0)
    probs = predict(crop)[2]
    tracker = FaceTracker()
    fake = Pipeline(attributes=False)
    fake.process_frame(frame)
    results = fake.process_frame(frame)
    stages = {'phát hiện (MediaPipe)': lambda: detect_faces(frame),
              'crop căn chỉnh': lambda: align_crop(frame, found['box'], found['angle'], 0.0),
              'tiền xử lý': lambda: prepare_face(crop),
              'tracking': lambda: tracker.update([found['box']]),
              'chất lượng': lambda: face_quality(frame, found['box'], found['angle'], found['yaw']),
              'cảm xúc': lambda: predict(crop),
              'giới tính (1 lần)': lambda: predict_attributes(frame, found['box']),
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


rows = []
for count in (1, 3):
    frame = scene(count)
    rows.append(run(f'{count} mặt, chỉ cảm xúc', frame, attributes=False))
    rows.append(run(f'{count} mặt, đầy đủ', frame, attributes=True))
has_stages = (args.root / 'core' / 'tracker.py').is_file() and args.model == 'efficientnet'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'root': str(args.root), 'model': args.model, 'frame': '640x480',
                                   'frames': args.frames, 'rows': rows,
                                   'stages': stage_times(scene(1)) if has_stages else None},
                                  ensure_ascii=False, indent=2), encoding='utf-8')
