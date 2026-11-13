"""E8: so efficientnet.onnx với Keras trên ảnh RAF-DB và ghi file khớp; predict() chỉ dùng ONNX khi file này đạt.

python scripts/check_onnx.py --images 10000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from predict import EFFICIENTNET_ONNX, MODEL_DIR, _get_inference, _onnx_session
from preprocessing import prepare_face
from webcam_data import configure_console, file_hash, read_image
from scripts.train_rafdb import load_samples

MAX_DIFFERENCE = 1e-3


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'rafdb' / 'DATASET')
    parser.add_argument('--images', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    paths = [path for split in load_samples(args.data_dir) for path, _ in split]
    paths = [paths[i] for i in np.random.default_rng(42).permutation(len(paths))[:args.images]]
    inference, session = _get_inference('efficientnet'), _onnx_session(EFFICIENTNET_ONNX)
    name = session.get_inputs()[0].name
    largest, total, changed = 0.0, 0.0, 0
    for offset in range(0, len(paths), args.batch_size):
        batch = np.stack([prepare_face(read_image(path)) for path in paths[offset:offset + args.batch_size]])
        first, second = inference(batch).numpy(), session.run(None, {name: batch})[0]
        difference = np.abs(first - second)
        largest, total = max(largest, float(difference.max())), total + float(difference.sum())
        changed += int((first.argmax(axis=1) != second.argmax(axis=1)).sum())
    report = {'keras_sha256': file_hash(Path(MODEL_DIR) / 'efficientnet.keras'), 'onnx_sha256': file_hash(EFFICIENTNET_ONNX),
              'dataset': 'RAF-DB', 'images': len(paths), 'seed': 42, 'max_abs_difference': largest, 'mean_abs_difference': total / (len(paths) * 7),
              'labels_changed': changed, 'threshold': MAX_DIFFERENCE, 'passed': bool(largest < MAX_DIFFERENCE and changed == 0)}
    EFFICIENTNET_ONNX.with_suffix('.check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
