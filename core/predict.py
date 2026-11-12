"""
predict.py – bộ phân loại cảm xúc
face_bgr: ảnh mặt đã crop từ OpenCV (BGR, kích thước bất kỳ). Nhãn theo models/emotion/labels.txt:
surprise, fear, disgust, happy, sad, angry, neutral. Đường dẫn models/ tính từ vị trí file này, chạy ở thư mục nào cũng được.
"""
import os
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocessing import prepare_face
from webcam_data import file_hash
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(_HERE), "models", "emotion")

with open(os.path.join(MODEL_DIR, "labels.txt"), encoding="utf-8") as f:
    LABELS = [l.strip() for l in f.read().split("\n") if l.strip()]

REPORT_MODELS = ['cnn', 'mobilenetv3', 'efficientnet']      # 3 model của báo cáo, train trên RAF-DB
MODEL_TITLES = {'cnn': 'CNN', 'mobilenetv3': 'MobileNetV3-Small', 'efficientnet': 'EfficientNet-B0'}


def discover_models(directory=MODEL_DIR):
    """efficientnet.keras (nhóm bàn giao) luôn có; model khác chỉ hiện khi train xong và có metadata hợp lệ."""
    names = ['efficientnet']
    for path in sorted(Path(directory).glob('*.keras')):
        if path.stem == 'efficientnet':
            continue
        metadata_path = path.with_suffix('.metadata.json')
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if (metadata.get('status') == 'complete' and metadata.get('labels') == LABELS
                and metadata.get('selection_passed') is not False):
            names.append(path.stem)
    return sorted(names, key=lambda name: (REPORT_MODELS.index(name) if name in REPORT_MODELS else 3, name))


MODEL_NAMES = discover_models()
DEFAULT_MODEL = 'efficientnet'
EFFICIENTNET_ONNX = Path(MODEL_DIR) / 'efficientnet.onnx'
ONNX_THREADS = 4
_CACHE = {}
_INFERENCE_CACHE = {}
_ONNX_CACHE = {}


def get_crop_config(model_name=DEFAULT_MODEL):
    path = os.path.join(MODEL_DIR, f'{model_name}.metadata.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as file:
            config = json.load(file).get('crop_config')
        if config:
            return config
    return {'margin': 0.15, 'align': True}


def _get(model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model '{model_name}' không tồn tại, chỉ có: {MODEL_NAMES}")
    if model_name not in _CACHE:
        path = os.path.join(MODEL_DIR, f"{model_name}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Thiếu file model: {path}")
        if model_name != 'efficientnet':
            metadata = json.loads(Path(path).with_suffix('.metadata.json').read_text(encoding='utf-8'))
            if metadata.get('sha256') != file_hash(path):
                raise ValueError('Checkpoint không khớp metadata, cần kiểm tra lại file model.')
        _CACHE[model_name] = keras.models.load_model(path, compile=False)
    return _CACHE[model_name]


def _get_inference(model_name):
    """Tạo graph một lần cho mỗi model; không thay đổi trọng số hay preprocessing."""
    model = _get(model_name)
    if model_name not in _INFERENCE_CACHE:
        @tf.function(input_signature=[tf.TensorSpec((None, 100, 100, 3), tf.float32)],
                     autograph=False)
        def inference(images):
            return model(images, training=False)

        _INFERENCE_CACHE[model_name] = inference
    return _INFERENCE_CACHE[model_name]


def _onnx_session(path):
    if path not in _ONNX_CACHE:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = ONNX_THREADS
        _ONNX_CACHE[path] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return _ONNX_CACHE[path]


def _efficientnet_session():
    """Bản ONNX chỉ dùng khi còn khớp đúng file Keras đã so số; nếu không thì quay về Keras."""
    if 'efficientnet_checked' not in _ONNX_CACHE:
        check = EFFICIENTNET_ONNX.with_suffix('.check.json')
        try:
            data = json.loads(check.read_text(encoding='utf-8'))
            valid = (data['keras_sha256'] == file_hash(Path(MODEL_DIR) / 'efficientnet.keras') and
                     data['onnx_sha256'] == file_hash(EFFICIENTNET_ONNX) and data['passed'] is True)
            _ONNX_CACHE['efficientnet_checked'] = _onnx_session(EFFICIENTNET_ONNX) if valid else None
        except (OSError, ValueError, KeyError, ImportError):
            _ONNX_CACHE['efficientnet_checked'] = None
    return _ONNX_CACHE['efficientnet_checked']


def probabilities(faces_bgr, model_name=DEFAULT_MODEL):
    """Mảng (N, 7) theo LABELS; mọi mặt trong một lần gọi model."""
    x = np.stack([prepare_face(face) for face in faces_bgr])   # (N, 100, 100, 3), giá trị 0-255
    session = _efficientnet_session() if model_name == 'efficientnet' else None
    return (session.run(None, {session.get_inputs()[0].name: x})[0] if session
            else _get_inference(model_name)(x).numpy())


def predict_batch(faces_bgr, model_name=DEFAULT_MODEL):
    """[(nhãn, xác suất, dict 7 lớp)] cùng thứ tự với faces_bgr."""
    if not faces_bgr:
        return []
    output = []
    for probs in probabilities(faces_bgr, model_name):
        i = int(probs.argmax())
        output.append((LABELS[i], float(probs[i]), {l: float(p) for l, p in zip(LABELS, probs)}))
    return output


def predict(face_bgr, model_name=DEFAULT_MODEL):
    return predict_batch([face_bgr], model_name)[0]
