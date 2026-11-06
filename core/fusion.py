"""Optional learned residual correction: EfficientNet probabilities + MediaPipe blendshapes."""
import json
from pathlib import Path

import numpy as np

from webcam_data import LABELS, CROP_VERSION, file_hash


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def features(probs, blendshapes, names):
    if not blendshapes or any(name not in blendshapes for name in names):
        raise ValueError("Thiếu blendshape; không thể dùng fusion.")
    values = np.array([probs[label] for label in LABELS] + [blendshapes[name] for name in names], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Xác suất/blendshape không hợp lệ.")
    return values


class Fusion:
    def __init__(self, path, model_path, margin, align):
        self.path = str(Path(path).resolve())
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = self.data
        if data.get('version') != 1 or data.get('labels') != LABELS:
            raise ValueError("Fusion không đúng định dạng/thứ tự nhãn.")
        if file_hash(model_path) != data['base_sha256']:
            raise ValueError("Fusion được train cho checkpoint EfficientNet khác.")
        if data['crop_config'] != {'margin': margin, 'align': align, 'version': CROP_VERSION}:
            raise ValueError("Lề crop/căn chỉnh khác cấu hình lúc train fusion.")
        self.names = data['blendshape_names']
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Tên blendshape trong fusion không hợp lệ.')
        if data.get('landmarker_sha256'):
            from detectors import LM_MODEL
            if file_hash(LM_MODEL) != data['landmarker_sha256']:
                raise ValueError('Fusion được train bằng phiên bản MediaPipe Face Landmarker khác.')
        dimension = len(LABELS) + len(self.names)
        self.mean, self.scale = np.array(data['mean']), np.array(data['scale'])
        self.weights, self.bias = np.array(data['weights']), np.array(data['bias'])
        if (self.mean.shape != (dimension,) or self.scale.shape != (dimension,) or
                self.weights.shape != (dimension, 7) or self.bias.shape != (7,) or
                not all(np.isfinite(value).all() for value in (self.mean, self.scale, self.weights, self.bias)) or
                np.any(self.scale <= 0)):
            raise ValueError("Trọng số fusion không hợp lệ.")

    def predict(self, probs, blendshapes):
        vector = (features(probs, blendshapes, self.names) - self.mean) / self.scale
        # Start from the original model; learn a small correction instead of replacing it.
        logits = np.log(np.clip([probs[label] for label in LABELS], 1e-7, 1))
        output = softmax(logits + vector @ self.weights + self.bias)
        return dict(zip(LABELS, map(float, output)))
