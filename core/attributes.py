"""Ước lượng tuổi/giới tính cục bộ bằng genderage.onnx, không phát hiện mặt lại."""
import json
from pathlib import Path
import threading

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'genderage.onnx'
CALIBRATION_PATH = MODEL_PATH.with_name('genderage_calibration.json')
INPUT_SIZE = 96
# Chọn theo macro-F1 nhóm tuổi trên validation UTKFace khi thử 1.2–1.6.
# 1.5 cũng chính là hệ số crop trong mã Attribute gốc của InsightFace.
CROP_SCALE = 1.5
_CACHE = None
_LOCK = threading.Lock()


def load_calibration():
    """Tham số fit trên validation, tách khỏi trọng số; thiếu file thì giữ điểm softmax gốc."""
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'gender_temperature': 1.0, 'age_intervals': {}}


def calibrate_probability(probability, temperature):
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(1 / (1 + np.exp(-np.log(probability / (1 - probability)) / temperature)))


def _get_model():
    """Chỉ nạp một session; lỗi cấu hình được pipeline báo mà không dừng cảm xúc."""
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f'Thiếu mô hình tuổi/giới tính: {MODEL_PATH}. '
                    'Hãy tải genderage.onnx từ gói buffalo_l theo README.md.')
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError('Chưa cài ONNX Runtime. Chạy: pip install onnxruntime.') from exc
            try:
                options = ort.SessionOptions()
                # Tránh chiếm luồng CPU của TensorFlow/MediaPipe cho mạng nhỏ này.
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(MODEL_PATH), options,
                                               providers=['CPUExecutionProvider'])
                inputs, outputs = session.get_inputs(), session.get_outputs()
                if (len(inputs) != 1 or inputs[0].type != 'tensor(float)' or
                        inputs[0].shape[1:] != [3, INPUT_SIZE, INPUT_SIZE] or
                        len(outputs) != 1 or outputs[0].shape[-1:] != [3]):
                    raise ValueError('Cần đầu vào float32 [N, 3, 96, 96], đầu ra [1, 3].')
                _CACHE = (session, inputs[0].name, outputs[0].name)
            except Exception as exc:
                raise RuntimeError(f'Không nạp được mô hình tuổi/giới tính: {exc}') from exc
        return _CACHE


def _make_blob(frame_bgr, box, crop_scale=CROP_SCALE, flip=False):
    """Affine từ ảnh gốc, tâm bbox, không xoay; RGB, mean=0, std=1."""
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0 or
            frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or
            frame_bgr.dtype not in (np.dtype('uint8'), np.dtype('float32'))):
        return None
    try:
        values = np.asarray(box, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            return None
        if not np.isfinite(crop_scale) or crop_scale <= 0:
            return None
        x, y, width, height = values
        image_h, image_w = frame_bgr.shape[:2]
        if (width <= 0 or height <= 0 or x >= image_w or y >= image_h or
                x + width <= 0 or y + height <= 0):
            return None
        scale = INPUT_SIZE / (max(width, height) * crop_scale)
        transform = np.array([[scale, 0, INPUT_SIZE / 2 - scale * (x + width / 2)],
                              [0, scale, INPUT_SIZE / 2 - scale * (y + height / 2)]], np.float64)
        if not np.isfinite(transform).all():
            return None
        crop = cv2.warpAffine(frame_bgr, transform, (INPUT_SIZE, INPUT_SIZE),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if not np.isfinite(crop).all():
            return None
        if flip:
            crop = cv2.flip(crop, 1)
        return cv2.dnn.blobFromImage(crop, 1.0, (INPUT_SIZE, INPUT_SIZE),
                                     (0, 0, 0), swapRB=True)
    except (TypeError, ValueError, OverflowError, cv2.error):
        return None


def predict_attributes(frame_bgr, box, angle=0.0, crop_scale=CROP_SCALE, tta=False):
    """Trả {age, gender, gender_prob}, hoặc None nếu dữ liệu không hợp lệ.

    angle được giữ trong API nhưng cố ý không dùng: model gốc không xoay crop.
    gender_prob là softmax thô của giới tính trả về; Pipeline áp temperature khi tổng hợp.
    Thiếu/sai model gây lỗi rõ ràng; Pipeline bắt lỗi riêng cho nhánh thuộc tính.
    tta=True lấy trung bình đầu ra của ảnh gốc và ảnh lật ngang (gấp đôi chi phí ONNX).
    """
    blobs = [_make_blob(frame_bgr, box, crop_scale, flip) for flip in ((False, True) if tta else (False,))]
    if blobs[0] is None:
        return None
    session, input_name, output_name = _get_model()
    try:
        outputs = [np.asarray(session.run([output_name], {input_name: blob})[0], dtype=np.float64)
                   for blob in blobs]
    except Exception as exc:
        raise RuntimeError(f'Lỗi suy luận tuổi/giới tính: {exc}') from exc
    output = np.mean(outputs, axis=0) if all(item.shape == (1, 3) for item in outputs) else outputs[0]
    if output.shape != (1, 3) or not np.isfinite(output).all():
        return None
    logits = output[0, :2]
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    index = int(np.argmax(probabilities))
    age = float(output[0, 2] * 100)
    if not np.isfinite(age) or age < 0:
        return None
    return {'age': age, 'gender': ('female', 'male')[index],
            'gender_prob': float(probabilities[index])}
