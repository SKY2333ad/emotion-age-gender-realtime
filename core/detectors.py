"""MediaPipe Face Landmarker: detection, alignment and expression features."""
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LM_MODEL = os.path.join(_HERE, "face_landmarker.task")
LM_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
          "face_landmarker/float16/1/face_landmarker.task")
EYE_R, EYE_L = 33, 263
NOSE_TIP = 1
_lm = None
_image_lm = None
_lm_failed = False
_lm_ts = 0
_lock = threading.Lock()


def _get_lm():
    global _lm, _lm_failed
    if _lm is None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not os.path.exists(LM_MODEL):
                temporary = LM_MODEL + ".download"
                urllib.request.urlretrieve(LM_URL, temporary)
                os.replace(temporary, LM_MODEL)
            _lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
                running_mode=vision.RunningMode.VIDEO, num_faces=5,
                output_facial_transformation_matrixes=True))
            _lm_failed = False
        except Exception as exc:
            _lm_failed = True
            raise RuntimeError(f"Không khởi tạo được MediaPipe: {exc}") from exc
    return _lm


def _faces_from_result(res, width, height):
    out = []
    for i, landmarks in enumerate(res.face_landmarks):
        xs = np.array([p.x for p in landmarks]) * width
        ys = np.array([p.y for p in landmarks]) * height
        x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
        x1, y1 = min(width, int(xs.max())), min(height, int(ys.max()))
        if x1 <= x0 or y1 <= y0:
            continue
        eye_indices = sorted((EYE_R, EYE_L), key=lambda index: xs[index])
        left, right = eye_indices
        angle = float(np.degrees(np.arctan2(ys[right] - ys[left], xs[right] - xs[left])))
        # 0 = nhìn thẳng, 1 = mũi lệch ra tới đuôi mắt; đủ dùng để hạ trọng số mặt quay ngang.
        eye_width = xs[right] - xs[left]
        yaw = float(min(abs((xs[NOSE_TIP] - xs[left]) / max(eye_width, 1.0) - 0.5) * 2, 1.0))
        # Góc đầu (độ) từ ma trận xoay 3D của MediaPipe; roll vẫn lấy từ đường nối hai mắt ở trên.
        pose = None
        matrices = getattr(res, "facial_transformation_matrixes", None)
        if matrices and i < len(matrices):
            rotation = np.asarray(matrices[i])[:3, :3]
            pose = {"yaw": float(np.degrees(np.arcsin(np.clip(-rotation[2, 0], -1, 1)))),
                    "pitch": float(np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2])))}
        out.append({"box": (x0, y0, x1 - x0, y1 - y0), "angle": angle, "yaw": yaw,
                    "pose": pose})
    return out


def detect_faces(frame_bgr, detector="mediapipe"):
    if detector != "mediapipe":
        raise ValueError("Dự án chỉ sử dụng detector mediapipe.")
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    import mediapipe as mp
    global _lm_ts
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _lock:
        landmarker = _get_lm()
        _lm_ts = max(_lm_ts + 1, time.monotonic_ns() // 1_000_000)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), _lm_ts)
    return _faces_from_result(result, frame_bgr.shape[1], frame_bgr.shape[0])


def detect_image_faces(frame_bgr):
    """Independent IMAGE graph for dataset images, without tracking across unrelated people."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    global _image_lm
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with _lock:
        if _image_lm is None:
            if not os.path.exists(LM_MODEL):
                temporary = LM_MODEL + '.download'
                urllib.request.urlretrieve(LM_URL, temporary)
                os.replace(temporary, LM_MODEL)
            _image_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
                running_mode=vision.RunningMode.IMAGE, num_faces=5, output_facial_transformation_matrixes=True))
        result = _image_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return _faces_from_result(result, frame_bgr.shape[1], frame_bgr.shape[0])


def align_crop(frame, box, angle, margin=0.15):
    """Square crop without stretching; rotate source before cropping to retain edges."""
    if not 0 <= margin <= 0.5:
        raise ValueError("Lề crop phải trong [0, 0.5].")
    if frame is None or frame.size == 0 or not np.isfinite(angle):
        raise ValueError('Ảnh/góc căn chỉnh không hợp lệ.')
    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError("Khung mặt không hợp lệ.")
    side = max(1, int(round(max(width, height) * (1 + 2 * margin))))
    center = (x + width / 2, y + height / 2)
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    transform[:, 2] += (side / 2 - center[0], side / 2 - center[1])
    return cv2.warpAffine(frame, transform, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_face(frame, box, margin=0.15):
    return align_crop(frame, box, 0.0, margin)
