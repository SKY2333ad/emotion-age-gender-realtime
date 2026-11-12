"""3 detector tìm mặt: Haar (OpenCV), MediaPipe Face Landmarker (kèm góc nghiêng và blendshape), YOLO26 (ONNX)."""
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LM_MODEL = os.path.join(os.path.dirname(_HERE), "models", "face_detection", "face_landmarker.task")
LM_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
          "face_landmarker/float16/1/face_landmarker.task")
YOLO_MODEL = os.path.join(os.path.dirname(_HERE), "models", "face_detection", "yolo26n-face.onnx")
YOLO_SIZE = 640
YOLO_CONF, YOLO_IOU, YOLO_MAX_FACES = 0.4, 0.7, 5     # như bảng cấu hình khóa của báo cáo (NMS mặc định ultralytics)
DETECTORS = ("haar", "mediapipe", "yolo")
DETECTOR_NAMES = {"haar": "Haar Cascade", "mediapipe": "MediaPipe", "yolo": "YOLO26"}
EYE_R, EYE_L = 33, 263
NOSE_TIP = 1
_lm = None
_image_lm = None
_lm_failed = False
_lm_ts = 0
_lock = threading.Lock()
_haar = None
_yolo = None


def _create_landmarker(video):
    """VIDEO cho webcam (theo dõi giữa các khung), IMAGE cho ảnh dataset độc lập; tải model lần đầu."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(LM_MODEL):
        temporary = LM_MODEL + ".download"
        urllib.request.urlretrieve(LM_URL, temporary)
        os.replace(temporary, LM_MODEL)
    mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
        running_mode=mode, num_faces=5, output_face_blendshapes=True))


def _get_lm():
    global _lm, _lm_failed
    if _lm is None:
        try:
            _lm = _create_landmarker(video=True)
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
        blendshapes = None
        if res.face_blendshapes and i < len(res.face_blendshapes):
            blendshapes = {c.category_name: float(c.score) for c in res.face_blendshapes[i]
                           if c.category_name != "_neutral"}
        out.append({"box": (x0, y0, x1 - x0, y1 - y0), "angle": angle, "yaw": yaw,
                    "blendshapes": blendshapes})
    return out


def _plain_face(x, y, width, height):
    """Haar và YOLO chỉ cho khung: không có landmark nên không xoay, không có blendshape."""
    return {"box": (int(x), int(y), int(width), int(height)), "angle": 0.0, "yaw": 0.0, "blendshapes": None}


def _haar_faces(frame_bgr):
    global _haar
    gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    with _lock:
        if _haar is None:
            _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if _haar.empty():
                _haar = None
                raise RuntimeError("Không nạp được haarcascade_frontalface_default.xml của OpenCV.")
        boxes = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [_plain_face(*box) for box in boxes]


def _yolo_session():
    global _yolo
    if _yolo is None:
        if not os.path.exists(YOLO_MODEL):
            raise RuntimeError(f"Thiếu model YOLO26: {YOLO_MODEL}. Xem mục YOLO26 trong README.md.")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        _yolo = ort.InferenceSession(YOLO_MODEL, options, providers=["CPUExecutionProvider"])
    return _yolo


def yolo_boxes(output, scale, width, height, conf=YOLO_CONF, iou=YOLO_IOU, max_faces=YOLO_MAX_FACES):
    """Hai dạng đầu ra ONNX của YOLO26:
    (1, N, 6) end-to-end = [x1, y1, x2, y2, điểm, lớp], đã lọc trùng;
    (1, 5, N) = [cx, cy, w, h, điểm] của 1 lớp "face", lọc trùng bằng NMS như ultralytics.
    yolo26n-face.onnx của dự án là dạng thứ hai (checkpoint tắt end2end)."""
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim == 3 and rows.shape[0] == 1 and rows.shape[2] == 6:
        boxes, scores = rows[0, :, :4], rows[0, :, 4]
        keep = np.flatnonzero(scores >= conf)
        keep = keep[np.argsort(-scores[keep])]
    elif rows.ndim == 3 and rows.shape[0] == 1 and rows.shape[1] == 5:
        cx, cy, w, h, scores = rows[0]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        candidates = np.flatnonzero(scores >= conf)
        chosen = cv2.dnn.NMSBoxes(np.stack([boxes[candidates, 0], boxes[candidates, 1], w[candidates], h[candidates]],
                                           axis=1).tolist(), scores[candidates].tolist(), conf, iou) if len(candidates) else []
        keep = candidates[np.asarray(chosen, dtype=int).reshape(-1)]
    else:
        raise ValueError(f"Đầu ra YOLO26 phải là (1, N, 6) hoặc (1, 5, N); nhận {rows.shape}.")
    faces = []
    for x0, y0, x1, y1 in boxes[keep[:max_faces]]:
        x0, x1 = np.clip(np.array([x0, x1]) / scale, 0, width)
        y0, y1 = np.clip(np.array([y0, y1]) / scale, 0, height)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            faces.append(_plain_face(x0, y0, x1 - x0, y1 - y0))
    return faces


def _yolo_faces(frame_bgr):
    height, width = frame_bgr.shape[:2]
    scale = YOLO_SIZE / max(height, width)
    resized = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)))
    canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, np.uint8)        # ảnh đặt góc trái trên, phần còn lại tô xám
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=True)          # (1, 3, 640, 640) RGB 0..1
    with _lock:
        session = _yolo_session()
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return yolo_boxes(output, scale, width, height)


def detect_faces(frame_bgr, detector="mediapipe"):
    if detector not in DETECTORS:
        raise ValueError(f"Detector phải là một trong {DETECTORS}.")
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    if detector == "haar":
        return _haar_faces(frame_bgr)
    if detector == "yolo":
        return _yolo_faces(frame_bgr)
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
    global _image_lm
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with _lock:
        if _image_lm is None:
            _image_lm = _create_landmarker(video=False)
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
