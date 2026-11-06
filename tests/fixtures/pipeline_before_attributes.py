# B?n Pipeline tr??c t?ch h?p thu?c t?nh, c? ??nh ?? ki?m tra h?i quy c?m x?c.
"""MediaPipe + EfficientNet webcam pipeline. Square aligned crops, per-face probability smoothing and optional learned blendshape correction. Run: python core/pipeline.py (a: align, 1/2: checkpoint, q: quit)."""
import collections
import sys
import threading
import time

import cv2
import numpy as np

from detectors import detect_faces, crop_face, align_crop
from predict import predict, LABELS, MODEL_NAMES, DEFAULT_MODEL
from predict import MODEL_DIR
from fusion import Fusion
from webcam_data import configure_console
import os

DETECT_EVERY = 1
DETECT_WIDTH = 480
SMOOTH_N = 3
MAX_FACES = 5
MATCH_DIST = 80


class Pipeline:
    def __init__(self, smooth_n=SMOOTH_N, detect_every=DETECT_EVERY, align=True, crop_margin=0.15,
                 fusion_path=None):
        if smooth_n < 1 or detect_every < 1 or not 0 <= crop_margin <= 0.5:
            raise ValueError("smooth_n/detect_every phải >= 1; crop_margin trong [0, 0.5].")
        self.smooth_n, self.detect_every, self.align = smooth_n, detect_every, align
        self.crop_margin = crop_margin
        self.fusion_path = fusion_path
        self._fusion = None
        self.reset()
        self._ctx = None

    def reset(self):
        self._n = 0
        self._faces = []
        self._hist = {}          # key = tâm mặt (cx, cy) -> deque xác suất
        self._hist_modes = {}

    def _match(self, box, taken):
        x, y, w, h = box
        c = (x + w // 2, y + h // 2)
        best, bd = None, MATCH_DIST
        for k in self._hist:
            d = ((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2) ** 0.5
            if k not in taken and d < bd:
                best, bd = k, d
        dq = self._hist.pop(best) if best is not None else collections.deque(maxlen=self.smooth_n)
        mode = self._hist_modes.pop(best, None) if best is not None else None
        self._hist[c] = dq
        self._hist_modes[c] = mode
        return c

    def process_frame(self, frame_bgr, detector="mediapipe", model=None):
        model = model or DEFAULT_MODEL
        if detector != 'mediapipe':
            raise ValueError('Dự án chỉ sử dụng MediaPipe.')
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        context = (detector, model, self.align, self.crop_margin, self.smooth_n,
                   self.detect_every, frame_bgr.shape[:2], str(self.fusion_path))
        if self._ctx != context:                # đổi cách crop/model -> bỏ lịch sử cũ
            correction = (Fusion(self.fusion_path, os.path.join(MODEL_DIR, model + '.keras'),
                                 self.crop_margin, self.align) if self.fusion_path else None)
            self.reset()
            self._ctx = context
            self._fusion = correction
        self._n += 1
        H, W = frame_bgr.shape[:2]

        if self._fusion or (self._n - 1) % self.detect_every == 0 or not self._faces:
            s = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
            small = cv2.resize(frame_bgr, (int(W * s), int(H * s))) if s < 1 else frame_bgr
            self._faces = [dict(f, box=tuple(int(v / s) for v in f["box"]))
                           for f in detect_faces(small, detector)]

        faces = self._faces[:MAX_FACES]
        if not faces:
            self._hist.clear()
            self._hist_modes.clear()
            return []

        out, taken = [], set()
        for f in faces:
            box = f["box"]
            face = self.crop_for_face(frame_bgr, f)
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            _, _, probs = predict(face, model)
            raw_probs = probs.copy()
            fusion_used = False
            if (self._fusion and f['blendshapes'] and
                    all(name in f['blendshapes'] for name in self._fusion.names)):
                probs = self._fusion.predict(probs, f['blendshapes'])
                fusion_used = True
            key = self._match(box, taken)
            taken.add(key)
            if self._hist_modes[key] != fusion_used:
                self._hist[key].clear()
            self._hist_modes[key] = fusion_used
            self._hist[key].append(probs)
            avg = {l: float(np.mean([d[l] for d in self._hist[key]])) for l in LABELS}
            lab = max(avg, key=avg.get)
            top3 = sorted(avg, key=avg.get, reverse=True)[:3]
            r = {"box": box, "label": lab, "prob": avg[lab], "probs": avg,
                 'raw_probs': raw_probs, 'top3': {key: avg[key] for key in top3}, 'fusion_used': fusion_used}
            if f["blendshapes"] is not None:
                r["blendshapes"] = f["blendshapes"]
            out.append(r)
        for k in [k for k in self._hist if k not in taken]:
            del self._hist[k]
            self._hist_modes.pop(k, None)
        return out

    def crop_for_face(self, frame_bgr, face):
        """Dùng đúng cùng một crop cho dự đoán và ảnh xem trước."""
        if self.align:
            return align_crop(frame_bgr, face["box"], face["angle"], self.crop_margin)
        return crop_face(frame_bgr, face["box"], self.crop_margin)

    def get_face_crop(self, frame_bgr, box):
        """Ảnh crop của một kết quả trong khung vừa xử lý, chưa chuẩn hóa màu."""
        face = next((face for face in self._faces if face["box"] == box), None)
        return None if face is None else self.crop_for_face(frame_bgr, face)

