"""Webcam pipeline: 3 detector (Haar, MediaPipe, YOLO26), các model cảm xúc, crop vuông, ID ổn định, làm mượt theo chất lượng, hiệu chỉnh blendshape (chỉ MediaPipe). Chạy: python core/pipeline.py (d: detector, a: căn chỉnh, 1-9: model, q: thoát)."""
import collections
import sys
import threading
import time

import cv2
import numpy as np

from detectors import DETECTORS, detect_faces, crop_face, align_crop
from predict import predict_batch, LABELS, MODEL_NAMES, DEFAULT_MODEL, MODEL_DIR
from fusion import Fusion
from attributes import calibrate_probability, load_calibration, predict_attributes
from quality import EmotionSmoother, face_quality, weighted_mean, weighted_median
from tracker import FaceTracker
from ui_helpers import age_group
from webcam_data import configure_console

DETECT_EVERY = 1
DETECT_WIDTH = 480
SMOOTH_N = 3
MAX_FACES = 5
TRACK_GRACE = 10                 # số khung giữ ID và lịch sử khi mặt tạm mất
ATTR_EVERY = 5
ATTR_HISTORY = 12                # 5 × 12 = 60 khung: khoảng 2 giây ở 30 FPS, thay cho 12 × 15 = 180 khung (6 giây)
GENDER_HYSTERESIS = 0.05


class Pipeline:
    def __init__(self, smooth_n=SMOOTH_N, detect_every=DETECT_EVERY, align=True, crop_margin=0.15,
                 fusion_path=None, attributes=True, attr_every=ATTR_EVERY, attr_history=ATTR_HISTORY,
                 track_grace=TRACK_GRACE, smoothing='adaptive'):
        if smooth_n < 1 or detect_every < 1 or not 0 <= crop_margin <= 0.5:
            raise ValueError("smooth_n/detect_every phải >= 1; crop_margin trong [0, 0.5].")
        self.smooth_n, self.detect_every, self.align = smooth_n, detect_every, align
        self.crop_margin = crop_margin
        self.fusion_path = fusion_path
        if attr_every < 1 or attr_history < 1:
            raise ValueError('Chu kỳ và kích thước bộ đệm thuộc tính phải >= 1.')
        if smoothing not in ('adaptive', 'mean'):
            raise ValueError("smoothing phải là 'adaptive' hoặc 'mean'.")
        self.smoothing = smoothing
        self.tracker = FaceTracker(track_grace)
        calibration = load_calibration()
        self.gender_temperature = calibration['gender_temperature']
        self.age_intervals = calibration['age_intervals']
        self.attributes = attributes
        self.attr_every, self.attr_history = attr_every, attr_history
        self._fusion = None
        self.reset()
        self._ctx = None

    def reset(self):
        self._n = 0
        self._faces = []
        self.tracker.reset()
        self._hist = {}          # key = track ID -> deque xác suất
        self._hist_modes = {}
        self._smoothers = {}
        self._attr_hist = {}
        self._attr_last = {}
        self._attr_results = {}
        self.attribute_error = None

    def _forget_lost_tracks(self):
        for key in [key for key in self._hist if key not in self.tracker.tracks]:
            for state in (self._hist, self._hist_modes, self._smoothers,
                          self._attr_hist, self._attr_last, self._attr_results):
                state.pop(key, None)

    def _ensure_state(self, key):
        self._hist.setdefault(key, collections.deque(maxlen=self.smooth_n))
        self._hist_modes.setdefault(key, None)
        if self.attributes and key not in self._attr_hist:
            self._attr_hist[key] = collections.deque(maxlen=self.attr_history)
            self._attr_last[key] = None
            self._attr_results[key] = None

    def _smooth(self, key, probs, fusion_used, quality):
        """Trả (xác suất đã làm mượt, nhãn); đổi nguồn fusion/EfficientNet thì bỏ lịch sử cũ."""
        if self._hist_modes[key] != fusion_used:
            self._hist[key].clear()
            self._smoothers.pop(key, None)
        self._hist_modes[key] = fusion_used
        if self.smoothing == 'adaptive':
            smoother = self._smoothers.setdefault(key, EmotionSmoother(LABELS))
            return smoother.update(probs, quality)
        self._hist[key].append(probs)
        avg = {l: float(np.mean([d[l] for d in self._hist[key]])) for l in LABELS}
        return avg, max(avg, key=avg.get)

    def _attributes_for_face(self, frame_bgr, face, key, quality):
        last = self._attr_last[key]
        if not self.attribute_error and (last is None or self._n - last >= self.attr_every):
            self._attr_last[key] = self._n
            try:
                result = predict_attributes(frame_bgr, face['box'], face.get('angle', 0.0))
            except (OSError, RuntimeError, ValueError) as exc:
                self.attribute_error = f'Không nhận diện được tuổi/giới tính: {exc} Cảm xúc vẫn tiếp tục chạy.'
                result = None
            if result is not None:
                p_male = calibrate_probability(result['gender_prob'] if result['gender'] == 'male'
                                               else 1 - result['gender_prob'], self.gender_temperature)
                history = self._attr_hist[key]
                history.append((result['age'], p_male, quality))
                weights = [entry[2] for entry in history]
                mean = weighted_mean([entry[1] for entry in history], weights)
                previous = self._attr_results[key]
                gender = previous['gender'] if previous else ('male' if mean >= 0.5 else 'female')
                # Giữ nhãn cũ khi trung bình dao động sát 0.5, vẫn trả xác suất trung bình thật.
                if mean > 0.5 + GENDER_HYSTERESIS:
                    gender = 'male'
                elif mean < 0.5 - GENDER_HYSTERESIS:
                    gender = 'female'
                age = weighted_median([entry[0] for entry in history], weights)
                offsets = self.age_intervals.get(age_group(age))
                self._attr_results[key] = {
                    'age': age, 'age_range': (max(age + offsets[0], 0.0), age + offsets[1]) if offsets else None,
                    'gender': gender, 'gender_prob': mean if gender == 'male' else 1 - mean}
        return self._attr_results[key]

    def process_frame(self, frame_bgr, detector="mediapipe", model=None):
        model = model or DEFAULT_MODEL
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        context = (detector, model, self.align, self.crop_margin, self.smooth_n,
                   self.detect_every, frame_bgr.shape[:2], str(self.fusion_path),
                   self.attributes, self.attr_every, self.attr_history, self.smoothing)
        if self._ctx != context:                # đổi cách crop/model -> bỏ lịch sử cũ
            correction = (Fusion(self.fusion_path, f'{MODEL_DIR}/{model}.keras',
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

        crops = [(f, self.crop_for_face(frame_bgr, f)) for f in self._faces[:MAX_FACES]]
        crops = [(f, crop) for f, crop in crops if crop.shape[0] >= 20 and crop.shape[1] >= 20]
        keys = self.tracker.update([f['box'] for f, _ in crops])
        self._forget_lost_tracks()

        predictions = predict_batch([face for _, face in crops], model)   # một lần gọi model cho mọi mặt
        out = []
        for (f, _), key, (_, _, probs) in zip(crops, keys, predictions):
            box = f["box"]
            quality = face_quality(frame_bgr, box, f.get('angle', 0.0), f.get('yaw', 0.0))
            raw_probs = probs.copy()
            fusion_used = False
            if (self._fusion and f['blendshapes'] and
                    all(name in f['blendshapes'] for name in self._fusion.names)):
                probs = self._fusion.predict(probs, f['blendshapes'])
                fusion_used = True
            self._ensure_state(key)
            avg, lab = self._smooth(key, probs, fusion_used, quality['score'])
            top3 = sorted(avg, key=avg.get, reverse=True)[:3]
            r = {"box": box, 'track_id': key, 'quality': quality['score'], "label": lab, "prob": avg[lab],
                 "probs": avg, 'raw_probs': raw_probs, 'top3': {label: avg[label] for label in top3},
                 'fusion_used': fusion_used}
            if f["blendshapes"] is not None:
                r["blendshapes"] = f["blendshapes"]
            attrs = self._attributes_for_face(frame_bgr, f, key, quality['score']) if self.attributes else None
            r.update(attrs or {'age': None, 'age_range': None, 'gender': None, 'gender_prob': None})
            out.append(r)
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


_default = Pipeline()


def process_frame(frame_bgr, detector="mediapipe", model=None):
    return _default.process_frame(frame_bgr, detector, model)


# ---------------------------------------------------------------- Webcam
def open_camera(index=0, warmup=40):
    """Mở webcam, thử lần lượt DirectShow, MSMF, mặc định (Windows hay lỗi 'không đọc được khung hình')."""
    for backend in (getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(warmup):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)
        cap.release()
    return None


class CameraThread:
    """Đọc camera ở luồng riêng, luôn giữ khung mới nhất để không xử lý khung cũ trong bộ đệm."""
    def __init__(self, index=0):
        self.cap = open_camera(index)             # thử DirectShow/MSMF/mặc định, chỉ nhận camera đọc được khung thật
        self.ok = self.cap is not None
        self.frame, self.running = None, self.ok
        self.count = 0            # tăng mỗi khung đọc được; người dùng bỏ qua khung đã xử lý
        self._thread = threading.Thread(target=self._loop, daemon=True)
        if self.ok:
            self._thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
                self.count += 1
            else:
                time.sleep(0.01)      # camera bị rút/lỗi: không quay vòng 100% CPU

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1)    # chờ lần read() đang dở xong rồi mới giải phóng camera
        if self.cap is not None:
            self.cap.release()


def main():
    configure_console()
    cam = CameraThread(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    if not cam.ok:
        sys.exit("Không mở được camera (kiểm tra webcam / quyền truy cập camera).")
    detector, model = "mediapipe", DEFAULT_MODEL
    model_keys = {ord(str(i + 1)): m for i, m in enumerate(MODEL_NAMES)}
    print("d = đổi detector | a = căn chỉnh bật/tắt | " + " | ".join(f"{i + 1} = {m}" for i, m in enumerate(MODEL_NAMES))
          + " | q = thoát")
    pipe, fps, prev = Pipeline(), 0.0, time.time()
    mirror = True     # lật gương như soi gương (đặt False nếu không cần)
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        for r in pipe.process_frame(frame, detector, model):
            x, y, w, h = r["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f'{r["label"]} {r["prob"]:.0%}', (x, max(y - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f} | {detector} | {model} | align {'on' if pipe.align else 'off'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, "d: detector | 1-9: model | a: align | q: quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Emotion realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("d"):
            detector = DETECTORS[(DETECTORS.index(detector) + 1) % len(DETECTORS)]
        elif k == ord("a"):
            pipe.align = not pipe.align
        elif k in model_keys:
            model = model_keys[k]
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
