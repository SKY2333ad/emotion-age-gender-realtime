"""MediaPipe + EfficientNet webcam pipeline. Square aligned crops, stable track IDs and quality-aware smoothing. Run: python core/pipeline.py (a: align, 1/2: checkpoint, q: quit)."""
import collections
import sys
import threading
import time

import cv2
import numpy as np

from detectors import detect_faces, crop_face, align_crop
from predict import predict, LABELS, MODEL_NAMES, DEFAULT_MODEL
from attributes import calibrate_probability, load_calibration, predict_attributes
from quality import EmotionSmoother, face_quality, weighted_mean
from quality import BLOCKING, POSE_ALIGN_DEG, input_status, max_pose_angle, output_status
from tracker import FaceTracker
from webcam_data import DEFAULT_CROP_MARGIN, configure_console

DETECT_EVERY = 1
DETECT_WIDTH = 480
SMOOTH_N = 3
MAX_FACES = 5
TRACK_GRACE = 10                 # số khung giữ ID và lịch sử khi mặt tạm mất
ATTR_EVERY = 5
ATTR_HISTORY = 12                # 5 × 12 = 60 khung: khoảng 2 giây ở 30 FPS, thay cho 12 × 15 = 180 khung (6 giây)
GENDER_HYSTERESIS = 0.05


class Pipeline:
    def __init__(self, smooth_n=SMOOTH_N, detect_every=DETECT_EVERY, align=True,
                 crop_margin=DEFAULT_CROP_MARGIN,
                 attributes=True, attr_every=ATTR_EVERY, attr_history=ATTR_HISTORY,
                 track_grace=TRACK_GRACE, smoothing='adaptive', quality_gate=False):
        if smooth_n < 1 or detect_every < 1 or not 0 <= crop_margin <= 0.5:
            raise ValueError("smooth_n/detect_every phải >= 1; crop_margin trong [0, 0.5].")
        self.smooth_n, self.detect_every, self.align = smooth_n, detect_every, align
        self.crop_margin = crop_margin
        if attr_every < 1 or attr_history < 1:
            raise ValueError('Chu kỳ và kích thước bộ đệm thuộc tính phải >= 1.')
        if smoothing not in ('adaptive', 'mean'):
            raise ValueError("smoothing phải là 'adaptive' hoặc 'mean'.")
        self.smoothing = smoothing
        self.quality_gate = quality_gate      # bật: trả 'status', chặn mặt kém trước khi gọi model
        self.tracker = FaceTracker(track_grace)
        calibration = load_calibration()
        self.gender_temperature = calibration['gender_temperature']
        self.attributes = attributes
        self.attr_every, self.attr_history = attr_every, attr_history
        self.reset()
        self._ctx = None

    def reset(self):
        self._n = 0
        self._faces = []
        self.tracker.reset()
        self._hist = {}          # key = track ID -> deque xác suất
        self._smoothers = {}
        self._attr_hist = {}
        self._attr_last = {}
        self._attr_results = {}
        self.attribute_error = None

    def _forget_lost_tracks(self):
        for key in [key for key in self._hist if key not in self.tracker.tracks]:
            for state in (self._hist, self._smoothers,
                          self._attr_hist, self._attr_last, self._attr_results):
                state.pop(key, None)

    def _ensure_state(self, key):
        self._hist.setdefault(key, collections.deque(maxlen=self.smooth_n))
        if self.attributes and key not in self._attr_hist:
            self._attr_hist[key] = collections.deque(maxlen=self.attr_history)
            self._attr_last[key] = None
            self._attr_results[key] = None

    def _smooth(self, key, probs, quality):
        """Trả (xác suất đã làm mượt, nhãn)."""
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
                self.attribute_error = f'Không nhận diện được giới tính: {exc} Cảm xúc vẫn tiếp tục chạy.'
                result = None
            if result is not None:
                p_male = calibrate_probability(result['gender_prob'] if result['gender'] == 'male'
                                               else 1 - result['gender_prob'], self.gender_temperature)
                history = self._attr_hist[key]
                history.append((p_male, quality))
                mean = weighted_mean([entry[0] for entry in history], [entry[1] for entry in history])
                previous = self._attr_results[key]
                gender = previous['gender'] if previous else ('male' if mean >= 0.5 else 'female')
                # Giữ nhãn cũ khi trung bình dao động sát 0.5, vẫn trả xác suất trung bình thật.
                if mean > 0.5 + GENDER_HYSTERESIS:
                    gender = 'male'
                elif mean < 0.5 - GENDER_HYSTERESIS:
                    gender = 'female'
                self._attr_results[key] = {'gender': gender, 'gender_prob': mean if gender == 'male' else 1 - mean}
        return self._attr_results[key]

    def process_frame(self, frame_bgr, detector="mediapipe", model=None):
        model = model or DEFAULT_MODEL
        if detector != 'mediapipe':
            raise ValueError('Dự án chỉ sử dụng MediaPipe.')
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        context = (detector, model, self.align, self.crop_margin, self.smooth_n,
                   self.detect_every, frame_bgr.shape[:2],
                   self.attributes, self.attr_every, self.attr_history, self.smoothing)
        if self._ctx != context:                # đổi cách crop/model -> bỏ lịch sử cũ
            self.reset()
            self._ctx = context
        self._n += 1
        H, W = frame_bgr.shape[:2]

        if (self._n - 1) % self.detect_every == 0 or not self._faces:
            s = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
            small = cv2.resize(frame_bgr, (int(W * s), int(H * s))) if s < 1 else frame_bgr
            self._faces = [dict(f, box=tuple(int(v / s) for v in f["box"]))
                           for f in detect_faces(small, detector)]

        crops = [(f, self.crop_for_face(frame_bgr, f)) for f in self._faces[:MAX_FACES]]
        crops = [(f, crop) for f, crop in crops if crop.shape[0] >= 20 and crop.shape[1] >= 20]
        keys = self.tracker.update([f['box'] for f, _ in crops])
        self._forget_lost_tracks()

        out = []
        for (f, face), key in zip(crops, keys):
            box = f["box"]
            quality = face_quality(frame_bgr, box, f.get('angle', 0.0), f.get('yaw', 0.0))
            status = input_status(f, quality) if self.quality_gate else None
            if status in BLOCKING:
                # Không gọi model, không đưa khung kém vào bộ làm mượt hay lịch sử giới tính.
                out.append({'box': box, 'track_id': key, 'quality': quality['score'], 'status': status,
                            'label': None, 'prob': 0.0, 'probs': {}, 'raw_probs': {}, 'top3': {},
                            'gender': None, 'gender_prob': None})
                continue
            _, _, probs = predict(face, model)
            raw_probs = probs.copy()
            self._ensure_state(key)
            avg, lab = self._smooth(key, probs, quality['score'])
            top3 = sorted(avg, key=avg.get, reverse=True)[:3]
            r = {"box": box, 'track_id': key, 'quality': quality['score'], "label": lab, "prob": avg[lab],
                 "probs": avg, 'raw_probs': raw_probs, 'top3': {label: avg[label] for label in top3}}
            if f["blendshapes"] is not None:
                r["blendshapes"] = f["blendshapes"]
            attrs = self._attributes_for_face(frame_bgr, f, key, quality['score']) if self.attributes else None
            r.update(attrs or {'gender': None, 'gender_prob': None})
            if status:
                r['status'] = output_status(status, f, r['prob'])
            out.append(r)
        return out

    def crop_for_face(self, frame_bgr, face):
        """Dùng đúng cùng một crop cho dự đoán và ảnh xem trước; cổng chất lượng bắt buộc căn chỉnh khi góc > 15°."""
        if self.align or (self.quality_gate and max_pose_angle(face) > POSE_ALIGN_DEG):
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
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f
                self.count += 1

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def main():
    configure_console()
    cam = CameraThread(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    if not cam.ok:
        sys.exit("Không mở được camera (kiểm tra webcam / quyền truy cập camera).")
    detector, model = "mediapipe", DEFAULT_MODEL
    model_keys = {ord(str(i + 1)): m for i, m in enumerate(MODEL_NAMES)}
    print("MediaPipe | a = căn chỉnh bật/tắt | " + " | ".join(f"{i + 1} = {m}" for i, m in enumerate(MODEL_NAMES))
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
        cv2.putText(frame, "MediaPipe | 1/2: EfficientNet checkpoint | a: align | q: quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Emotion realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("a"):
            pipe.align = not pipe.align
        elif k in model_keys:
            model = model_keys[k]
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
