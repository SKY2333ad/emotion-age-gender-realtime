"""Điểm chất lượng mặt 0..1 và các phép gộp theo thời gian có trọng số chất lượng."""
import cv2
import numpy as np

MIN_WEIGHT = 0.05                # mẫu kém vẫn tính rất ít, tránh tổng trọng số bằng 0


def _ramp(value, low, high):
    return float(np.clip((value - low) / (high - low), 0, 1))


def sharpness(gray):
    """Độ nét tương đối 0..1 (Crete 2007): ảnh mất bao nhiêu chi tiết khi làm mờ thêm; ít phụ thuộc camera hơn Laplacian."""
    gray = gray.astype(np.float32)
    values = []
    for axis, kernel in ((0, (1, 9)), (1, (9, 1))):
        original = np.abs(np.diff(gray, axis=axis))
        blurred = np.abs(np.diff(cv2.blur(gray, kernel), axis=axis))
        values.append(np.maximum(original - blurred, 0).sum() / max(original.sum(), 1e-6))
    return float(min(values))


def face_quality(frame_bgr, box, angle=0.0, yaw=0.0):
    """Thành phần rẻ, không cần model: cỡ mặt, độ nét, độ sáng, hướng đầu.

    FaceLandmarker không trả detection confidence và không có đo che khuất, nên hai yếu tố này không có ở đây.
    Thang độ nét lấy từ UTKFace validation: ảnh gốc trung vị 0.52, mờ σ=3 là 0.33, σ=6 là 0.21.
    """
    x, y, width, height = [int(value) for value in box]
    image_h, image_w = frame_bgr.shape[:2]
    patch = frame_bgr[max(y, 0):min(y + height, image_h), max(x, 0):min(x + width, image_w)]
    if patch.size == 0:
        return {'score': 0.0, 'size': 0.0, 'sharpness': 0.0, 'lighting': 0.0, 'pose': 0.0,
                'brightness_value': 0.0, 'sharpness_value': 0.0}
    gray = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
    brightness, sharp = float(gray.mean()), sharpness(gray)
    parts = {'size': _ramp(min(width, height), 30, 110),
             'sharpness': _ramp(sharp, 0.2, 0.5),
             'lighting': min(_ramp(brightness, 25, 70), _ramp(255 - brightness, 25, 70)),
             'pose': min(1 - _ramp(abs(angle), 30, 60), 1 - _ramp(yaw, 0.25, 0.7))}
    # Lấy tích: một thành phần rất kém kéo điểm xuống, không bù được bằng thành phần khác.
    return dict(parts, score=float(np.prod(list(parts.values()))),
                brightness_value=brightness, sharpness_value=sharp)


# Cổng chất lượng đầu vào (phương án điều kiện thực tế, 13/11/2026). Mặt bị chặn không được đưa vào model.
# Độ nét/độ sáng kiểm trên RAF-DB train: chặn 0% và 0,7% ảnh thường; độ nét 0,25 ứng với mờ σ≈2 trên mặt 100 px.
POSE_ALIGN_DEG, POSE_MAX_DEG = 15.0, 30.0
MIN_FACE_PX, GOOD_FACE_PX = 80, 120
MIN_BRIGHTNESS = 45.0
MIN_SHARPNESS = 0.25
MIN_CONFIDENCE = 0.45
BLOCKING = ('LARGE_POSE', 'TOO_FAR', 'LOW_LIGHT', 'BLURRY')
STATUS_TEXT = {'OK': 'Đầu vào tốt', 'NEED_ALIGNMENT': 'Mặt hơi nghiêng, đã căn chỉnh',
               'LARGE_POSE': 'Góc mặt trên 30°, vui lòng nhìn về camera',
               'TOO_FAR': 'Mặt quá nhỏ, vui lòng đến gần camera', 'LOW_LIGHT': 'Vui lòng tăng ánh sáng',
               'BLURRY': 'Giữ khuôn mặt ổn định', 'LOW_CONFIDENCE': 'Không chắc chắn'}


def max_pose_angle(face):
    """max(|yaw|, |pitch|, |roll|) theo độ; thiếu yaw/pitch thì chỉ dùng roll."""
    pose = face.get('pose') or {}
    return max(abs(face.get('angle', 0.0)), abs(pose.get('yaw', 0.0)), abs(pose.get('pitch', 0.0)))


def input_status(face, quality):
    """Trạng thái trước khi gọi model, xét theo thứ tự: góc, cỡ mặt, độ sáng, độ nét."""
    angle = max_pose_angle(face)
    if angle > POSE_MAX_DEG:
        return 'LARGE_POSE'
    if face['box'][2] < MIN_FACE_PX:
        return 'TOO_FAR'
    if quality['brightness_value'] < MIN_BRIGHTNESS:
        return 'LOW_LIGHT'
    if quality['sharpness_value'] < MIN_SHARPNESS:
        return 'BLURRY'
    return 'NEED_ALIGNMENT' if angle > POSE_ALIGN_DEG else 'OK'


def output_status(status, face, probability):
    """Sau khi dự đoán: xác suất thấp hoặc mặt 80–119 px thì đánh dấu không chắc chắn."""
    if probability < MIN_CONFIDENCE or face['box'][2] < GOOD_FACE_PX:
        return 'LOW_CONFIDENCE'
    return status


def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.maximum(np.asarray(weights, float), MIN_WEIGHT)
    order = np.argsort(values)
    values, cumulative = values[order], np.cumsum(weights[order])
    half = cumulative[-1] / 2
    index = int(np.searchsorted(cumulative, half))
    if index + 1 < len(values) and np.isclose(cumulative[index], half):
        return float((values[index] + values[index + 1]) / 2)   # trọng số đều thì trùng np.median
    return float(values[index])


def weighted_mean(values, weights):
    return float(np.average(values, weights=np.maximum(np.asarray(weights, float), MIN_WEIGHT)))


class EmotionSmoother:
    """EMA theo chất lượng + phát hiện đổi biểu cảm + trễ nhãn, tách biệt với tuổi/giới tính."""

    def __init__(self, labels, base=0.35, fast=0.75, change=0.6, margin=0.08):
        self.labels, self.base, self.fast, self.change, self.margin = labels, base, fast, change, margin
        self.state, self.label = None, None

    def update(self, probs, quality=1.0):
        vector = np.array([probs[label] for label in self.labels], float)
        if self.state is None:
            self.state = vector
        else:
            # Thay đổi lớn ở khung chất lượng tốt là biểu cảm mới, không phải nhiễu: bám nhanh.
            changed = np.abs(vector - self.state).sum() >= self.change
            # Khung kém vẫn đóng góp ít, để điểm thấp kéo dài không làm nhãn đứng yên.
            alpha = (self.fast if changed else self.base) * (0.25 + 0.75 * quality)
            self.state = self.state + alpha * (vector - self.state)
        top = self.labels[int(self.state.argmax())]
        current = self.state[self.labels.index(self.label)] if self.label else -1.0
        if self.label is None or self.state.max() - current >= self.margin:
            self.label = top
        return dict(zip(self.labels, map(float, self.state))), self.label
