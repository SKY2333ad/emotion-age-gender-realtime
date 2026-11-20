"""E7/E9: so bộ ghép tâm cũ với tracker mới và làm mượt cũ với thích ứng trên chuỗi tổng hợp có đáp án.

Đây là mô phỏng hành vi thuật toán, không thay thế video thật nhiều người có nhãn ID.
python scripts/evaluate_tracking.py --output results/tracking_evaluation.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from quality import EmotionSmoother
from tracker import FaceTracker
from webcam_data import configure_console

GENDER_NOISE = 0.08    # độ lệch chuẩn nhiễu giả định giữa các lần suy luận

MATCH_DIST = 80
LABELS = ['neutral', 'happy', 'sad']


class CenterMatcher:
    """Bản sao logic ghép tâm tham lam của Pipeline cũ (mốc so sánh)."""

    def __init__(self):
        self.keys, self.next_id = {}, 1

    def update(self, boxes):
        taken, output, updated = set(), [], {}
        for x, y, width, height in boxes:
            center = (x + width // 2, y + height // 2)
            best, distance = None, MATCH_DIST
            for key in self.keys:
                gap = ((center[0] - key[0]) ** 2 + (center[1] - key[1]) ** 2) ** .5
                if key not in taken and gap < distance:
                    best, distance = key, gap
            track = self.keys[best] if best is not None else self.next_id
            if best is None:
                self.next_id += 1
            else:
                taken.add(best)
            updated[center] = track
            output.append(track)
        self.keys = updated
        return output


def scenarios():
    """Mỗi kịch bản trả (khung → danh sách box, khung → danh tính thật cùng thứ tự)."""
    result = {}

    def lane(start, step, y=100, size=80):
        return lambda t: (start + step * t, y, size, size)

    # Hai người đi ngược chiều, cắt nhau không che khuất.
    first, second = lane(50, 12, 60), lane(600, -12, 120)
    result['cắt nhau'] = ([[first(t), second(t)] for t in range(45)], [[0, 1] for _ in range(45)])
    # Thứ tự detector đổi từng khung, hai mặt sát nhau.
    near = [[(100 + t, 100, 80, 80), (195 + t, 100, 80, 80)] for t in range(40)]
    result['sát nhau, đổi thứ tự'] = ([boxes[::-1] if t % 2 else boxes for t, boxes in enumerate(near)],
                                      [[1, 0] if t % 2 else [0, 1] for t in range(40)])
    # Che khuất 8 khung trong khi di chuyển đều.
    walker = lane(50, 10)
    frames = [[] if 15 <= t < 23 else [walker(t)] for t in range(40)]
    result['che khuất ngắn'] = (frames, [[] if not boxes else [0] for boxes in frames])
    # Di chuyển nhanh: 90 px/khung, vượt MATCH_DIST của bộ ghép cũ.
    runner = lane(0, 90, size=100)
    result['di chuyển nhanh'] = ([[runner(t)] for t in range(10)], [[0] for _ in range(10)])
    # Một người bị che, người kia đi ngang qua đúng vị trí đó.
    still, passer = (300, 100, 90, 90), lane(120, 15, 110)
    frames, truth = [], []
    for t in range(30):
        visible = [] if 10 <= t < 16 else [still]
        frames.append(visible + [passer(t)])
        truth.append(([0] if visible else []) + [1])
    result['đi ngang vị trí bị che'] = (frames, truth)
    return result


def id_switches(predicted, truth):
    """Đếm số lần một danh tính đổi track ID, hoặc một track ID nhảy sang danh tính khác."""
    last, owners, switches = {}, {}, 0
    for tracks, people in zip(predicted, truth):
        for track, person in zip(tracks, people):
            switches += person in last and last[person] != track
            switches += track in owners and owners[track] != person
            last[person], owners[track] = track, person
    return switches


def smoothing_trial(mode, seed, noise, frames=300, change_every=60):
    """Chuỗi biểu cảm đổi theo chu kỳ cộng nhiễu; trả số lần đổi nhãn thừa và độ trễ phản hồi (khung)."""
    rng = np.random.default_rng(seed)
    smoother, window = EmotionSmoother(LABELS), collections.deque(maxlen=3)
    predicted, truth = [], []
    for t in range(frames):
        target = (t // change_every) % len(LABELS)
        vector = rng.dirichlet([1.0] * len(LABELS)) * noise
        vector[target] += 1 - noise
        probs = dict(zip(LABELS, vector))
        truth.append(target)
        if mode == 'raw':
            predicted.append(int(vector.argmax()))
        elif mode == 'mean3':
            window.append(vector)
            predicted.append(int(np.mean(window, axis=0).argmax()))
        else:
            predicted.append(LABELS.index(smoother.update(probs, 0.8)[1]))
    changes = sum(first != second for first, second in zip(predicted, predicted[1:]))
    delays = []
    for start in range(change_every, frames, change_every):
        hit = next((t - start for t in range(start, min(start + change_every, frames))
                    if predicted[t] == truth[start]), change_every)
        delays.append(hit)
    return {'flips': changes - (frames - 1) // change_every, 'delay': float(np.mean(delays))}


def attribute_trial(every, history, seed, frames=600, change_at=300, quality=0.8):
    """Đi qua Pipeline thật với model giả: đổi giới tính ở khung change_at, đếm số khung tới khi kết quả đổi."""
    from unittest.mock import patch
    import pipeline
    rng = np.random.default_rng(seed)
    frame = np.full((300, 400, 3), 100, np.uint8)
    face = {'box': (100, 100, 100, 100), 'angle': 0.0, 'blendshapes': None}
    probs = dict.fromkeys(pipeline.LABELS, 1 / len(pipeline.LABELS))
    step = {'t': 0}

    def fake_attributes(*_):
        before = step['t'] < change_at
        p_male = np.clip((0.9 if before else 0.1) + rng.normal(0, GENDER_NOISE), .01, .99)
        return {'gender': 'male' if p_male >= .5 else 'female', 'gender_prob': float(max(p_male, 1 - p_male))}

    with patch.object(pipeline, 'detect_faces', return_value=[face]),             patch.object(pipeline, 'predict', return_value=('neutral', 1.0, probs)),             patch.object(pipeline, 'predict_attributes', fake_attributes),             patch.object(pipeline, 'face_quality', return_value={'score': quality}),             patch.object(pipeline, 'load_calibration', return_value={'gender_temperature': 1.0}):
        pipe = pipeline.Pipeline(attr_every=every, attr_history=history)
        genders = []
        for t in range(frames):
            step['t'] = t
            result = pipe.process_frame(frame)[0]
            genders.append(result['gender'])
    stable = genders[60:change_at]
    return {'gender_delay': next(t - change_at for t in range(change_at, frames) if genders[t] == 'female'),
            'gender_flips': sum(a != b for a, b in zip(stable, stable[1:]))}


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'tracking_evaluation.json')
    args = parser.parse_args()
    tracking = {}
    for name, (frames, truth) in scenarios().items():
        tracker, matcher = FaceTracker(), CenterMatcher()
        tracking[name] = {
            'frames': len(frames),
            'center_matcher_id_switches': id_switches([matcher.update(boxes) for boxes in frames], truth),
            'tracker_id_switches': id_switches([tracker.update(boxes) for boxes in frames], truth)}
    smoothing = {}
    for noise in (0.3, 0.6, 0.9):
        for mode in ('raw', 'mean3', 'adaptive'):
            trials = [smoothing_trial(mode, seed, noise) for seed in range(30)]
            smoothing[f'noise={noise} {mode}'] = {key: float(np.mean([trial[key] for trial in trials]))
                                                  for key in ('flips', 'delay')}
    attributes = {}
    for every, history in ((12, 15), (5, 12)):
        trials = [attribute_trial(every, history, seed) for seed in range(30)]
        attributes[f'every={every} history={history}'] = {key: float(np.mean([trial[key] for trial in trials]))
                                                          for key in trials[0]}
    report = {'method': 'synthetic; 30 seeds cho làm mượt và thuộc tính', 'tracking': tracking,
              'smoothing': smoothing, 'attributes': attributes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
