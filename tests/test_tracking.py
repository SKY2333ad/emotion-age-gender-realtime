"""Tracker, điểm chất lượng và làm mượt thích ứng; dữ liệu tổng hợp, không cần model."""
import sys
import unittest
from pathlib import Path

import cv2
import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT))
from quality import EmotionSmoother, face_quality, weighted_mean
from scripts.evaluate_tracking import CenterMatcher, id_switches, scenarios
from tracker import FaceTracker, iou
from webcam_data import read_image

LABELS = ['happy', 'sad']


class TrackerTests(unittest.TestCase):
    def test_iou_bounds(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(iou((0, 0, 10, 10), (20, 0, 10, 10)), 0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3)

    def test_ids_follow_movement_regardless_of_detection_order(self):
        tracker = FaceTracker()
        first = tracker.update([(0, 0, 50, 50), (300, 0, 50, 50)])
        second = tracker.update([(310, 0, 50, 50), (8, 0, 50, 50)])
        self.assertEqual(second, first[::-1])

    def test_id_survives_short_gap_and_expires_after_grace(self):
        tracker = FaceTracker(max_missed=2)
        track = tracker.update([(100, 100, 60, 60)])[0]
        tracker.update([])
        tracker.update([])
        self.assertEqual(tracker.update([(104, 100, 60, 60)]), [track])
        for _ in range(3):
            tracker.update([])
        self.assertEqual(tracker.tracks, {})
        self.assertNotEqual(tracker.update([(104, 100, 60, 60)]), [track])

    def test_new_far_face_gets_new_id_and_ids_are_not_reused(self):
        tracker = FaceTracker(max_missed=0)
        first = tracker.update([(0, 0, 40, 40)])[0]
        second = tracker.update([(300, 300, 40, 40)])[0]
        self.assertNotEqual(first, second)
        tracker.update([])
        self.assertNotIn(tracker.update([(0, 0, 40, 40)])[0], (first, second))

    def test_velocity_predicts_through_crossing(self):
        for name, (frames, truth) in scenarios().items():
            with self.subTest(name):
                tracker = FaceTracker()
                predicted = [tracker.update(boxes) for boxes in frames]
                self.assertEqual(id_switches(predicted, truth), 0)

    def test_new_tracker_never_worse_than_center_matcher(self):
        for name, (frames, truth) in scenarios().items():
            with self.subTest(name):
                tracker, matcher = FaceTracker(), CenterMatcher()
                new = id_switches([tracker.update(boxes) for boxes in frames], truth)
                old = id_switches([matcher.update(boxes) for boxes in frames], truth)
                self.assertLessEqual(new, old)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.sharp = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')[132:334, 170:355]

    def test_score_orders_good_blurred_dark_small_and_tilted(self):
        good = face_quality(self.sharp, (20, 20, 150, 150))['score']
        self.assertGreater(good, face_quality(cv2.GaussianBlur(self.sharp, (0, 0), 8), (20, 20, 150, 150))['score'])
        self.assertGreater(good, face_quality((self.sharp * 0.08).astype(np.uint8), (20, 20, 150, 150))['score'])
        self.assertGreater(good, face_quality(self.sharp, (20, 20, 30, 30))['score'])
        self.assertGreater(good, face_quality(self.sharp, (20, 20, 150, 150), angle=50)['score'])

    def test_real_face_blur_and_darkness_lower_score(self):
        image = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')
        box = (170, 132, 185, 202)
        good = face_quality(image, box)['score']
        self.assertGreater(good, .5)
        moderate = face_quality(cv2.GaussianBlur(image, (0, 0), 3), box)['score']
        self.assertLess(moderate, good)
        self.assertLess(face_quality(cv2.GaussianBlur(image, (0, 0), 6), box)['score'], moderate * .5)
        self.assertLess(face_quality((image * .2).astype(np.uint8), box)['score'], good * .5)
        self.assertLess(face_quality(image, box, yaw=.6)['score'], good * .4)

    def test_box_outside_frame_scores_zero(self):
        self.assertEqual(face_quality(self.sharp, (500, 500, 30, 30))['score'], 0)

    def test_weighted_mean(self):
        self.assertAlmostEqual(weighted_mean([0, 1], [3, 1]), .25)

    def test_low_quality_frame_cannot_flip_emotion(self):
        smoother = EmotionSmoother(LABELS)
        smoother.update({'happy': .9, 'sad': .1}, 1.0)
        for _ in range(3):
            _, label = smoother.update({'happy': .05, 'sad': .95}, 0.02)
        self.assertEqual(label, 'happy')


class EmotionSmootherTests(unittest.TestCase):
    def test_real_change_is_followed_quickly(self):
        smoother = EmotionSmoother(LABELS)
        for _ in range(5):
            smoother.update({'happy': .9, 'sad': .1})
        delay = next(step for step in range(1, 12)
                     if smoother.update({'happy': .1, 'sad': .9})[1] == 'sad')
        self.assertLessEqual(delay, 3)

    def test_noise_around_tie_does_not_flicker(self):
        smoother = EmotionSmoother(LABELS)
        rng = np.random.default_rng(1)
        labels = [smoother.update(dict(zip(LABELS, (p, 1 - p))))[1]
                  for p in .5 + rng.normal(0, .04, 300)]
        changes = sum(first != second for first, second in zip(labels, labels[1:]))
        self.assertLessEqual(changes, 2)


if __name__ == '__main__':
    unittest.main()
