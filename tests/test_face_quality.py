"""Cổng chất lượng đầu vào: góc mặt, cỡ mặt, độ sáng, độ nét, độ tin cậy và làm mượt."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
import pipeline
from quality import face_quality, input_status, output_status

LABELS = pipeline.LABELS
FEAR = {label: float(label == 'fear') for label in LABELS}
BOX = (40, 40, 140, 140)


def face(yaw=0.0, pitch=0.0, roll=0.0, box=BOX):
    return {'box': box, 'angle': roll, 'yaw': 0.0, 'blendshapes': None, 'pose': {'yaw': yaw, 'pitch': pitch}}


class FaceQualityTests(unittest.TestCase):
    def setUp(self):
        # Ảnh thật có chi tiết khuôn mặt; nhiễu ngẫu nhiên vẫn "nét" sau khi làm mờ nên không dùng được.
        image = cv2.imread(str(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg'))
        self.frame = cv2.resize(image[60:420, 80:440], (240, 240), interpolation=cv2.INTER_AREA)

    def status(self, frame, detected):
        return input_status(detected, face_quality(frame, detected['box'], detected['angle']))

    def test_pose_levels_use_largest_of_yaw_pitch_roll(self):
        for angle in (0, 10, 15):
            self.assertEqual(self.status(self.frame, face(yaw=angle)), 'OK')
        for angle in (16, 20, 30):
            self.assertEqual(self.status(self.frame, face(pitch=-angle)), 'NEED_ALIGNMENT')
        for angle in (31, 45):
            self.assertEqual(self.status(self.frame, face(yaw=angle)), 'LARGE_POSE')
        self.assertEqual(self.status(self.frame, face(yaw=12, pitch=8, roll=19)), 'NEED_ALIGNMENT')
        no_pose = dict(face(), pose=None, angle=20)
        self.assertEqual(self.status(self.frame, no_pose), 'NEED_ALIGNMENT')

    def test_size_light_and_blur(self):
        self.assertEqual(self.status(self.frame, face(box=(40, 40, 79, 79))), 'TOO_FAR')
        self.assertEqual(self.status(self.frame, face(box=(40, 40, 80, 80))), 'OK')
        self.assertEqual(self.status((self.frame * 0.25).astype(np.uint8), face()), 'LOW_LIGHT')
        self.assertEqual(self.status(cv2.GaussianBlur(self.frame, (0, 0), 6), face()), 'BLURRY')
        self.assertEqual(output_status('OK', face(), 0.44), 'LOW_CONFIDENCE')
        self.assertEqual(output_status('OK', face(box=(0, 0, 100, 100)), 0.9), 'LOW_CONFIDENCE')
        self.assertEqual(output_status('NEED_ALIGNMENT', face(), 0.9), 'NEED_ALIGNMENT')

    def run_pipeline(self, frames, faces, **kwargs):
        pipe = pipeline.Pipeline(attributes=False, quality_gate=True, align=False, **kwargs)
        with patch.object(pipeline, 'detect_faces', side_effect=faces), \
                patch.object(pipeline, 'predict', return_value=('fear', 1.0, FEAR)) as predict:
            results = [pipe.process_frame(frame) for frame in frames]
        return pipe, results, predict

    def test_blocked_faces_skip_model_and_smoothing(self):
        dark = (self.frame * 0.25).astype(np.uint8)
        pipe, results, predict = self.run_pipeline([self.frame, dark, self.frame],
                                                   [[face()], [face()], [face(yaw=40)]])
        self.assertEqual(predict.call_count, 1)
        self.assertEqual([r[0]['status'] for r in results], ['OK', 'LOW_LIGHT', 'LARGE_POSE'])
        self.assertIsNone(results[1][0]['label'])
        self.assertEqual(results[1][0]['track_id'], results[0][0]['track_id'])
        state = pipe._smoothers[results[0][0]['track_id']].state
        np.testing.assert_array_equal(state, [FEAR[label] for label in LABELS])

    def test_tilted_face_is_aligned_even_when_alignment_is_off(self):
        tilted = face(roll=20)
        _, _, predict = self.run_pipeline([self.frame], [[tilted]])
        expected = pipeline.align_crop(self.frame, tilted['box'], 20, 0.0)
        np.testing.assert_array_equal(predict.call_args.args[0], expected)

    def test_new_track_id_starts_with_empty_history(self):
        pipe, results, _ = self.run_pipeline([self.frame] * 3, [[face()], [], [face()]], track_grace=0)
        first, second = results[0][0]['track_id'], results[2][0]['track_id']
        self.assertNotEqual(first, second)
        self.assertEqual(set(pipe._smoothers), {second})

    def test_gate_off_keeps_old_result_keys(self):
        pipe = pipeline.Pipeline(attributes=False)
        with patch.object(pipeline, 'detect_faces', return_value=[face(yaw=60)]), \
                patch.object(pipeline, 'predict', return_value=('fear', 1.0, FEAR)):
            result = pipe.process_frame(self.frame)[0]
        self.assertNotIn('status', result)
        self.assertEqual(result['label'], 'fear')


if __name__ == '__main__':
    unittest.main()
