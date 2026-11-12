"""3 detector của báo cáo: Haar, MediaPipe, YOLO26 cùng một hợp đồng đầu ra; pipeline và app đổi được detector."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
import detectors
import pipeline
from tracker import iou
from webcam_data import read_image

IMAGE = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')


class DetectorTests(unittest.TestCase):
    def test_haar_and_mediapipe_find_the_same_face(self):
        haar, = detectors.detect_faces(IMAGE, 'haar')
        landmarks, = detectors.detect_faces(IMAGE, 'mediapipe')
        self.assertEqual(set(haar), {'box', 'angle', 'yaw', 'blendshapes'})
        self.assertIsNone(haar['blendshapes'])
        self.assertGreater(iou(haar['box'], landmarks['box']), 0.4)
        self.assertEqual(detectors.detect_faces(np.full((200, 200, 3), 90, np.uint8), 'haar'), [])

    def test_yolo26_end_to_end_output_maps_back_to_frame(self):
        # Khung 640×480 thu về 640 cạnh dài: scale 1; khung 1280×960: scale 0.5.
        output = np.array([[[100, 50, 200, 170, 0.9, 0], [0, 0, 10, 10, 0.2, 0], [600, 400, 700, 500, 0.8, 0]]])
        faces = detectors.yolo_boxes(output, 0.5, 1280, 960)
        self.assertEqual([face['box'] for face in faces], [(200, 100, 200, 240), (1200, 800, 80, 160)])
        self.assertTrue(all(face['angle'] == 0 and face['blendshapes'] is None for face in faces))
        with self.assertRaisesRegex(ValueError, r'\(1, 5, N\)'):
            detectors.yolo_boxes(np.zeros((1, 7, 10)), 1, 640, 480)

    def test_yolo26_raw_output_is_filtered_by_nms_like_ultralytics(self):
        # Dạng (1, 5, N) = cx, cy, w, h, điểm: 2 khung trùng nhau chỉ giữ khung điểm cao, khung điểm thấp bị bỏ.
        raw = np.zeros((1, 5, 4), np.float32)
        raw[0, :, 0] = [150, 110, 100, 120, 0.9]
        raw[0, :, 1] = [152, 112, 100, 120, 0.8]
        raw[0, :, 2] = [450, 300, 80, 80, 0.6]
        raw[0, :, 3] = [300, 300, 80, 80, 0.1]
        faces = detectors.yolo_boxes(raw, 1.0, 640, 480)
        self.assertEqual([face['box'] for face in faces], [(100, 50, 100, 120), (410, 260, 80, 80)])
        self.assertEqual(detectors.yolo_boxes(np.zeros((1, 5, 8400)), 1, 640, 480), [])

    def test_missing_yolo_model_has_actionable_error(self):
        with patch.object(detectors, 'YOLO_MODEL', str(ROOT / 'models' / 'face_detection' / 'khong_co.onnx')), \
                patch.object(detectors, '_yolo', None):
            with self.assertRaisesRegex(RuntimeError, 'README'):
                detectors.detect_faces(IMAGE, 'yolo')

    def test_pipeline_runs_with_each_available_detector(self):
        available = ['haar', 'mediapipe'] + (['yolo'] if Path(detectors.YOLO_MODEL).exists() else [])
        for name in available:
            result = pipeline.Pipeline(attributes=False).process_frame(IMAGE, name)
            self.assertEqual(len(result), 1, name)
            self.assertEqual(result[0]['track_id'], 1)
            self.assertFalse(result[0]['fusion_used'])

    def test_app_switches_detector_without_errors(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        box = next(item for item in app.selectbox if item.label == 'Phát hiện khuôn mặt')
        self.assertEqual(box.value, 'mediapipe')
        box.set_value('haar').run()
        self.assertEqual(len(app.exception), 0)
        fusion = next(item for item in app.checkbox if item.label == 'Kết hợp đặc trưng MediaPipe')
        self.assertTrue(fusion.disabled)
        next(item for item in app.selectbox if item.label == 'Phát hiện khuôn mặt').set_value('yolo').run()
        self.assertEqual(len(app.exception), 0)
        if not Path(detectors.YOLO_MODEL).exists():
            self.assertTrue(any('yolo26n-face.onnx' in item.value for item in app.error))


if __name__ == '__main__':
    unittest.main()
