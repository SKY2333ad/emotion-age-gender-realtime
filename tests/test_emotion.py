"""Hợp đồng dự đoán cảm xúc: ONNX giống Keras, quay về Keras khi không khớp, 3 model của báo cáo cùng hợp đồng."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
import predict as emotion
from preprocessing import prepare_face
from webcam_data import read_image

FACE = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')[100:340, 150:380]


class EfficientNetOnnxTests(unittest.TestCase):
    def setUp(self):
        emotion._ONNX_CACHE.pop('efficientnet_checked', None)
        self.addCleanup(emotion._ONNX_CACHE.pop, 'efficientnet_checked', None)

    @unittest.skipUnless(emotion.EFFICIENTNET_ONNX.is_file(), 'chưa có efficientnet.onnx')
    def test_default_prediction_uses_onnx_and_matches_keras(self):
        self.assertIsNotNone(emotion._efficientnet_session())
        label, score, probs = emotion.predict(FACE)
        keras = emotion._get_inference('efficientnet')(prepare_face(FACE)[None]).numpy()[0]
        self.assertEqual(label, emotion.LABELS[int(keras.argmax())])
        np.testing.assert_allclose([probs[name] for name in emotion.LABELS], keras, atol=1e-3)
        self.assertEqual(list(probs), emotion.LABELS)
        self.assertAlmostEqual(probs[label], score)

    def test_mismatched_or_missing_check_falls_back_to_keras(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'efficientnet.onnx'
            path.write_bytes(b'not a model')
            with patch.object(emotion, 'EFFICIENTNET_ONNX', path):
                self.assertIsNone(emotion._efficientnet_session())
                emotion._ONNX_CACHE.pop('efficientnet_checked')
                path.with_suffix('.check.json').write_text(json.dumps(
                    {'keras_sha256': 'khác', 'onnx_sha256': 'khác', 'passed': True}), encoding='utf-8')
                self.assertIsNone(emotion._efficientnet_session())
                self.assertEqual(list(emotion.predict(FACE)[2]), emotion.LABELS)


class BatchTests(unittest.TestCase):
    def test_batch_matches_one_face_at_a_time_and_keeps_order(self):
        faces = [FACE, FACE[:, ::-1].copy(), FACE[40:200, 30:190].copy()]
        models = emotion.MODEL_NAMES
        for model in models:
            batch = emotion.predict_batch(faces, model)
            for face, (label, score, probs) in zip(faces, batch):
                single = emotion.predict(face, model)
                self.assertEqual(label, single[0])
                np.testing.assert_allclose(list(probs.values()), list(single[2].values()), atol=1e-4)
        self.assertEqual(emotion.predict_batch([], 'efficientnet'), [])


class CameraThreadTests(unittest.TestCase):
    def test_failed_reads_back_off_and_stop_waits_for_reader(self):
        import pipeline

        class BrokenCapture:
            reads = released = 0

            def read(self):
                BrokenCapture.reads += 1
                return False, None

            def release(self):
                BrokenCapture.released += 1

        with patch.object(pipeline, 'open_camera', return_value=BrokenCapture()):
            camera = pipeline.CameraThread()
            import time
            time.sleep(0.2)
            camera.stop()
        self.assertLess(BrokenCapture.reads, 50)          # trước đây quay vòng hàng nghìn lần trong 0,2 giây
        self.assertFalse(camera._thread.is_alive())
        self.assertEqual(BrokenCapture.released, 1)
        self.assertEqual(camera.count, 0)


class ReportModelTests(unittest.TestCase):
    def test_three_report_models_are_available_and_follow_the_contract(self):
        self.assertEqual(emotion.MODEL_NAMES[:3], emotion.REPORT_MODELS)
        for model in emotion.REPORT_MODELS:
            label, score, probs = emotion.predict(FACE, model)
            self.assertEqual(list(probs), emotion.LABELS)
            self.assertAlmostEqual(sum(probs.values()), 1, places=4)
            self.assertEqual(label, max(probs, key=probs.get))


if __name__ == '__main__':
    unittest.main()
