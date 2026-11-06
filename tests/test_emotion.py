"""Hợp đồng dự đoán cảm xúc: ONNX giống Keras, quay về Keras khi không khớp, HSEmotion đúng thứ tự nhãn."""
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


@unittest.skipUnless(emotion.HSEMOTION_PATH.is_file(), 'chưa có model HSEmotion')
class HsemotionTests(unittest.TestCase):
    def test_contract_matches_project_labels(self):
        label, score, probs = emotion.predict(FACE, 'hsemotion')
        self.assertEqual(list(probs), emotion.LABELS)
        self.assertAlmostEqual(sum(probs.values()), 1, places=5)
        self.assertEqual(label, max(probs, key=probs.get))

    def test_label_order_maps_each_project_label_to_its_logit(self):
        class Session:
            def run(self, _, feed):
                logits = np.zeros((1, 8), np.float32)
                logits[0, emotion.HSEMOTION_ORDER.index('fear')] = 10
                return [logits]

        with patch.dict(emotion._ONNX_CACHE, {emotion.HSEMOTION_PATH: Session()}):
            self.assertEqual(emotion.predict(FACE, 'hsemotion')[0], 'fear')


if __name__ == '__main__':
    unittest.main()
