"""Thuộc tính độc lập với cảm xúc: ONNX, tracking, lịch chạy, UI và hồi quy bản cũ."""
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
import attributes
import pipeline
from ui_helpers import attribute_text, draw_results, result_row


def attribute(p_male=0.8):
    return {'gender': 'male' if p_male >= 0.5 else 'female',
            'gender_prob': p_male if p_male >= 0.5 else 1 - p_male}


def face(x=30, y=40, blendshapes=None):
    return {'box': (x, y, 60, 70), 'angle': 5.0, 'blendshapes': blendshapes}


class AttributeModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((150, 200, 3), (10, 80, 240), np.uint8)
        self.box = (60, 40, 60, 70)

    def test_empty_images_invalid_boxes_do_not_load_model(self):
        invalid = [(None, self.box), (np.empty((0, 0, 3), np.uint8), self.box),
                   (np.zeros((30, 30), np.uint8), self.box), ([], self.box),
                   (self.frame, None), (self.frame, (0, 0, 0, 10)),
                   (self.frame, (0, 0, 10, -1)), (self.frame, (0, 0, float('nan'), 30)),
                   (self.frame, (float('inf'), 0, 30, 30)), (self.frame, (0, 0, 20)),
                   (self.frame, (900, 0, 20, 20)), (self.frame, (-30, 0, 20, 20))]
        with patch.object(attributes, '_get_model') as loader:
            for image, box in invalid:
                with self.subTest(box=box):
                    self.assertIsNone(attributes.predict_attributes(image, box))
            loader.assert_not_called()

    def test_affine_rgb_unscaled_without_rotation_and_input_unchanged(self):
        before = self.frame.copy()
        blob = attributes._make_blob(self.frame, self.box)
        self.assertEqual(blob.shape, (1, 3, 96, 96))
        self.assertEqual(blob.dtype, np.float32)
        np.testing.assert_array_equal(blob[0, :, 48, 48], (240, 80, 10))
        session = Mock()
        session.run.return_value = [np.array([[0.1, 1.1, 0.314]], np.float32)]
        with patch.object(attributes, '_get_model', return_value=(session, 'data', 'fc1')):
            result = attributes.predict_attributes(self.frame, self.box, angle=45)
        np.testing.assert_array_equal(session.run.call_args.args[1]['data'], blob)
        self.assertNotIn('age', result)          # model trả tuổi nhưng dự án không dùng
        self.assertEqual(result['gender'], 'male')
        self.assertAlmostEqual(result['gender_prob'], 1 / (1 + np.exp(-1)), places=6)
        np.testing.assert_array_equal(self.frame, before)

    def test_nonfinite_input_or_output_and_wrong_output_shape_return_none(self):
        with patch.object(attributes, '_get_model') as loader:
            self.assertIsNone(attributes.predict_attributes(np.full_like(self.frame, np.nan, dtype=np.float32), self.box))
            loader.assert_not_called()
        session = Mock()
        with patch.object(attributes, '_get_model', return_value=(session, 'data', 'fc1')):
            for output in ([[0, np.nan, .3]], [[0, 1, np.inf]], [[0, 1]]):
                session.run.return_value = [np.array(output)]
                self.assertIsNone(attributes.predict_attributes(self.frame, self.box))

    def test_boundary_box_is_supported(self):
        blob = attributes._make_blob(self.frame, (-10, -10, 60, 70))
        self.assertEqual(blob.shape, (1, 3, 96, 96))
        self.assertTrue(np.isfinite(blob).all())

    def test_real_onnx_contract_when_model_is_available(self):
        if not attributes.MODEL_PATH.is_file():
            self.skipTest('Chưa tải models/genderage.onnx; các test mock và thiếu model vẫn chạy.')
        result = attributes.predict_attributes(self.frame, self.box)
        self.assertEqual(set(result), {'gender', 'gender_prob'})
        self.assertIn(result['gender'], ('male', 'female'))
        self.assertGreaterEqual(result['gender_prob'], .5)
        self.assertLessEqual(result['gender_prob'], 1)
        cache = attributes._get_model()
        self.assertIs(cache, attributes._get_model())

    def test_missing_model_has_actionable_vietnamese_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(attributes, 'MODEL_PATH', Path(tmp) / 'genderage.onnx'), \
                patch.object(attributes, '_CACHE', None):
            with self.assertRaisesRegex(FileNotFoundError, 'Thiếu mô hình giới tính.*genderage.onnx'):
                attributes.predict_attributes(self.frame, self.box)

    def test_lazy_cache_is_thread_safe(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name='data', type='tensor(float)', shape=['None', 3, 96, 96])]
        session.get_outputs.return_value = [SimpleNamespace(name='fc1', shape=[1, 3])]
        runtime = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Mock(return_value=session))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'genderage.onnx'
            path.write_bytes(b'onnx-test')
            with patch.object(attributes, 'MODEL_PATH', path), patch.object(attributes, '_CACHE', None), \
                    patch.dict(sys.modules, {'onnxruntime': runtime}):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(lambda _: attributes._get_model(), range(32)))
                self.assertTrue(all(result is results[0] for result in results))
                runtime.InferenceSession.assert_called_once()
                options = runtime.InferenceSession.call_args.args[1]
                self.assertEqual(options.intra_op_num_threads, 1)

    def test_wrong_model_contract_is_rejected(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name='data', type='tensor(float)', shape=[1, 3, 112, 112])]
        session.get_outputs.return_value = [SimpleNamespace(name='fc1', shape=[1, 3])]
        runtime = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Mock(return_value=session))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'genderage.onnx'
            path.touch()
            with patch.object(attributes, 'MODEL_PATH', path), patch.object(attributes, '_CACHE', None), \
                    patch.dict(sys.modules, {'onnxruntime': runtime}):
                with self.assertRaisesRegex(RuntimeError, '96, 96'):
                    attributes._get_model()
                self.assertIsNone(attributes._CACHE)


class AttributeTrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((300, 400, 3), 100, np.uint8)
        self.detector = patch.object(pipeline, 'detect_faces', return_value=[face()]).start()
        self.probs = {label: float(label == 'happy') for label in pipeline.LABELS}
        self.emotion = patch.object(pipeline, 'predict', return_value=('happy', 1.0, self.probs)).start()
        self.inference = patch.object(pipeline, 'predict_attributes', return_value=attribute()).start()
        patch.object(pipeline, 'load_calibration',
                     return_value={'gender_temperature': 1.0}).start()
        self.addCleanup(patch.stopall)

    def test_new_face_runs_immediately_and_only_every_twelve_frames(self):
        pipe = pipeline.Pipeline(attr_every=12)
        for index in range(1, 26):
            result = pipe.process_frame(self.frame)[0]
            self.assertEqual(self.inference.call_count, 1 + (index - 1) // 12)
            self.assertEqual(result['gender'], 'male')
        # Mặt mới đến ở khung giữa chu kỳ vẫn suy luận ngay.
        self.detector.return_value = [face(), face(260)]
        pipe.process_frame(self.frame)
        self.assertEqual(self.inference.call_count, 4)

    def test_probability_mean_and_hysteresis(self):
        pipe = pipeline.Pipeline(attr_every=1)
        genders = []
        probabilities = [.51, .48, .51, .48, .51, .48, .51]
        for probability in probabilities:
            self.inference.return_value = attribute(probability)
            result = pipe.process_frame(self.frame)[0]
            genders.append(result['gender'])
        self.assertEqual(set(genders), {'male'})
        self.assertAlmostEqual(result['gender_prob'], np.mean(probabilities))
        self.inference.return_value = attribute(.02)
        result = pipe.process_frame(self.frame)[0]
        self.assertEqual(result['gender'], 'female')
        self.assertAlmostEqual(result['gender_prob'], 1 - np.mean(probabilities + [.02]))

    def test_calibration_softens_gender_without_changing_label(self):
        with patch.object(pipeline, 'load_calibration', return_value={'gender_temperature': 2.0}):
            pipe = pipeline.Pipeline()
        self.inference.return_value = attribute(.9)
        result = pipe.process_frame(self.frame)[0]
        self.assertEqual(result['gender'], 'male')
        self.assertAlmostEqual(result['gender_prob'], attributes.calibrate_probability(.9, 2.0))
        self.assertLess(result['gender_prob'], .9)
        self.assertNotIn('age', result)

    def test_each_face_has_own_history_and_movement_preserves_it(self):
        pipe = pipeline.Pipeline(attr_every=1)
        self.detector.return_value = [face(), face(260)]
        self.inference.side_effect = [attribute(.9), attribute(.1), attribute(.8), attribute(.2)]
        first = pipe.process_frame(self.frame)
        self.detector.return_value = [face(32), face(262)]
        second = pipe.process_frame(self.frame)
        self.assertEqual([r['gender'] for r in first], ['male', 'female'])
        np.testing.assert_allclose([r['gender_prob'] for r in second], [.85, .85])
        self.assertEqual([r['gender'] for r in second], ['male', 'female'])
        self.assertEqual(set(pipe._attr_hist), set(pipe._hist))
        self.assertEqual(sorted(map(len, pipe._attr_hist.values())), [2, 2])

    def test_disappearance_replacement_and_empty_frame_clear_buffers(self):
        pipe = pipeline.Pipeline(attr_every=12, track_grace=0)
        self.detector.return_value = [face(), face(260)]
        pipe.process_frame(self.frame)
        self.detector.return_value = [face(260)]
        pipe.process_frame(self.frame)
        self.assertEqual(set(pipe._attr_hist), {2})
        self.detector.return_value = [face(30, 200)]  # Quá xa mọi track: phải là ID mới.
        self.inference.return_value = attribute(.1)
        self.assertEqual(pipe.process_frame(self.frame)[0]['gender'], 'female')
        self.assertEqual(self.inference.call_count, 3)
        self.assertEqual(set(pipe._attr_hist), {3})
        self.detector.return_value = []
        self.assertEqual(pipe.process_frame(self.frame), [])
        self.assertEqual(pipe._attr_hist, {})
        self.assertEqual(pipe._attr_results, {})
        self.assertEqual(pipe._attr_last, {})
        self.detector.return_value = [face(30, 200)]
        pipe.process_frame(self.frame)
        self.assertEqual(self.inference.call_count, 4)

    def test_short_disappearance_keeps_track_id_and_history(self):
        pipe = pipeline.Pipeline(attr_every=12, track_grace=3)
        first = pipe.process_frame(self.frame)[0]
        self.detector.return_value = []
        for _ in range(3):
            self.assertEqual(pipe.process_frame(self.frame), [])
        self.detector.return_value = [face(36)]
        second = pipe.process_frame(self.frame)[0]
        self.assertEqual(second['track_id'], first['track_id'])
        self.assertEqual(self.inference.call_count, 1)
        self.detector.return_value = []
        for _ in range(4):
            pipe.process_frame(self.frame)
        self.assertEqual(pipe._attr_hist, {})

    def test_history_is_bounded_and_toggles_reset_context(self):
        pipe = pipeline.Pipeline(attr_every=1, attr_history=3)
        for probability in (.9, .8, .7, .6, .55):
            self.inference.return_value = attribute(probability)
            result = pipe.process_frame(self.frame)[0]
        self.assertAlmostEqual(result['gender_prob'], np.mean([.7, .6, .55]))
        self.assertEqual(len(next(iter(pipe._attr_hist.values()))), 3)
        pipe.attributes = False
        result = pipe.process_frame(self.frame)[0]
        self.assertIsNone(result['gender'])
        self.assertEqual(pipe._attr_hist, {})
        self.assertEqual(self.inference.call_count, 5)
        pipe.attributes = True
        self.inference.return_value = attribute(.3)
        self.assertEqual(pipe.process_frame(self.frame)[0]['gender'], 'female')
        self.assertEqual(self.inference.call_count, 6)
        pipe.attr_every = 12
        self.assertEqual(pipe.process_frame(self.frame)[0]['gender'], 'female')
        self.assertEqual(self.inference.call_count, 7)
        pipe.reset()
        self.assertEqual(pipe._attr_hist, {})

    def test_none_predictions_do_not_repeat_each_frame(self):
        self.inference.return_value = None
        pipe = pipeline.Pipeline(attr_every=12)
        for _ in range(12):
            result = pipe.process_frame(self.frame)[0]
            self.assertEqual([result[k] for k in ('gender', 'gender_prob')], [None] * 2)
        self.assertEqual(self.inference.call_count, 1)
        self.inference.return_value = attribute(.8)
        self.assertEqual(pipe.process_frame(self.frame)[0]['gender'], 'male')
        self.assertEqual(self.inference.call_count, 2)

    def test_missing_model_keeps_emotion_and_reports_error_once_until_reset(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(attributes, 'MODEL_PATH', Path(tmp) / 'genderage.onnx'), \
                patch.object(attributes, '_CACHE', None):
            self.inference.side_effect = attributes.predict_attributes
            pipe = pipeline.Pipeline()
            for _ in range(25):
                result = pipe.process_frame(self.frame)[0]
                self.assertEqual(result['probs'], self.probs)
                self.assertIsNone(result['gender'])
            self.assertIn('Thiếu mô hình giới tính', pipe.attribute_error)
            self.assertIn('genderage.onnx', pipe.attribute_error)
            self.assertEqual(self.inference.call_count, 1)

    def test_runtime_error_keeps_emotion(self):
        self.inference.side_effect = RuntimeError('ONNX không chạy được')
        pipe = pipeline.Pipeline()
        self.assertEqual(pipe.process_frame(self.frame)[0]['probs'], self.probs)
        self.assertIn('ONNX không chạy được', pipe.attribute_error)


class EmotionRegressionTests(unittest.TestCase):
    def test_real_emotion_model_is_identical_with_attributes_on_off(self):
        # Dùng ảnh tổng hợp trong RAM; detector được cố định để chỉ so nhánh cảm xúc thật.
        frame = np.random.default_rng(31).integers(0, 256, (180, 300, 3), dtype=np.uint8)
        detected = face(blendshapes=None)
        for model, margin, align in [('efficientnet', 0.0, True)]:
            disabled = pipeline.Pipeline(attributes=False, crop_margin=margin, align=align)
            enabled = pipeline.Pipeline(attributes=True, crop_margin=margin, align=align)
            with patch.object(pipeline, 'detect_faces', return_value=[detected]):
                for _ in range(3):
                    old = disabled.process_frame(frame, model=model)[0]
                    new = enabled.process_frame(frame, model=model)[0]
                    for key in ('label', 'prob', 'probs', 'raw_probs', 'top3'):
                        self.assertEqual(old[key], new[key])
                    if attributes.MODEL_PATH.is_file():
                        self.assertIsNone(enabled.attribute_error)


class AttributeUITests(unittest.TestCase):
    def test_vietnamese_text_and_export_preserve_raw_values(self):
        result = dict(attribute(.1), box=(30, 30, 60, 60), label='happy', prob=.8, probs={'happy': .8})
        self.assertEqual(attribute_text(result), 'giới tính ước lượng: nữ')
        row = result_row(result, 'now', 'webcam', 'mediapipe', 'efficientnet')
        self.assertNotIn('age', row)
        self.assertEqual(row['gender'], 'female')
        self.assertEqual(row['prob_happy'], .8)
        result.pop('gender')
        self.assertIsNone(result_row(result, '', '', '', '')['gender'])

    def test_drawing_attributes_preserves_input_and_disabled_overlay(self):
        frame = np.zeros((160, 300, 3), np.uint8)
        result = dict(box=(40, 50, 100, 100), label='happy', prob=.9)
        disabled = draw_results(frame, [result])
        explicit_none = draw_results(frame, [dict(result, gender=None, gender_prob=None)])
        np.testing.assert_array_equal(disabled, explicit_none)
        enabled = draw_results(frame, [dict(result, **attribute(.1))])
        self.assertFalse(np.array_equal(enabled, disabled))
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_streamlit_sidebar_and_session_statistics(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        toggle = next(c for c in app.checkbox if c.label == 'Nhận diện giới tính')
        self.assertTrue(toggle.value)
        app.session_state['rows'] = [
            result_row(dict(attribute(.9), label='happy', prob=.8, probs={'happy': .8}),
                       '2026-11-06T08:00:00', 'webcam', 'mediapipe', 'efficientnet'),
            result_row(dict(attribute(.1), label='sad', prob=.7, probs={'sad': .7}),
                       '2026-11-06T08:00:01', 'webcam', 'mediapipe', 'efficientnet')]
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Tỉ lệ nam/nữ' in c.value for c in app.caption))
        self.assertEqual(set(app.dataframe[0].value['Giới tính ước lượng']), {'nam', 'nữ'})
        next(c for c in app.checkbox if c.label == 'Nhận diện giới tính').uncheck().run()
        self.assertEqual(len(app.exception), 0)
if __name__ == '__main__':
    unittest.main()
