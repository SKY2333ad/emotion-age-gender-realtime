"""Tests for subject isolation, leakage, learned fusion and production crop behavior."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from detectors import align_crop, detect_faces
from fusion import Fusion, features
from webcam_data import LABELS, CROP_VERSION, split_subjects, check_content_leakage, file_hash
from scripts.emotion.train_fusion import fit_correction
from webcam_data import write_image, load_manifest


class WebcamTests(unittest.TestCase):
    def test_split_is_subject_disjoint_complete_and_reproducible(self):
        rows = [{'subject': f'person-{person}', 'label': label, 'session': f'{person}-{session}'}
                for person in range(6) for label in LABELS for session in range(2)]
        splits = split_subjects(rows)
        self.assertEqual(splits, split_subjects(rows))
        subjects = [{row['subject'] for row in values} for values in splits.values()]
        self.assertFalse(subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2])
        self.assertTrue(all({row['label'] for row in values} == set(LABELS) for values in splits.values()))
        with self.assertRaises(ValueError):
            split_subjects([row for row in rows if row['subject'] == 'person-0'])

    def test_renamed_copies_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / 'a.png', Path(temporary) / 'renamed.png'
            first.write_bytes(b'same-image')
            second.write_bytes(first.read_bytes())
            with self.assertRaises(ValueError):
                check_content_leakage({'train': [(first, 0)], 'validation': [(second, 0)]})

    def test_square_crop_preserves_geometry_and_handles_frame_boundary(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[30:60, 60:90] = 255
        cropped = align_crop(frame, (40, 20, 80, 50), 0, margin=0)
        self.assertEqual(cropped.shape, (80, 80, 3))
        _, _, width, height = cv2.boundingRect(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))
        self.assertEqual(width, height)
        self.assertEqual(align_crop(frame, (0, 0, 60, 80), 25, 0.15).shape, (104, 104, 3))
        with self.assertRaises(ValueError):
            detect_faces(frame, 'unsupported')

    def test_fusion_learns_and_guards_checkpoint_and_crop(self):
        labels = np.tile(np.arange(7), 5)
        probabilities = np.full((35, 7), 0.1)
        probabilities[:, 6] = 0.4
        x = np.concatenate((probabilities, np.eye(7)[labels]), axis=1)
        result = fit_correction(x, probabilities, labels, x, probabilities, labels,
                                epochs=100, learning_rate=0.1, regularization=0.01)
        self.assertGreater(result['best_validation']['macro_f1'], result['baseline_validation']['macro_f1'])
        with tempfile.TemporaryDirectory() as temporary:
            model, artifact = Path(temporary) / 'base.keras', Path(temporary) / 'fusion.json'
            model.write_bytes(b'checkpoint')
            data = {key: result[key] for key in ('weights', 'bias', 'mean', 'scale')}
            data.update({'version': 1, 'labels': LABELS, 'base_sha256': file_hash(model),
                         'blendshape_names': [f'bs{i}' for i in range(7)],
                         'crop_config': {'margin': 0.15, 'align': True, 'version': CROP_VERSION}})
            artifact.write_text(json.dumps(data), encoding='utf-8')
            fusion = Fusion(artifact, model, 0.15, True)
            probs = dict(zip(LABELS, probabilities[0]))
            output = fusion.predict(probs, dict(zip(data['blendshape_names'], np.eye(7)[0])))
            self.assertAlmostEqual(sum(output.values()), 1)
            self.assertEqual(max(output, key=output.get), LABELS[0])
            with self.assertRaises(ValueError):
                Fusion(artifact, model, 0.05, True)
            with self.assertRaises(ValueError):
                features(probs, {}, data['blendshape_names'])
            model.write_bytes(b'changed-checkpoint')
            with self.assertRaises(ValueError):
                Fusion(artifact, model, 0.15, True)

    def test_fusion_uses_fresh_blendshapes_every_frame(self):
        import pipeline
        frame = np.full((120, 120, 3), 100, np.uint8)
        face = {'box': (20, 20, 60, 60), 'angle': 0, 'blendshapes': {'jawOpen': 0.3}}
        probs = {label: float(label == 'sad') for label in LABELS}
        with patch.object(pipeline, 'Fusion') as correction, \
                patch.object(pipeline, 'detect_faces', return_value=[face]) as detector, \
                patch.object(pipeline, 'predict_batch', side_effect=lambda faces, model: [('sad', 1, probs)] * len(faces)):
            correction.return_value.predict.return_value = probs
            pipe = pipeline.Pipeline(detect_every=3, fusion_path='test.json')
            for _ in range(3):
                result = pipe.process_frame(frame)
                self.assertTrue(result[0]['fusion_used'])
                self.assertEqual(result[0]['label'], 'sad')
            self.assertEqual(detector.call_count, 3)

    def test_manifest_recrop_fusion_and_test_report_work_together(self):
        from scripts.data import prepare_webcam
        from scripts.emotion import train_fusion, evaluate_webcam
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'manifest.jsonl'
            rows = []
            for person in range(5):
                for index, label in enumerate(LABELS):
                    path = root / f'{person}_{label}.png'
                    write_image(path, np.full((100, 100, 3), (50 + person, 80 + index, 100), np.uint8))
                    rows.append({'path': path.name, 'frame_path': path.name, 'subject': f'p{person}',
                                 'label': label, 'box': [10, 10, 80, 80], 'angle': 0,
                                 'crop_margin': 0.15, 'align': True, 'crop_version': CROP_VERSION,
                                 'blendshapes': {f'bs{i}': float(i == index) for i in range(7)}})
            manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            prepared = root / 'prepared'
            prepare_webcam.main(['--manifest', str(manifest), '--output', str(prepared), '--margin', '0.10'])
            prepared_manifest = prepared / 'manifest.jsonl'
            self.assertEqual(len(load_manifest(prepared_manifest)), 35)
            model = root / 'efficientnet.keras'
            model.write_bytes(b'test-checkpoint')
            artifact = root / 'fusion.json'
            probs = {label: (0.4 if label == 'neutral' else 0.1) for label in LABELS}
            with patch.object(train_fusion, 'MODEL_DIR', str(root)), \
                    patch.object(train_fusion, 'predict_row', return_value=probs):
                self.assertEqual(train_fusion.main(['--manifest', str(prepared_manifest), '--model', 'efficientnet',
                                                   '--margin', '0.10', '--output', str(artifact),
                                                   '--learning-rate', '0.1', '--regularization', '0.01']), 0)
            self.assertTrue(artifact.exists())
            output = root / 'evaluation'
            with patch.object(evaluate_webcam, 'MODEL_DIR', str(root)), \
                    patch.object(evaluate_webcam, 'predict_row', return_value=probs):
                evaluate_webcam.main(['--manifest', str(prepared_manifest), '--model', 'efficientnet',
                                      '--split', 'test', '--margins', '0.10', '--fusion', str(artifact),
                                      '--output', str(output)])
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertGreater(report['test_margin_0.10_fusion']['macro_f1'], report['test_margin_0.10']['macro_f1'])
            self.assertTrue((output / 'predictions.csv').exists())
            self.assertTrue((output / 'confusion_test_margin_0.10_fusion.png').exists())

    def test_invalid_fusion_never_silently_falls_back_on_retry(self):
        import pipeline
        frame = np.ones((120, 120, 3), np.uint8)
        with patch.object(pipeline, 'Fusion', side_effect=ValueError('wrong checkpoint')) as correction, \
                patch.object(pipeline, 'predict_batch') as classifier:
            pipe = pipeline.Pipeline(fusion_path='wrong.json')
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'wrong checkpoint'):
                    pipe.process_frame(frame)
            self.assertEqual(correction.call_count, 2)
            classifier.assert_not_called()

    def test_missing_blendshapes_fallback_clears_fused_history(self):
        import pipeline
        frame = np.full((120, 120, 3), 100, np.uint8)
        face = {'box': (20, 20, 60, 60), 'angle': 0, 'blendshapes': {'jawOpen': 0.3}}
        sad = {label: float(label == 'sad') for label in LABELS}
        fear = {label: float(label == 'fear') for label in LABELS}
        with patch.object(pipeline, 'Fusion') as correction, \
                patch.object(pipeline, 'detect_faces', side_effect=[[face], [dict(face, blendshapes=None)]]), \
                patch.object(pipeline, 'predict_batch', side_effect=lambda faces, model: [('sad', 1, sad)] * len(faces)):
            correction.return_value.predict.return_value = fear
            pipe = pipeline.Pipeline(fusion_path='test.json')
            self.assertEqual(pipe.process_frame(frame)[0]['label'], 'fear')
            result = pipe.process_frame(frame)[0]
            self.assertFalse(result['fusion_used'])
            self.assertEqual(result['label'], 'sad')
            self.assertEqual(result['probs']['sad'], 1)


if __name__ == '__main__':
    unittest.main()
