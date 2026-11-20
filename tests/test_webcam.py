"""Tests for subject isolation, leakage and production crop behavior."""
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
from webcam_data import LABELS, CROP_VERSION, split_subjects, check_content_leakage, quality
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

    def test_capture_quality_rejects_dark_blurry_and_tiny_faces(self):
        self.assertEqual(quality(np.zeros((90, 90, 3), np.uint8), (0, 0, 90, 90))['reason'],
                         'too dark / too bright')
        self.assertEqual(quality(np.full((90, 90, 3), 100, np.uint8), (0, 0, 90, 90))['reason'], 'blurry')
        self.assertEqual(quality(np.full((90, 90, 3), 100, np.uint8), (0, 0, 30, 30))['reason'], 'face too small')

    def test_manifest_recrop_and_test_report_work_together(self):
        from scripts import prepare_webcam, evaluate_webcam
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
            self.assertEqual(len(load_manifest(prepared_manifest)), 5 * len(LABELS))
            model = root / 'efficientnet.keras'
            model.write_bytes(b'test-checkpoint')
            probs = {label: (0.4 if label == 'neutral' else 0.1) for label in LABELS}
            output = root / 'evaluation'
            with patch.object(evaluate_webcam, 'MODEL_DIR', str(root)), \
                    patch.object(evaluate_webcam, 'predict_row', return_value=probs):
                evaluate_webcam.main(['--manifest', str(prepared_manifest), '--model', 'efficientnet',
                                      '--split', 'test', '--margins', '0.10',
                                      '--output', str(output)])
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(sum(map(sum, report['test_margin_0.10']['confusion_matrix'])), len(LABELS))
            self.assertTrue((output / 'predictions.csv').exists())
            self.assertTrue((output / 'confusion_test_margin_0.10.png').exists())



if __name__ == '__main__':
    unittest.main()
