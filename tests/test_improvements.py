"""Verify alignment, image isolation, leakage checks and sad/fear selection constraints."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
import detectors
from selection import qualifies
from webcam_data import LABELS, CROP_VERSION, load_manifest, split_manifest, check_content_leakage, write_image, file_hash


class ImprovementTests(unittest.TestCase):
    def test_alignment_levels_eyes_without_stretching(self):
        frame = np.zeros((200, 200, 3), np.uint8)
        cv2.circle(frame, (70, 80), 5, (0, 0, 255), -1)
        cv2.circle(frame, (130, 100), 5, (0, 255, 0), -1)
        angle = float(np.degrees(np.arctan2(20, 60)))
        crop = detectors.align_crop(frame, (40, 50, 120, 120), angle, 0.15)
        red = np.argwhere((crop[:, :, 2] > 150) & (crop[:, :, 1] < 50))
        green = np.argwhere((crop[:, :, 1] > 150) & (crop[:, :, 2] < 50))
        self.assertLess(abs(red[:, 0].mean() - green[:, 0].mean()), 1)
        self.assertEqual(crop.shape[0], crop.shape[1])
        for points in (red, green):
            self.assertLessEqual(abs(np.ptp(points[:, 0]) - np.ptp(points[:, 1])), 1)

    def test_eye_order_does_not_rotate_face_upside_down(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        points[0], points[1] = SimpleNamespace(x=0.1, y=0.1), SimpleNamespace(x=0.9, y=0.9)
        points[detectors.EYE_R], points[detectors.EYE_L] = SimpleNamespace(x=0.7, y=0.5), SimpleNamespace(x=0.3, y=0.4)
        result = SimpleNamespace(face_landmarks=[points], face_blendshapes=[])
        self.assertLess(abs(detectors._faces_from_result(result, 100, 100)[0]['angle']), 30)

    def test_static_images_use_image_graph_without_video_tracking(self):
        graph = SimpleNamespace(detect=lambda image: SimpleNamespace(face_landmarks=[], face_blendshapes=[]))
        with patch.object(detectors, '_image_lm', graph), patch.object(detectors, '_get_lm') as video:
            self.assertEqual(detectors.detect_image_faces(np.zeros((100, 100, 3), np.uint8)), [])
            video.assert_not_called()

    def test_reencoded_cross_split_copy_and_label_conflict_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((40, 50, 3), 100, np.uint8)
            first, second = root / 'a.png', root / 'b.bmp'
            write_image(first, image)
            write_image(second, image)
            with self.assertRaisesRegex(ValueError, 'giữa'):
                check_content_leakage({'train': [(first, 1)], 'test': [(second, 1)]})
            with self.assertRaisesRegex(ValueError, 'mâu thuẫn'):
                check_content_leakage({'train': [(first, 1), (second, 4)]})

    def test_raf_manifest_preserves_explicit_splits_without_fake_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for split_index, split in enumerate(('train', 'validation', 'test')):
                for label_index, label in enumerate(LABELS):
                    path = root / f'{split}_{label}.png'
                    write_image(path, np.full((30, 40, 3), (20 + split_index, 60 + label_index, 130), np.uint8))
                    rows.append({'path': path.name, 'label': label, 'subject': None,
                                 'dataset': 'rafdb', 'split_basis': 'official-rafdb', 'split': split})
            manifest = root / 'manifest.jsonl'
            manifest.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            splits = split_manifest(load_manifest(manifest))
            self.assertEqual({key: len(values) for key, values in splits.items()},
                             {'train': 7, 'validation': 7, 'test': 7})
            self.assertTrue(all(row['subject'] is None for row in splits['train']))
            rows[0].pop('split_basis')
            manifest.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_selection_requires_both_focus_classes_and_macro_improvement(self):
        baseline = {'macro_f1': 0.5, 'per_class': {label: {'f1': 0.5} for label in LABELS}}
        better = {'macro_f1': 0.6, 'per_class': {label: {'f1': 0.6} for label in LABELS}}
        self.assertTrue(qualifies(better, baseline, 0.5))
        better['per_class']['fear']['f1'] = 0.49
        self.assertFalse(qualifies(better, baseline, 0.5))
        better['per_class']['fear']['f1'] = 0.6
        better['per_class']['sad']['f1'] = 0.49
        self.assertFalse(qualifies(better, baseline, 0.5))
        better['per_class']['sad']['f1'] = 0.6
        self.assertFalse(qualifies(better, baseline, 0.6))

    def test_fine_tuning_preserves_batchnorm_and_frozen_convolutions(self):
        from tensorflow import keras
        from scripts.emotion.train_rafdb import configure_fine_tuning
        model = keras.Sequential([keras.layers.Input((20, 20, 3)), keras.layers.Conv2D(4, 3),
                                  keras.layers.BatchNormalization(), keras.layers.GlobalAveragePooling2D(),
                                  keras.layers.Dense(7, activation='softmax')])
        configure_fine_tuning(model, keras, 1)
        before = [value.copy() for layer in model.layers[:-1] for value in layer.get_weights()]
        head_before = model.layers[-1].get_weights()[0].copy()
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy')
        model.train_on_batch(np.ones((3, 20, 20, 3), np.float32), np.array([1, 1, 4]))
        after = [value for layer in model.layers[:-1] for value in layer.get_weights()]
        for first, second in zip(before, after):
            np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(head_before, model.layers[-1].get_weights()[0]))

    def test_fusion_selection_keeps_baseline_for_missing_blendshapes(self):
        from scripts.emotion.train_fusion import fit_correction
        labels = np.tile(np.arange(7), 3)
        probabilities = np.full((len(labels), 7), 0.1)
        probabilities[:, 6] = 0.4
        x = np.concatenate((probabilities, np.eye(7)[labels]), axis=1)
        result = fit_correction(x, probabilities, labels, x, probabilities, labels,
                                epochs=20, learning_rate=0.1, val_mask=np.zeros(len(labels), dtype=bool))
        self.assertEqual(result['best_epoch'], 0)
        self.assertEqual(result['baseline_validation'], result['best_validation'])

    def test_batched_evaluation_keeps_test_images_without_blendshapes(self):
        from scripts.emotion import evaluate_rafdb
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for split_index, split in enumerate(('train', 'validation', 'test')):
                for index, label in enumerate(LABELS):
                    path = root / f'{split}_{label}.png'
                    write_image(path, np.full((30, 40, 3), (20 + split_index, 60 + index, 130), np.uint8))
                    rows.append({'path': path.name, 'label': label, 'subject': None,
                                 'dataset': 'rafdb', 'split_basis': 'official-rafdb', 'split': split,
                                 'crop_margin': 0.15, 'align': True, 'crop_version': CROP_VERSION,
                                 'blendshapes': {'jawOpen': 0.3} if index else None})
            manifest = root / 'manifest.jsonl'
            manifest.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            model = root / 'efficientnet.keras'
            model.write_bytes(b'checkpoint')
            fusion_file = root / 'fusion.json'
            fusion_file.write_text(json.dumps({
                'version': 1, 'labels': LABELS, 'base_model': 'efficientnet', 'base_sha256': file_hash(model),
                'blendshape_names': ['jawOpen'], 'mean': [0] * 8, 'scale': [1] * 8,
                'weights': np.zeros((8, 7)).tolist(), 'bias': [0] * 7,
                'crop_config': {'margin': 0.15, 'align': True, 'version': CROP_VERSION},
                'train_subjects': [], 'train_image_hashes': []}), encoding='utf-8')

            def inference(images):
                values = np.zeros((len(images), 7), np.float32)
                values[:, 6] = 1
                return SimpleNamespace(numpy=lambda: values)

            output = root / 'report'
            with patch.object(evaluate_rafdb, 'MODEL_DIR', str(root)), \
                    patch.object(evaluate_rafdb, '_get_inference', return_value=inference):
                evaluate_rafdb.main(['--manifest', str(manifest), '--split', 'test', '--fusion', str(fusion_file),
                                     '--output', str(output)])
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['efficientnet_fusion']['fusion_coverage'], {'used': 6, 'total': 7, 'fallback': 1})
            self.assertEqual(np.array(report['efficientnet_fusion']['confusion_matrix']).sum(), 7)


if __name__ == '__main__':
    unittest.main()
