"""Vòng lặp Live đầu-cuối với camera giả: model thật, tracker, hiệu chuẩn và giao diện Streamlit."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
import pipeline
from webcam_data import read_image


class FakeCamera:
    """Trả ảnh mặt thật một số khung rồi đứng yên, như camera bị treo."""
    ok = True
    stopped = False

    def __init__(self, index=0):
        self.image = read_image(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg')
        self.reads = 0

    @property
    def count(self):
        return min(self.reads + 1, 6)

    def read(self):
        self.reads += 1
        return self.image.copy()

    def stop(self):
        FakeCamera.stopped = True


class LiveLoopTests(unittest.TestCase):
    def test_live_loop_records_track_and_stops_when_camera_hangs(self):
        from streamlit.testing.v1 import AppTest
        with patch.object(pipeline, 'CameraThread', FakeCamera):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60)
            app.session_state['live'] = True
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(FakeCamera.stopped)
        rows = app.session_state['rows']
        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual({row['track_id'] for row in rows}, {1})
        self.assertTrue(all(0 < row['quality'] <= 1 for row in rows))
        self.assertTrue(all(row['age_low'] < row['age'] < row['age_high'] for row in rows))
        self.assertTrue(any('Không đọc được khung hình' in item.value for item in app.warning))
        self.assertTrue(np.isfinite([row['gender_prob'] for row in rows]).all())


if __name__ == '__main__':
    unittest.main()
