import sys
import os
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("src"))

from preprocessor import ArtefactPreprocessor
from augmentor import MuseumAugmentor

class TestARComponents(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ArtefactPreprocessor(beta_passing_grade=50.0, target_size=(640, 640))
        self.augmentor = MuseumAugmentor(target_size=(640, 640))

        self.synthetic_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.synthetic_img, (100, 80), (540, 400), (200, 200, 200), -1)
        cv2.rectangle(self.synthetic_img, (80, 60), (560, 420), (50, 50, 50), 5)
        cv2.putText(self.synthetic_img, "TEST ARTIFACT", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    def test_preprocessor_quality_assessment(self):
        passed, details = self.preprocessor.assess_quality(self.synthetic_img)
        self.assertIn("beta_score", details)
        self.assertIn("laplacian_var", details)
        self.assertGreater(details["beta_score"], 0)

    def test_preprocessor_standardize(self):
        standardized, meta = self.preprocessor.standardize(self.synthetic_img)
        self.assertEqual(standardized.shape, (640, 640, 3))
        self.assertIn("scale", meta)

    def test_augmentor_clahe(self):
        enhanced = self.augmentor.apply_clahe(self.synthetic_img)
        self.assertEqual(enhanced.shape, self.synthetic_img.shape)

    def test_augmentor_glare(self):
        glared = self.augmentor.add_museum_glare(self.synthetic_img)
        self.assertEqual(glared.shape, self.synthetic_img.shape)

    def test_augmentor_rotation_box(self):
        initial_box = (0.5, 0.5, 0.5, 0.5)
        rot_img, rot_box = self.augmentor.rotate_with_box(self.synthetic_img, initial_box, 15.0)
        self.assertIsNotNone(rot_box)
        bx, by, bw, bh = rot_box
        self.assertTrue(0.0 <= bx <= 1.0)
        self.assertTrue(0.0 <= by <= 1.0)
        self.assertTrue(0.0 <= bw <= 1.0)
        self.assertTrue(0.0 <= bh <= 1.0)

    def test_preprocessor_archival_monochrome(self):
        # Grayscale antique photo simulation
        sepia_img = np.full((400, 400, 3), 140, dtype=np.uint8)
        cv2.putText(sepia_img, "ARCHIVE 1919", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 3)
        passed, details = self.preprocessor.assess_quality(sepia_img)
        self.assertTrue(details.get("is_archival_monochrome"))
        self.assertIn("tenengrad_score", details)
        self.assertTrue(passed)

    def test_mobile_ar_session_smoothing(self):
        sys.path.insert(0, os.path.abspath("model_package"))
        from infer_example import MobileARSession

        class MockEngine:
            def predict(self, frame, conf_thresh=0.45):
                return [{
                    "class_id": 2,
                    "hero_name": "R.A. Kartini",
                    "confidence": 0.88,
                    "bbox": [50, 60, 200, 220]
                }]

        session = MobileARSession(MockEngine(), detect_interval=3, ema_alpha=0.60)
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)

        r1 = session.process_frame(dummy)
        self.assertEqual(len(r1), 1)
        self.assertEqual(r1[0]["hero_name"], "R.A. Kartini")

        # In-between frames should preserve locked detection
        r2 = session.process_frame(dummy)
        self.assertEqual(len(r2), 1)
        self.assertEqual(r2[0]["bbox"], [50, 60, 200, 220])

if __name__ == "__main__":
    unittest.main()

