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

if __name__ == "__main__":
    unittest.main()
