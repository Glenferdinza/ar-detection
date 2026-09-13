import sys
import os
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from two_stage_verifier import TwoStageHeroVerifier
from test_media import detect_heroes_fast
from ultralytics import YOLO

class TestTwoStageCascade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = TwoStageHeroVerifier("weights/stage2_classifier.pt", beta_thresh=10.0)
        cls.yolo_model = YOLO("weights/best.pt")

    def test_living_human_shield_rejects_user(self):
        face_path = "scratch/verified_user_face.jpg"
        if not os.path.exists(face_path):
            self.skipTest(f"{face_path} not found")

        img = cv2.imread(face_path)
        valid, cls_id, conf, meta = self.verifier.verify_candidate(img, yolo_cls_id=0, yolo_conf=0.75)
        self.assertFalse(valid, "Living human user face must be rejected by Stage 2 Living Human Shield")
        self.assertEqual(meta.get("reason"), "rejected_as_living_face_or_background")
        self.assertGreater(meta["probs"]["negative"], 0.60)

    def test_phone_exhibit_kartini_accepted(self):
        crop_path = "crop_check.jpg"
        if not os.path.exists(crop_path):
            self.skipTest(f"{crop_path} not found")

        img = cv2.imread(crop_path)
        valid, cls_id, conf, meta = self.verifier.verify_candidate(img, yolo_cls_id=2, yolo_conf=0.55)
        self.assertTrue(valid, "Phone exhibit Kartini crop must be accepted")
        self.assertEqual(cls_id, 2, "Class ID must be 2 (Kartini)")
        self.assertGreater(conf, 0.70)

    def test_detect_heroes_fast_no_false_positive_on_user(self):
        face_path = "scratch/verified_user_face.jpg"
        if not os.path.exists(face_path):
            self.skipTest(f"{face_path} not found")

        img = cv2.imread(face_path)
        boxes = detect_heroes_fast(img, self.yolo_model, verifier=self.verifier, conf_thresh=0.35)
        self.assertEqual(len(boxes), 0, "No bounding box must be returned for living human face")

    def test_diverse_human_visitors_rejected(self):
        import glob
        neg_files = sorted(glob.glob("data/negative_faces/neg_face_*.jpg"))[:12]
        if len(neg_files) == 0:
            self.skipTest("Diverse negative visitor dataset not found")
        for f in neg_files:
            img = cv2.imread(f)
            if img is None:
                continue
            valid, cls_id, conf, meta = self.verifier.verify_candidate(img, yolo_cls_id=0, yolo_conf=0.70)
            self.assertFalse(valid, f"Diverse human visitor {os.path.basename(f)} must be rejected by Living Human Shield")
            if "probs" in meta:
                self.assertGreater(meta["probs"]["negative"], 0.40)

    def test_full_body_aspect_ratio_suppressed(self):
        # Standing human from head to toe with vertical aspect ratio 3.0
        tall_human = np.full((900, 300, 3), 160, dtype=np.uint8)
        boxes = detect_heroes_fast(tall_human, self.yolo_model, verifier=self.verifier, conf_thresh=0.25)
        self.assertEqual(len(boxes), 0, "Full-body vertical standing human must be suppressed by aspect ratio filter")

if __name__ == "__main__":
    unittest.main()
