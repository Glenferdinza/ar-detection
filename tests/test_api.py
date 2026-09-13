import sys
import os
import unittest
import numpy as np
import cv2
import io

sys.path.insert(0, os.path.abspath("src"))

from starlette.testclient import TestClient
from app import app

client = TestClient(app)

class TestARAPI(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")

    def test_detect_valid_artifact(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (80, 60), (560, 420), (180, 180, 180), -1)
        cv2.rectangle(img, (70, 50), (570, 430), (50, 40, 30), 8)
        cv2.putText(img, "SOEKARNO ARTIFACT", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        _, img_encoded = cv2.imencode(".jpg", img)

        response = client.post(
            "/rest_api/last_request",
            files={"file": ("soekarno_sample.jpg", img_encoded.tobytes(), "image/jpeg")},
            data={"usr_id": "mobile_user_01"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("odr_id", data)
        self.assertIn("odr_usr_id", data)
        self.assertIn("odr_status", data)
        self.assertIn("assessment_beta", data)

    def test_reject_blurry_artifact(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", blank)

        response = client.post(
            "/rest_api/last_request",
            files={"file": ("blurry_sample.jpg", img_encoded.tobytes(), "image/jpeg")},
            data={"usr_id": "mobile_user_02"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["odr_status"], "REJECTED_QUALITY")
        self.assertFalse(data["assessment_beta"]["is_passed"])

    def test_reject_visitor_face(self):
        face_path = "scratch/verified_user_face.jpg"
        if not os.path.exists(face_path):
            self.skipTest("scratch/verified_user_face.jpg not found")

        with open(face_path, "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/rest_api/last_request",
            files={"file": ("user_visitor.jpg", file_bytes, "image/jpeg")},
            data={"usr_id": "visitor_99"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["odr_status"], "NOT_DETECTED")

if __name__ == "__main__":
    unittest.main()
