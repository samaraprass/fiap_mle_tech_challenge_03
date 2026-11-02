import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app

class TestMedicalApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 999)
        data = response.json()
        self.assertIn("app", data)
        self.assertEqual(data["status"], "online")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_version"], "1.1.0")
        self.assertIn("timestamp", data)

    def test_predict_cardiovascular(self):
        payload = {
            "medical_abstract": (
                "Patient presented with acute retrosternal chest pressure and dyspnea. "
                "Coronary angiography showed occlusion of the left anterior descending artery. "
                "Diagnosis confirmed as acute myocardial infarction."
            )
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["condition_label"], 4)
        self.assertEqual(data["condition_name"], "cardiovascular diseases")
        self.assertGreater(data["confidence"], 0.5)
        self.assertEqual(len(data["probabilities"]), 5)
        self.assertGreater(data["inference_time_ms"], 0)

    def test_predict_neoplasms(self):
        payload = {
            "medical_abstract": (
                "Histopathological examination of the biopsy revealed malignant carcinoma cells with "
                "metastasis to regional lymph nodes. The tumor showed high mitotic activity."
            )
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["condition_label"], 1)
        self.assertEqual(data["condition_name"], "neoplasms")
        self.assertGreater(data["confidence"], 0.5)

    def test_predict_validation_short_text(self):
        payload = {"medical_abstract": "Too short"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_predict_validation_missing_field(self):
        payload = {}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_predict_batch(self):
        payload = {
            "items": [
                {
                    "medical_abstract": (
                        "Patient with acute myocardial infarction, persistent chest pain and ischemic ECG changes."
                    )
                },
                {
                    "medical_abstract": (
                        "Endoscopic biopsy confirmed active gastric ulcer with Helicobacter pylori colonization and bleeding."
                    )
                }
            ]
        }
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_items"], 2)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertGreater(data["total_time_ms"], 0)
        self.assertEqual(data["predictions"][0]["condition_label"], 4)

if __name__ == "__main__":
    unittest.main()
