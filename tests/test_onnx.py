import os
import sys
import unittest
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import onnxruntime as ort

class TestOnnxOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx_model_path = ROOT_DIR / "models" / "model_optimized.onnx"
        cls.sklearn_model_path = ROOT_DIR / "models" / "baseline_model.joblib"

    def test_onnx_model_exists(self):
        """Verifica se o arquivo do modelo ONNX foi gerado com sucesso."""
        self.assertTrue(self.onnx_model_path.exists())
        self.assertGreater(os.path.getsize(self.onnx_model_path), 1024 * 100)  # > 100 KB

    def test_onnx_session_and_inference(self):
        """Verifica se a sessão ONNX Runtime executa e retorna classes válidas."""
        session = ort.InferenceSession(str(self.onnx_model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        pipe = joblib.load(self.sklearn_model_path)
        vec = pipe.named_steps["vec"]
        clf = pipe.named_steps["clf"]

        text = "Patient admitted with severe chest pain and acute myocardial infarction."
        vec_text = vec.transform([text]).astype(np.float32).toarray()

        # Inferência ONNX
        onnx_pred = session.run([output_name], {input_name: vec_text})[0][0]
        # Inferência Scikit-Learn
        sklearn_pred = clf.predict(vec_text)[0]

        self.assertIn(int(onnx_pred), [1, 2, 3, 4, 5])
        self.assertEqual(int(onnx_pred), int(sklearn_pred))

if __name__ == "__main__":
    unittest.main()
