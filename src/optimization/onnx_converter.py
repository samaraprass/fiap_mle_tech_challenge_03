import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger("onnx_optimizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def export_model_to_onnx(
    sklearn_model_path: str = "models/baseline_model.joblib",
    onnx_output_path: str = "models/model_optimized.onnx",
    target_opset: int = 15
) -> Dict:
    """Converte o classificador do pipeline Scikit-Learn para o formato otimizado ONNX Runtime."""
    logger.info(f"Carregando pipeline Scikit-Learn de: {sklearn_model_path}")
    if not os.path.exists(sklearn_model_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {sklearn_model_path}")

    pipeline = joblib.load(sklearn_model_path)
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]

    n_features = len(vec.vocabulary_)
    logger.info(f"Dimensão do vocabulário TF-IDF: {n_features} features")

    # Define o tipo de entrada do grafo ONNX: tensor de float32 [batch_size, n_features]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    logger.info(f"Convertendo estimador {clf.__class__.__name__} para ONNX (Opset {target_opset})...")
    t0 = time.time()
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=target_opset)
    conversion_time = round(time.time() - t0, 3)

    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    with open(onnx_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    file_size_kb = round(os.path.getsize(onnx_output_path) / 1024, 2)
    logger.info(f"Modelo ONNX gerado com sucesso em: {onnx_output_path} ({file_size_kb} KB)")

    # Validação do grafo ONNX
    onnx.checker.check_model(onnx_model)
    logger.info("Grafo ONNX verificado e aprovado pelo ONNX Checker.")

    # Sanity check com ONNX Runtime
    session = ort.InferenceSession(onnx_output_path, providers=["CPUExecutionProvider"])
    dummy_input = np.zeros((1, n_features), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: dummy_input})
    logger.info(f"Sanity check ONNX Runtime executado com sucesso: classe predita dummy = {out[0]}")

    metadata = {
        "source_model": sklearn_model_path,
        "onnx_model": onnx_output_path,
        "n_features": n_features,
        "target_opset": target_opset,
        "conversion_time_s": conversion_time,
        "file_size_kb": file_size_kb,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return metadata

if __name__ == "__main__":
    export_model_to_onnx()
