import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)

class MedicalClassifierService:
    def __init__(self, model_path: str, metadata_path: str, onnx_path: Optional[str] = "models/model_optimized.onnx"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.onnx_path = onnx_path
        self.pipeline = None
        self.onnx_session = None
        self.onnx_input_name: Optional[str] = None
        self.metadata: Dict = {}
        self.labels_map: Dict[int, str] = {}
        self.model_version: str = "unknown"

    def load(self) -> None:
        """Carrega o pipeline serializado, metadados e opcionalmente a sessão ONNX Runtime."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {self.model_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Arquivo de metadados não encontrado em: {self.metadata_path}")

        logger.info(f"Carregando modelo a partir de: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

        logger.info(f"Carregando metadados a partir de: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        raw_map = self.metadata.get("labels_map", {})
        self.labels_map = {int(k): str(v) for k, v in raw_map.items()}
        self.model_version = self.metadata.get("version", "1.0.0")

        # Inicializa ONNX Runtime se disponível
        resolved_onnx = self.onnx_path
        if resolved_onnx and not os.path.isabs(resolved_onnx):
            from src.api.config import ROOT_DIR
            resolved_onnx = str(ROOT_DIR / resolved_onnx)

        if HAS_ONNX and resolved_onnx and os.path.exists(resolved_onnx):
            try:
                logger.info(f"Carregando sessão otimizada ONNX de: {resolved_onnx}")
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                self.onnx_session = ort.InferenceSession(resolved_onnx, sess_options, providers=["CPUExecutionProvider"])
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                logger.info("ONNX Runtime pronto para inferência ultra-rápida.")
            except Exception as e:
                logger.warning(f"Não foi possível inicializar ONNX Runtime: {e}")
                self.onnx_session = None

        # Sanity check interno (aquecimento)
        _ = self.predict("Test clinical warm-up query for medical triage.")
        logger.info("Modelo carregado e validado com sucesso!")

    def is_ready(self) -> bool:
        """Verifica se o modelo está pronto para receber requisições."""
        return self.pipeline is not None

    def is_onnx_ready(self) -> bool:
        """Verifica se o runtime otimizado ONNX está ativo."""
        return self.onnx_session is not None

    def predict(self, text: str) -> Dict:
        """Executa a inferência para um único laudo médico via Scikit-Learn."""
        if not self.is_ready():
            raise RuntimeError("O modelo ainda não foi carregado na memória.")

        t0 = time.perf_counter()
        pred_label = int(self.pipeline.predict([text])[0])
        probabilities_vec = self.pipeline.predict_proba([text])[0]
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        classes = self.pipeline.classes_
        probabilities_dict = {}
        for cls_id, prob in zip(classes, probabilities_vec):
            cls_name = self.labels_map.get(int(cls_id), f"Class {cls_id}")
            probabilities_dict[cls_name] = round(float(prob), 4)

        confidence = round(float(np.max(probabilities_vec)), 4)
        condition_name = self.labels_map.get(pred_label, f"Class {pred_label}")

        return {
            "condition_label": pred_label,
            "condition_name": condition_name,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "inference_time_ms": inference_time_ms
        }

    def predict_onnx(self, text: str) -> Dict:
        """Executa a inferência compilada em C++ via ONNX Runtime (alta performance)."""
        if not self.is_onnx_ready():
            raise RuntimeError("O runtime ONNX não está carregado. Verifique se models/model_optimized.onnx existe.")

        t0 = time.perf_counter()
        vec = self.pipeline.named_steps["vec"]
        X_mat = vec.transform([text]).astype(np.float32).toarray()

        # Execução no grafo ONNX
        outputs = self.onnx_session.run(None, {self.onnx_input_name: X_mat})
        pred_label = int(outputs[0][0])
        prob_map_raw = outputs[1][0]  # dict {1: 0.05, 2: 0.1, ...}
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        probabilities_dict = {
            self.labels_map.get(int(cls_id), f"Class {cls_id}"): round(float(prob), 4)
            for cls_id, prob in prob_map_raw.items()
        }
        confidence = round(float(max(prob_map_raw.values())), 4)
        condition_name = self.labels_map.get(pred_label, f"Class {pred_label}")

        return {
            "condition_label": pred_label,
            "condition_name": condition_name,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "inference_time_ms": inference_time_ms
        }

    def predict_batch(self, texts: List[str]) -> Tuple[List[Dict], float]:
        """Executa inferência em lote vetorizada."""
        if not self.is_ready():
            raise RuntimeError("O modelo ainda não foi carregado na memória.")

        t0 = time.perf_counter()
        pred_labels = self.pipeline.predict(texts)
        probabilities_mat = self.pipeline.predict_proba(texts)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 3)

        classes = self.pipeline.classes_
        results = []
        for label, prob_vec in zip(pred_labels, probabilities_mat):
            lbl_int = int(label)
            prob_dict = {
                self.labels_map.get(int(c), f"Class {c}"): round(float(p), 4)
                for c, p in zip(classes, prob_vec)
            }
            results.append({
                "condition_label": lbl_int,
                "condition_name": self.labels_map.get(lbl_int, f"Class {lbl_int}"),
                "confidence": round(float(np.max(prob_vec)), 4),
                "probabilities": prob_dict,
                "inference_time_ms": round(total_time_ms / len(texts), 3)
            })

        return results, total_time_ms

