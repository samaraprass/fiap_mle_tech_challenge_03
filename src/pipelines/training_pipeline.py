import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger("medical_training_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Lista curada de stopwords clínicas
CLINICAL_STOPWORDS = list(ENGLISH_STOP_WORDS.union({
    'patient', 'patients', 'case', 'cases', 'study', 'studies', 'results',
    'result', 'conclusion', 'conclusions', 'method', 'methods', 'treated',
    'treatment', 'underwent', 'using', 'used', 'clinical', 'associated',
    'group', 'data', 'years', 'age', 'year', 'performed', 'evaluated'
}))

def ingest_and_validate_data(train_path: str, labels_path: str) -> Dict:
    """Task 1: Ingestão e validação da integridade dos dados brutos."""
    logger.info(f"Iniciando ingestão de dados a partir de: {train_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado em: {train_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado em: {labels_path}")

    df_train = pd.read_csv(train_path)
    df_labels = pd.read_csv(labels_path)

    # Validação de schema
    required_cols = {"condition_label", "medical_abstract"}
    if not required_cols.issubset(df_train.columns):
        raise ValueError(f"Dataset inválido! Colunas necessárias: {required_cols}. Encontradas: {set(df_train.columns)}")

    # Validação de dados nulos
    null_counts = df_train[list(required_cols)].isnull().sum().to_dict()
    if any(count > 0 for count in null_counts.values()):
        logger.warning(f"Valores nulos detectados: {null_counts}. Realizando limpeza...")
        df_train = df_train.dropna(subset=list(required_cols))

    total_samples = len(df_train)
    if total_samples < 500:
        raise ValueError(f"Quantidade insuficiente de laudos para treino: {total_samples}")

    class_distribution = df_train["condition_label"].value_counts().to_dict()
    logger.info(f"Ingestão concluída com sucesso! Total de amostras válidas: {total_samples}")

    return {
        "status": "validated",
        "total_samples": total_samples,
        "class_distribution": class_distribution,
        "train_path": train_path,
        "labels_path": labels_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def train_and_evaluate_model(
    train_path: str,
    test_path: str,
    labels_path: str,
    min_f1_threshold: float = 0.50
) -> Tuple[Pipeline, Dict]:
    """Task 2: Pré-processamento, treinamento e validação com Quality Gate."""
    logger.info("Iniciando treinamento do pipeline de NLP...")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_labels = pd.read_csv(labels_path)

    label_map = dict(zip(df_labels["condition_label"], df_labels["condition_name"]))

    # Pipeline Scikit-Learn de Alta Performance
    pipeline = Pipeline([
        ('vec', TfidfVectorizer(
            min_df=2,
            max_df=0.7,
            stop_words=CLINICAL_STOPWORDS,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

    t0 = time.time()
    pipeline.fit(df_train["medical_abstract"], df_train["condition_label"])
    train_duration_s = round(time.time() - t0, 2)
    logger.info(f"Treinamento finalizado em {train_duration_s} segundos.")

    # Avaliação no conjunto de teste
    preds = pipeline.predict(df_test["medical_abstract"])
    acc = round(float(accuracy_score(df_test["condition_label"], preds)), 4)
    f1_m = round(float(f1_score(df_test["condition_label"], preds, average="macro")), 4)
    f1_w = round(float(f1_score(df_test["condition_label"], preds, average="weighted")), 4)

    logger.info(f"Métricas obtidas: Acurácia={acc} | F1-Macro={f1_m} | F1-Weighted={f1_w}")

    # Quality Gate (critério de promoção)
    if f1_m < min_f1_threshold:
        raise ValueError(
            f"Quality Gate falhou! F1-Macro obtido ({f1_m}) inferior ao mínimo exigido ({min_f1_threshold})"
        )

    eval_metrics = {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "train_duration_s": train_duration_s,
        "quality_gate_passed": True,
        "labels_map": {int(k): v for k, v in label_map.items()}
    }

    return pipeline, eval_metrics

def save_and_register_model(
    model: Pipeline,
    metrics: Dict,
    model_output_path: str = "models/baseline_model.joblib",
    metadata_output_path: str = "models/classes_mapping.json",
    version: str = "1.2.0"
) -> Dict:
    """Task 3: Serialização e registro de metadados do modelo homologado."""
    logger.info(f"Persistindo modelo treinado em: {model_output_path}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)

    # 1. Salva pipeline joblib
    joblib.dump(model, model_output_path)
    file_size_kb = round(os.path.getsize(model_output_path) / 1024, 1)

    # 2. Salva metadados operacionais
    metadata = {
        "model_name": "Tfidf_LogisticRegression_Balanced",
        "version": version,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels_map": metrics.get("labels_map", {}),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "train_duration_s": metrics["train_duration_s"]
        },
        "artifact_size_kb": file_size_kb
    }

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"Modelo e metadados registrados com sucesso! Tamanho: {file_size_kb} KB")
    return metadata

