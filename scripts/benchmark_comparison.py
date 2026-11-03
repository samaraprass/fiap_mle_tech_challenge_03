import json
import logging
import os
import sys
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_comparison")

def run_comparative_benchmark(num_samples: int = 500, warm_up: int = 25):
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPARATIVO OFICIAL: SCIKIT-LEARN vs. ONNX RUNTIME")
    logger.info("=" * 70)

    sklearn_path = ROOT_DIR / "models" / "baseline_model.joblib"
    onnx_path = ROOT_DIR / "models" / "model_optimized.onnx"
    test_csv_path = ROOT_DIR / "raw_data" / "medical_tc_test.csv"

    if not sklearn_path.exists():
        raise FileNotFoundError(f"Modelo Scikit-Learn não encontrado: {sklearn_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Modelo ONNX não encontrado: {onnx_path}")

    # 1. Carregar modelos
    logger.info("Carregando pipeline Scikit-Learn...")
    pipe = joblib.load(sklearn_path)
    vec = pipe.named_steps["vec"]
    clf = pipe.named_steps["clf"]

    logger.info("Carregando sessão otimizada do ONNX Runtime (CPUExecutionProvider)...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. Carregar amostras de laudos clínicos reais
    df_test = pd.read_csv(test_csv_path)
    sample_texts = df_test["medical_abstract"].tolist()[:num_samples]
    total_eval = len(sample_texts)
    logger.info(f"Total de laudos selecionados para o benchmark: {total_eval}")

    # Pré-vetorização para teste isolado de inferência matricial do classificador
    logger.info("Pré-computando matrizes esparsas de TF-IDF...")
    X_tfidf = vec.transform(sample_texts).astype(np.float32).toarray()

    # 3. Warm-up
    logger.info(f"Executando {warm_up} iterações de aquecimento (warm-up)...")
    for i in range(warm_up):
        row = X_tfidf[i:i+1]
        _ = clf.predict_proba(row)
        _ = session.run([output_name], {input_name: row})

    # 4. Benchmark Scikit-Learn
    logger.info(f"Medindo latência Scikit-Learn ({total_eval} execuções)...")
    sklearn_latencies = []
    sklearn_preds = []

    t0_global_sk = time.perf_counter()
    for i in range(total_eval):
        row = X_tfidf[i:i+1]
        t0 = time.perf_counter()
        pred = clf.predict(row)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sklearn_latencies.append(elapsed_ms)
        sklearn_preds.append(int(pred))
    total_time_sk = time.perf_counter() - t0_global_sk

    # 5. Benchmark ONNX Runtime
    logger.info(f"Medindo latência ONNX Runtime ({total_eval} execuções)...")
    onnx_latencies = []
    onnx_preds = []

    t0_global_onnx = time.perf_counter()
    for i in range(total_eval):
        row = X_tfidf[i:i+1]
        t0 = time.perf_counter()
        pred = session.run([output_name], {input_name: row})[0][0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        onnx_latencies.append(elapsed_ms)
        onnx_preds.append(int(pred))
    total_time_onnx = time.perf_counter() - t0_global_onnx

    # 6. Cálculo das métricas estatísticas
    sk_arr = np.array(sklearn_latencies)
    onnx_arr = np.array(onnx_latencies)

    # Paridade de predição
    matches = sum(1 for p_sk, p_ox in zip(sklearn_preds, onnx_preds) if p_sk == p_ox)
    parity_rate_pct = round((matches / total_eval) * 100, 2)

    mean_sk = float(np.mean(sk_arr))
    mean_onnx = float(np.mean(onnx_arr))
    speedup = round(mean_sk / mean_onnx, 2)
    latency_reduction_pct = round(((mean_sk - mean_onnx) / mean_sk) * 100, 2)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_samples": total_eval,
        "parity_accuracy_pct": parity_rate_pct,
        "speedup_factor": speedup,
        "latency_reduction_pct": latency_reduction_pct,
        "scikit_learn": {
            "mean_ms": round(mean_sk, 4),
            "p50_median_ms": round(float(np.percentile(sk_arr, 50)), 4),
            "p90_ms": round(float(np.percentile(sk_arr, 90)), 4),
            "p95_ms": round(float(np.percentile(sk_arr, 95)), 4),
            "p99_ms": round(float(np.percentile(sk_arr, 99)), 4),
            "min_ms": round(float(np.min(sk_arr)), 4),
            "max_ms": round(float(np.max(sk_arr)), 4),
            "throughput_rps": round(total_eval / total_time_sk, 1)
        },
        "onnx_runtime": {
            "mean_ms": round(mean_onnx, 4),
            "p50_median_ms": round(float(np.percentile(onnx_arr, 50)), 4),
            "p90_ms": round(float(np.percentile(onnx_arr, 90)), 4),
            "p95_ms": round(float(np.percentile(onnx_arr, 95)), 4),
            "p99_ms": round(float(np.percentile(onnx_arr, 99)), 4),
            "min_ms": round(float(np.min(onnx_arr)), 4),
            "max_ms": round(float(np.max(onnx_arr)), 4),
            "throughput_rps": round(total_eval / total_time_onnx, 1)
        }
    }

    # Impressão formatada
    print("\n" + "=" * 70)
    print("QUADRO COMPARATIVO CONSOLIDADO (ETAPA 4):")
    print("=" * 70)
    print(f"Amostras avaliadas:           {total_eval} laudos clínicos")
    print(f"Paridade de Predição:         {parity_rate_pct}% (predições idênticas)")
    print(f"Ganho de Velocidade (Speedup): {speedup}x mais rápido")
    print(f"Redução de Latência:          {latency_reduction_pct}%")
    print("-" * 70)
    print(f"{'Métrica':<25} | {'Scikit-Learn':<18} | {'ONNX Runtime':<18}")
    print("-" * 70)
    print(f"{'Latência Média':<25} | {results['scikit_learn']['mean_ms']:>14.3f} ms | {results['onnx_runtime']['mean_ms']:>14.3f} ms")
    print(f"{'Latência P50 (Mediana)':<25} | {results['scikit_learn']['p50_median_ms']:>14.3f} ms | {results['onnx_runtime']['p50_median_ms']:>14.3f} ms")
    print(f"{'Latência P90':<25} | {results['scikit_learn']['p90_ms']:>14.3f} ms | {results['onnx_runtime']['p90_ms']:>14.3f} ms")
    print(f"{'Latência P95':<25} | {results['scikit_learn']['p95_ms']:>14.3f} ms | {results['onnx_runtime']['p95_ms']:>14.3f} ms")
    print(f"{'Latência P99':<25} | {results['scikit_learn']['p99_ms']:>14.3f} ms | {results['onnx_runtime']['p99_ms']:>14.3f} ms")
    print(f"{'Throughput Máximo':<25} | {results['scikit_learn']['throughput_rps']:>12.1f} req/s | {results['onnx_runtime']['throughput_rps']:>12.1f} req/s")
    print("=" * 70)

    # Persistência
    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_file = docs_dir / "latency_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Resultados consolidados gravados em: {out_file}")

    return results

if __name__ == "__main__":
    run_comparative_benchmark()
