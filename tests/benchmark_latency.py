import argparse
import json
import os
import sys
import time
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from src.api.main import app

def run_benchmark(num_requests: int = 300, warm_up: int = 20):
    print("=" * 65)
    print(f"BENCHMARK DE TEMPO DE RESPOSTA LOCAL (FASTAPI BASELINE)")
    print("=" * 65)

    test_csv_path = Path("raw_data/medical_tc_test.csv")
    if test_csv_path.exists():
        df_test = pd.read_csv(test_csv_path)
        sample_texts = df_test["medical_abstract"].tolist()[:100]
    else:
        sample_texts = [
            "Patient presented with acute retrosternal chest pressure and dyspnea consistent with myocardial infarction.",
            "Endoscopy and biopsy confirmed active gastric ulcer with Helicobacter pylori colonization.",
            "Histopathological examination of tumor biopsy revealed malignant carcinoma cells."
        ]

    latencies_ms = []
    status_codes = []

    with TestClient(app) as client:
        # 1. Warm-up
        print(f"Executando {warm_up} requisições de aquecimento (warm-up)...")
        for i in range(warm_up):
            text = sample_texts[i % len(sample_texts)]
            client.post("/predict", json={"medical_abstract": text})

        # 2. Benchmark Measurement
        print(f"Iniciando benchmark de {num_requests} requisições individuais síncronas...")
        t_global_start = time.perf_counter()

        for i in range(num_requests):
            text = sample_texts[i % len(sample_texts)]
            payload = {"medical_abstract": text}

            t0 = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            status_codes.append(response.status_code)
            latencies_ms.append(elapsed_ms)

        total_elapsed_s = time.perf_counter() - t_global_start

    latencies = np.array(latencies_ms)
    successful_requests = sum(1 for c in status_codes if c == 200)
    throughput_rps = round(num_requests / total_elapsed_s, 2)

    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "error_rate_pct": round(((num_requests - successful_requests) / num_requests) * 100, 2),
        "total_benchmark_time_s": round(total_elapsed_s, 3),
        "throughput_rps": throughput_rps,
        "latency_metrics_ms": {
            "mean": round(float(np.mean(latencies)), 3),
            "std": round(float(np.std(latencies)), 3),
            "min": round(float(np.min(latencies)), 3),
            "p50_median": round(float(np.percentile(latencies, 50)), 3),
            "p90": round(float(np.percentile(latencies, 90)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
            "max": round(float(np.max(latencies)), 3),
        }
    }

    print("\n" + "-" * 65)
    print("RESULTADOS DO BENCHMARK DE LATÊNCIA:")
    print("-" * 65)
    print(f"Total de requisições:    {stats['total_requests']}")
    print(f"Taxa de sucesso:         {successful_requests}/{num_requests} ({(successful_requests/num_requests)*100:.1f}%)")
    print(f"Taxa de erro:            {stats['error_rate_pct']}%")
    print(f"Throughput (RPS):        {throughput_rps} req/s")
    print(f"Latência Média:          {stats['latency_metrics_ms']['mean']} ms")
    print(f"Latência Mediana (P50):  {stats['latency_metrics_ms']['p50_median']} ms")
    print(f"Latência P90:            {stats['latency_metrics_ms']['p90']} ms")
    print(f"Latência P95:            {stats['latency_metrics_ms']['p95']} ms")
    print(f"Latência P99:            {stats['latency_metrics_ms']['p99']} ms")
    print(f"Latência Mínima/Máxima:  {stats['latency_metrics_ms']['min']} ms / {stats['latency_metrics_ms']['max']} ms")
    print("-" * 65)

    output_path = Path(__file__).resolve().parent / "latency_baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Métricas salvas em: {output_path}")

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=300)
    args = parser.parse_args()
    run_benchmark(num_requests=args.requests)
