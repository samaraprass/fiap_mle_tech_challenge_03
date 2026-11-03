import json
import random
import time
from pathlib import Path
import urllib.request
import urllib.error

API_URL = "http://localhost:8000"

SAMPLE_REPORTS = [
    "Patient presented with acute substernal chest pressure, radiation to jaw and dyspnea. Electrocardiogram reveals ST elevation in leads V1-V4 consistent with anterior acute myocardial infarction.",
    "Histopathological examination of needle biopsy confirmed malignant invasive ductal carcinoma of the breast with positive lymph node involvement.",
    "Endoscopy revealed severe erosive gastritis with mucosal ulceration and active bleeding in the duodenal bulb.",
    "Patient suffering from focal seizures, persistent headache and left-sided hemiparesis. Brain MRI demonstrates high-grade glioblastoma in right temporal lobe.",
    "Patient presented with fever, malaise, chills and generalized inflammatory response secondary to bacterial bacteremia.",
    "Coronary angiography showed critical 90% stenosis in the right coronary artery requiring immediate percutaneous coronary intervention.",
    "Colonoscopy revealed ulcerative colitis with extensive pseudopolyps and severe mucosal inflammation."
]

def send_request(endpoint: str, payload: dict = None, method: str = "POST"):
    url = f"{API_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if payload else {}

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.reason
    except Exception as e:
        return 0, str(e)

def generate_traffic(total_requests: int = 60, delay_range: tuple = (0.05, 0.25)):
    print(f"Iniciando gerador de tráfego sintético ({total_requests} requisições)...")
    success_count = 0
    error_count = 0

    for i in range(total_requests):
        # 1. 85% requisições válidas de predição
        # 2. 10% requisições de health
        # 3. 5% requisições com erro 422 para testar a taxa de erro no Grafana
        r = random.random()

        if r < 0.85:
            text = random.choice(SAMPLE_REPORTS)
            status, res = send_request("/predict", {"medical_abstract": text})
            if status == 200:
                success_count += 1
                pred_name = res.get("condition_name", "unknown")
                print(f"[{i+1}/{total_requests}] POST /predict -> 200 OK ({pred_name})")
            else:
                error_count += 1
                print(f"[{i+1}/{total_requests}] POST /predict -> {status}")
        elif r < 0.95:
            status, _ = send_request("/health", method="GET")
            success_count += 1
            print(f"[{i+1}/{total_requests}] GET /health -> {status}")
        else:
            # Envia texto curto para gerar HTTP 422 intencional e alimentar o painel de Erro
            status, _ = send_request("/predict", {"medical_abstract": "Curto"})
            error_count += 1
            print(f"[{i+1}/{total_requests}] POST /predict (Erro Intencional 422) -> {status}")

        time.sleep(random.uniform(*delay_range))

    print("-" * 50)
    print(f"Tráfego concluído! Sucessos: {success_count} | Erros gerados: {error_count}")

if __name__ == "__main__":
    generate_traffic(total_requests=60)

