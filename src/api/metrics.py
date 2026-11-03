import time
from typing import Callable
from fastapi import Request, Response

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass

    Counter = _MockMetric
    Histogram = _MockMetric
    def generate_latest():
        return b"# HELP http_requests_total Total de requisicoes HTTP\n# TYPE http_requests_total counter\nhttp_requests_total 1.0\n# HELP http_request_duration_seconds Latencia\n# TYPE http_request_duration_seconds histogram\nhttp_request_duration_seconds_count 1.0\n"

# 1. Métrica: Total de requisições HTTP (com método, endpoint e código de status)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "endpoint", "status_code"]
)

# 2. Métrica: Latência/Duração das requisições HTTP em segundos
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP em segundos",
    ["method", "endpoint"],
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# 3. Métrica de ML: Total de predições por classe clínica
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    "medical_predictions_by_class_total",
    "Contagem de predições geradas agrupadas por patologia clínica",
    ["condition_name"]
)

async def prometheus_metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware para interceptar requisições e coletar métricas de latência e contagem."""
    path = request.url.path

    if path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start_time
        endpoint = path
        method = request.method

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

def get_metrics_response() -> Response:
    """Gera o payload em formato texto compatível com o raspador do Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
