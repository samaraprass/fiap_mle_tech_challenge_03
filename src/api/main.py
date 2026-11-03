from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.metrics import (
    PREDICTIONS_BY_CLASS_TOTAL,
    get_metrics_response,
    prometheus_metrics_middleware
)
from src.api.model import MedicalClassifierService
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse
)

logging.basicConfig(
    level=settings.app.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("medical_api")

# Instância do serviço de classificação
classifier_service = MedicalClassifierService(
    model_path=settings.model.model_path,
    metadata_path=settings.model.metadata_path
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Gerencia o ciclo de vida da aplicação (carga inicial do modelo e encerramento)."""
    logger.info("Inicializando Medical Triage API...")
    try:
        classifier_service.load()
        logger.info(f"Modelo v{classifier_service.model_version} pronto para inferência.")
    except Exception as e:
        logger.error(f"Falha crítica ao carregar o modelo de ML: {str(e)}", exc_info=True)
        # O modelo pode tentar ser recarregado posteriormente ou tratado no healthcheck
    yield
    logger.info("Encerrando Medical Triage API.")

app = FastAPI(
    title=settings.app.title,
    version=settings.app.version,
    description=settings.app.description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware de instrumentação do Prometheus
app.middleware("http")(prometheus_metrics_middleware)

@app.get("/", tags=["Info"])
async def root():
    """Endpoint raiz com informações operacionais e atalho para a documentação interativa."""
    return {
        "app": settings.app.title,
        "version": settings.app.version,
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Observabilidade"],
    summary="Verificação de integridade e prontidão da API e modelo"
)
async def health():
    """Retorna o estado de saúde da aplicação e confirmação de prontidão do modelo em memória."""
    is_ready = classifier_service.is_ready()
    status_str = "healthy" if is_ready else "degraded"
    
    return HealthResponse(
        status=status_str,
        version=settings.app.version,
        model_loaded=is_ready,
        model_version=classifier_service.model_version if is_ready else None,
        onnx_loaded=classifier_service.is_onnx_ready(),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get(
    "/metrics",
    tags=["Observabilidade"],
    summary="Endpoint de métricas de telemetria no formato Prometheus"
)
async def metrics():
    """Exporta métricas de latência, requisições HTTP e predições para raspagem do Prometheus."""
    return get_metrics_response()

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inferência"],
    summary="Triagem e classificação de um laudo médico individual"
)
async def predict(
    payload: PredictionRequest,
    engine: str = Query("sklearn", description="Motor de inferência: 'sklearn' (padrão) ou 'onnx' (otimizado)")
):
    """Recebe o texto de um laudo médico (`medical_abstract`) e retorna a patologia clínica prevista com grau de confiança."""
    if not classifier_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo de Machine Learning não está disponível no momento."
        )

    try:
        if engine.lower() == "onnx":
            if not classifier_service.is_onnx_ready():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Motor ONNX não está carregado ou arquivo models/model_optimized.onnx ausente."
                )
            result = classifier_service.predict_onnx(payload.medical_abstract)
        else:
            result = classifier_service.predict(payload.medical_abstract)

        PREDICTIONS_BY_CLASS_TOTAL.labels(condition_name=result["condition_name"]).inc()
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro durante a predição unitária: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna ao processar laudo: {str(e)}"
        )

@app.post(
    "/predict/onnx",
    response_model=PredictionResponse,
    tags=["Inferência Otimizada (ONNX)"],
    summary="Triagem de laudo com motor ONNX Runtime compilado em C++ (baixa latência)"
)
async def predict_onnx(payload: PredictionRequest):
    """Executa a inferência direta utilizando o grafo ONNX Runtime acelerado em C++."""
    if not classifier_service.is_onnx_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Motor ONNX não está carregado ou arquivo models/model_optimized.onnx ausente."
        )

    try:
        result = classifier_service.predict_onnx(payload.medical_abstract)
        PREDICTIONS_BY_CLASS_TOTAL.labels(condition_name=result["condition_name"]).inc()
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Erro durante inferência ONNX: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna no motor ONNX: {str(e)}"
        )

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inferência"],
    summary="Classificação em lote de múltiplos laudos médicos"
)
async def predict_batch(payload: BatchPredictionRequest):
    """Processa uma lista de laudos de forma vetorizada, otimizando o throughput para processamento em lote."""
    if not classifier_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo de Machine Learning não está disponível no momento."
        )

    try:
        texts = [item.medical_abstract for item in payload.items]
        predictions, total_time_ms = classifier_service.predict_batch(texts)
        for p in predictions:
            PREDICTIONS_BY_CLASS_TOTAL.labels(condition_name=p["condition_name"]).inc()
        
        response_items = [PredictionResponse(**p) for p in predictions]
        return BatchPredictionResponse(
            total_items=len(response_items),
            total_time_ms=total_time_ms,
            predictions=response_items
        )
    except Exception as e:
        logger.error(f"Erro durante a predição em lote: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna ao processar lote: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False
    )

