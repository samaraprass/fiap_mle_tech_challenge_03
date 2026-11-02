from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.config import settings
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
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inferência"],
    summary="Triagem e classificação de um laudo médico individual"
)
async def predict(payload: PredictionRequest):
    """Recebe o texto de um laudo médico (`medical_abstract`) e retorna a patologia clínica prevista com grau de confiança."""
    if not classifier_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo de Machine Learning não está disponível no momento."
        )

    try:
        result = classifier_service.predict(payload.medical_abstract)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Erro durante a predição unitária: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna ao processar laudo: {str(e)}"
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

