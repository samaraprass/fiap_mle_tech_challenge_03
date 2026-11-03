from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy", description="Status operacional do serviço")
    version: str = Field(..., example="1.0.0", description="Versão da API")
    model_loaded: bool = Field(..., example=True, description="Indica se o modelo está carregado em memória")
    model_version: Optional[str] = Field(None, example="1.1.0", description="Versão do modelo carregado")
    onnx_loaded: Optional[bool] = Field(False, example=True, description="Indica se o runtime otimizado ONNX está disponível")
    timestamp: str = Field(..., description="Timestamp ISO da verificação")

class PredictionRequest(BaseModel):
    medical_abstract: str = Field(
        ...,
        min_length=15,
        max_length=20000,
        example="Patient presents with acute substernal chest pain radiating to the left arm and dyspnea. Electrocardiogram reveals ST-segment elevation consistent with acute myocardial infarction.",
        description="Texto completo do laudo médico ou resumo clínico para triagem"
    )

class PredictionResponse(BaseModel):
    condition_label: int = Field(..., example=4, description="ID numérico da condição clínica predita (1 a 5)")
    condition_name: str = Field(..., example="cardiovascular diseases", description="Nome descritivo da patologia/condição")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.8845, description="Probabilidade calibrada da classe predita")
    probabilities: Dict[str, float] = Field(..., description="Distribuição de probabilidades entre todas as classes")
    inference_time_ms: float = Field(..., example=1.42, description="Tempo de inferência do modelo em milissegundos")

class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de laudos médicos para classificação em lote"
    )

class BatchPredictionResponse(BaseModel):
    total_items: int = Field(..., example=2, description="Quantidade total de laudos processados")
    total_time_ms: float = Field(..., example=3.25, description="Tempo total de processamento em milissegundos")
    predictions: List[PredictionResponse] = Field(..., description="Resultados das predições individuais")

