# syntax=docker/dockerfile:1
# ------------------------------------------------------------------------------
# Dockerfile - Medical Triage API (FastAPI + Scikit-Learn NLP)
# Fase 3: Tech Challenge - FIAP MLE
# ------------------------------------------------------------------------------

FROM python:3.12-slim AS base

# Configurações de ambiente para Python em container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Instalação de dependências do sistema necessárias para compilação e healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python com cache otimizado
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Criação de usuário não-root por segurança (Best Practice de Produção)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Cópia do código-fonte, modelos e configurações
COPY config.yaml .
COPY models/ ./models/
COPY src/ ./src/

# Ajuste de permissões de leitura/execução para o usuário não-root
RUN chown -R appuser:appgroup /app

USER appuser

# Porta exposta pela aplicação
EXPOSE 8000

# Healthcheck interno do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização com Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]

