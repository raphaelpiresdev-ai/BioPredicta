"""Módulo principal da API REST FastAPI do MultiTarget BioPredict.

Configura a instância da aplicação FastAPI com metadados, documentação OpenAPI,
gerenciador de ciclo de vida (lifespan) para carregamento dos modelos científicos
e roteadores HTTP para monitoramento (GET /health) e predição (POST /predict).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router
from src.model_utils import load_model

logger = logging.getLogger(__name__)

# Resolução canônica dos caminhos do projeto e artefatos de modelos
DIRETORIO_RAIZ = Path(__file__).resolve().parents[2]
DIRETORIO_MODELOS = Path(os.getenv("BIOPREDICTA_MODELS_DIR", DIRETORIO_RAIZ / "models"))
CAMINHO_MODELO_A2A = Path(os.getenv("A2A_MODEL_PATH", DIRETORIO_MODELOS / "a2a_model.joblib"))
CAMINHO_MODELO_HERG = Path(os.getenv("HERG_MODEL_PATH", DIRETORIO_MODELOS / "herg_model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerenciador do ciclo de vida da aplicação FastAPI.

    Responsável por:
    1. Carregar os modelos serializados A2A (regressão) e hERG (classificação)
       uma única vez durante a inicialização da API.
    2. Disponibilizar os modelos em `app.state.a2a_model` e `app.state.herg_model`
       para acesso compartilhado durante toda a vida útil da aplicação.
    3. Abortar explicitamente a inicialização caso ocorra falha de arquivo ou desserialização.
    4. Limpar as referências de estado ao encerrar a aplicação.
    """
    logger.info("Iniciando ciclo de vida da API: carregamento dos modelos de Machine Learning...")

    # Verificação explícita de existência dos artefatos
    if not CAMINHO_MODELO_A2A.exists():
        msg_erro = f"Artefato do modelo A2A não encontrado: {CAMINHO_MODELO_A2A}"
        logger.error(msg_erro)
        raise FileNotFoundError(msg_erro)

    if not CAMINHO_MODELO_HERG.exists():
        msg_erro = f"Artefato do modelo hERG não encontrado: {CAMINHO_MODELO_HERG}"
        logger.error(msg_erro)
        raise FileNotFoundError(msg_erro)

    try:
        logger.info("Carregando modelo A2A a partir de %s", CAMINHO_MODELO_A2A)
        a2a_model = load_model(CAMINHO_MODELO_A2A)

        logger.info("Carregando modelo hERG a partir de %s", CAMINHO_MODELO_HERG)
        herg_model = load_model(CAMINHO_MODELO_HERG)
    except Exception as exc:
        msg_erro = f"Falha na desserialização dos modelos de Machine Learning: {exc}"
        logger.error(msg_erro, exc_info=True)
        raise RuntimeError(msg_erro) from exc

    # Armazena os modelos no estado compartilhado da aplicação
    app.state.a2a_model = a2a_model
    app.state.herg_model = herg_model
    logger.info("Modelos A2A e hERG carregados com sucesso no estado da aplicação (app.state).")

    try:
        yield
    finally:
        logger.info("Finalizando ciclo de vida da API: liberando referências aos modelos...")
        app.state.a2a_model = None
        app.state.herg_model = None
        logger.info("Ciclo de vida finalizado com sucesso.")


app = FastAPI(
    title="MultiTarget BioPredict API",
    description=(
        "Plataforma de Triagem Virtual Multiobjetivo (A2A × hERG). "
        "REST API para predição de afinidade ao receptor purinérgico A2A (pIC50) "
        "e avaliação toxicológica de bloqueio do canal de potássio hERG."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Inclusão dos roteadores oficiais de monitoramento e inferência
app.include_router(health_router)
app.include_router(predict_router)
