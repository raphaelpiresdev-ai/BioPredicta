"""Roteador para inferência preditiva multiobjetivo (POST /predict)."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas.prediction import PredictionRequest, PredictionResponse
from src.api.services.prediction_service import (
    InvalidSmilesError,
    ModelsNotLoadedError,
    run_prediction_pipeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inferência"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predição multiobjetivo para afinidade A2A e bloqueio hERG",
    description=(
        "Recebe uma estrutura molecular representada por código linear SMILES e retorna as estimativas "
        "computacionais de pIC50 (modelo A2A) e probabilidade de bloqueio (modelo hERG) em uma única requisição."
    ),
)
def predict_molecule(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Processa a inferência síncrona multiobjetivo para a molécula submetida."""
    a2a_model = getattr(request.app.state, "a2a_model", None)
    herg_model = getattr(request.app.state, "herg_model", None)

    try:
        return run_prediction_pipeline(
            smiles=payload.smiles,
            a2a_model=a2a_model,
            herg_model=herg_model,
        )
    except InvalidSmilesError as err:
        logger.warning("Requisição rejeitada com HTTP 400 (SMILES inválido): %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except ModelsNotLoadedError as err:
        logger.error("Falha de infraestrutura: modelos não carregados em memória: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Modelos preditivos indisponíveis no servidor.",
        ) from err
    except Exception as exc:
        logger.error("Erro inesperado durante a execução da predição: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocorreu um erro interno durante o processamento da predição.",
        ) from exc

