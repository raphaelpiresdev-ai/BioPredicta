"""Roteador para monitoramento operacional da API (GET /health)."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from src.api.schemas.health import HealthResponse, ModelsLoadedStatus

router = APIRouter(tags=["Monitoramento"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Verificação de integridade operacional do serviço",
    description=(
        "Retorna o status operacional da API e a disponibilidade dos modelos de Machine "
        "Learning em memória. Retorna sempre HTTP 200 com informações estruturadas de prontidão."
    ),
)
def get_health(request: Request) -> HealthResponse:
    """Verifica se a API está respondendo e se os modelos A2A e hERG estão carregados em app.state."""
    a2a_carregado = getattr(request.app.state, "a2a_model", None) is not None
    herg_carregado = getattr(request.app.state, "herg_model", None) is not None
    pronto_inferencia = a2a_carregado and herg_carregado

    estado_geral = "ok" if pronto_inferencia else "degraded"

    return HealthResponse(
        status=estado_geral,
        service="MultiTarget BioPredict API",
        models_loaded=ModelsLoadedStatus(
            a2a=a2a_carregado,
            herg=herg_carregado,
        ),
        ready_for_inference=pronto_inferencia,
    )

