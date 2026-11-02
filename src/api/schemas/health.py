"""Modelos e contratos de dados para monitoramento operacional da API (Pydantic).

Define os esquemas formais para a rota de health check (GET /health),
apresentando o estado do serviço e o carregamento dos modelos serializados.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelsLoadedStatus(BaseModel):
    """Status de disponibilidade em memória dos modelos preditivos."""

    model_config = ConfigDict(extra="forbid", strict=True)

    a2a: bool = Field(
        ...,
        description="Indica se o modelo de regressão de potência no receptor de adenosina A2A está carregado e pronto para inferência.",
    )
    herg: bool = Field(
        ...,
        description="Indica se o modelo de classificação de bloqueio do canal hERG está carregado e pronto para inferência.",
    )


class HealthResponse(BaseModel):
    """Resposta estruturada do monitoramento de integridade e prontidão da API."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(
        ...,
        description="Estado operacional geral da API ('ok' quando totalmente operacional, 'degraded' se algum modelo estiver ausente).",
        examples=["ok"],
    )
    service: str = Field(
        default="MultiTarget BioPredict API",
        description="Nome do serviço de predição molecular.",
        examples=["MultiTarget BioPredict API"],
    )
    models_loaded: ModelsLoadedStatus = Field(
        ...,
        description="Status individual de carregamento dos modelos serializados.",
    )
    ready_for_inference: bool = Field(
        ...,
        description="Indica se o serviço está plenamente habilitado para processar inferências multiobjetivo.",
        examples=[True],
    )

