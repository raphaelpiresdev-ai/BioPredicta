"""Subpacote de esquemas (schemas) da API MultiTarget BioPredict.

Fornece os modelos e contratos de dados tipados (Pydantic) para validação
estrita de entrada, saída e monitoramento na interface REST da plataforma.
"""

from src.api.schemas.health import HealthResponse, ModelsLoadedStatus
from src.api.schemas.prediction import PredictionRequest, PredictionResponse

__all__ = [
    "HealthResponse",
    "ModelsLoadedStatus",
    "PredictionRequest",
    "PredictionResponse",
]
