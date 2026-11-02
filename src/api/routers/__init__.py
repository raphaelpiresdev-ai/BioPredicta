"""Subpacote de roteadores (routers) da API MultiTarget BioPredict.

Organiza as rotas HTTP expostas pela API para monitoramento e inferência.
"""

from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
