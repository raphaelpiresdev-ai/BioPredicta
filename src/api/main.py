"""Main application module for MultiTarget BioPredict FastAPI REST API.

Provides the FastAPI application instance configured with metadata, title,
and OpenAPI documentation paths.
"""

from __future__ import annotations

from fastapi import FastAPI

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
)

