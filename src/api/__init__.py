"""MultiTarget BioPredict — API Package.

This package provides the FastAPI REST interface for the MultiTarget BioPredict platform.
Subpackages:
- schemas: Request and response data models (Pydantic).
- services: Application services and integration with scientific core.
- routers: HTTP route definitions.
"""

from src.api.main import app

__all__ = ["app"]

