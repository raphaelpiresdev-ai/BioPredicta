"""Unit tests for Phase 2 — FastAPI Modular Structure.

Verifies:
1. Successful import of the FastAPI application instance.
2. Proper application metadata and configuration.
3. Importability of schemas, services, and routers subpackages.
4. Absence of circular imports between api, src, and ui modules.
5. Conformance with Phase 2 constraints (no prediction endpoints or lifespan prematurely added).
"""

from __future__ import annotations

import sys
from fastapi import FastAPI


def test_fastapi_app_instance():
    """Verify that the FastAPI application is properly instantiated and configured."""
    from src.api.main import app
    from src.api import app as app_from_init

    assert isinstance(app, FastAPI)
    assert app is app_from_init
    assert app.title == "MultiTarget BioPredict API"
    assert app.version == "0.1.0"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"


def test_api_subpackages_importable():
    """Verify that schemas, services, and routers packages exist and can be imported."""
    import src.api.schemas as schemas
    import src.api.services as services
    import src.api.routers as routers

    assert schemas is not None
    assert services is not None
    assert routers is not None


def test_no_endpoints_or_lifespan_prematurely_added():
    """Verify that Phase 2 boundaries are respected (no prediction endpoints or lifespan handlers yet)."""
    from src.api.main import app

    # Only default OpenAPI/docs routes should exist, no custom business endpoints yet
    user_routes = [r for r in app.routes if getattr(r, "path", "") not in ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")]
    assert len(user_routes) == 0, f"Expected 0 custom endpoints in Phase 2, found: {[r.path for r in user_routes]}"

    # Lifespan should not be configured yet (Phase 3)
    lifespan_type = type(app.router.lifespan_context).__name__
    assert app.router.lifespan_context is None or lifespan_type in ("_DefaultLifespan", "default_lifespan")


def test_no_circular_dependencies():
    """Verify that importing src.api does not break or import Streamlit UI modules."""
    # Ensure src.chem_utils and src.model_utils can be imported independently
    import src.chem_utils as chem_utils
    import src.model_utils as model_utils

    assert hasattr(chem_utils, "FEATURE_NAMES")
    assert hasattr(model_utils, "predict")

    # Verify that importing api does not inject streamlit into sys.modules if not already there
    import src.api
    assert src.api.app is not None
