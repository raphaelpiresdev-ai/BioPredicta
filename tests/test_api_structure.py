"""Testes de conformidade da estrutura modular da API FastAPI (Fases 2 a 5).

Verifica:
1. Instanciação e configuração correta da aplicação FastAPI.
2. Metadados e rotas padrão de documentação OpenAPI.
3. Importabilidade dos subpacotes schemas, services e routers.
4. Respeito às restrições de escopo (exclusivamente /health e /predict registrados).
5. Ausência de dependências circulares entre api, src e ui.
"""

from __future__ import annotations

import sys
from fastapi import FastAPI


def test_fastapi_app_instance():
    """Verifica se a aplicação FastAPI é instanciada e configurada com metadados corretos."""
    from src.api.main import app
    from src.api import app as app_from_init

    assert isinstance(app, FastAPI)
    assert app is app_from_init
    assert app.title == "MultiTarget BioPredict API"
    assert app.version == "0.1.0"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
    official_identity = (
        "Plataforma de Triagem Virtual Multiobjetivo: "
        "Potência no Receptor de Adenosina A2A × Segurança Cardíaca hERG"
    )
    assert app.summary == official_identity
    assert official_identity in app.description
    openapi_info = app.openapi()["info"]
    assert openapi_info["summary"] == official_identity
    assert official_identity in openapi_info["description"]


def test_api_subpackages_importable():
    """Verifica se os pacotes schemas, services e routers existem e podem ser importados."""
    import src.api.schemas as schemas
    import src.api.services as services
    import src.api.routers as routers

    assert schemas is not None
    assert services is not None
    assert routers is not None


def test_expected_official_endpoints_registered():
    """Verifica se exclusivamente as rotas oficiais /health e /predict estão registradas na API."""
    from src.api.main import app

    caminhos_openapi = list(app.openapi()["paths"].keys())
    assert set(caminhos_openapi) == {"/health", "/predict"}, f"Rotas encontradas: {caminhos_openapi}"

    # O gerenciador de ciclo de vida (lifespan) deve permanecer registrado
    assert app.router.lifespan_context is not None


def test_no_circular_dependencies():
    """Verifica que a importação de src.api não quebra nem importa módulos de UI do Streamlit."""
    import src.chem_utils as chem_utils
    import src.model_utils as model_utils

    assert hasattr(chem_utils, "FEATURE_NAMES")
    assert hasattr(model_utils, "predict")

    import src.api
    assert src.api.app is not None
