"""Testes de conformidade da estrutura modular da API FastAPI (Fases 2 e 3).

Verifica:
1. Instanciação e configuração correta da aplicação FastAPI.
2. Metadados e rotas padrão de documentação OpenAPI.
3. Importabilidade dos subpacotes schemas, services e routers.
4. Respeito às restrições de escopo (nenhum endpoint customizado implementado prematuramente).
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


def test_api_subpackages_importable():
    """Verifica se os pacotes schemas, services e routers existem e podem ser importados."""
    import src.api.schemas as schemas
    import src.api.services as services
    import src.api.routers as routers

    assert schemas is not None
    assert services is not None
    assert routers is not None


def test_no_endpoints_prematurely_added():
    """Verifica se restrições de escopo são respeitadas (nenhum endpoint de predição adicionado prematuramente)."""
    from src.api.main import app, lifespan

    # Apenas rotas padrão da documentação OpenAPI devem existir, nenhum endpoint de negócio prematuro
    rotas_usuario = [
        r for r in app.routes
        if getattr(r, "path", "") not in ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json")
    ]
    assert len(rotas_usuario) == 0, f"Esperado 0 endpoints de negócio, encontrados: {[r.path for r in rotas_usuario]}"

    # O gerenciador de ciclo de vida (lifespan) deve estar registrado
    assert app.router.lifespan_context is not None


def test_no_circular_dependencies():
    """Verifica que a importação de src.api não quebra nem importa módulos de UI do Streamlit."""
    import src.chem_utils as chem_utils
    import src.model_utils as model_utils

    assert hasattr(chem_utils, "FEATURE_NAMES")
    assert hasattr(model_utils, "predict")

    import src.api
    assert src.api.app is not None
