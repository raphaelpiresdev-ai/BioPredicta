"""Pacote da API REST do projeto MultiTarget BioPredict.

Plataforma de Triagem Virtual Multiobjetivo: Potência no Receptor de Adenosina A2A × Segurança Cardíaca hERG.

Fornece a interface programática REST desenvolvida em FastAPI.
Subpacotes:
- schemas: Modelos e contratos de dados de entrada e saída (Pydantic).
- services: Serviços de orquestração e integração com o núcleo científico.
- routers: Definições de rotas e controladores HTTP.
"""

from src.api.main import app

__all__ = ["app"]
