"""Pacote da API REST MultiTarget BioPredict.

Fornece a interface REST desenvolvida em FastAPI para a plataforma MultiTarget BioPredict.
Subpacotes:
- schemas: Modelos e contratos de dados de entrada e saída (Pydantic).
- services: Serviços de orquestração e integração com o núcleo científico.
- routers: Definições de rotas e controladores HTTP.
"""

from src.api.main import app

__all__ = ["app"]
