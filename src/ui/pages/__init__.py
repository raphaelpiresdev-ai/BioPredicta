"""
Subpacote de Páginas da Aplicação Multipágina do MultiTarget BioPredict.
"""

from src.ui.pages.p2_nova_analise import render_nova_analise
from src.ui.pages.p3_resultados import render_resultados
from src.ui.pages.p4_interpretacao import render_interpretacao
from src.ui.pages.p6_como_utilizar import render_como_utilizar

__all__ = [
    "render_nova_analise",
    "render_resultados",
    "render_interpretacao",
    "render_como_utilizar",
]
