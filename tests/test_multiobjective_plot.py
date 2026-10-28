"""
Suíte de testes de regressão da Fase 3B — Geometria e Limiares do Gráfico Multiobjetivo (A2A × hERG).

Verifica:
1. Identificação explícita de pIC50 no eixo X e 1 - P(hERG) no eixo Y.
2. Presença e posição exata das 4 linhas de referência (x = 5.0, x = 6.5, y = 0.30, y = 0.70).
3. Sombreamento em coordenadas cartesianas absolutas de dados (ax.transData via fill_between),
   confirmando que a região ideal ocupa exatamente x >= 6.5 e 0.70 <= y <= 1.00 (P_hERG < 0.30),
   sem distorção causada por ylim = (-0.05, 1.05).
4. Ausência de chamadas a axvspan com coordenadas mistas em app.py e no notebook 07.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
NB07_PATH = PROJECT_ROOT / "notebooks" / "07_multitarget_shap_evaluation.ipynb"


def _load_build_multiobjective_figure() -> Callable[[float, float, str], Tuple[plt.Figure, plt.Axes]]:
    """Extrai e compila a função build_multiobjective_figure diretamente da AST de app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))

    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_multiobjective_figure":
            func_node = node
            break

    assert func_node is not None, "Função build_multiobjective_figure não encontrada em app.py."

    mod = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"plt": plt, "Tuple": Tuple}
    exec(compile(mod, filename=str(APP_PATH), mode="exec"), ns)
    return ns["build_multiobjective_figure"]


def _extract_reference_lines(ax: plt.Axes) -> tuple[list[float], list[float]]:
    """Extrai as posições X das linhas verticais (axvline) e Y das linhas horizontais (axhline)."""
    vlines_x: list[float] = []
    hlines_y: list[float] = []

    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) == 2 and np.allclose(xdata[0], xdata[1]) and np.allclose(ydata, [0.0, 1.0]):
            vlines_x.append(float(xdata[0]))
        elif len(ydata) == 2 and np.allclose(ydata[0], ydata[1]) and np.allclose(xdata, [0.0, 1.0]):
            hlines_y.append(float(ydata[0]))

    return sorted(vlines_x), sorted(hlines_y)


def _extract_shaded_bboxes(ax: plt.Axes) -> list[tuple[float, float, float, float]]:
    """Retorna os bounding boxes (xmin, xmax, ymin, ymax) das regiões preenchidas com fill_between."""
    bboxes: list[tuple[float, float, float, float]] = []
    for coll in ax.collections:
        # PathCollection do scatter tem offsets; PolyCollection do fill_between usa ax.transData puro
        if isinstance(coll, mcollections.PolyCollection) and not isinstance(coll, mcollections.PathCollection):
            assert coll.get_transform() == ax.transData, (
                "Região sombreada não está em coordenadas estritas de dados (ax.transData)."
            )
            paths = coll.get_paths()
            assert len(paths) >= 1
            verts = paths[0].vertices
            xmin, ymin = np.min(verts, axis=0)
            xmax, ymax = np.max(verts, axis=0)
            bboxes.append((float(xmin), float(xmax), float(ymin), float(ymax)))
    return bboxes


def test_no_axvspan_mixed_coordinates_in_app_or_notebook():
    """Confirma que axvspan (que mistura X em dados com Y relativo) foi removido de app.py e do notebook 07."""
    app_src = APP_PATH.read_text(encoding="utf-8")
    assert "axvspan" not in app_src, "Uso residual de axvspan encontrado em app.py."

    nb_data = json.loads(NB07_PATH.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") == "code":
            cell_src = "".join(cell.get("source", []))
            assert "axvspan" not in cell_src, f"Uso residual de axvspan encontrado na célula {idx} do notebook 07."


def test_app_multiobjective_figure_geometry_and_thresholds():
    """Valida eixos, limites, as 4 linhas de referência e os vértices exatos da região ideal em app.py."""
    build_fig = _load_build_multiobjective_figure()
    fig, ax = build_fig(7.25, 0.85, "Composto Teste")
    try:
        # 1. Rótulos dos eixos e limites visuais
        assert "pIC50" in ax.get_xlabel()
        assert "1 - P(hERG)" in ax.get_ylabel()
        np.testing.assert_allclose(ax.get_xlim(), (3.5, 11.5), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ax.get_ylim(), (-0.05, 1.05), rtol=1e-12, atol=1e-12)

        # 2. Quatro linhas de referência explícitas: pIC50 = 5.0, 6.5 e 1 - P_hERG = 0.30, 0.70
        vlines_x, hlines_y = _extract_reference_lines(ax)
        np.testing.assert_allclose(vlines_x, [5.0, 6.5], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hlines_y, [0.30, 0.70], rtol=1e-12, atol=1e-12)

        # 3. Sombreamento em coordenadas de dados (sem deslocamento de 0.70 para 0.72 ou 0.30 para 0.28)
        bboxes = _extract_shaded_bboxes(ax)
        assert len(bboxes) >= 2

        # Região ideal: x in [6.5, 11.5], y in [0.70, 1.00]
        ideal_matches = [
            b for b in bboxes
            if np.allclose(b, (6.5, 11.5, 0.70, 1.00), rtol=1e-12, atol=1e-12)
        ]
        assert len(ideal_matches) == 1, f"Região ideal [6.5, 11.5, 0.70, 1.00] não encontrada em {bboxes}"

        # Região potente porém cardiotóxica: x in [6.5, 11.5], y in [0.00, 0.30] (y = 0.30 pertence à região de alto risco)
        toxic_potent_matches = [
            b for b in bboxes
            if np.allclose(b, (6.5, 11.5, 0.00, 0.30), rtol=1e-12, atol=1e-12)
        ]
        assert len(toxic_potent_matches) == 1, (
            f"Região cardiotóxica [6.5, 11.5, 0.00, 0.30] não encontrada em {bboxes}"
        )
        assert toxic_potent_matches[0][3] == pytest.approx(1.0 - 0.70, abs=1e-12)
    finally:
        plt.close(fig)


def test_notebook07_multiobjective_cell_geometry_and_thresholds():
    """Executa a célula do gráfico multiobjetivo do notebook 07 e valida seus limiares e geometria."""
    nb_data = json.loads(NB07_PATH.read_text(encoding="utf-8"))
    plot_cell_src = None
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "Matriz de Triagem Virtual Multiobjetivo" in src:
                plot_cell_src = src
                break

    assert plot_cell_src is not None, "Célula do gráfico multiobjetivo não encontrada no notebook 07."

    # Substitui plt.show() apenas durante o teste para evitar abertura de janela interativa
    exec_src = plot_cell_src.replace("plt.show()", "")
    ns = {
        "plt": plt,
        "preds_a2a": np.array([5.51, 4.95, 5.24, 5.45, 5.80]),
        "probs_herg": np.array([0.459, 0.105, 0.805, 0.076, 0.114]),
        "controls": {k: {} for k in ["Istradefilina", "Cafeina", "Terfenadina", "Taurina", "L-Teanina"]},
    }
    exec(compile(exec_src, filename=str(NB07_PATH), mode="exec"), ns)
    fig = ns["fig"]
    ax = ns["ax"]
    try:
        np.testing.assert_allclose(ax.get_xlim(), (3.5, 10.5), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ax.get_ylim(), (-0.05, 1.05), rtol=1e-12, atol=1e-12)

        vlines_x, hlines_y = _extract_reference_lines(ax)
        np.testing.assert_allclose(vlines_x, [5.0, 6.5], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hlines_y, [0.30, 0.70], rtol=1e-12, atol=1e-12)

        bboxes = _extract_shaded_bboxes(ax)
        ideal_matches = [
            b for b in bboxes
            if np.allclose(b, (6.5, 10.5, 0.70, 1.00), rtol=1e-12, atol=1e-12)
        ]
        assert len(ideal_matches) == 1, f"Região ideal [6.5, 10.5, 0.70, 1.00] não encontrada em {bboxes}"
    finally:
        plt.close(fig)

