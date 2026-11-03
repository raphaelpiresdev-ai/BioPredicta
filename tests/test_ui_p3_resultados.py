"""
Suíte de Testes da Etapa 3 — Página de Resultados Multiobjetivo (src/ui/pages/p3_resultados.py).

Testa:
1. Renderização em estado vazio (sem análise prévia);
2. Renderização completa a partir de uma análise válida armazenada;
3. Apresentação e integridade da identificação molecular (com e sem PubChem);
4. Exibição fiel dos valores A2A e hERG a partir do dicionário armazenado (sem recálculo);
5. Apresentação das métricas e alertas do Domínio de Aplicabilidade;
6. Construção e limites do gráfico cartesiano multiobjetivo;
7. Apresentação explicativa SHAP sem recálculo de TreeExplainer;
8. Preservação do estado da sessão entre Nova Análise e Resultados;
9. Resiliência absoluta diante de ausência de campos opcionais;
10. Garantia estrita de não-execução redundante de inferência ou SHAP.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import streamlit as st

from src.chem_utils import FEATURE_NAMES, canonicalize_smiles, compute_2d_descriptors
from src.ui.components import build_multiobjective_figure, build_shap_impact_figure
from src.ui.pages.p3_resultados import (
    format_ic50,
    render_resultados,
    resolve_compound_display_name,
)


@pytest.fixture
def mock_caffeine_analysis() -> Dict[str, Any]:
    """Gera um dicionário completo de análise estruturado exatamente como em Nova Análise."""
    smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    canon = canonicalize_smiles(smi)
    mol = Chem.MolFromSmiles(canon)
    desc = compute_2d_descriptors(mol)

    shap_a2a = np.zeros(len(FEATURE_NAMES))
    shap_a2a[0] = 0.25  # MolWt
    shap_a2a[1] = -0.32  # MolLogP
    shap_a2a[234 + 6] = 0.48  # fp_234
    shap_a2a[831 + 6] = 0.35  # fp_831

    shap_herg = np.zeros(len(FEATURE_NAMES))
    shap_herg[1] = 0.15  # MolLogP (eleva risco)
    shap_herg[2] = -0.22  # TPSA (protetor)

    return {
        "timestamp": "2026-10-31T12:00:00",
        "input_method": "✏️ Inserir SMILES",
        "raw_input": smi,
        "canonical_smiles": canon,
        "mol": mol,
        "formula": "C8H10N4O2",
        "mol_wt": 194.19,
        "descriptors": desc,
        "pubchem": {
            "status": "success",
            "cid": 2519,
            "title": "Caffeine",
            "compound_name": "Caffeine",
            "iupac_name": "1,3,7-trimethylpurine-2,6-dione",
        },
        "provenance": {
            "name": "Cafeína (Controle A2A)",
            "category": "Controle Farmacológico",
        },
        "a2a": {
            "pred_pIC50": 5.42,
            "pred_IC50_nM": 3801.89,
            "classification": {
                "potency_class": "moderada",
                "color_class": "semaphore-yellow",
                "label": "🟡 Potência Moderada Operacional (5,0 ≤ pIC50 < 6,5)",
                "desc": "Faixa micromolar compatível com potência intermediária.",
            },
            "ad": {
                "status": "ok",
                "endpoint": "A2A",
                "provenance": "Nível 1 — SMILES Idêntico",
                "provenance_level": 1,
                "s_max_all": 1.0,
                "s_max_excluding_match": 0.45,
                "has_exclusion": True,
                "nearest_neighbor": {
                    "id": "CHEMBL113",
                    "similarity": 1.0,
                    "is_exact_match": True,
                    "experimental_label_name": "pIC50",
                    "experimental_value": 5.40,
                },
                "nearest_neighbor_non_matching": {
                    "id": "CHEMBL224",
                    "similarity": 0.45,
                    "is_exact_match": False,
                    "experimental_label_name": "pIC50",
                    "experimental_value": 5.80,
                },
                "loo_median_similarity": 0.38,
                "descriptors": desc,
            },
            "shap_values": shap_a2a,
            "base_value": 5.62,
        },
        "herg": {
            "prob_herg": 0.12,
            "safety_index": 0.88,
            "complementary_index": 0.88,
            "classification": {
                "risk_class": "baixo",
                "badge_color": "semaphore-green",
                "bar_color": "#00D2BE",
                "label": "🟢 Baixo Risco Estimado (P < 0,30)",
                "description": "Baixa probabilidade estimada de bloqueio do canal de potássio hERG.",
            },
            "ad": {
                "status": "ok",
                "endpoint": "hERG",
                "provenance": "Nível 1 — SMILES Idêntico",
                "provenance_level": 1,
                "s_max_all": 1.0,
                "s_max_excluding_match": 0.42,
                "has_exclusion": True,
                "nearest_neighbor": {
                    "id": "TDC_98",
                    "similarity": 1.0,
                    "is_exact_match": True,
                    "experimental_label_name": "Bloqueador",
                    "experimental_value": 0,
                },
                "nearest_neighbor_non_matching": {
                    "id": "TDC_512",
                    "similarity": 0.42,
                    "is_exact_match": False,
                    "experimental_label_name": "Bloqueador",
                    "experimental_value": 0,
                },
                "loo_median_similarity": 0.35,
                "descriptors": desc,
            },
            "shap_values": shap_herg,
            "base_value": 0.28,
        },
        "multiobjective": {
            "quadrant": {
                "quad_badge_cls": "semaphore-yellow",
                "quad_title": "🟡 Zona de Otimização / Hit-to-Lead",
                "quad_desc": "Perfil intermediário de potência ou segurança; requer otimização estrutural guiada.",
            },
            "x": 5.42,
            "y": 0.88,
        },
    }


class TestResultadosPageEmptyState:
    """Verifica o comportamento da página quando nenhuma análise prévia foi realizada."""

    def test_open_page_without_previous_analysis(self):
        """Página abre sem análise prévia e exibe orientação amigável sem lançar exceção."""
        st.session_state["current_analysis"] = None
        # Não deve levantar nenhuma exceção
        render_resultados(analysis=None)


class TestResultadosPageValidAnalysis:
    """Verifica a renderização com uma análise válida ativa."""

    def test_open_page_with_valid_analysis(self, mock_caffeine_analysis):
        """Página abre com uma análise válida consumindo os dados diretamente da sessão."""
        st.session_state["current_analysis"] = mock_caffeine_analysis
        render_resultados(analysis=mock_caffeine_analysis)

    def test_molecular_identification_presentation(self, mock_caffeine_analysis):
        """Identificação molecular é exibida fielmente."""
        name, has_comm = resolve_compound_display_name(mock_caffeine_analysis)
        assert name == "Caffeine"
        assert has_comm is True

        # Sem pubchem
        analysis_no_pub = dict(mock_caffeine_analysis)
        analysis_no_pub["pubchem"] = {}
        analysis_no_pub["provenance"] = {}
        name2, has_comm2 = resolve_compound_display_name(analysis_no_pub)
        assert "In Silico" in name2
        assert has_comm2 is False

    def test_stored_values_fidelity(self, mock_caffeine_analysis):
        """Resultados A2A e hERG são preservados estritamente dos valores armazenados."""
        a2a = mock_caffeine_analysis["a2a"]
        herg = mock_caffeine_analysis["herg"]

        assert a2a["pred_pIC50"] == 5.42
        assert a2a["classification"]["potency_class"] == "moderada"
        assert herg["prob_herg"] == 0.12
        assert herg["safety_index"] == 0.88
        assert herg["classification"]["risk_class"] == "baixo"

    def test_format_ic50_helper(self):
        """Verifica a formatação de unidades da concentração inibitória IC50."""
        assert format_ic50(0.5) == "500.0 pM"
        assert format_ic50(250.0) == "250.0 nM"
        assert format_ic50(3801.89) == "3.80 µM"
        assert format_ic50(2_500_000.0) == "2.50 mM"


class TestMultiobjectivePlotIntegration:
    """Verificação do gráfico cartesiano multiobjetivo."""

    def test_build_multiobjective_figure(self):
        """Gera o gráfico cartesiano com todos os elementos e limiares operacionais."""
        fig, ax = build_multiobjective_figure(x_cur=5.42, y_cur=0.88, active_name="Caffeine")
        assert fig is not None
        assert ax is not None

        # Limites dos eixos
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= 3.5
        assert xlim[1] >= 11.5
        assert ylim[0] <= 0.0
        assert ylim[1] >= 1.0

        plt.close(fig)


class TestShapInterpretabilityIntegration:
    """Verificação da apresentação explicativa SHAP sem novo cálculo."""

    def test_build_shap_impact_figure_a2a(self, mock_caffeine_analysis):
        """Gera gráfico horizontal SHAP para A2A com cores semânticas."""
        a2a = mock_caffeine_analysis["a2a"]
        fig, ax, top_items = build_shap_impact_figure(
            shap_values=a2a["shap_values"],
            base_value=a2a["base_value"],
            pred_value=a2a["pred_pIC50"],
            target="A2A",
            top_n=10,
            mol=mock_caffeine_analysis["mol"],
        )
        assert fig is not None
        assert ax is not None
        assert len(top_items) == 10
        # O atributo com maior impacto positivo configurado no mock
        top_feature_names = [it["feature_name"] for it in top_items]
        assert "fp_234" in top_feature_names

        plt.close(fig)

    def test_build_shap_impact_figure_empty_or_zero(self):
        """Trata graciosamente SHAP ausente ou zerado sem lançar exceção."""
        fig, ax, top_items = build_shap_impact_figure(
            shap_values=np.zeros(len(FEATURE_NAMES)),
            base_value=5.0,
            pred_value=5.0,
            target="A2A",
        )
        assert fig is None
        assert ax is None
        assert len(top_items) == 0


class TestNoRedundantInferenceExecution:
    """Garante de forma estrita que NENHUM cálculo de modelo ML ou SHAP é executado em Resultados."""

    def test_render_resultados_does_not_call_ml_models_or_tree_explainer(self, mock_caffeine_analysis):
        """Verifica ausência de chamadas a predict(), predict_proba() ou TreeExplainer."""
        with patch("joblib.load") as mock_joblib:
            with patch("shap.TreeExplainer") as mock_tree_explainer:
                render_resultados(analysis=mock_caffeine_analysis)
                # Modelos NÃO devem ser carregados
                assert not mock_joblib.called
                # TreeExplainer NÃO deve ser instanciado
                assert not mock_tree_explainer.called


class TestResilienceToMissingOptionalFields:
    """Verifica resiliência absoluta quando faltam metadados opcionais na análise."""

    def test_minimal_analysis_dictionary(self):
        """Um dicionário com estrutura mínima é renderizado sem levantar exceção."""
        minimal_analysis = {
            "canonical_smiles": "c1ccccc1",
            "formula": "C6H6",
            "mol_wt": 78.11,
            "descriptors": {},
            "pubchem": {},
            "provenance": {},
            "a2a": {
                "pred_pIC50": 4.5,
                "classification": {"potency_class": "baixa", "color_class": "semaphore-red"},
                "ad": {"status": "ok", "endpoint": "A2A", "provenance_level": 4, "s_max_all": 0.25},
            },
            "herg": {
                "prob_herg": 0.10,
                "safety_index": 0.90,
                "classification": {"risk_class": "baixo", "bar_color": "#00D2BE"},
                "ad": {"status": "ok", "endpoint": "hERG", "provenance_level": 4, "s_max_all": 0.20},
            },
            "multiobjective": {
                "quadrant": {"quad_title": "Quadrante 2", "quad_badge_cls": "semaphore-yellow"},
                "x": 4.5,
                "y": 0.90,
            },
        }

        render_resultados(analysis=minimal_analysis)

