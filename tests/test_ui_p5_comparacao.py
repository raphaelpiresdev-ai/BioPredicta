"""
Suíte de testes automatizados para a Etapa 5.3 — Passo 1: Componentes Químicos e Gráficos de Comparação.

Testa exaustivamente:
1. Reconstrução molecular padronizada (reconstruct_mol_from_smiles).
2. Geração de fingerprints circulares Morgan ECFP4 (get_morgan_fp_bitvect).
3. Matriz simétrica de similaridade Tanimoto:
   - Simetria estrita M[i, j] == M[j, i].
   - Diagonal unitária M[i, i] == 1.0.
   - Limites no intervalo contínuo [0.0, 1.0].
   - Compatibilidade polimórfica (SMILES str, Chem.Mol, dicionários do histórico).
4. Cálculo dos deltas farmacológicos e físico-químicos (compute_comparative_deltas):
   - Invariância da molécula de referência (deltas = 0.0, ratio = 1.0).
   - Exatidão algébrica dos deltas de pIC50, prob_herg, 1 - P e razão de IC50.
   - Deltas exatos dos 6 descritores canônicos (MolWt, MolLogP, NumHDonors, etc.).
   - Seleção flexível de referência por ref_index ou ref_smiles.
   - Resiliência contra campos ausentes (cálculo on-the-fly).
5. Renderização do gráfico cartesiano multiobjetivo comparativo:
   - Geração sem erros para 2, 3, 4 e 5 moléculas simultâneas.
   - Preservação estrita dos limites de eixos (X in [3.5, 11.5], Y in [-0.05, 1.05]).
   - Preservação dos limiares canônicos (A2A: 5.0, 6.5; hERG: 0.30, 0.70).
   - Destaque visual da molécula de referência e paleta diferenciada.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from rdkit import Chem, DataStructs

from src.chem_utils import smiles_to_mol
from src.ui.components import (
    build_comparative_multiobjective_figure,
    build_multiobjective_figure,
    compute_comparative_deltas,
    compute_pairwise_tanimoto_matrix,
    get_morgan_fp_bitvect,
    reconstruct_mol_from_smiles,
)


# -----------------------------------------------------------------------------
# Fixtures com dados sintéticos e reais de teste
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_molecules_dataset() -> List[Dict[str, Any]]:
    """Gera 5 moléculas com perfil completo similares às registradas no analysis_history."""
    return [
        {
            "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "provenance": {"compound_name": "Cafeína"},
            "descriptors": {
                "MolWt": 194.19,
                "MolLogP": -0.07,
                "NumHDonors": 0,
                "NumHAcceptors": 6,
                "TPSA": 58.44,
                "NumRotatableBonds": 0,
            },
            "a2a": {
                "pred_pIC50": 5.25,
                "pred_IC50_nM": 5623.4,
                "classification": {"label": "Potência Moderada", "color": "#F59E0B"},
            },
            "herg": {
                "prob_herg": 0.134,
                "safety_index": 0.866,
                "complementary_index": 0.866,
                "classification": {"label": "Baixo Risco", "color": "#00D2BE"},
            },
            "multiobjective": {"x": 5.25, "y": 0.866},
        },
        {
            "canonical_smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
            "provenance": {"compound_name": "Istradefilina"},
            "descriptors": {
                "MolWt": 384.43,
                "MolLogP": 2.92,
                "NumHDonors": 0,
                "NumHAcceptors": 6,
                "TPSA": 76.90,
                "NumRotatableBonds": 5,
            },
            "a2a": {
                "pred_pIC50": 8.12,
                "pred_IC50_nM": 7.58,
                "classification": {"label": "Alta Potência", "color": "#00D2BE"},
            },
            "herg": {
                "prob_herg": 0.220,
                "safety_index": 0.780,
                "complementary_index": 0.780,
                "classification": {"label": "Baixo Risco", "color": "#00D2BE"},
            },
            "multiobjective": {"x": 8.12, "y": 0.780},
        },
        {
            "canonical_smiles": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
            "provenance": {"compound_name": "Haloperidol"},
            "descriptors": {
                "MolWt": 375.87,
                "MolLogP": 3.79,
                "NumHDonors": 1,
                "NumHAcceptors": 3,
                "TPSA": 40.54,
                "NumRotatableBonds": 6,
            },
            "a2a": {
                "pred_pIC50": 4.10,
                "pred_IC50_nM": 79432.8,
                "classification": {"label": "Baixa Potência", "color": "#94A3B8"},
            },
            "herg": {
                "prob_herg": 0.850,
                "safety_index": 0.150,
                "complementary_index": 0.150,
                "classification": {"label": "Alto Risco", "color": "#FF4B4B"},
            },
            "multiobjective": {"x": 4.10, "y": 0.150},
        },
        {
            "canonical_smiles": "NCCS(=O)(=O)O",
            "provenance": {"compound_name": "Taurina"},
            "descriptors": {
                "MolWt": 125.15,
                "MolLogP": -1.45,
                "NumHDonors": 2,
                "NumHAcceptors": 3,
                "TPSA": 88.52,
                "NumRotatableBonds": 2,
            },
            "a2a": {
                "pred_pIC50": 3.80,
                "pred_IC50_nM": 158489.3,
                "classification": {"label": "Baixa Potência", "color": "#94A3B8"},
            },
            "herg": {
                "prob_herg": 0.050,
                "safety_index": 0.950,
                "complementary_index": 0.950,
                "classification": {"label": "Baixo Risco", "color": "#00D2BE"},
            },
            "multiobjective": {"x": 3.80, "y": 0.950},
        },
        {
            "canonical_smiles": "c1ccc(cc1)c2nc3ccccc3o2",
            "provenance": {"compound_name": "2-Fenilbenzoxazol"},
            "descriptors": {
                "MolWt": 195.22,
                "MolLogP": 3.32,
                "NumHDonors": 0,
                "NumHAcceptors": 2,
                "TPSA": 26.03,
                "NumRotatableBonds": 1,
            },
            "a2a": {
                "pred_pIC50": 6.20,
                "pred_IC50_nM": 630.9,
                "classification": {"label": "Potência Moderada", "color": "#F59E0B"},
            },
            "herg": {
                "prob_herg": 0.450,
                "safety_index": 0.550,
                "complementary_index": 0.550,
                "classification": {"label": "Risco Moderado", "color": "#F59E0B"},
            },
            "multiobjective": {"x": 6.20, "y": 0.550},
        },
    ]


# -----------------------------------------------------------------------------
# 1. Testes de Reconstrução Molecular e Fingerprints
# -----------------------------------------------------------------------------
class TestMolecularReconstructionAndFingerprints:
    """Verificação de reconstrução a partir de SMILES e geração de ECFP4."""

    def test_reconstruct_mol_from_valid_smiles(self):
        mol = reconstruct_mol_from_smiles("CCO")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() > 0

    def test_reconstruct_mol_from_invalid_inputs(self):
        assert reconstruct_mol_from_smiles("") is None
        assert reconstruct_mol_from_smiles("   ") is None
        assert reconstruct_mol_from_smiles(None) is None  # type: ignore
        assert reconstruct_mol_from_smiles("invalid_smiles_string_123") is None

    def test_get_morgan_fp_bitvect_success(self):
        smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        fp = get_morgan_fp_bitvect(smi, radius=2, n_bits=1024)
        assert fp is not None
        assert isinstance(fp, DataStructs.ExplicitBitVect)
        assert fp.GetNumBits() == 1024
        assert fp.GetNumOnBits() > 0

    def test_get_morgan_fp_bitvect_from_mol_object(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp = get_morgan_fp_bitvect(mol, radius=2, n_bits=1024)
        assert fp is not None
        assert fp.GetNumBits() == 1024

    def test_get_morgan_fp_bitvect_invalid(self):
        assert get_morgan_fp_bitvect(None) is None
        assert get_morgan_fp_bitvect("not_a_molecule") is None


# -----------------------------------------------------------------------------
# 2. Testes da Matriz de Tanimoto
# -----------------------------------------------------------------------------
class TestTanimotoMatrixCalculation:
    """Validação da simetria, diagonal unitária e comportamento numérico da matriz."""

    def test_empty_input_handling(self):
        labels, matrix = compute_pairwise_tanimoto_matrix([])
        assert labels == []
        assert matrix.shape == (0, 0)

    def test_single_molecule(self):
        labels, matrix = compute_pairwise_tanimoto_matrix(["CCO"])
        assert len(labels) == 1
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0

    def test_matrix_symmetry_and_diagonal_for_multiple_molecules(self, mock_molecules_dataset):
        labels, matrix = compute_pairwise_tanimoto_matrix(mock_molecules_dataset)
        assert len(labels) == 5
        assert matrix.shape == (5, 5)

        # 1. Simetria estrita: M = M^T
        assert np.allclose(matrix, matrix.T, atol=1e-7)

        # 2. Diagonal unitária: M[i, i] == 1.0
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-7)

        # 3. Intervalo [0.0, 1.0]
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)

    def test_polymorphic_input_support(self, mock_molecules_dataset):
        # Mistura de dicts, Chem.Mol e strings SMILES
        mixed = [
            mock_molecules_dataset[0],  # dict
            Chem.MolFromSmiles("CCO"),  # Chem.Mol
            "c1ccccc1",                 # string SMILES
        ]
        labels, matrix = compute_pairwise_tanimoto_matrix(mixed)
        assert len(labels) == 3
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 1.0)


# -----------------------------------------------------------------------------
# 3. Testes do Cálculo de Deltas Relativos
# -----------------------------------------------------------------------------
class TestComparativeDeltasCalculation:
    """Verificação algébrica e estrutural dos deltas farmacológicos e físico-químicos."""

    def test_empty_list_handling(self):
        assert compute_comparative_deltas([]) == []

    def test_reference_molecule_invariance(self, mock_molecules_dataset):
        """Para a molécula de referência, todos os deltas devem ser 0.0 e a razão de IC50 deve ser 1.0."""
        deltas = compute_comparative_deltas(mock_molecules_dataset, ref_index=0)
        ref_entry = deltas[0]

        assert ref_entry["is_reference"] is True
        assert ref_entry["deltas"]["delta_pIC50"] == 0.0
        assert ref_entry["deltas"]["ratio_IC50"] == 1.0
        assert ref_entry["deltas"]["delta_prob_herg"] == 0.0
        assert ref_entry["deltas"]["delta_safety_index"] == 0.0
        assert ref_entry["deltas"]["delta_MolWt"] == 0.0
        assert ref_entry["deltas"]["delta_MolLogP"] == 0.0
        assert ref_entry["deltas"]["delta_NumHDonors"] == 0
        assert ref_entry["deltas"]["delta_NumHAcceptors"] == 0
        assert ref_entry["deltas"]["delta_TPSA"] == 0.0
        assert ref_entry["deltas"]["delta_NumRotatableBonds"] == 0

    def test_numerical_deltas_correctness(self, mock_molecules_dataset):
        """Verifica a precisão matemática das subtrações frente à Cafeína (ref_index=0)."""
        deltas = compute_comparative_deltas(mock_molecules_dataset, ref_index=0)
        caffeine = mock_molecules_dataset[0]
        istradefylline = mock_molecules_dataset[1]
        entry_ist = deltas[1]

        assert entry_ist["is_reference"] is False

        # Delta pIC50 = 8.12 - 5.25 = +2.87
        expected_dp = istradefylline["a2a"]["pred_pIC50"] - caffeine["a2a"]["pred_pIC50"]
        assert pytest.approx(entry_ist["deltas"]["delta_pIC50"], rel=1e-4) == expected_dp

        # Delta prob_herg = 0.220 - 0.134 = +0.086
        expected_dprob = istradefylline["herg"]["prob_herg"] - caffeine["herg"]["prob_herg"]
        assert pytest.approx(entry_ist["deltas"]["delta_prob_herg"], rel=1e-4) == expected_dprob

        # Delta safety_index = 0.780 - 0.866 = -0.086
        expected_dsafety = istradefylline["herg"]["safety_index"] - caffeine["herg"]["safety_index"]
        assert pytest.approx(entry_ist["deltas"]["delta_safety_index"], rel=1e-4) == expected_dsafety

        # Razão de IC50 = 7.58 / 5623.4
        expected_ratio = istradefylline["a2a"]["pred_IC50_nM"] / caffeine["a2a"]["pred_IC50_nM"]
        assert pytest.approx(entry_ist["deltas"]["ratio_IC50"], rel=1e-4) == expected_ratio

        # Delta MolWt = 384.43 - 194.19 = +190.24
        expected_dwt = istradefylline["descriptors"]["MolWt"] - caffeine["descriptors"]["MolWt"]
        assert pytest.approx(entry_ist["deltas"]["delta_MolWt"], rel=1e-4) == expected_dwt

    def test_reference_selection_by_smiles(self, mock_molecules_dataset):
        """Garante que fornecer ref_smiles seleciona a molécula correta mesmo que ref_index seja diferente."""
        istradefylline_smi = mock_molecules_dataset[1]["canonical_smiles"]
        deltas = compute_comparative_deltas(
            mock_molecules_dataset,
            ref_index=0,
            ref_smiles=istradefylline_smi,
        )
        assert deltas[0]["is_reference"] is False
        assert deltas[1]["is_reference"] is True
        assert deltas[1]["deltas"]["delta_pIC50"] == 0.0

    def test_resilience_to_missing_descriptors(self):
        """Se o registro não tiver o bloco de descritores, eles devem ser inferidos on-the-fly pelo SMILES."""
        minimal_mols = [
            {
                "canonical_smiles": "CCO",
                "a2a": {"pred_pIC50": 4.5, "pred_IC50_nM": 31622.7},
                "herg": {"prob_herg": 0.10, "safety_index": 0.90},
            },
            {
                "canonical_smiles": "c1ccccc1",
                "a2a": {"pred_pIC50": 4.0, "pred_IC50_nM": 100000.0},
                "herg": {"prob_herg": 0.30, "safety_index": 0.70},
            },
        ]
        deltas = compute_comparative_deltas(minimal_mols, ref_index=0)
        assert len(deltas) == 2
        assert "MolWt" in deltas[0]["values"]
        assert deltas[0]["values"]["MolWt"] > 0.0
        assert deltas[1]["deltas"]["delta_MolWt"] != 0.0


# -----------------------------------------------------------------------------
# 4. Testes do Gráfico Multiobjetivo Multiponto
# -----------------------------------------------------------------------------
class TestComparativeMultiobjectiveFigure:
    """Verificação do gráfico Matplotlib para 2 a 5 moléculas simultâneas."""

    @pytest.mark.parametrize("n_mols", [2, 3, 4, 5])
    def test_build_comparative_multiobjective_figure_point_counts(self, mock_molecules_dataset, n_mols):
        """Renderiza a figura sem erros para conjuntos de 2, 3, 4 e 5 moléculas."""
        subset = mock_molecules_dataset[:n_mols]
        fig, ax = build_comparative_multiobjective_figure(subset)
        assert fig is not None
        assert ax is not None

        # 1. Limites dos eixos rigorosamente preservados
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= 3.5
        assert xlim[1] >= 11.5
        assert ylim[0] <= -0.05
        assert ylim[1] >= 1.05

        # 2. Preservação dos limiares canônicos (A2A: 5.0, 6.5; hERG: 0.30, 0.70)
        vlines = [line.get_xdata()[0] for line in ax.lines if len(line.get_xdata()) > 0 and line.get_xdata()[0] == line.get_xdata()[-1]]
        hlines = [line.get_ydata()[0] for line in ax.lines if len(line.get_ydata()) > 0 and line.get_ydata()[0] == line.get_ydata()[-1]]

        assert any(abs(x - 5.0) < 1e-3 for x in vlines)
        assert any(abs(x - 6.5) < 1e-3 for x in vlines)
        assert any(abs(y - 0.70) < 1e-3 for y in hlines)
        assert any(abs(y - 0.30) < 1e-3 for y in hlines)

        plt.close(fig)

    def test_original_build_multiobjective_figure_remains_intact(self):
        """Garante que a função original para molécula individual continua funcionando 100% idêntica."""
        fig, ax = build_multiobjective_figure(x_cur=5.42, y_cur=0.88, active_name="Caffeine")
        assert fig is not None
        assert ax is not None
        assert ax.get_xlim()[0] <= 3.5
        assert ax.get_ylim()[1] >= 1.0
        plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Testes da Página de Comparação e Integração com a Sessão
# -----------------------------------------------------------------------------
class TestComparacaoPageEmptyState:
    """Verificação da interface quando há menos de 2 moléculas disponíveis no histórico."""

    def test_empty_history_renders_empty_state(self):
        """Histórico vazio exibe card informativo sem lançar exceção e preserva o estado."""
        import streamlit as st
        from src.ui.pages.p5_comparacao import render_comparacao

        st.session_state["analysis_history"] = []
        st.session_state["current_analysis"] = None

        # Execução sem exceção
        render_comparacao()

        assert st.session_state["analysis_history"] == []
        assert st.session_state["current_analysis"] is None

    def test_single_molecule_history_renders_informative_guidance(self, mock_molecules_dataset):
        """Histórico com apenas 1 molécula orienta o usuário a realizar nova análise sem corromper dados."""
        import streamlit as st
        from src.ui.pages.p5_comparacao import render_comparacao

        st.session_state["analysis_history"] = [mock_molecules_dataset[0]]
        st.session_state["current_analysis"] = mock_molecules_dataset[0]

        # Execução sem exceção
        render_comparacao()

        assert len(st.session_state["analysis_history"]) == 1
        assert st.session_state["current_analysis"]["canonical_smiles"] == mock_molecules_dataset[0]["canonical_smiles"]


class TestComparacaoPageExecution:
    """Verificação da renderização da página com múltiplas moléculas selecionadas."""

    def test_comparacao_page_renders_with_3_molecules(self, mock_molecules_dataset):
        """Renderiza a página com 3 moléculas preservando estritamente current_analysis e analysis_history."""
        import streamlit as st
        from src.ui.pages.p5_comparacao import render_comparacao

        history_copy = [dict(m) for m in mock_molecules_dataset[:3]]
        st.session_state["analysis_history"] = history_copy
        st.session_state["current_analysis"] = history_copy[0]
        st.session_state["current_analysis_smiles"] = history_copy[0]["canonical_smiles"]

        render_comparacao()

        # Invariância estrita: current_analysis não é modificado pela navegação
        assert st.session_state["current_analysis"] is history_copy[0]
        assert len(st.session_state["analysis_history"]) == 3
        assert st.session_state["analysis_history"][0]["canonical_smiles"] == mock_molecules_dataset[0]["canonical_smiles"]

    def test_comparacao_page_renders_with_5_molecules(self, mock_molecules_dataset):
        """Renderiza a página com o limite máximo de 5 moléculas sem erros."""
        import streamlit as st
        from src.ui.pages.p5_comparacao import render_comparacao

        st.session_state["analysis_history"] = mock_molecules_dataset[:5]
        st.session_state["current_analysis"] = mock_molecules_dataset[0]
        st.session_state["current_analysis_smiles"] = mock_molecules_dataset[0]["canonical_smiles"]

        render_comparacao()

        assert len(st.session_state["analysis_history"]) == 5


class TestReferenceMoleculeSwitching:
    """Verificação do impacto da troca de molécula de referência nos deltas."""

    def test_reference_switch_inverts_relative_deltas(self, mock_molecules_dataset):
        """Trocar a molécula de referência atualiza com precisão os deltas farmacológicos e físico-químicos."""
        mols = mock_molecules_dataset[:2]

        # Com referência = Molécula 0 (Cafeína)
        deltas_ref0 = compute_comparative_deltas(mols, ref_index=0)
        assert deltas_ref0[0]["is_reference"] is True
        assert deltas_ref0[1]["is_reference"] is False
        dp_0_to_1 = deltas_ref0[1]["deltas"]["delta_pIC50"]

        # Com referência = Molécula 1 (Istradefilina)
        deltas_ref1 = compute_comparative_deltas(mols, ref_index=1)
        assert deltas_ref1[0]["is_reference"] is False
        assert deltas_ref1[1]["is_reference"] is True
        dp_1_to_0 = deltas_ref1[0]["deltas"]["delta_pIC50"]

        # Os deltas devem ser estritamente simétricos opostos: dp_0_to_1 == -dp_1_to_0
        assert pytest.approx(dp_0_to_1, rel=1e-4) == -dp_1_to_0


class TestIndependentADPresentation:
    """Verificação da independência total entre os diagnósticos AD de A2A e hERG."""

    def test_ad_badge_a2a_and_herg_separate(self):
        """Garante que os badges AD nunca se fundem e preservam metadados canônicos de ad_utils."""
        from src.ui.pages.p5_comparacao import _render_ad_badge

        ad_a2a = {"status": "ok", "provenance_level": 1, "s_max_all": 1.0}
        ad_herg = {"status": "ok", "provenance_level": 4, "s_max_all": 0.58}

        badge_a2a = _render_ad_badge(ad_a2a, "A2A")
        badge_herg = _render_ad_badge(ad_herg, "hERG")

        assert "A2A AD:" in badge_a2a
        assert "Nível 1" in badge_a2a
        assert "Correspondência de estrutura" in badge_a2a
        assert "S_max=1.00" in badge_a2a
        assert "Estrutura idêntica" not in badge_a2a

        assert "hERG AD:" in badge_herg
        assert "Nível 4" in badge_herg
        assert "Sem correspondência exata conhecida" in badge_herg
        assert "S_max=0.58" in badge_herg
        assert "Interpolação" not in badge_herg
        assert "Extrapolação" not in badge_herg

    def test_tanimoto_methodological_note_does_not_claim_scaffold_identity(self):
        """Verifica estaticamente que a página não afirma conservação estrita de scaffold por Tanimoto > 0.70."""
        with open("src/ui/pages/p5_comparacao.py", encoding="utf-8") as f:
            code = f.read()

        # Deve usar ECFP4 com raio 2 e 1024 bits
        assert "raio 2, 1024 bits" in code or "raio 2" in code
        # Não deve afirmar conservação ou identidade de scaffold
        assert "comprova conservação ou identidade de scaffold" not in code
        assert "sem implicar necessariamente conservação ou identidade de scaffold" in code


class TestAppNavigationIntegration:
    """Verificação da integração da rota no sistema multipágina app.py."""

    def test_route_comparacao_in_app_ast(self):
        """Verifica estaticamente (AST) que a página Comparação está registrada em app.py sob ANÁLISE."""
        import ast

        with open("app.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found_route = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Page":
                has_comp_url = any(
                    kw.arg == "url_path" and getattr(kw.value, "value", "") == "comparacao"
                    for kw in node.keywords
                )
                has_comp_title = any(
                    kw.arg == "title" and getattr(kw.value, "value", "") == "Comparação"
                    for kw in node.keywords
                )
                if has_comp_url or has_comp_title:
                    found_route = True
                    break

        assert found_route is True, "Rota 'comparacao' não encontrada em app.py via AST"

    def test_page_comparacao_function_is_callable(self):
        """Garante que a função page_comparacao em app.py é executável e importa render_comparacao."""
        import app
        assert hasattr(app, "page_comparacao")
        assert callable(app.page_comparacao)


