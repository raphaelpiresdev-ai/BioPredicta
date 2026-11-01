"""
Suíte de testes automatizados para a Etapa 5.2 — Gerenciamento do Histórico Não-Destrutivo de Análises.

Verifica:
1. Inicialização do histórico (init_history).
2. Inserção de análises válidas (add_to_history).
3. Preservação integral dos campos do registro (predições A2A/hERG, AD, SHAP, descritores, metadados).
4. Deduplicação inteligente por SMILES canônico.
5. Atualização da posição da molécula reanalisada (reposicionamento no topo/final como mais recente).
6. Funcionamento estrito do limite FIFO de 10 registros.
7. Seleção e recuperação de análises anteriores (set_active_molecule).
8. Independência entre os registros armazenados (isolamento de mutações por cópias profundas).
9. Preservação do histórico durante a digitação/edição de novos SMILES.
10. Compatibilidade de current_analysis gerada pelo histórico com p3_resultados e p4_interpretacao.
11. Rejeição graciosa de entradas inválidas ou incompletas (None, dicionário sem smiles, string vazia).
12. Limpeza segura do histórico (clear_history) sem destruir a análise ativa.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from src.chem_utils import smiles_to_mol
from src.ui.history_manager import (
    MAX_HISTORY_ITEMS,
    SESSION_ACTIVE_KEY,
    SESSION_ACTIVE_SMILES_KEY,
    SESSION_HISTORY_KEY,
    add_to_history,
    clear_history,
    get_history,
    get_history_summary_list,
    init_history,
    set_active_molecule,
)
from src.ui.pages.p3_resultados import render_resultados
from src.ui.pages.p4_interpretacao import render_interpretacao


@pytest.fixture
def mock_session() -> Dict[str, Any]:
    """Cria um dicionário isolado simulando st.session_state para cada teste."""
    return {}


@pytest.fixture
def sample_analysis_caffeine() -> Dict[str, Any]:
    """Retorna uma estrutura de análise completa para Cafeína."""
    smi = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    mol = smiles_to_mol(smi)
    return {
        "timestamp": "2026-10-31T20:00:00",
        "input_method": "smiles",
        "raw_input": smi,
        "canonical_smiles": smi,
        "mol": mol,
        "formula": "C8H10N4O2",
        "mol_wt": 194.19,
        "descriptors": {
            "MolWt": 194.19,
            "MolLogP": -0.07,
            "NumHDonors": 0,
            "NumHAcceptors": 6,
            "TPSA": 58.44,
            "NumRotatableBonds": 0,
        },
        "provenance": {
            "compound_name": "Cafeína",
            "chembl_id": "CHEMBL113",
            "is_in_a2a_curated": True,
        },
        "pubchem": {
            "cid": 2519,
            "title": "Caffeine",
            "iupac_name": "1,3,7-trimethylpurine-2,6-dione",
        },
        "features": pd.DataFrame([{"feat_0": 0.0, "feat_1": 1.0}]),
        "a2a": {
            "pred_pIC50": 5.25,
            "pred_IC50_nM": 5623.4,
            "classification": {"label": "Potência Moderada", "color": "#F59E0B"},
            "ad": {
                "status": "ok",
                "provenance": "Correspondência de estrutura",
                "provenance_level": 1,
                "s_max_all": 1.0,
                "s_max_excluding_match": 0.52,
                "has_exclusion": True,
                "loo_median_similarity": 0.8219,
                "nearest_neighbor": {"id": "CHEMBL113", "similarity": 1.0},
            },
            "shap_values": np.array([0.15, -0.08, 0.42]),
            "base_value": 5.48,
        },
        "herg": {
            "prob_herg": 0.134,
            "safety_index": 0.866,
            "complementary_index": 0.866,
            "classification": {"label": "Baixo Risco", "color": "#00D2BE"},
            "ad": {
                "status": "ok",
                "provenance": "Correspondência de estrutura",
                "provenance_level": 1,
                "s_max_all": 1.0,
                "s_max_excluding_match": 0.52,
                "has_exclusion": True,
                "loo_median_similarity": 0.8219,
                "nearest_neighbor": {"id": "CHEMBL113", "similarity": 1.0},
            },
            "shap_values": np.array([-0.22, 0.05, -0.11]),
            "base_value": 0.50,
        },
        "multiobjective": {
            "quadrant": {
                "quad_title": "Quadrante 2 — Baixo Interesse",
                "quad_badge_cls": "semaphore-yellow",
            },
            "x": 5.25,
            "y": 0.866,
        },
    }


@pytest.fixture
def sample_analysis_istradefylline() -> Dict[str, Any]:
    """Retorna uma estrutura de análise completa para Istradefilina."""
    smi = "Cc1nc2c(n1)c(=O)n(C)c(=O)n2C"
    mol = smiles_to_mol(smi)
    return {
        "timestamp": "2026-10-31T20:05:00",
        "canonical_smiles": smi,
        "mol": mol,
        "formula": "C9H10N4O2",
        "mol_wt": 206.20,
        "pubchem": {"title": "Istradefylline Analogue"},
        "a2a": {
            "pred_pIC50": 7.85,
            "pred_IC50_nM": 14.1,
            "classification": {"label": "Alta Potência", "color": "#00D2BE"},
            "shap_values": np.array([0.65, 0.25]),
            "base_value": 5.48,
        },
        "herg": {
            "prob_herg": 0.180,
            "safety_index": 0.820,
            "classification": {"label": "Baixo Risco", "color": "#00D2BE"},
            "shap_values": np.array([-0.18, 0.02]),
            "base_value": 0.50,
        },
        "multiobjective": {
            "quadrant": {"quad_title": "Quadrante 1 — Zona Ideal"},
            "x": 7.85,
            "y": 0.820,
        },
    }


# -----------------------------------------------------------------------------
# 1. Inicialização do Histórico
# -----------------------------------------------------------------------------
def test_init_history(mock_session: Dict[str, Any]) -> None:
    """Verifica que o histórico é inicializado como lista vazia em estado limpo ou pré-existente."""
    hist = init_history(mock_session)
    assert isinstance(hist, list)
    assert len(hist) == 0
    assert mock_session[SESSION_HISTORY_KEY] == []

    # Chamada idempotente
    hist2 = init_history(mock_session)
    assert hist2 is hist


# -----------------------------------------------------------------------------
# 2. Inserção de Análises Válidas
# -----------------------------------------------------------------------------
def test_add_valid_analysis(mock_session: Dict[str, Any], sample_analysis_caffeine: Dict[str, Any]) -> None:
    """Verifica a adição correta de uma análise válida ao histórico."""
    rec = add_to_history(sample_analysis_caffeine, session_state=mock_session)
    assert rec is not None
    assert rec["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]

    history = get_history(mock_session)
    assert len(history) == 1
    assert history[0]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]


# -----------------------------------------------------------------------------
# 3. Preservação dos Campos do Registro
# -----------------------------------------------------------------------------
def test_field_preservation(mock_session: Dict[str, Any], sample_analysis_caffeine: Dict[str, Any]) -> None:
    """Garante que predições, AD, tensores SHAP e identificação são preservados fielmente."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    stored = get_history(mock_session)[0]

    # Predições A2A e hERG
    assert stored["a2a"]["pred_pIC50"] == 5.25
    assert stored["a2a"]["pred_IC50_nM"] == 5623.4
    assert stored["herg"]["prob_herg"] == 0.134
    assert stored["herg"]["safety_index"] == 0.866

    # AD
    assert stored["a2a"]["ad"]["provenance_level"] == 1
    assert stored["a2a"]["ad"]["s_max_all"] == 1.0

    # SHAP (deve manter o array idêntico)
    assert np.allclose(stored["a2a"]["shap_values"], np.array([0.15, -0.08, 0.42]))
    assert np.allclose(stored["herg"]["shap_values"], np.array([-0.22, 0.05, -0.11]))

    # Identificação PubChem e Proveniência
    assert stored["pubchem"]["cid"] == 2519
    assert stored["provenance"]["chembl_id"] == "CHEMBL113"

    # Objetos pesados RDKit Mol não são armazenados no histórico
    assert stored["mol"] is None


# -----------------------------------------------------------------------------
# 4. Deduplicação por SMILES Canônico
# -----------------------------------------------------------------------------
def test_deduplication_by_canonical_smiles(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
) -> None:
    """Ao analisar a mesma molécula novamente, não deve criar entrada duplicada."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    assert len(get_history(mock_session)) == 1

    # Reanálise da mesma molécula
    reanalysis = copy.deepcopy(sample_analysis_caffeine)
    reanalysis["timestamp"] = "2026-10-31T20:10:00"
    reanalysis["a2a"]["pred_pIC50"] = 5.28

    add_to_history(reanalysis, session_state=mock_session)
    history = get_history(mock_session)
    assert len(history) == 1
    # Registro foi atualizado
    assert history[0]["a2a"]["pred_pIC50"] == 5.28
    assert history[0]["timestamp"] == "2026-10-31T20:10:00"


# -----------------------------------------------------------------------------
# 5. Atualização da Posição da Molécula Reanalisada (Mais Recente)
# -----------------------------------------------------------------------------
def test_reanalyzed_molecule_repositioned_to_most_recent(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
    sample_analysis_istradefylline: Dict[str, Any],
) -> None:
    """Quando uma molécula já existente é reanalisada, deve ir para o final da lista (mais recente)."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    add_to_history(sample_analysis_istradefylline, session_state=mock_session)

    # Ordem inicial: [Cafeína, Istradefilina]
    assert get_history(mock_session)[0]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]
    assert get_history(mock_session)[1]["canonical_smiles"] == sample_analysis_istradefylline["canonical_smiles"]

    # Reanalisa Cafeína
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    history = get_history(mock_session)
    assert len(history) == 2
    # Nova ordem: [Istradefilina, Cafeína]
    assert history[0]["canonical_smiles"] == sample_analysis_istradefylline["canonical_smiles"]
    assert history[1]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]


# -----------------------------------------------------------------------------
# 6. Funcionamento Estrito do Limite FIFO de 10 Registros
# -----------------------------------------------------------------------------
def test_fifo_limit_enforcement(mock_session: Dict[str, Any]) -> None:
    """Garante que a política FIFO descarta os registros mais antigos ao atingir 10 itens."""
    for i in range(15):
        smi = "C" * (i + 1)
        analysis = {
            "canonical_smiles": smi,
            "formula": f"C{i+1}",
            "a2a": {"pred_pIC50": float(i)},
        }
        add_to_history(analysis, session_state=mock_session, max_size=MAX_HISTORY_ITEMS)

    history = get_history(mock_session)
    assert len(history) == 10
    # Os primeiros 5 ('C', 'CC', 'CCC', 'CCCC', 'CCCCC') foram descartados
    assert history[0]["canonical_smiles"] == "CCCCCC"
    assert history[-1]["canonical_smiles"] == "C" * 15


# -----------------------------------------------------------------------------
# 7. Seleção e Recuperação de Análises Anteriores (set_active_molecule)
# -----------------------------------------------------------------------------
def test_set_active_molecule(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
    sample_analysis_istradefylline: Dict[str, Any],
) -> None:
    """Verifica a restauração de uma análise do histórico como a molécula ativa."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    add_to_history(sample_analysis_istradefylline, session_state=mock_session)

    # Ativa Cafeína
    act = set_active_molecule(sample_analysis_caffeine["canonical_smiles"], session_state=mock_session)
    assert act is not None
    assert mock_session[SESSION_ACTIVE_KEY] is not None
    assert mock_session[SESSION_ACTIVE_SMILES_KEY] == sample_analysis_caffeine["canonical_smiles"]
    assert mock_session[SESSION_ACTIVE_KEY]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]

    # Objeto RDKit Mol foi reconstruído com sucesso
    assert act["mol"] is not None
    assert isinstance(act["mol"], Chem.Mol)
    assert act["mol"].GetNumAtoms() > 0

    # Busca por molécula inexistente deve retornar None sem alterar estado
    res_none = set_active_molecule("NON_EXISTENT_SMILES", session_state=mock_session)
    assert res_none is None
    assert mock_session[SESSION_ACTIVE_SMILES_KEY] == sample_analysis_caffeine["canonical_smiles"]


# -----------------------------------------------------------------------------
# 8. Independência entre Registros (Ausência de Referência Compartilhada)
# -----------------------------------------------------------------------------
def test_independent_copies_no_side_effects(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
) -> None:
    """Modificações no dicionário original não devem alterar o registro armazenado no histórico."""
    original = copy.deepcopy(sample_analysis_caffeine)
    add_to_history(original, session_state=mock_session)

    # Modifica o dicionário original
    original["a2a"]["pred_pIC50"] = 99.99
    original["a2a"]["shap_values"][0] = 999.0

    stored = get_history(mock_session)[0]
    assert stored["a2a"]["pred_pIC50"] == 5.25
    assert stored["a2a"]["shap_values"][0] == 0.15


# -----------------------------------------------------------------------------
# 9. Preservação do Histórico Durante Edição de Novos SMILES
# -----------------------------------------------------------------------------
def test_history_preserved_during_form_editing(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
) -> None:
    """Verifica que o histórico permanece intacto quando o usuário altera o campo de entrada."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    set_active_molecule(sample_analysis_caffeine["canonical_smiles"], session_state=mock_session)

    # Simula edição do formulário pelo usuário para uma nova estrutura ainda não analisada
    mock_session["input_smiles_field"] = "c1ccccc1"
    mock_session["field_smiles_text"] = "c1ccccc1"

    # O histórico e a molécula ativa anterior não foram apagados
    assert len(get_history(mock_session)) == 1
    assert mock_session[SESSION_ACTIVE_SMILES_KEY] == sample_analysis_caffeine["canonical_smiles"]
    assert mock_session[SESSION_ACTIVE_KEY]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]


# -----------------------------------------------------------------------------
# 10. Compatibilidade de current_analysis com p3_resultados e p4_interpretacao
# -----------------------------------------------------------------------------
def test_active_analysis_compatibility_with_pages(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
) -> None:
    """Garante que a análise restaurada pelo histórico é consumível por p3 e p4 sem erro."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    active_analysis = set_active_molecule(sample_analysis_caffeine["canonical_smiles"], session_state=mock_session)

    assert active_analysis is not None
    # Executa renderização de Resultados passando a análise recuperada
    try:
        render_resultados(analysis=active_analysis)
    except Exception as exc:
        pytest.fail(f"render_resultados falhou ao consumir análise recuperada do histórico: {exc}")

    # Executa renderização de Interpretação passando a análise recuperada
    try:
        render_interpretacao(analysis=active_analysis)
    except Exception as exc:
        pytest.fail(f"render_interpretacao falhou ao consumir análise recuperada do histórico: {exc}")


# -----------------------------------------------------------------------------
# 11. Rejeição Graciosa de Entradas Inválidas ou Incompletas
# -----------------------------------------------------------------------------
def test_rejection_of_invalid_or_incomplete_records(mock_session: Dict[str, Any]) -> None:
    """Entradas nulas, vazias ou sem canonical_smiles devem ser rejeitadas sem corromper o histórico."""
    init_history(mock_session)

    # None
    assert add_to_history(None, session_state=mock_session) is None
    # Dicionário vazio
    assert add_to_history({}, session_state=mock_session) is None
    # Dicionário sem canonical_smiles
    assert add_to_history({"formula": "C6H6"}, session_state=mock_session) is None
    # canonical_smiles vazio
    assert add_to_history({"canonical_smiles": ""}, session_state=mock_session) is None
    # canonical_smiles apenas espaços
    assert add_to_history({"canonical_smiles": "   "}, session_state=mock_session) is None

    assert len(get_history(mock_session)) == 0


# -----------------------------------------------------------------------------
# 12. Limpeza Segura do Histórico (clear_history)
# -----------------------------------------------------------------------------
def test_clear_history(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
) -> None:
    """Verifica que clear_history limpa o histórico sem anular a análise ativa."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    set_active_molecule(sample_analysis_caffeine["canonical_smiles"], session_state=mock_session)

    assert len(get_history(mock_session)) == 1
    clear_history(session_state=mock_session)
    assert len(get_history(mock_session)) == 0
    # current_analysis permanece intocada
    assert mock_session[SESSION_ACTIVE_KEY] is not None
    assert mock_session[SESSION_ACTIVE_SMILES_KEY] == sample_analysis_caffeine["canonical_smiles"]


# -----------------------------------------------------------------------------
# 13. Formatação da Lista de Resumo do Histórico (get_history_summary_list)
# -----------------------------------------------------------------------------
def test_history_summary_list_presentation(
    mock_session: Dict[str, Any],
    sample_analysis_caffeine: Dict[str, Any],
    sample_analysis_istradefylline: Dict[str, Any],
) -> None:
    """Verifica a correta ordenação e extração de métricas para os cartões de histórico."""
    add_to_history(sample_analysis_caffeine, session_state=mock_session)
    add_to_history(sample_analysis_istradefylline, session_state=mock_session)
    set_active_molecule(sample_analysis_istradefylline["canonical_smiles"], session_state=mock_session)

    summaries = get_history_summary_list(session_state=mock_session)
    assert len(summaries) == 2

    # Ordem reversa: a mais recente (Istradefilina) é o primeiro item
    assert summaries[0]["canonical_smiles"] == sample_analysis_istradefylline["canonical_smiles"]
    assert summaries[0]["is_active"] is True
    assert summaries[0]["pIC50"] == 7.85

    # O segundo item é Cafeína
    assert summaries[1]["canonical_smiles"] == sample_analysis_caffeine["canonical_smiles"]
    assert summaries[1]["is_active"] is False
    assert summaries[1]["pIC50"] == 5.25
