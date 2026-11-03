"""
Suíte de Testes Automatizados — Etapa 4: Interpretação Química e Computacional (Auditoria Corretiva).

Verifica:
1. Recuperação dos subambientes Morgan (radius=2, nBits=1024).
2. Tratamento de colisões de hashing e multiplicidade com análise de subgrafos canônicos.
3. Destaque estrutural dos fragmentos (subambiente isolado e contexto molecular).
4. Interpretação correta da orientação do sinal SHAP para A2A e hERG (terminologia neutra).
5. Quadro comparativo dos 6 descritores físico-químicos e ressalvas.
6. Resiliência à ausência ou integridade parcial de dados SHAP.
7. Tratamento de bits ausentes (garantia de ausência de imagem e explicação do valor zero).
8. Ausência de alucinações bibliográficas e conformidade do bit fp_926.
9. Resiliência a dados incompletos ou erros de Domínio de Aplicabilidade (AD).
10. Estado vazio amigável da página quando executada sem análise ativa.
11. Teste de aditividade exata do SHAP no modelo hERG (Random Forest e escala de probabilidade).
12. Ausência total de terminologia de "proteção biológica" no classificador hERG.
"""

from __future__ import annotations

import base64
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
import shap

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    featurize_dataframe,
    smiles_to_mol,
)
import src.ui.interpretation_engine as ie
from src.ui.pages.p4_interpretacao import render_interpretacao


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def caffeine_mol() -> Chem.Mol:
    """Cafeína: composto com anel purínico, 3 metilas e colisões conhecidas no bit 33."""
    mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert mol is not None
    return mol


@pytest.fixture
def mock_analysis_caffeine(caffeine_mol: Chem.Mol) -> Dict[str, Any]:
    """Cria payload representativo de current_analysis com Cafeína."""
    shap_dummy_a2a = np.zeros(len(FEATURE_NAMES))
    shap_dummy_a2a[831] = 0.425  # fp_831 positivo (eleva pIC50)
    shap_dummy_a2a[926] = -0.210  # fp_926 negativo (reduz pIC50)
    logp_idx = FEATURE_NAMES.index("MolLogP")
    shap_dummy_a2a[logp_idx] = 0.150

    shap_dummy_herg = np.zeros(len(FEATURE_NAMES))
    shap_dummy_herg[831] = -0.310  # fp_831 negativo (reduz saída positiva hERG)
    shap_dummy_herg[767] = 0.520   # fp_767 positivo (eleva saída positiva hERG)

    return {
        "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "mol": caffeine_mol,
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
            "name": "Cafeína",
            "chembl_id": "CHEMBL113",
            "a2a_in_sample": True,
        },
        "pubchem": {
            "title": "Caffeine",
            "cid": 2519,
        },
        "a2a": {
            "pred_pIC50": 6.82,
            "pred_IC50_nM": 151.3,
            "classification": "Alta Potência",
            "shap_values": shap_dummy_a2a,
            "base_value": 6.20,
            "ad": {
                "status": "ok",
                "provenance": "Correspondência de estrutura",
                "provenance_level": 1,
                "s_max_all": 1.0,
                "s_max_excluding_match": 0.5214,
                "has_exclusion": True,
                "nearest_neighbor": {
                    "id": "CHEMBL113",
                    "similarity": 1.0,
                    "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                    "experimental_label_name": "pIC50 experimental",
                    "experimental_value": 6.85,
                },
                "loo_median_similarity": 0.8219,
            },
        },
        "herg": {
            "prob_herg": 0.18,
            "safety_index": 0.82,
            "classification": "Baixo Risco",
            "shap_values": shap_dummy_herg,
            "base_value": 0.35,
            "ad": {
                "status": "ok",
                "provenance": "Sem correspondência exata conhecida",
                "provenance_level": 4,
                "s_max_all": 0.4850,
                "s_max_excluding_match": None,
                "has_exclusion": False,
                "nearest_neighbor": {
                    "id": "CHEMBL5000",
                    "similarity": 0.4850,
                    "canonical_smiles": "c1ccccc1",
                    "experimental_label_name": "Classificação Experimental",
                    "experimental_value": "Inativo",
                },
                "loo_median_similarity": 0.8308,
            },
        },
        "multiobjective": {
            "quadrant": "Quadrante 1 (Ideal)",
            "x": 6.82,
            "y": 0.82,
        },
    }


# -----------------------------------------------------------------------------
# 1. Recuperação dos Ambientes Morgan
# -----------------------------------------------------------------------------
def test_morgan_environment_recovery(caffeine_mol: Chem.Mol) -> None:
    """Verifica a extração fiel de bitInfo Morgan preservando radius=2 e nBits=1024."""
    bi = ie.extract_morgan_bit_info(caffeine_mol, radius=2, n_bits=1024)
    assert isinstance(bi, dict)
    assert len(bi) > 0

    for bit_id, envs in bi.items():
        assert 0 <= bit_id < 1024
        assert isinstance(envs, tuple)
        assert len(envs) > 0
        for atom_idx, radius in envs:
            assert 0 <= atom_idx < caffeine_mol.GetNumAtoms()
            assert 0 <= radius <= 2


# -----------------------------------------------------------------------------
# 2. Tratamento de Colisões e Múltiplos Ambientes via Subgrafos Canônicos
# -----------------------------------------------------------------------------
def test_collision_handling_and_multiple_environments(caffeine_mol: Chem.Mol) -> None:
    """Verifica a diferenciação entre colisões de hashing e repetições simétricas."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    # Bit 33 na Cafeína possui colisão comprovada: 3 metilas em r=0 e 1 carbono de anel em r=2
    diag_33 = ie.inspect_bit_environments(caffeine_mol, 33, bi)
    assert diag_33["status"] == "present"
    assert diag_33["count"] == 4
    assert diag_33["has_collision"] is True
    assert diag_33["ambiguity_type"] == "hash_collision"
    assert len(diag_33["environments"]) == 4

    # Bit 121 possui 3 ocorrências simétricas em raio 1 (C-N das metilas: submol enraizada idêntica 'n[CH3:1]')
    diag_121 = ie.inspect_bit_environments(caffeine_mol, 121, bi)
    assert diag_121["status"] == "present"
    assert diag_121["count"] == 3
    assert diag_121["has_collision"] is False
    assert diag_121["ambiguity_type"] == "multiple_occurrences"
    for env in diag_121["environments"]:
        assert env["submol_smi"] in ("Cn", "n[CH3:1]")


def test_subtle_collision_detection_same_radius_same_center_symbol() -> None:
    """Verifica colisão em ambientes que possuem o mesmo raio e símbolo central mas subestruturas distintas."""
    # Teste com CHEMBL5182250 (subestrutura com colisão sutil comprovada)
    mol = Chem.MolFromSmiles("COc1ccc(cc1)C(=O)NCC1CCC(CC1)N1CCN(CC1)c1nc2ccccc2s1")
    if mol is not None:
        bi = ie.extract_morgan_bit_info(mol)
        # Se houver qualquer bit com mesmo raio e centro mas subestruturas distintas
        for b_id, envs in bi.items():
            if len(envs) > 1:
                diag = ie.inspect_bit_environments(mol, b_id, bi)
                radii = set(e["radius"] for e in diag["environments"])
                symbols = set(e["symbol"] for e in diag["environments"])
                submols = set(e["submol_smi"] for e in diag["environments"])
                if len(radii) == 1 and len(symbols) == 1 and len(submols) > 1:
                    # Colisão sutil detectada pelo subgrafo canônico
                    assert diag["has_collision"] is True
                    assert diag["ambiguity_type"] == "hash_collision"
                    break


# -----------------------------------------------------------------------------
# 3. Destaque Estrutural dos Fragmentos
# -----------------------------------------------------------------------------
def test_structural_fragment_highlights(caffeine_mol: Chem.Mol) -> None:
    """Verifica a renderização 2D de subambientes e de moléculas destacadas em base64 PNG."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    b64_sub = ie.render_morgan_subenv_png(caffeine_mol, 33, bi, which_example=0)
    assert b64_sub is not None
    assert len(b64_sub) > 500
    raw_sub = base64.b64decode(b64_sub)
    assert raw_sub.startswith(b"\x89PNG\r\n\x1a\n")

    b64_high = ie.render_mol_highlight_png(caffeine_mol, highlight_atoms=[0, 1], highlight_bonds=[0])
    assert b64_high is not None
    assert len(b64_high) > 500
    raw_high = base64.b64decode(b64_high)
    assert raw_high.startswith(b"\x89PNG\r\n\x1a\n")

    assert ie.render_morgan_subenv_png(None, 33, bi) is None
    assert ie.render_mol_highlight_png(None) is None


# -----------------------------------------------------------------------------
# 4. Interpretação Correta do Sinal SHAP para Cada Modelo (Terminologia Neutra)
# -----------------------------------------------------------------------------
def test_shap_sign_orientation_for_each_model(caffeine_mol: Chem.Mol) -> None:
    """Verifica as convenções de direção SHAP e estrita neutralidade terminológica."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    # A2A: Positivo (+) aumenta pIC50, Negativo (-) reduz pIC50
    card_a2a_pos = ie.build_attribute_explanation_card("fp_831", 0.42, "A2A", mol=caffeine_mol, bit_info=bi)
    assert card_a2a_pos["shap_contribution"]["is_positive"] is True
    assert "Eleva a saída predita de afinidade" in card_a2a_pos["shap_contribution"]["direction_desc"]
    assert "aumenta o pIC" in card_a2a_pos["shap_contribution"]["direction_desc"]

    card_a2a_neg = ie.build_attribute_explanation_card("fp_831", -0.31, "A2A", mol=caffeine_mol, bit_info=bi)
    assert card_a2a_neg["shap_contribution"]["is_positive"] is False
    assert "Reduz a saída predita de afinidade" in card_a2a_neg["shap_contribution"]["direction_desc"]
    assert "diminui o pIC" in card_a2a_neg["shap_contribution"]["direction_desc"]

    # hERG: Terminologia obrigatória (sem alegação de proteção biológica)
    card_herg_pos = ie.build_attribute_explanation_card("fp_767", 0.55, "hERG", mol=caffeine_mol, bit_info=bi)
    assert card_herg_pos["shap_contribution"]["is_positive"] is True
    assert "Eleva a saída do modelo associada à classe hERG positiva" in card_herg_pos["shap_contribution"]["direction_desc"]
    assert "protetiv" not in card_herg_pos["shap_contribution"]["direction_desc"].lower()

    card_herg_neg = ie.build_attribute_explanation_card("fp_831", -0.25, "hERG", mol=caffeine_mol, bit_info=bi)
    assert card_herg_neg["shap_contribution"]["is_positive"] is False
    assert "Reduz a saída do modelo associada à classe hERG positiva" in card_herg_neg["shap_contribution"]["direction_desc"]
    assert "protetiv" not in card_herg_neg["shap_contribution"]["direction_desc"].lower()


# -----------------------------------------------------------------------------
# 5. Descritores Físico-Químicos
# -----------------------------------------------------------------------------
def test_physicochemical_descriptors_interpretation(caffeine_mol: Chem.Mol) -> None:
    """Verifica a extração didática dos 6 descritores 2D e ressalvas de não-universalidade."""
    desc_dict = {
        "MolWt": 194.19,
        "MolLogP": -0.07,
        "NumHDonors": 0,
        "NumHAcceptors": 6,
        "TPSA": 58.44,
        "NumRotatableBonds": 0,
    }
    dummy_shap = np.zeros(len(FEATURE_NAMES))
    desc_rows = ie.explain_physicochemical_descriptors(desc_dict, dummy_shap, dummy_shap)

    assert len(desc_rows) == 6
    names = [row["name"] for row in desc_rows]
    for d in DESCRIPTOR_NAMES:
        assert d in names

    card_desc = ie.build_attribute_explanation_card("MolLogP", 0.12, "A2A", mol=caffeine_mol, feature_value=-0.07)
    assert card_desc["computational_attribute"]["is_descriptor"] is True
    assert card_desc["structural_subenvironment"]["status"] == "scalar_descriptor"
    assert "Propriedade físico-química global" in card_desc["structural_subenvironment"]["message"]


# -----------------------------------------------------------------------------
# 6. Ausência de Dados SHAP
# -----------------------------------------------------------------------------
def test_absence_of_shap_data(caffeine_mol: Chem.Mol) -> None:
    """Verifica o comportamento resiliente na ausência de SHAP."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    res_none = ie.get_ranked_explanations(None, "A2A", mol=caffeine_mol, bit_info=bi)
    assert res_none["top_positive"] == []
    assert res_none["top_negative"] == []
    assert res_none["top_absolute"] == []

    res_short = ie.get_ranked_explanations(np.zeros(10), "A2A", mol=caffeine_mol, bit_info=bi)
    assert res_short["top_positive"] == []

    res_desc = ie.explain_physicochemical_descriptors({}, None, None)
    assert len(res_desc) == 6
    for r in res_desc:
        assert r["shap_a2a"] == 0.0
        assert r["shap_herg"] == 0.0


# -----------------------------------------------------------------------------
# 7. Ausência de Subambiente e Garantia de Não Exibição de Imagem
# -----------------------------------------------------------------------------
def test_absence_of_recoverable_subenvironment(caffeine_mol: Chem.Mol) -> None:
    """Verifica que bits ausentes (valor zero) não exibem fragmento 2D e explicam o impacto da ausência."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    diag_904 = ie.inspect_bit_environments(caffeine_mol, 904, bi)
    assert diag_904["status"] == "absent"
    assert diag_904["count"] == 0

    card_904 = ie.build_attribute_explanation_card("fp_904", -0.18, "A2A", mol=caffeine_mol, bit_info=bi)
    assert card_904["structural_subenvironment"]["status"] == "absent"
    assert card_904["structural_subenvironment"]["b64_subenv"] is None
    assert card_904["structural_subenvironment"]["b64_highlight"] is None
    assert "ausente na molécula avaliada (valor = 0" in card_904["structural_subenvironment"]["message"]
    assert "Nenhuma imagem de fragmento é exibida" in card_904["structural_subenvironment"]["message"]


# -----------------------------------------------------------------------------
# 8. Ausência de Referências Bibliográficas e Conformidade do fp_926
# -----------------------------------------------------------------------------
def test_absence_of_bibliographic_references(caffeine_mol: Chem.Mol) -> None:
    """Verifica que o sistema não inventa hipóteses e preserva a definição corrigida do fp_926."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)

    card_unk = ie.build_attribute_explanation_card("fp_1023", 0.04, "A2A", mol=caffeine_mol, bit_info=bi)
    assert card_unk["bibliographic_hypothesis"]["has_reference"] is False
    assert "Sem hipótese bibliográfica documentada" in card_unk["bibliographic_hypothesis"]["text"]

    card_926 = ie.build_attribute_explanation_card("fp_926", -0.15, "A2A", mol=caffeine_mol, bit_info=bi)
    obs = card_926["observational_description"]
    assert "Carbono alifático (raio 0)" in obs
    assert "Sistema Aromático Condensado Substituído" not in obs


# -----------------------------------------------------------------------------
# 9. Dados Incompletos de Domínio de Aplicabilidade (AD)
# -----------------------------------------------------------------------------
def test_incomplete_ad_data_handling() -> None:
    """Verifica a resiliência a diagnósticos AD nulos, incompletos ou em falha técnica."""
    ad_res_none = ie.explain_ad_coverage(None, "A2A")
    assert ad_res_none["status"] == "unavailable"
    assert "não disponível" in ad_res_none["provenance_desc"]

    ad_res_err = ie.explain_ad_coverage({"status": "technical_error"}, "hERG")
    assert ad_res_err["status"] == "unavailable"

    diag_valid = {
        "status": "ok",
        "provenance": "Correspondência de estrutura",
        "provenance_level": 1,
        "s_max_all": 1.0,
        "s_max_excluding_match": 0.52,
        "has_exclusion": True,
        "nearest_neighbor": {"id": "CHEMBL113", "similarity": 1.0},
        "loo_median_similarity": 0.8219,
    }
    ad_res_ok = ie.explain_ad_coverage(diag_valid, "A2A")
    assert ad_res_ok["status"] == "ok"
    assert ad_res_ok["provenance_level"] == 1
    assert ad_res_ok["has_exclusion"] is True


# -----------------------------------------------------------------------------
# 10. Estado Vazio da Página
# -----------------------------------------------------------------------------
def test_empty_state_page_rendering() -> None:
    """Verifica que a função de renderização não quebra quando executada sem análise ativa."""
    try:
        render_interpretacao(analysis=None)
    except Exception as exc:
        pytest.fail(f"render_interpretacao falhou no estado vazio com exceção inesperada: {exc}")


# -----------------------------------------------------------------------------
# 11. Auditoria de Concordância Numérica Aditiva SHAP no Modelo hERG
# -----------------------------------------------------------------------------
def test_herg_shap_exact_additivity_and_scale() -> None:
    """Verifica a concordância numérica aditiva entre a reconstrução SHAP e a saída do classificador hERG (tolerância observada < 1e-3)."""
    model_herg = joblib.load("models/herg_model.joblib")
    explainer_herg = shap.TreeExplainer(model_herg)

    df_test = pd.DataFrame([
        {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "name": "Caffeine"},
        {"smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1", "name": "Terfenadine"}
    ])
    X_feat = featurize_dataframe(df_test, smiles_col="smiles")
    X_sample = X_feat[list(FEATURE_NAMES)]

    probas = model_herg.predict_proba(X_sample)
    shap_raw = explainer_herg(X_sample)

    assert len(shap_raw.values.shape) == 3
    assert shap_raw.values.shape[2] == 2  # Classes [0, 1]

    # Verifica para cada composto a concordância aditiva numérica com tolerância observada < 1e-3
    for i in range(len(df_test)):
        base_val_class1 = shap_raw.base_values[i, 1]
        shap_sum_class1 = np.sum(shap_raw.values[i, :, 1])
        reconstructed = base_val_class1 + shap_sum_class1
        actual_proba = probas[i, 1]

        diff = abs(reconstructed - actual_proba)
        # Resíduos numéricos esperados em agregações de float em ensembles de árvores
        assert diff < 1e-3, f"Falha na concordância aditiva SHAP hERG: diff={diff} (reconstructed={reconstructed}, proba={actual_proba})"


# -----------------------------------------------------------------------------
# 12. Ausência Total de Terminologia de Proteção Biológica
# -----------------------------------------------------------------------------
def test_herg_no_protective_claims(caffeine_mol: Chem.Mol) -> None:
    """Garante que termos como 'protetivo', 'proteção biológica' ou 'cardiac_safe' estão ausentes."""
    bi = ie.extract_morgan_bit_info(caffeine_mol)
    card_neg = ie.build_attribute_explanation_card("fp_831", -0.40, "hERG", mol=caffeine_mol, bit_info=bi)

    desc = card_neg["shap_contribution"]["direction_desc"].lower()
    assert "protetiv" not in desc
    assert "cardiac_safe" not in desc
    assert "proteção" not in desc
    assert "Reduz a saída do modelo associada à classe hERG positiva" in card_neg["shap_contribution"]["direction_desc"]


# -----------------------------------------------------------------------------
# 13. Distinção de Átomo Central via Assinatura Enraizada em Subgrafos Idênticos
# -----------------------------------------------------------------------------
def test_rooted_submol_distinguishes_different_central_atoms_with_same_unrooted_subgraph() -> None:
    """Verifica que ambientes com mesma estrutura não enraizada mas diferentes átomos centrais geram assinaturas distintas."""
    mol = Chem.MolFromSmiles("CCCC")
    # Raio 1 ao redor do átomo 1 (C secundário: C-C-C)
    sig_c1 = ie.compute_subenvironment_signature(mol, 1, 1)
    # Raio 2 ao redor do átomo 0 (C primário: C-C-C)
    sig_c0 = ie.compute_subenvironment_signature(mol, 0, 2)

    # Ambos cobrem os mesmos 3 átomos de carbono (C0, C1, C2), mas com raízes distintas
    assert sig_c1["atom_indices"] == sig_c0["atom_indices"]
    # A assinatura enraizada deve distingui-los categoricamente através do atom map :1
    assert sig_c1["signature"] != sig_c0["signature"]
    assert sig_c1["submol_smi"] != sig_c0["submol_smi"]
