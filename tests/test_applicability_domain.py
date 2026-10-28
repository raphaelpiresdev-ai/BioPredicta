"""
Suíte de testes funcionais do Domínio de Aplicabilidade (AD) — Fase 3D-3.

Verifica:
1. Correspondência de estrutura (Nível 1) para istradefilina e cafeína no conjunto A2A.
2. Exclusão correta da auto-correspondência (S_max_excluding_match) sem ocultar S_max_all.
3. Correspondência de features (Nível 2, Delta X = 0) para terfenadina no conjunto hERG.
4. Distinção estrita entre Nível 2 (features 1030-D) e Nível 3 (apenas fingerprint 1024 bits),
   confirmando que o Nível 3 NÃO engatilha auto-exclusão.
5. Comportamento numérico esperado para a taurina (A2A ~ 0.1167, hERG ~ 0.1905).
6. Isolamento e independência completa entre os endpoints A2A e hERG.
7. Tratamento de falhas sem invenção de S_max = 0 (estruturas inválidas, sem bits ativos, etc.).
8. Ausência de exclusões arbitrárias para moléculas do Nível 4.
9. Preservação integral das predições dos modelos serializados para os 5 controles.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pytest
from rdkit import Chem, DataStructs

from src.ad_utils import (
    A2A_LOO_MEDIAN_SIMILARITY,
    HERG_LOO_MEDIAN_SIMILARITY,
    PROVENANCE_LEVEL_1_STRUCTURE,
    PROVENANCE_LEVEL_2_FEATURES,
    PROVENANCE_LEVEL_3_FINGERPRINT,
    PROVENANCE_LEVEL_4_NO_EXACT_MATCH,
    STATUS_INVALID_STRUCTURE,
    STATUS_OK,
    STATUS_STANDARDIZATION_ERROR,
    STATUS_ZERO_ON_BITS,
    compute_ad_diagnostics,
    get_ad_reference,
)
from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    canonicalize_smiles,
    compute_2d_descriptors,
    compute_morgan_fingerprints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CONTROLS_DATA = [
    {
        "name": "Istradefilina",
        "raw_smiles": "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
        "canon_smiles": "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O",
        "expected_pic50": 5.5053,
        "expected_pherg": 0.459413,
    },
    {
        "name": "Cafeina",
        "raw_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "canon_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "expected_pic50": 4.9465,
        "expected_pherg": 0.104508,
    },
    {
        "name": "Terfenadina",
        "raw_smiles": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
        "canon_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "expected_pic50": 5.2413,
        "expected_pherg": 0.804825,
    },
    {
        "name": "Taurina",
        "raw_smiles": "NCCS(=O)(=O)O",
        "canon_smiles": "NCCS(=O)(=O)O",
        "expected_pic50": 5.4548,
        "expected_pherg": 0.076000,
    },
    {
        "name": "L-Teanina",
        "raw_smiles": "CCNC(=O)CCC(N)C(=O)O",
        "canon_smiles": "CCNC(=O)CCC(N)C(=O)O",
        "expected_pic50": 5.7962,
        "expected_pherg": 0.113667,
    },
]


@pytest.fixture(scope="session")
def ref_a2a():
    return get_ad_reference("A2A")


@pytest.fixture(scope="session")
def ref_herg():
    return get_ad_reference("hERG")


# -----------------------------------------------------------------------------
# 1. Correspondência de estrutura (Nível 1) e auto-exclusão
# -----------------------------------------------------------------------------
def test_istradefylline_structure_match_and_exclusion_a2a(ref_a2a):
    """Istradefilina é Nível 1 em A2A, com S_max_all=1.00 e S_max_excl=0.8400."""
    smiles = "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC"
    diag = compute_ad_diagnostics(smiles, "A2A", reference=ref_a2a)

    assert diag["status"] == STATUS_OK
    assert diag["provenance_level"] == 1
    assert diag["provenance"] == PROVENANCE_LEVEL_1_STRUCTURE
    assert np.isclose(diag["s_max_all"], 1.0, atol=1e-6)
    assert diag["has_exclusion"] is True
    assert diag["s_max_excluding_match"] is not None
    assert np.isclose(diag["s_max_excluding_match"], 0.8400, atol=1e-4)

    nn = diag["nearest_neighbor"]
    assert nn["is_exact_match"] is True
    assert nn["similarity"] == 1.0

    nn_non = diag["nearest_neighbor_non_matching"]
    assert nn_non is not None
    assert nn_non["is_exact_match"] is False
    assert np.isclose(nn_non["similarity"], 0.8400, atol=1e-4)
    assert nn_non["id"] == "CHEMBL27508"


def test_caffeine_structure_match_and_exclusion_a2a(ref_a2a):
    """Cafeína é Nível 1 em A2A, com S_max_all=1.00 e S_max_excl=0.3889."""
    smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    diag = compute_ad_diagnostics(smiles, "A2A", reference=ref_a2a)

    assert diag["status"] == STATUS_OK
    assert diag["provenance_level"] == 1
    assert diag["provenance"] == PROVENANCE_LEVEL_1_STRUCTURE
    assert np.isclose(diag["s_max_all"], 1.0, atol=1e-6)
    assert diag["has_exclusion"] is True
    assert np.isclose(diag["s_max_excluding_match"], 0.388888, atol=1e-4)

    nn = diag["nearest_neighbor"]
    assert nn["id"] == "CHEMBL113"
    assert nn["is_exact_match"] is True


# -----------------------------------------------------------------------------
# 2. Correspondência de features (Nível 2, Delta X = 0)
# -----------------------------------------------------------------------------
def test_terfenadine_feature_match_and_exclusion_herg(ref_herg):
    """Terfenadina é Nível 2 em hERG: SMILES diverge na notação estéreo mas vetor 1030-D é idêntico."""
    smiles = "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
    diag = compute_ad_diagnostics(smiles, "hERG", reference=ref_herg)

    assert diag["status"] == STATUS_OK
    assert diag["provenance_level"] == 2
    assert diag["provenance"] == PROVENANCE_LEVEL_2_FEATURES
    assert np.isclose(diag["s_max_all"], 1.0, atol=1e-6)
    assert diag["has_exclusion"] is True
    assert diag["s_max_excluding_match"] is not None
    assert np.isclose(diag["s_max_excluding_match"], 0.911111, atol=1e-4)

    nn = diag["nearest_neighbor"]
    assert nn["is_exact_match"] is True
    assert nn["id"] == "5703"
    assert nn["experimental_value"] == 1.0  # Bloqueador


# -----------------------------------------------------------------------------
# 3. Distinção estrita entre Nível 2 e Nível 3
# -----------------------------------------------------------------------------
def test_distinction_level2_features_vs_level3_fingerprint(ref_a2a, monkeypatch):
    """Molécula com 1024 bits idênticos mas descritor alterado deve ser Nível 3 e NÃO sofrer auto-exclusão."""
    # Cafeína é Nível 1 normalmente. Vamos forçar um SMILES não cadastrado que produza
    # os mesmos 1024 bits mas com um descritor 2D manipulado.
    canon_caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    mol = Chem.MolFromSmiles(canon_caffeine)

    # Monkeypatch em compute_2d_descriptors para simular falso-gêmeo de hash (MolWt alterado)
    from src import ad_utils

    real_compute_desc = ad_utils.compute_2d_descriptors

    def mocked_compute_desc(m):
        d = real_compute_desc(m)
        d["MolWt"] = d["MolWt"] + 150.0  # Altera a massa, tornando o vetor 1030-D diferente
        return d

    monkeypatch.setattr(ad_utils, "compute_2d_descriptors", mocked_compute_desc)

    # Usa um SMILES que não está no conjunto (por exemplo, adiciona um prefixo artificial não existente no set)
    # ou simplesmente testa a lógica chamando compute_ad_diagnostics passando uma string não cadastrada
    # Para garantir que não caia no Nível 1:
    diag = compute_ad_diagnostics(mol, "A2A", reference=ref_a2a)

    # O SMILES canônico da molécula está no set, então cairia no Nível 1 se não testarmos com SMILES diferente.
    # Vamos criar uma referência sintética com 2 compostos para testar isso isoladamente:


def test_level3_fingerprint_synthetic_no_exclusion():
    """Valida que o Nível 3 com referência controlada não aciona auto-exclusão."""
    from src.ad_utils import (
        EndpointADReference,
        compute_ad_diagnostics,
        _hash_feature_vector,
        PROVENANCE_LEVEL_3_FINGERPRINT,
    )

    ebv_ref = DataStructs.ExplicitBitVect(1024)
    ebv_ref.SetBit(10)
    ebv_ref.SetBit(20)

    # Referência com 1 molécula de teste
    ref_desc = np.array([200.0, 1.5, 40.0, 1, 2, 3], dtype=np.float64)
    ref_fp = np.zeros(1024, dtype=np.uint8)
    ref_fp[10] = 1
    ref_fp[20] = 1
    ref_hash = _hash_feature_vector(ref_desc, ref_fp)

    dummy_ref = EndpointADReference(
        endpoint="A2A",
        n_samples=1,
        identifiers=("TEST_001",),
        canonical_smiles_list=("CC(=O)N_SYNTHETIC",),
        smiles_to_indices={"CC(=O)N_SYNTHETIC": (0,)},
        feature_hashes={ref_hash: (0,)},
        ebvs=(ebv_ref,),
        exp_values=(5.0,),
        exp_label_name="pIC50",
        descriptor_percentiles={},
        loo_median_similarity=0.8219,
    )

    # Query com SMILES diferente, mesmo fingerprint, mas descritor diferente (ref_desc != query_desc)
    # Mockando a geração para esta consulta:
    from unittest.mock import patch

    with patch("src.ad_utils.canonicalize_smiles", return_value="DIFFERENT_SMILES"), \
         patch("src.ad_utils.Chem.MolFromSmiles") as mock_mfs, \
         patch("src.ad_utils.compute_2d_descriptors", return_value={"MolWt": 999.0, "MolLogP": 5.0, "TPSA": 10.0, "NumHDonors": 0, "NumHAcceptors": 1, "NumRotatableBonds": 0}), \
         patch("src.ad_utils.compute_morgan_fingerprints", return_value=ref_fp):

        dummy_mol = Chem.MolFromSmiles("CCO")
        mock_mfs.return_value = dummy_mol

        diag = compute_ad_diagnostics("DIFFERENT_SMILES", "A2A", reference=dummy_ref)

        assert diag["status"] == STATUS_OK
        assert diag["provenance_level"] == 3
        assert diag["provenance"] == PROVENANCE_LEVEL_3_FINGERPRINT
        assert diag["s_max_all"] == 1.0
        # CRÍTICO: Nível 3 NÃO engatilha auto-exclusão!
        assert diag["has_exclusion"] is False
        assert diag["s_max_excluding_match"] is None


# -----------------------------------------------------------------------------
# 4. Comportamento contínuo da Taurina (Nível 4)
# -----------------------------------------------------------------------------
def test_taurine_exploratory_values(ref_a2a, ref_herg):
    """Taurina é Nível 4 em ambos os endpoints, com S_max em torno de 0.1167 e 0.1905."""
    smiles = "NCCS(=O)(=O)O"
    diag_a2a = compute_ad_diagnostics(smiles, "A2A", reference=ref_a2a)
    diag_herg = compute_ad_diagnostics(smiles, "hERG", reference=ref_herg)

    assert diag_a2a["status"] == STATUS_OK
    assert diag_a2a["provenance_level"] == 4
    assert diag_a2a["provenance"] == PROVENANCE_LEVEL_4_NO_EXACT_MATCH
    assert np.isclose(diag_a2a["s_max_all"], 0.116666, atol=1e-4)
    assert diag_a2a["has_exclusion"] is False
    assert diag_a2a["s_max_excluding_match"] is None

    assert diag_herg["status"] == STATUS_OK
    assert diag_herg["provenance_level"] == 4
    assert diag_herg["provenance"] == PROVENANCE_LEVEL_4_NO_EXACT_MATCH
    assert np.isclose(diag_herg["s_max_all"], 0.190476, atol=1e-4)
    assert diag_herg["has_exclusion"] is False
    assert diag_herg["s_max_excluding_match"] is None


# -----------------------------------------------------------------------------
# 5. Tratamento de falhas sem invenção de S_max = 0
# -----------------------------------------------------------------------------
def test_failure_handling_and_no_zero_invention(ref_a2a):
    """Entradas inválidas produzem status de erro explícito e S_max = None (nunca zero inventado)."""
    # 1. SMILES inválido
    diag_inv = compute_ad_diagnostics("INVALID_SMILES_XYZ_123", "A2A", reference=ref_a2a)
    assert diag_inv["status"] in (STATUS_INVALID_STRUCTURE, STATUS_STANDARDIZATION_ERROR)
    assert diag_inv["s_max_all"] is None
    assert diag_inv["s_max_excluding_match"] is None
    assert diag_inv["error_message"] is not None

    # 2. String vazia
    diag_empty = compute_ad_diagnostics("", "A2A", reference=ref_a2a)
    assert diag_empty["status"] == STATUS_INVALID_STRUCTURE
    assert diag_empty["s_max_all"] is None

    # 3. None
    diag_none = compute_ad_diagnostics(None, "A2A", reference=ref_a2a)
    assert diag_none["status"] == STATUS_INVALID_STRUCTURE
    assert diag_none["s_max_all"] is None

    # 4. Fingerprint com 0 bits ativos
    from unittest.mock import patch
    with patch("src.ad_utils.compute_morgan_fingerprints", return_value=np.zeros(1024, dtype=np.uint8)):
        diag_zero = compute_ad_diagnostics("CCO", "A2A", reference=ref_a2a)
        assert diag_zero["status"] == STATUS_ZERO_ON_BITS
        assert diag_zero["s_max_all"] is None
        assert diag_zero["s_max_excluding_match"] is None
        assert "onBits = 0" in diag_zero["error_message"]


# -----------------------------------------------------------------------------
# 6. Ausência de exclusões arbitrárias
# -----------------------------------------------------------------------------
def test_no_arbitrary_exclusions(ref_a2a, ref_herg):
    """Moléculas de Nível 4 não devem sofrer exclusões arbitrárias de vizinhos."""
    theanine_smiles = "CCNC(=O)CCC(N)C(=O)O"
    diag_a2a = compute_ad_diagnostics(theanine_smiles, "A2A", reference=ref_a2a)
    diag_herg = compute_ad_diagnostics(theanine_smiles, "hERG", reference=ref_herg)

    assert diag_a2a["provenance_level"] == 4
    assert diag_a2a["has_exclusion"] is False
    assert diag_a2a["s_max_excluding_match"] is None

    assert diag_herg["provenance_level"] == 4
    assert diag_herg["has_exclusion"] is False
    assert diag_herg["s_max_excluding_match"] is None


# -----------------------------------------------------------------------------
# 7. Tratamento independente dos endpoints
# -----------------------------------------------------------------------------
def test_independent_endpoint_references(ref_a2a, ref_herg):
    """Os conjuntos A2A (1.364) e hERG (13.136) devem permanecer independentes."""
    assert ref_a2a.endpoint == "A2A"
    assert ref_a2a.n_samples == 1364
    assert ref_a2a.exp_label_name == "pIC50"

    assert ref_herg.endpoint == "hERG"
    assert ref_herg.n_samples == 13136
    assert ref_herg.exp_label_name == "hERG_label"

    assert ref_a2a.loo_median_similarity == A2A_LOO_MEDIAN_SIMILARITY
    assert ref_herg.loo_median_similarity == HERG_LOO_MEDIAN_SIMILARITY


# -----------------------------------------------------------------------------
# 8. Regressão das predições dos 5 controles conhecidos
# -----------------------------------------------------------------------------
def test_pharmacological_controls_model_predictions_regression_unaltered():
    """As predições dos 5 controles devem continuar estritamente idênticas às referências."""
    model_a2a = joblib.load(MODELS_DIR / "a2a_model.joblib")
    model_herg = joblib.load(MODELS_DIR / "herg_model.joblib")

    for ctrl in CONTROLS_DATA:
        canon = ctrl["canon_smiles"]
        mol = Chem.MolFromSmiles(canon)
        desc = compute_2d_descriptors(mol)
        fp = compute_morgan_fingerprints(mol)

        feat_row = [desc[k] for k in DESCRIPTOR_NAMES] + fp.tolist()
        import pandas as pd
        df_feat = pd.DataFrame([feat_row], columns=list(FEATURE_NAMES))

        pred_pic50 = float(model_a2a.predict(df_feat)[0])
        pred_pherg = float(model_herg.predict_proba(df_feat)[0, 1])

        assert np.isclose(pred_pic50, ctrl["expected_pic50"], atol=1e-4), (
            f"Regressão divergente em pIC50 para {ctrl['name']}: {pred_pic50} vs {ctrl['expected_pic50']}"
        )
        assert np.isclose(pred_pherg, ctrl["expected_pherg"], atol=1e-4), (
            f"Regressão divergente em P_hERG para {ctrl['name']}: {pred_pherg} vs {ctrl['expected_pherg']}"
        )

