"""
Suíte de testes de regressão da Fase 3A — Featurização em Lote e Padronização Química.

Verifica:
1. Paridade estrutural entre SMILES bruto (Kekulé) e SMILES canônico da istradefilina.
2. Remoção de sais/contraíons (cafeína neutra vs. sais) produzindo features idênticas.
3. Tratamento resiliente de SMILES inválidos em lote (preservação de shape e índice com NaN/zeros).
4. Dimensão (1.030 features), tipos de dados (uint8 para fingerprints) e ordem canônica (FEATURE_NAMES).
5. Regressão das predições dos modelos serializados (A2A e hERG) para os 5 controles farmacológicos.
6. Paridade integral com os arquivos Parquet curados existentes (sem regravá-los).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    FINGERPRINT_NAMES,
    canonicalize_smiles,
    featurize_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Tolerância numérica para descritores contínuos 2D (MolWt, MolLogP, TPSA):
# justificada pela precisão de ponto flutuante IEEE 754 (float64 ~ 1e-15).
DESCRIPTOR_RTOL = 1e-12
DESCRIPTOR_ATOL = 1e-12

# Tolerância numérica para comparação das predições dos modelos contra os valores
# de referência auditados na Fase 2A (registrados com 4 a 6 casas decimais).
PREDICTION_RTOL = 1e-4
PREDICTION_ATOL = 1e-4


def _sha256_file(path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em disco para comprovar imutabilidade."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 1. Paridade estrutural (SMILES bruto vs. canônico da Istradefilina)
# -----------------------------------------------------------------------------
def test_istradefylline_structural_parity():
    """SMILES bruto (Kekulé) e canônico da istradefilina devem gerar as mesmas 1.030 features."""
    raw_smiles = "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC"
    canon_smiles = "CCn1c(=O)c2c(nc(/C=C/c3ccc(OC)c(OC)c3)n2C)n(CC)c1=O"

    assert canonicalize_smiles(raw_smiles) == canon_smiles
    assert canonicalize_smiles(canon_smiles) == canon_smiles

    df_pair = pd.DataFrame(
        {"smiles": [raw_smiles, canon_smiles]},
        index=["istradefylline_raw", "istradefylline_canon"],
    )
    df_feat = featurize_dataframe(df_pair, smiles_col="smiles")

    vec_raw = df_feat.loc["istradefylline_raw", list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    vec_canon = df_feat.loc["istradefylline_canon", list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(
        vec_raw,
        vec_canon,
        rtol=DESCRIPTOR_RTOL,
        atol=DESCRIPTOR_ATOL,
        err_msg="Diferença detectada entre SMILES bruto e canônico da istradefilina.",
    )


# -----------------------------------------------------------------------------
# 2. Padronização de sais e contraíons
# -----------------------------------------------------------------------------
def test_caffeine_salt_stripping_and_feature_parity():
    """Cafeína neutra e representações com contraíon/sal devem produzir as mesmas features."""
    caffeine_neutral = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    caffeine_hcl = "Cn1c(=O)c2c(ncn2C)n(C)c1=O.Cl"
    caffeine_nacl = "Cn1c(=O)c2c(ncn2C)n(C)c1=O.[Na+].[Cl-]"

    assert canonicalize_smiles(caffeine_hcl) == caffeine_neutral
    assert canonicalize_smiles(caffeine_nacl) == caffeine_neutral

    df_salts = pd.DataFrame(
        {"smiles": [caffeine_neutral, caffeine_hcl, caffeine_nacl]},
        index=["neutral", "hcl_salt", "nacl_mixture"],
    )
    df_feat = featurize_dataframe(df_salts, smiles_col="smiles")

    ref_vec = df_feat.loc["neutral", list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    hcl_vec = df_feat.loc["hcl_salt", list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    nacl_vec = df_feat.loc["nacl_mixture", list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(hcl_vec, ref_vec, rtol=DESCRIPTOR_RTOL, atol=DESCRIPTOR_ATOL)
    np.testing.assert_allclose(nacl_vec, ref_vec, rtol=DESCRIPTOR_RTOL, atol=DESCRIPTOR_ATOL)


# -----------------------------------------------------------------------------
# 3. Tratamento de SMILES inválidos em lote e preservação de índices
# -----------------------------------------------------------------------------
def test_batch_with_invalid_smiles_preserves_index_and_shape():
    """Lote misto (válidos + inválidos) não deve lançar exceção e deve sinalizar inválidos com NaN/zeros."""
    custom_index = [
        "ctrl_caffeine",
        "inv_dev_md_unkekulizable",
        "ctrl_istradefylline",
        "inv_syntax",
        "inv_empty",
        "inv_none",
        "ctrl_taurine",
    ]
    df_batch = pd.DataFrame(
        {
            "mol_id": [101, 102, 103, 104, 105, 106, 107],
            "smiles": [
                "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "Cc1nc2c(n1)n(c(=O)n(c2=O)C)C",  # SMILES inválido documentado no relatório
                "CCN1C(=O)C2=C(N=C(N2C)/C=C/C3=CC(=C(C=C3)OC)OC)N(C1=O)CC",
                "INVALID_SMILES_STRING_###",
                "   ",
                None,
                "NCCS(=O)(=O)O",
            ],
        },
        index=custom_index,
    )

    df_out = featurize_dataframe(df_batch, smiles_col="smiles")

    # Preservação exata do shape e do alinhamento de índices
    assert df_out.shape == (len(df_batch), len(df_batch.columns) + 1030)
    assert list(df_out.index) == custom_index
    assert (df_out["mol_id"] == df_batch["mol_id"]).all()

    valid_keys = ["ctrl_caffeine", "ctrl_istradefylline", "ctrl_taurine"]
    invalid_keys = ["inv_dev_md_unkekulizable", "inv_syntax", "inv_empty", "inv_none"]

    # Linhas inválidas devem ter NaN em todos os 6 descritores e 0 nos 1024 bits
    desc_isna_all = df_out[list(DESCRIPTOR_NAMES)].isna().all(axis=1)
    assert desc_isna_all.loc[invalid_keys].all()
    assert not desc_isna_all.loc[valid_keys].any()

    fp_invalid = df_out.loc[invalid_keys, list(FINGERPRINT_NAMES)].to_numpy()
    assert np.all(fp_invalid == 0)

    # Linhas válidas no lote misto devem coincidir com a featurização individual
    df_valid_only = featurize_dataframe(df_batch.loc[valid_keys], smiles_col="smiles")
    np.testing.assert_allclose(
        df_out.loc[valid_keys, list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
        df_valid_only.loc[valid_keys, list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
        rtol=DESCRIPTOR_RTOL,
        atol=DESCRIPTOR_ATOL,
    )


# -----------------------------------------------------------------------------
# 4. Dimensão, ordem das 1.030 features e tipos de dados
# -----------------------------------------------------------------------------
def test_feature_dimension_and_canonical_order():
    """Cada vetor válido deve conter exatamente 1.030 features na ordem definida por FEATURE_NAMES."""
    assert len(DESCRIPTOR_NAMES) == 6
    assert len(FINGERPRINT_NAMES) == 1024
    assert len(FEATURE_NAMES) == 1030
    assert FEATURE_NAMES == DESCRIPTOR_NAMES + FINGERPRINT_NAMES

    df_in = pd.DataFrame({"canonical_smiles": ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]})
    df_out = featurize_dataframe(df_in, smiles_col="canonical_smiles")

    feature_cols = list(df_out.columns[1:])
    assert feature_cols == list(FEATURE_NAMES)
    assert df_out[list(FEATURE_NAMES)].shape == (1, 1030)
    assert not df_out[list(FEATURE_NAMES)].isna().any().any()

    for fp_col in FINGERPRINT_NAMES:
        assert df_out[fp_col].dtype == np.uint8


# -----------------------------------------------------------------------------
# 5. Regressão das predições dos 5 controles farmacológicos
# -----------------------------------------------------------------------------
def test_pharmacological_controls_model_predictions_regression():
    """As predições de pIC50 (A2A) e P_hERG para os 5 controles devem permanecer inalteradas."""
    a2a_path = MODELS_DIR / "a2a_model.joblib"
    herg_path = MODELS_DIR / "herg_model.joblib"

    hash_a2a_before = _sha256_file(a2a_path)
    hash_herg_before = _sha256_file(herg_path)

    model_a2a = joblib.load(a2a_path)
    model_herg = joblib.load(herg_path)

    controls = [
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

    df_raw = pd.DataFrame(controls)
    df_feat_raw = featurize_dataframe(df_raw[["name", "raw_smiles"]], smiles_col="raw_smiles")
    df_feat_canon = featurize_dataframe(df_raw[["name", "canon_smiles"]], smiles_col="canon_smiles")

    X_raw = df_feat_raw[list(FEATURE_NAMES)]
    X_canon = df_feat_canon[list(FEATURE_NAMES)]

    pred_pic50_raw = model_a2a.predict(X_raw)
    pred_pic50_canon = model_a2a.predict(X_canon)
    pred_pherg_raw = model_herg.predict_proba(X_raw)[:, 1]
    pred_pherg_canon = model_herg.predict_proba(X_canon)[:, 1]

    expected_pic50 = np.array([c["expected_pic50"] for c in controls], dtype=np.float64)
    expected_pherg = np.array([c["expected_pherg"] for c in controls], dtype=np.float64)

    # Paridade entre entrada bruta e canônica
    np.testing.assert_allclose(pred_pic50_raw, pred_pic50_canon, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pred_pherg_raw, pred_pherg_canon, rtol=1e-7, atol=1e-7)

    # Paridade contra os valores de referência da auditoria (Fase 2A)
    np.testing.assert_allclose(pred_pic50_raw, expected_pic50, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
    np.testing.assert_allclose(pred_pherg_raw, expected_pherg, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)

    # Garantia de que os modelos serializados não foram modificados
    assert _sha256_file(a2a_path) == hash_a2a_before
    assert _sha256_file(herg_path) == hash_herg_before


# -----------------------------------------------------------------------------
# 6. Comparação com os Parquets curados existentes (sem regravação)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dataset_name", "curated_csv_name", "parquet_name"),
    [
        ("A2A", "a2a_curated.csv", "a2a_featurized.parquet"),
        ("hERG", "herg_curated.csv", "herg_featurized.parquet"),
    ],
)
def test_curated_datasets_match_existing_parquets_without_modification(
    dataset_name: str,
    curated_csv_name: str,
    parquet_name: str,
):
    """As features geradas a partir dos CSVs curados devem coincidir com os Parquets existentes."""
    csv_path = PROCESSED_DIR / curated_csv_name
    parquet_path = PROCESSED_DIR / parquet_name

    hash_csv_before = _sha256_file(csv_path)
    hash_parquet_before = _sha256_file(parquet_path)

    df_curated = pd.read_csv(csv_path)
    df_parquet = pd.read_parquet(parquet_path)

    df_recomputed = featurize_dataframe(df_curated, smiles_col="canonical_smiles")

    assert df_recomputed.shape == df_parquet.shape, (
        f"Divergência de shape em {dataset_name}: {df_recomputed.shape} vs {df_parquet.shape}"
    )

    # Comparação dos 6 descritores 2D com tolerância de ponto flutuante
    np.testing.assert_allclose(
        df_recomputed[list(DESCRIPTOR_NAMES)].to_numpy(dtype=np.float64),
        df_parquet[list(DESCRIPTOR_NAMES)].to_numpy(dtype=np.float64),
        rtol=DESCRIPTOR_RTOL,
        atol=DESCRIPTOR_ATOL,
        err_msg=f"Divergência nos descritores 2D do dataset {dataset_name}.",
    )

    # Comparação exata dos 1024 bits de Morgan Fingerprint (uint8)
    fp_recomputed = df_recomputed[list(FINGERPRINT_NAMES)].to_numpy(dtype=np.uint8)
    fp_parquet = df_parquet[list(FINGERPRINT_NAMES)].to_numpy(dtype=np.uint8)
    assert np.array_equal(fp_recomputed, fp_parquet), (
        f"Divergência nos bits de Morgan Fingerprint do dataset {dataset_name}."
    )

    # Confirmação de que nenhum arquivo de dados foi regravado ou modificado
    assert _sha256_file(csv_path) == hash_csv_before
    assert _sha256_file(parquet_path) == hash_parquet_before

