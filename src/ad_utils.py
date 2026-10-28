"""
ad_utils — Motor de Diagnóstico de Cobertura Estrutural e Domínio de Aplicabilidade (AD).

MultiTarget BioPredict — Fase 3D-3:
Implementação controlada do cálculo de similaridade estrutural (Tanimoto / Morgan ECFP4)
e determinação hierárquica de proveniência molecular para os modelos A2A e hERG.

Diretrizes epistemológicas e metodológicas:
1. Contextualização informativa sem promessa de confiabilidade ou validação clínica.
2. Não rotula moléculas como "Dentro/Fora do Domínio" ou "Confiável/Não Confiável".
3. Auto-exclusão estrita apenas nos Níveis 1 e 2 (estrutura ou features 1030-D).
4. O Nível 3 (fingerprint idêntico com descritores distintos) NÃO sofre auto-exclusão.
5. Tratamento explícito de falhas; S_max = None (nunca inventa zero).
6. Independência completa entre os endpoints A2A e hERG.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.chem_utils import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    canonicalize_smiles,
    compute_2d_descriptors,
    compute_morgan_fingerprints,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# -----------------------------------------------------------------------------
# Constantes do Domínio de Aplicabilidade
# -----------------------------------------------------------------------------
A2A_LOO_MEDIAN_SIMILARITY: float = 0.8219
HERG_LOO_MEDIAN_SIMILARITY: float = 0.8308

PROVENANCE_LEVEL_1_STRUCTURE: str = "Correspondência de estrutura"
PROVENANCE_LEVEL_2_FEATURES: str = "Correspondência de features (vetor 1.030-D)"
PROVENANCE_LEVEL_3_FINGERPRINT: str = "Fingerprint idêntico (1.024 bits)"
PROVENANCE_LEVEL_4_NO_EXACT_MATCH: str = "Sem correspondência exata conhecida"

STATUS_OK: str = "ok"
STATUS_INVALID_STRUCTURE: str = "invalid_structure"
STATUS_STANDARDIZATION_ERROR: str = "standardization_error"
STATUS_MISSING_FINGERPRINT: str = "missing_fingerprint"
STATUS_ZERO_ON_BITS: str = "zero_on_bits"
STATUS_TECHNICAL_ERROR: str = "technical_error"
STATUS_NO_ELIGIBLE_NEIGHBOR: str = "no_eligible_neighbor"


def _hash_feature_vector(desc_vals: np.ndarray, fp_vals: np.ndarray) -> bytes:
    """Gera um digest compacto de 16 bytes (blake2b) do vetor completo de 1.030 atributos.

    Permite busca exata O(1) de equivalência informacional (Delta X = 0)
    sem duplicar matrizes densas em memória.
    """
    return hashlib.blake2b(desc_vals.tobytes() + fp_vals.tobytes(), digest_size=16).digest()


@dataclass(frozen=True)
class EndpointADReference:
    """Estrutura imutável em memória com os dados de referência de um endpoint."""

    endpoint: str
    n_samples: int
    identifiers: Tuple[str, ...]
    canonical_smiles_list: Tuple[str, ...]
    smiles_to_indices: Dict[str, Tuple[int, ...]]
    feature_hashes: Dict[bytes, Tuple[int, ...]]
    ebvs: Tuple[DataStructs.ExplicitBitVect, ...]
    exp_values: Tuple[Optional[float], ...]
    exp_label_name: str
    descriptor_percentiles: Dict[str, Dict[str, float]]
    loo_median_similarity: float


# Cache em memória no processo Python (independente entre A2A e hERG)
_AD_REFERENCES: Dict[str, EndpointADReference] = {}


def compute_descriptor_percentiles_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calcula estatísticas descritivas (percentis) dos 6 descritores 2D no conjunto de treino."""
    stats: Dict[str, Dict[str, float]] = {}
    for desc in DESCRIPTOR_NAMES:
        col = df[desc]
        stats[desc] = {
            "min": float(col.min()),
            "p01": float(col.quantile(0.01)),
            "p05": float(col.quantile(0.05)),
            "median": float(col.median()),
            "p95": float(col.quantile(0.95)),
            "p99": float(col.quantile(0.99)),
            "max": float(col.max()),
        }
    return stats


def load_ad_reference(endpoint: str, data_dir: Optional[Path] = None) -> EndpointADReference:
    """Carrega os dados de referência featurizados do endpoint a partir do Parquet existente.

    Args:
        endpoint: 'A2A' ou 'hERG'.
        data_dir: Diretório onde se encontram os Parquets (padrão: data/processed).

    Returns:
        Instância de EndpointADReference com fingerprints e índices pré-computados.
    """
    ep = endpoint.upper()
    if ep in ("A2A", "A_2A"):
        canon_ep = "A2A"
        filename = "a2a_featurized.parquet"
        id_col = "molecule_chembl_id"
        exp_col = "pIC50"
        loo_median = A2A_LOO_MEDIAN_SIMILARITY
    elif ep in ("HERG", "H_ERG"):
        canon_ep = "hERG"
        filename = "herg_featurized.parquet"
        id_col = "Drug_ID"
        exp_col = "hERG_label"
        loo_median = HERG_LOO_MEDIAN_SIMILARITY
    else:
        raise ValueError(f"Endpoint não suportado: '{endpoint}'. Use 'A2A' ou 'hERG'.")

    base_dir = data_dir if data_dir is not None else DATA_DIR
    parquet_path = base_dir / filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Arquivo de referência Parquet não encontrado: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    n_samples = len(df)

    # Identificadores e rótulos
    identifiers = tuple(str(x) for x in df[id_col].tolist())
    canonical_smiles_list = tuple(df["canonical_smiles"].astype(str).tolist())
    exp_values = tuple(
        float(v) if pd.notna(v) else None for v in df[exp_col].tolist()
    )

    # Dicionário de busca por SMILES canônico exato (Nível 1)
    smiles_to_indices: Dict[str, List[int]] = {}
    for idx, smi in enumerate(canonical_smiles_list):
        smiles_to_indices.setdefault(smi, []).append(idx)
    smiles_to_indices_frozen = {k: tuple(v) for k, v in smiles_to_indices.items()}

    # Matrizes e representações
    fp_cols = [f"fp_{i}" for i in range(1024)]
    X_fps = df[fp_cols].to_numpy(dtype=np.uint8)
    desc_arr = df[list(DESCRIPTOR_NAMES)].to_numpy(dtype=np.float64)

    # Construção dos ExplicitBitVect para BulkTanimotoSimilarity ultrarrápido
    ebvs_list: List[DataStructs.ExplicitBitVect] = []
    for row in X_fps:
        ebv = DataStructs.ExplicitBitVect(1024)
        for bit in np.flatnonzero(row):
            ebv.SetBit(int(bit))
        ebvs_list.append(ebv)
    ebvs_tuple = tuple(ebvs_list)

    # Dicionário de hashes para verificação exata de features 1030-D (Nível 2)
    feat_hashes: Dict[bytes, List[int]] = {}
    for idx in range(n_samples):
        h = _hash_feature_vector(desc_arr[idx], X_fps[idx])
        feat_hashes.setdefault(h, []).append(idx)
    feat_hashes_frozen = {k: tuple(v) for k, v in feat_hashes.items()}

    # Percentis descritivos
    desc_percentiles = compute_descriptor_percentiles_from_df(df)

    ref = EndpointADReference(
        endpoint=canon_ep,
        n_samples=n_samples,
        identifiers=identifiers,
        canonical_smiles_list=canonical_smiles_list,
        smiles_to_indices=smiles_to_indices_frozen,
        feature_hashes=feat_hashes_frozen,
        ebvs=ebvs_tuple,
        exp_values=exp_values,
        exp_label_name=exp_col,
        descriptor_percentiles=desc_percentiles,
        loo_median_similarity=loo_median,
    )
    return ref


def get_ad_reference(endpoint: str, data_dir: Optional[Path] = None) -> EndpointADReference:
    """Obtém a referência AD do cache ou a carrega do disco de forma thread-safe."""
    ep = "A2A" if endpoint.upper() in ("A2A", "A_2A") else "hERG"
    if ep not in _AD_REFERENCES:
        _AD_REFERENCES[ep] = load_ad_reference(ep, data_dir=data_dir)
    return _AD_REFERENCES[ep]


def compute_ad_diagnostics(
    mol: Any,
    endpoint: str,
    reference: Optional[EndpointADReference] = None,
    raw_smiles: Optional[str] = None,
) -> Dict[str, Any]:
    """Executa o diagnóstico de cobertura estrutural e proveniência para uma molécula e endpoint.

    Args:
        mol: Chem.Mol ou string SMILES.
        endpoint: 'A2A' ou 'hERG'.
        reference: Referência pré-carregada (se None, carrega do cache).
        raw_smiles: String SMILES bruta original (opcional).

    Returns:
        Dicionário com o resultado detalhado do diagnóstico.
    """
    ep = "A2A" if endpoint.upper() in ("A2A", "A_2A") else "hERG"

    # Carrega a referência com segurança
    try:
        ref = reference if reference is not None else get_ad_reference(ep)
    except Exception as exc:
        logger.error(f"Falha técnica ao carregar referência AD para {ep}: {exc}")
        return {
            "status": STATUS_TECHNICAL_ERROR,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": getattr(ref, "loo_median_similarity", None) if "ref" in locals() else None,
            "descriptors": None,
            "error_message": f"Falha ao carregar banco de referência: {exc}",
        }

    # 1. Validação de estrutura e padronização química
    if mol is None:
        return {
            "status": STATUS_INVALID_STRUCTURE,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": ref.loo_median_similarity,
            "descriptors": None,
            "error_message": "Estrutura nula ou inválida fornecida para diagnóstico.",
        }

    canon_smi: Optional[str] = None
    mol_obj: Optional[Chem.Mol] = None

    if isinstance(mol, str):
        raw_str = mol.strip()
        if not raw_str:
            return {
                "status": STATUS_INVALID_STRUCTURE,
                "endpoint": ep,
                "provenance": None,
                "provenance_level": None,
                "s_max_all": None,
                "s_max_excluding_match": None,
                "has_exclusion": False,
                "nearest_neighbor": None,
                "nearest_neighbor_non_matching": None,
                "loo_median_similarity": ref.loo_median_similarity,
                "descriptors": None,
                "error_message": "String SMILES vazia fornecida.",
            }
        canon_smi = canonicalize_smiles(raw_str)
        if canon_smi is None:
            return {
                "status": STATUS_STANDARDIZATION_ERROR,
                "endpoint": ep,
                "provenance": None,
                "provenance_level": None,
                "s_max_all": None,
                "s_max_excluding_match": None,
                "has_exclusion": False,
                "nearest_neighbor": None,
                "nearest_neighbor_non_matching": None,
                "loo_median_similarity": ref.loo_median_similarity,
                "descriptors": None,
                "error_message": "Falha na padronização química do SMILES (estrutura inválida ou sem átomos válidos).",
            }
        mol_obj = Chem.MolFromSmiles(canon_smi)
    elif isinstance(mol, Chem.Mol):
        mol_obj = mol
        canon_smi = Chem.MolToSmiles(mol_obj, canonical=True, isomericSmiles=True)
    else:
        return {
            "status": STATUS_INVALID_STRUCTURE,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": ref.loo_median_similarity,
            "descriptors": None,
            "error_message": f"Tipo de molécula não suportado: {type(mol)}.",
        }

    if mol_obj is None or mol_obj.GetNumAtoms() == 0:
        return {
            "status": STATUS_STANDARDIZATION_ERROR,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": ref.loo_median_similarity,
            "descriptors": None,
            "error_message": "Molécula desprovida de átomos após padronização.",
        }

    # 2. Featurização do query (6 descritores + 1024 bits Morgan ECFP4)
    try:
        desc_dict = compute_2d_descriptors(mol_obj)
        fp_np = compute_morgan_fingerprints(mol_obj, radius=2, n_bits=1024)
    except Exception as exc:
        logger.error(f"Erro ao featurizar molécula para diagnóstico AD: {exc}")
        return {
            "status": STATUS_MISSING_FINGERPRINT,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": ref.loo_median_similarity,
            "descriptors": None,
            "error_message": f"Falha na featurização molecular: {exc}",
        }

    # 3. Conversão para ExplicitBitVect e verificação de bits ativos
    on_bits = np.flatnonzero(fp_np)
    if len(on_bits) == 0:
        return {
            "status": STATUS_ZERO_ON_BITS,
            "endpoint": ep,
            "provenance": None,
            "provenance_level": None,
            "s_max_all": None,
            "s_max_excluding_match": None,
            "has_exclusion": False,
            "nearest_neighbor": None,
            "nearest_neighbor_non_matching": None,
            "loo_median_similarity": ref.loo_median_similarity,
            "descriptors": desc_dict,
            "error_message": (
                "Fingerprint de Morgan não possui bits ativos (onBits = 0). "
                "Impossível calcular similaridade de Tanimoto sem divisão por zero."
            ),
        }

    query_ebv = DataStructs.ExplicitBitVect(1024)
    for b in on_bits:
        query_ebv.SetBit(int(b))

    # 4. Cálculo de similaridade de Tanimoto em lote contra todos os fingerprints
    sims = DataStructs.BulkTanimotoSimilarity(query_ebv, list(ref.ebvs))
    s_max_all = float(np.max(sims))
    max_idx = int(np.argmax(sims))

    # 5. Determinação hierárquica e rigorosa de proveniência
    desc_vec = np.array([desc_dict[k] for k in DESCRIPTOR_NAMES], dtype=np.float64)
    query_feat_hash = _hash_feature_vector(desc_vec, fp_np)

    provenance: str
    provenance_level: int
    match_indices: Set[int] = set()

    # Nível 1: Correspondência de estrutura (SMILES canônico idêntico)
    if canon_smi in ref.smiles_to_indices:
        provenance = PROVENANCE_LEVEL_1_STRUCTURE
        provenance_level = 1
        match_indices = set(ref.smiles_to_indices[canon_smi])

    # Nível 2: Correspondência de features (vetor 1.030-D idêntico, Delta X = 0)
    elif query_feat_hash in ref.feature_hashes:
        provenance = PROVENANCE_LEVEL_2_FEATURES
        provenance_level = 2
        match_indices = set(ref.feature_hashes[query_feat_hash])

    # Nível 3: Fingerprint idêntico (1.024 bits idênticos, mas descritores diferem)
    elif np.isclose(s_max_all, 1.0, atol=1e-7):
        provenance = PROVENANCE_LEVEL_3_FINGERPRINT
        provenance_level = 3
        # O Nível 3 NÃO engatilha auto-exclusão (evita descarte de falsos gêmeos por colisão)
        match_indices = set()

    # Nível 4: Sem correspondência exata conhecida
    else:
        provenance = PROVENANCE_LEVEL_4_NO_EXACT_MATCH
        provenance_level = 4
        match_indices = set()

    # 6. Informações do vizinho mais próximo global
    nearest_neighbor = {
        "index": max_idx,
        "id": ref.identifiers[max_idx],
        "similarity": s_max_all,
        "is_exact_match": (provenance_level in (1, 2)),
        "canonical_smiles": ref.canonical_smiles_list[max_idx],
        "experimental_label_name": ref.exp_label_name,
        "experimental_value": ref.exp_values[max_idx],
    }

    # 7. Cálculo de S_max_excluding_match (apenas para Níveis 1 e 2)
    s_max_excluding_match: Optional[float] = None
    nearest_neighbor_non_matching: Optional[Dict[str, Any]] = None
    has_exclusion = False

    if provenance_level in (1, 2) and match_indices:
        has_exclusion = True
        remaining_indices = [i for i in range(ref.n_samples) if i not in match_indices]
        if not remaining_indices:
            # Caso limite: todo o conjunto foi excluído
            return {
                "status": STATUS_NO_ELIGIBLE_NEIGHBOR,
                "endpoint": ep,
                "provenance": provenance,
                "provenance_level": provenance_level,
                "s_max_all": s_max_all,
                "s_max_excluding_match": None,
                "has_exclusion": True,
                "nearest_neighbor": nearest_neighbor,
                "nearest_neighbor_non_matching": None,
                "loo_median_similarity": ref.loo_median_similarity,
                "descriptors": desc_dict,
                "error_message": "Nenhum vizinho elegível remanescente após exclusão da correspondência.",
            }

        remaining_sims = [sims[i] for i in remaining_indices]
        s_max_excluding_match = float(np.max(remaining_sims))
        best_rem_pos = int(np.argmax(remaining_sims))
        best_rem_idx = remaining_indices[best_rem_pos]

        nearest_neighbor_non_matching = {
            "index": best_rem_idx,
            "id": ref.identifiers[best_rem_idx],
            "similarity": s_max_excluding_match,
            "is_exact_match": False,
            "canonical_smiles": ref.canonical_smiles_list[best_rem_idx],
            "experimental_label_name": ref.exp_label_name,
            "experimental_value": ref.exp_values[best_rem_idx],
        }

    return {
        "status": STATUS_OK,
        "endpoint": ep,
        "canonical_smiles": canon_smi,
        "provenance": provenance,
        "provenance_level": provenance_level,
        "s_max_all": s_max_all,
        "s_max_excluding_match": s_max_excluding_match,
        "has_exclusion": has_exclusion,
        "nearest_neighbor": nearest_neighbor,
        "nearest_neighbor_non_matching": nearest_neighbor_non_matching,
        "loo_median_similarity": ref.loo_median_similarity,
        "descriptors": desc_dict,
        "descriptor_percentiles": ref.descriptor_percentiles,
        "error_message": None,
    }


# -----------------------------------------------------------------------------
# Componente de Apresentação HTML (Streamlit)
# -----------------------------------------------------------------------------
def build_ad_card_html(diag: Dict[str, Any]) -> str:
    """Gera o card HTML estilizado para um único endpoint."""
    ep = diag["endpoint"]
    status = diag["status"]

    if status != STATUS_OK:
        err_msg = diag.get("error_message") or "Diagnóstico não pôde ser calculado."
        return f"""
        <div class="ad-card" style="background:#1E222D; border:1px solid rgba(255,75,75,0.35); border-radius:10px; padding:18px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:1.05rem; font-weight:700; color:#FFFFFF;">Alvo: {ep}</span>
                <span style="background:rgba(255,75,75,0.20); color:#FF4B4B; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600;">Indisponível</span>
            </div>
            <p style="color:#CBD5E1; font-size:0.88rem; margin:0;">{err_msg}</p>
        </div>
        """

    prov_lvl = diag["provenance_level"]
    prov_label = diag["provenance"]
    s_max_all = diag["s_max_all"]
    s_max_excl = diag["s_max_excluding_match"]
    has_excl = diag["has_exclusion"]
    nn = diag["nearest_neighbor"]
    nn_non = diag["nearest_neighbor_non_matching"]
    loo_med = diag["loo_median_similarity"]

    # Cores e distintivos de proveniência
    if prov_lvl == 1:
        badge_bg = "rgba(0, 210, 190, 0.18)"
        badge_color = "#00D2BE"
        prov_desc = "Molécula idêntica (SMILES canônico padronizado) a um registro do conjunto de treinamento."
    elif prov_lvl == 2:
        badge_bg = "rgba(168, 85, 247, 0.20)"
        badge_color = "#C084FC"
        prov_desc = "SMILES diferente, mas vetor completo de 1.030 atributos idêntico ao treino (invariância estéreo 2D)."
    elif prov_lvl == 3:
        badge_bg = "rgba(245, 158, 11, 0.20)"
        badge_color = "#FBBF24"
        prov_desc = "Fingerprint Morgan (1.024 bits) idêntico, mas com divergência em descritores físico-químicos."
    else:
        badge_bg = "rgba(148, 163, 184, 0.16)"
        badge_color = "#94A3B8"
        prov_desc = "Sem correspondência exata conhecida no conjunto de treino; valor contínuo baseado em Tanimoto."

    # Bloco de exclusão
    if has_excl and s_max_excl is not None:
        excl_html = f"""
        <div style="background:#252A37; border-radius:6px; padding:10px 12px; margin-top:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.82rem; color:#CBD5E1;">S_max sem auto-correspondência:</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:1.0rem; font-weight:700; color:#00D2BE;">{s_max_excl:.1%} <span style="font-size:0.75rem; color:#94A3B8;">({s_max_excl:.4f})</span></span>
            </div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                Vizinho não correspondente: <b>ID {nn_non['id']}</b> ({nn_non['experimental_label_name']} = {nn_non['experimental_value']})
            </div>
        </div>
        """
    else:
        excl_html = """
        <div style="background:#252A37; border-radius:6px; padding:8px 12px; margin-top:8px;">
            <span style="font-size:0.78rem; color:#94A3B8;">Auto-exclusão não aplicável (sem correspondência exata de estrutura ou features 1030-D).</span>
        </div>
        """

    # Formatador de valor experimental
    exp_val_str = f"{nn['experimental_value']:.2f}" if isinstance(nn['experimental_value'], float) else f"{nn['experimental_value']}"
    if ep == "hERG":
        exp_val_label = "Bloqueador (1)" if nn['experimental_value'] == 1 else "Não bloqueador (0)"
    else:
        exp_val_label = f"pIC50 = {exp_val_str}"

    return f"""
    <div class="ad-card" style="background:#1E222D; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
            <span style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">Alvo Farmacológico: {ep}</span>
            <span style="background:{badge_bg}; color:{badge_color}; padding:4px 12px; border-radius:14px; font-size:0.80rem; font-weight:600; border:1px solid {badge_color}33;">
                Nível {prov_lvl}: {prov_label}
            </span>
        </div>
        <p style="color:#CBD5E1; font-size:0.82rem; margin:0 0 12px 0; line-height:1.4;">{prov_desc}</p>
        
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:10px;">
            <div style="background:#252A37; border-radius:8px; padding:12px;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:600; letter-spacing:0.5px;">Similaridade Máxima Geral (S_max)</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#FFFFFF; margin-top:4px;">
                    {s_max_all:.1%} <span style="font-size:0.85rem; color:#94A3B8;">({s_max_all:.4f})</span>
                </div>
                <div style="font-size:0.75rem; color:#CBD5E1; margin-top:4px;">
                    Vizinho mais próximo: <b>ID {nn['id']}</b> ({exp_val_label})
                </div>
            </div>
            
            <div style="background:#252A37; border-radius:8px; padding:12px;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:600; letter-spacing:0.5px;">Referência Leave-One-Out do Treino</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#CBD5E1; margin-top:4px;">
                    {loo_med:.1%} <span style="font-size:0.85rem; color:#94A3B8;">({loo_med:.4f})</span>
                </div>
                <div style="font-size:0.72rem; color:#94A3B8; margin-top:4px;">
                    Mediana LOO exploratória (influenciada por pares com S_max = 1,00; não é corte rígido).
                </div>
            </div>
        </div>

        {excl_html}
    </div>
    """


def build_ad_section_html(diag_a2a: Dict[str, Any], diag_herg: Dict[str, Any]) -> str:
    """Gera o bloco completo da seção de Domínio de Aplicabilidade para a interface Streamlit."""
    card_a2a = build_ad_card_html(diag_a2a)
    card_herg = build_ad_card_html(diag_herg)

    return f"""
    <div class="ad-section-container" style="margin-top:20px; margin-bottom:24px;">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
            <div>{card_a2a}</div>
            <div>{card_herg}</div>
        </div>
        
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:14px 16px; margin-top:12px;">
            <div style="display:flex; align-items:flex-start; gap:10px;">
                <span style="font-size:1.1rem; line-height:1.2;">ℹ️</span>
                <div style="font-size:0.80rem; color:#CBD5E1; line-height:1.45;">
                    <b>Nota de Interpretação e Governança Metodológica:</b><br/>
                    A similaridade estrutural máxima (S_max / Tanimoto Morgan ECFP4) e os descritores físico-químicos fornecem contexto exploratório sobre a cobertura do espaço químico em relação aos dados de treinamento.
                    <b>Similaridade elevada não garante acurácia preditiva</b> (devido a falésias de atividade e interações moleculares sutis), e <b>similaridade reduzida não prova que a previsão esteja incorreta</b>, mas sinaliza extrapolação química em relação ao espaço amostrado. Os valores não constituem certificação de eficácia clínica ou segurança toxicológica.
                </div>
            </div>
        </div>
    </div>
    """

