"""
chem_utils — Utilitários de quimioinformática (RDKit) do MultiTarget BioPredict.

Funções compartilhadas por todos os notebooks e pelo app Streamlit, garantindo
que treino e inferência usem EXATAMENTE a mesma padronização e featurização.

Pipeline padrão:
    SMILES bruto -> canonicalize_smiles() -> Mol
                 -> compute_2d_descriptors()     (6 descritores físico-químicos)
                 -> compute_morgan_fingerprints() (ECFP4, 1024 bits)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# Silencia avisos do RDKit (SMILES inválidos são tratados retornando None).
RDLogger.DisableLog("rdApp.*")

MolLike = Union[Chem.Mol, str]

#: Ordem canônica dos descritores 2D (usar como ordem de colunas nos DataFrames).
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
)

#: Nomes canônicos dos 1024 bits de Morgan ECFP4
FINGERPRINT_NAMES: tuple[str, ...] = tuple(f"fp_{i}" for i in range(1024))

#: Espaço vetorial completo de atributos (1030 colunas: 6 descritores + 1024 bits)
FEATURE_NAMES: tuple[str, ...] = DESCRIPTOR_NAMES + FINGERPRINT_NAMES


# -----------------------------------------------------------------------------
# Padronização estrutural
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _standardizers():
    """Instancia (uma única vez) os objetos de padronização do RDKit."""
    return (
        rdMolStandardize.LargestFragmentChooser(preferOrganic=True),
        rdMolStandardize.Uncharger(),
    )


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Padroniza uma molécula RDKit.

    Etapas:
        1. ``Cleanup``: sanitização, remoção de Hs explícitos, normalização de
           grupos funcionais (ex.: nitro, N-óxidos) e reionização.
        2. ``LargestFragmentChooser``: remove contraíons/solventes (sais), ex.:
           ``CCN.Cl`` -> ``CCN``.
        3. ``Uncharger``: neutraliza cargas simples quando possível, ex.:
           ``CC(=O)[O-]`` -> ``CC(=O)O``.

    Returns:
        Molécula padronizada ou ``None`` se a padronização falhar.
    """
    if mol is None:
        return None
    try:
        chooser, uncharger = _standardizers()
        mol = rdMolStandardize.Cleanup(mol)
        mol = chooser.choose(mol)
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:  # noqa: BLE001 — qualquer falha do RDKit invalida a molécula
        return None


def _mol_from_molblock(text: str) -> Optional[Chem.Mol]:
    """Interpreta robustamente um bloco MDL Molfile (V2000 ou V3000)."""
    if not isinstance(text, str) or not text.strip():
        return None
    # 1. Tentativa de parsing direto
    mol = Chem.MolFromMolBlock(text)
    if mol is not None:
        return mol
    # 2. Tentativa com rstrip (preserva quebras de linha iniciais de cabeçalho)
    mol = Chem.MolFromMolBlock(text.rstrip())
    if mol is not None:
        return mol
    # 3. Localiza a linha de contagens V2000/V3000 e ajusta para 3 linhas de cabeçalho
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "V2000" in line or "V3000" in line:
            if i < 3:
                adjusted = ([""] * (3 - i)) + lines
            elif i > 3:
                adjusted = lines[i - 3 :]
            else:
                adjusted = lines
            mol = Chem.MolFromMolBlock("\n".join(adjusted))
            if mol is not None:
                return mol
            break
    return None


def smiles_to_mol(smiles: str, standardize: bool = True) -> Optional[Chem.Mol]:
    """Converte SMILES ou bloco MDL Molfile em ``Chem.Mol`` (opcionalmente padronizado).

    Suporta strings lineares SMILES e representações MDL Molfile (V2000/V3000),
    como as geradas pelo editor molecular Ketcher ao submeter a estrutura.

    Returns:
        ``Chem.Mol`` válido ou ``None`` se a representação for vazia/inválida.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    raw = smiles.strip()
    mol = None
    # 1. Se contiver demarcadores característicos de bloco MDL Molfile
    if "M  END" in raw or "V2000" in raw or "V3000" in raw:
        mol = _mol_from_molblock(smiles)

    # 2. Se não for MolBlock ou falhou, tenta parse como SMILES
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(raw)
        except Exception:
            mol = None

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    return standardize_mol(mol) if standardize else mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Retorna o SMILES canônico (isomérico) padronizado.

    Remove sais/contraíons, neutraliza cargas simples e preserva
    estereoquímica. Duas representações da mesma estrutura produzem a mesma
    string — essencial para deduplicação entre ChEMBL e TDC.

    Examples:
        >>> canonicalize_smiles("OCC")
        'CCO'
        >>> canonicalize_smiles("CC(=O)[O-].[Na+]")
        'CC(=O)O'
        >>> canonicalize_smiles("not_a_smiles") is None
        True

    Returns:
        SMILES canônico ou ``None`` caso o SMILES seja inválido.
    """
    mol = smiles_to_mol(smiles, standardize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _ensure_mol(mol: MolLike) -> Chem.Mol:
    """Aceita ``Chem.Mol`` ou SMILES; levanta ``ValueError`` se inválido."""
    if isinstance(mol, str):
        canon = canonicalize_smiles(mol)
        mol = Chem.MolFromSmiles(canon) if canon is not None else None
    if mol is None or not isinstance(mol, Chem.Mol):
        raise ValueError("Molécula inválida: forneça um Chem.Mol ou SMILES válido.")
    return mol


# -----------------------------------------------------------------------------
# Featurização
# -----------------------------------------------------------------------------
def compute_2d_descriptors(mol: MolLike) -> dict[str, float]:
    """Calcula os 6 descritores físico-químicos 2D definidos no PRD.

    ============  =====================================================
    MolWt         Massa molecular média (Da)
    MolLogP       LogP de Wildman–Crippen (lipofilicidade)
    TPSA          Área de superfície polar topológica (Å²)
    NumHDonors    Doadores de ligação de H (Lipinski)
    NumHAcceptors Aceptores de ligação de H (Lipinski)
    NumRotatableBonds  Ligações rotacionáveis (flexibilidade)
    ============  =====================================================

    Args:
        mol: ``Chem.Mol`` ou SMILES.

    Returns:
        Dicionário ``{nome: valor}`` na ordem de :data:`DESCRIPTOR_NAMES`.
    """
    mol = _ensure_mol(mol)
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHDonors": int(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
        "NumRotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
    }


@lru_cache(maxsize=8)
def _morgan_generator(radius: int, n_bits: int):
    """Gerador Morgan reutilizável (API moderna ``rdFingerprintGenerator``)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def compute_morgan_fingerprints(mol: MolLike, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Gera o vetor binário de Morgan (ECFP4 por padrão: raio 2, 1024 bits).

    Args:
        mol: ``Chem.Mol`` ou SMILES.
        radius: Raio do ambiente circular (2 ≙ ECFP4).
        n_bits: Comprimento do vetor de bits.

    Returns:
        ``np.ndarray`` de shape ``(n_bits,)`` e dtype ``uint8`` (0/1) —
        ``uint8`` economiza 8x memória frente a ``float64`` (crítico em 8 GB RAM).
    """
    mol = _ensure_mol(mol)
    fp = _morgan_generator(radius, n_bits).GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8, copy=False)


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 1024,
) -> pd.DataFrame:
    """Calcula os 6 descritores 2D e Morgan Fingerprints (1024 bits uint8) em lote para um DataFrame.

    Para cada linha do DataFrame, aplica o mesmo procedimento de padronização química
    do treinamento via :func:`canonicalize_smiles` (limpeza, remoção de sais/contraíons,
    neutralização e geração de SMILES canônico isomérico), converte em molécula RDKit,
    calcula os descritores físico-químicos e os Morgan Fingerprints (ECFP4),
    anexando as 1030 colunas de atributos ao DataFrame original.

    Entradas com SMILES inválidos, vazios ou nulos não interrompem o lote: a linha
    correspondente preserva seu índice original com ``NaN`` nos 6 descritores 2D e
    zeros (``0``, ``uint8``) nos ``n_bits`` de Morgan Fingerprint.

    Args:
        df: DataFrame de entrada contendo a coluna de SMILES.
        smiles_col: Nome da coluna contendo os SMILES (brutos ou canônicos).
        radius: Raio do ambiente circular de Morgan (2 para ECFP4).
        n_bits: Quantidade de bits do vetor de Morgan (padrão 1024).

    Returns:
        DataFrame com colunas originais + 6 descritores 2D + 1024 bits fp_0..fp_1023 (uint8),
        preservando exatamente o índice e o número de linhas de ``df``.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Coluna {smiles_col!r} não encontrada no DataFrame.")

    if len(df) == 0:
        df_desc = pd.DataFrame(columns=list(DESCRIPTOR_NAMES), index=df.index, dtype=float)
        df_fps = pd.DataFrame(columns=list(FINGERPRINT_NAMES[:n_bits]), index=df.index, dtype=np.uint8)
        return pd.concat([df, df_desc, df_fps], axis=1)

    canon_smiles = [
        canonicalize_smiles(s) if isinstance(s, str) else None
        for s in df[smiles_col]
    ]
    mols = [
        Chem.MolFromSmiles(cs) if cs is not None else None
        for cs in canon_smiles
    ]

    # 1. Descritores 2D (NaN para linhas com SMILES inválido)
    nan_desc = {name: np.nan for name in DESCRIPTOR_NAMES}
    desc_list = [
        compute_2d_descriptors(m) if m is not None else nan_desc.copy()
        for m in mols
    ]
    df_desc = pd.DataFrame(desc_list, columns=list(DESCRIPTOR_NAMES), index=df.index)

    # 2. Morgan Fingerprints (uint8; zeros para linhas com SMILES inválido)
    gen = _morgan_generator(radius, n_bits)
    zero_fp = np.zeros(n_bits, dtype=np.uint8)
    fp_arrays = [
        gen.GetFingerprintAsNumPy(m).astype(np.uint8, copy=False) if m is not None else zero_fp.copy()
        for m in mols
    ]
    fp_mat = np.vstack(fp_arrays)
    df_fps = pd.DataFrame(fp_mat, columns=list(FINGERPRINT_NAMES[:n_bits]), index=df.index)

    return pd.concat([df, df_desc, df_fps], axis=1)
