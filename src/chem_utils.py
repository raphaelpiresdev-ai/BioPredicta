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


def smiles_to_mol(smiles: str, standardize: bool = True) -> Optional[Chem.Mol]:
    """Converte SMILES em ``Chem.Mol`` (opcionalmente padronizado).

    Returns:
        ``Chem.Mol`` válido ou ``None`` se o SMILES for vazio/inválido.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
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
        mol = smiles_to_mol(mol, standardize=True)
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
