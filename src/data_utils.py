"""
data_utils — Aquisição e curadoria de dados para MultiTarget BioPredict.

Módulo responsável pela ingestão e curadoria dos conjuntos de dados:
1. Receptor Adenosina A2A (ChEMBL): Target CHEMBL251 (IC50, afinidade/potência).
2. Canal Cardíaco hERG (TDC / Harvard Dataverse): herg_karim (~13.4k compostos).

Ambos os pipelines realizam padronização química estrita com RDKit (via chem_utils),
resolução de conflitos de medição/duplicatas e exportação de datasets prontos para modelagem.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.chem_utils import canonicalize_smiles

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Harvard Dataverse API URL
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# ChEMBL REST API URL para atividades
CHEMBL_ACTIVITY_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    "?target_chembl_id={target_id}&standard_type={standard_type}&limit={limit}&offset={offset}"
)

#: Datasets hERG do TDC (grupo Tox) -> ID do arquivo no Harvard Dataverse.
TDC_DATASETS: dict[str, int] = {
    "herg": 4259588,          # ~648 compostos (Wang et al.)
    "herg_karim": 6822246,    # ~13.4k compostos, balanceado (Karim et al.)
    "herg_central": 5740618,  # ~300k compostos (HTS) — pesado para 8 GB RAM
}


# =============================================================================
# 1. Aquisição de Dados (Raw)
# =============================================================================

def download_tdc_dataset(
    name: str,
    dest_dir: Path | str = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Baixa (com cache local) um dataset TDC e retorna ``DataFrame [Drug_ID, Drug, Y]``.

    Args:
        name: Chave em :data:`TDC_DATASETS` (ex.: ``"herg_karim"``).
        dest_dir: Diretório de destino do CSV bruto.
        force: Rebaixa mesmo se o arquivo já existir.

    Returns:
        DataFrame com colunas Drug_ID, Drug (SMILES) e Y (rótulo binário).
    """
    key = name.lower()
    if key not in TDC_DATASETS:
        raise ValueError(f"Dataset desconhecido: {name!r}. Opções: {sorted(TDC_DATASETS)}")

    dest = Path(dest_dir) / f"tdc_{key}.csv"
    if dest.exists() and not force:
        return pd.read_csv(dest)

    resp = requests.get(DATAVERSE_URL.format(file_id=TDC_DATASETS[key]), timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.content.decode("utf-8")), sep="\t")

    # Normaliza para o esquema padrão do TDC: Drug_ID, Drug (SMILES), Y.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    rename = {c: "Drug_ID" for c in df.columns if str(c).lower() in ("drug_id", "id", "")}
    rename.update({c: "Drug" for c in df.columns if str(c).lower() in ("drug", "smiles", "x")})
    df = df.rename(columns=rename)
    if "Drug_ID" not in df.columns:
        df.insert(0, "Drug_ID", [f"{key}_{i}" for i in range(len(df))])

    dest.parent.mkdir(parents=True, exist_ok=True)
    df[["Drug_ID", "Drug", "Y"]].to_csv(dest, index=False)
    return df[["Drug_ID", "Drug", "Y"]]


def fetch_chembl_a2a_dataset(
    target_id: str = "CHEMBL251",
    standard_type: str = "IC50",
    dest_dir: Path | str = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Baixa (com cache local) bioatividades brutas do ChEMBL via REST API paginada.

    Args:
        target_id: ChEMBL Target ID (padrão: CHEMBL251 - Adenosine receptor A2a).
        standard_type: Tipo de ensaio (padrão: IC50).
        dest_dir: Diretório para salvar o CSV bruto.
        force: Força re-download se True.

    Returns:
        DataFrame com todas as atividades retornadas pela API.
    """
    dest = Path(dest_dir) / "chembl_a2a_raw.csv"
    if dest.exists() and not force:
        return pd.read_csv(dest, low_memory=False)

    activities: list[dict] = []
    offset = 0
    limit = 1000

    while True:
        url = CHEMBL_ACTIVITY_URL.format(
            target_id=target_id,
            standard_type=standard_type,
            limit=limit,
            offset=offset,
        )
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("activities", [])
        activities.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    df = pd.DataFrame(activities)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


# =============================================================================
# 2. Curadoria e Limpeza Química
# =============================================================================

def curate_a2a_dataset(
    df_raw: pd.DataFrame,
    max_std: float = 1.0,
) -> pd.DataFrame:
    """Aplica curadoria estrita aos dados brutos de afinidade A2A do ChEMBL.

    Critérios de Curadoria:
        1. Relação exata: ``standard_relation == '='``.
        2. Tipo e Unidade: ``standard_type == 'IC50'`` e ``standard_units == 'nM'``.
        3. Valores numéricos positivos e SMILES válidos não-nulos.
        4. Padronização química via RDKit (remoção de sais/contraíons e neutralização).
        5. Cálculo do pIC50: ``pIC50 = 9.0 - log10(standard_value)``.
        6. Deduplicação por SMILES canônico único com cálculo da mediana.
        7. Descarte de alta discordância experimental (desvio padrão > max_std).

    Returns:
        DataFrame [molecule_chembl_id, canonical_smiles, pIC50, num_records, pIC50_std].
    """
    # 1. Filtros preliminares de integridade
    mask = (
        (df_raw["standard_relation"] == "=")
        & (df_raw["standard_type"] == "IC50")
        & (df_raw["standard_units"] == "nM")
        & (df_raw["standard_value"].notna())
        & (df_raw["canonical_smiles"].notna())
    )
    df = df_raw.loc[mask].copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0].copy()

    # 2. Padronização química
    df["clean_smiles"] = df["canonical_smiles"].astype(str).apply(canonicalize_smiles)
    df = df[df["clean_smiles"].notna()].copy()

    # 3. Conversão para pIC50
    df["pIC50"] = 9.0 - np.log10(df["standard_value"].to_numpy())

    # 4. Agregação e deduplicação
    def _aggregate(group: pd.DataFrame) -> pd.Series:
        pvals = group["pIC50"].to_numpy()
        std_val = float(np.std(pvals, ddof=1)) if len(pvals) > 1 else 0.0
        return pd.Series({
            "molecule_chembl_id": group["molecule_chembl_id"].iloc[0],
            "pIC50": float(np.median(pvals)),
            "pIC50_std": round(std_val, 3),
            "num_records": int(len(pvals)),
        })

    curated = (
        df.groupby("clean_smiles", as_index=False)
        .apply(_aggregate, include_groups=False)
        .rename(columns={"clean_smiles": "canonical_smiles"})
    )

    # 5. Filtragem de discordância experimental severa
    curated = curated[curated["pIC50_std"] <= max_std].copy()
    curated["num_records"] = curated["num_records"].astype(int)
    curated["pIC50"] = curated["pIC50"].round(4)
    curated = curated.sort_values(by="pIC50", ascending=False).reset_index(drop=True)
    return curated[["molecule_chembl_id", "canonical_smiles", "pIC50", "num_records", "pIC50_std"]]


def curate_herg_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica curadoria estrita ao dataset toxicológico hERG (TDC).

    Critérios de Curadoria:
        1. Remoção de registros com SMILES ou rótulo ausente.
        2. Padronização química via RDKit (remoção de solventes, sais e neutralização).
        3. Identificação de moléculas duplicadas por SMILES canônico único.
        4. Resolução de conflitos: descarte de moléculas com rótulos antagônicos
           (0 e 1 simultaneamente reportados em ensaios distintos).
        5. Atribuição do rótulo consensual binário hERG (1 = Bloqueador, 0 = Não-bloqueador).

    Returns:
        DataFrame [Drug_ID, canonical_smiles, hERG_label, num_records].
    """
    df = df_raw.dropna(subset=["Drug", "Y"]).copy()
    df["clean_smiles"] = df["Drug"].astype(str).apply(canonicalize_smiles)
    df = df[df["clean_smiles"].notna()].copy()

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        labels = group["Y"].to_numpy()
        mean_y = float(np.mean(labels))
        return pd.Series({
            "Drug_ID": str(group["Drug_ID"].iloc[0]),
            "mean_label": mean_y,
            "num_records": int(len(labels)),
        })

    grouped = (
        df.groupby("clean_smiles", as_index=False)
        .apply(_aggregate, include_groups=False)
        .rename(columns={"clean_smiles": "canonical_smiles"})
    )

    # Descarta conflitos irreconciliáveis (compostos onde 0 < mean_label < 1)
    curated = grouped[(grouped["mean_label"] == 0.0) | (grouped["mean_label"] == 1.0)].copy()
    curated["hERG_label"] = curated["mean_label"].astype(int)
    curated["num_records"] = curated["num_records"].astype(int)
    curated["Drug_ID"] = curated["Drug_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    curated = curated.drop(columns=["mean_label"])
    curated = curated.sort_values(by="hERG_label", ascending=False).reset_index(drop=True)
    return curated[["Drug_ID", "canonical_smiles", "hERG_label", "num_records"]]
