"""
data_utils — Aquisição de datasets sem depender do PyTDC.

Os datasets do Therapeutics Data Commons (TDC) ficam hospedados no Harvard
Dataverse. Aqui eles são baixados diretamente pelo ID do arquivo (os mesmos
IDs usados internamente pelo PyTDC em ``tdc/metadata.py``), evitando as
dependências pesadas/incompatíveis do PyTDC no Windows.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

#: Datasets hERG do TDC (grupo Tox) -> ID do arquivo no Harvard Dataverse.
TDC_DATASETS: dict[str, int] = {
    "herg": 4259588,          # ~648 compostos (Wang et al.)
    "herg_karim": 6822246,    # ~13.4k compostos, balanceado (Karim et al.)
    "herg_central": 5740618,  # ~300k compostos (HTS) — pesado para 8 GB RAM
}


def download_tdc_dataset(name: str, dest_dir: Path | str = RAW_DIR, force: bool = False) -> pd.DataFrame:
    """Baixa (com cache local) um dataset TDC e retorna ``DataFrame [Drug_ID, Drug, Y]``.

    Args:
        name: Chave em :data:`TDC_DATASETS` (ex.: ``"herg_karim"``).
        dest_dir: Diretório de destino do CSV bruto.
        force: Rebaixa mesmo se o arquivo já existir.
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
