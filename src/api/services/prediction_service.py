"""Serviço de orquestração de inferência multiobjetivo (A2A e hERG).

Integra a interface da API com as funções especializadas do núcleo científico
(chem_utils e model_utils), realizando padronização química, featurização 1030-D
e invocação dos modelos de Machine Learning (XGBoost e Random Forest).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.chem_utils import FEATURE_NAMES, canonicalize_smiles, featurize_dataframe
from src.api.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)


class PredictionServiceError(Exception):
    """Exceção base para falhas na camada de serviço de predição."""


class InvalidSmilesError(PredictionServiceError):
    """Exceção lançada quando a molécula SMILES não pode ser interpretada pelo RDKit."""


class ModelsNotLoadedError(PredictionServiceError):
    """Exceção lançada quando um ou ambos os modelos estão ausentes em app.state."""


def run_prediction_pipeline(
    smiles: str,
    a2a_model: Any,
    herg_model: Any,
) -> PredictionResponse:
    """Executa o pipeline unificado de predição multiobjetivo para uma única molécula.

    1. Verifica a presença dos modelos preditivos em memória.
    2. Aplica padronização química e gera SMILES canônico (canonicalize_smiles).
    3. Executa a featurização 1030-D (6 descritores 2D + 1024 bits Morgan ECFP4).
    4. Executa a predição contínua de pIC50 via modelo A2A (XGBoost).
    5. Executa a predição probabilística de bloqueio via modelo hERG (Random Forest),
       identificando dinamicamente o índice da classe positiva a partir de classes_.
    6. Retorna a instância validada de PredictionResponse.
    """
    if a2a_model is None or herg_model is None:
        raise ModelsNotLoadedError("Modelos preditivos (A2A e/ou hERG) não estão disponíveis em memória.")

    # 1. Padronização e canonicalização química via RDKit
    canonical_smi = canonicalize_smiles(smiles)
    if not canonical_smi:
        logger.warning("Falha na interpretação/canonicalização do SMILES: %s", smiles)
        raise InvalidSmilesError(
            "O código SMILES fornecido é quimicamente inválido ou não pôde ser interpretado pelo RDKit."
        )

    # 2. Featurização tabular 1030-D
    try:
        df_single = pd.DataFrame([{"canonical_smiles": canonical_smi}])
        df_feat = featurize_dataframe(df_single, smiles_col="canonical_smiles")
        X_sample = df_feat[list(FEATURE_NAMES)]

        # Verifica integridade dos descritores gerados
        if X_sample.isna().any().any():
            raise InvalidSmilesError(
                "Falha ao extrair descritores físico-químicos e fingerprints para a estrutura molecular."
            )
    except InvalidSmilesError:
        raise
    except Exception as exc:
        logger.error("Erro inesperado durante a featurização da molécula: %s", exc, exc_info=True)
        raise PredictionServiceError(f"Erro durante a featurização molecular: {exc}") from exc

    # 3. Predição A2A (XGBoost Regressor -> pIC50)
    try:
        pred_p_ic50 = float(a2a_model.predict(X_sample)[0])
    except Exception as exc:
        logger.error("Falha na execução do modelo A2A: %s", exc, exc_info=True)
        raise PredictionServiceError(f"Falha na inferência do modelo A2A: {exc}") from exc

    # 4. Predição hERG (Random Forest Classifier -> P(hERG=1))
    try:
        if hasattr(herg_model, "classes_"):
            classes_list = list(herg_model.classes_)
            if 1 in classes_list:
                pos_idx = classes_list.index(1)
            else:
                pos_idx = -1
        else:
            pos_idx = 1

        prob_matrix = np.asarray(herg_model.predict_proba(X_sample))
        pred_p_herg_positive = float(prob_matrix[0, pos_idx])
        # Garante limites estritos [0.0, 1.0] contra variações infinitesimais de float
        pred_p_herg_positive = max(0.0, min(1.0, pred_p_herg_positive))
    except Exception as exc:
        logger.error("Falha na execução do modelo hERG: %s", exc, exc_info=True)
        raise PredictionServiceError(f"Falha na inferência do modelo hERG: {exc}") from exc

    return PredictionResponse(
        smiles=canonical_smi,
        p_ic50=pred_p_ic50,
        p_herg_positive=pred_p_herg_positive,
    )
