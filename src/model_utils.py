"""
model_utils — Métricas estatísticas, validação cruzada e rotinas de avaliação de modelos.

Módulo de avaliação supervisionada do MultiTarget BioPredict:
- Métricas contínuas de regressão para potência A2A (RMSE, MAE, R²).
- Métricas de classificação binária para cardiotoxicidade hERG (ROC-AUC, PR-AUC, F1, MCC, Balanced Accuracy).
- Notação canônica para intervalos operacionais de risco hERG (sem sobreposição):
  * Baixo Risco: P < 0.30
  * Risco Moderado: 0.30 <= P <= 0.70
  * Alto Risco: P > 0.70
- Protocolos de validação cruzada 5-fold (K-Fold e Stratified K-Fold) com paralelismo local seguro (n_jobs=2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

RANDOM_STATE = 42
PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Métricas de regressão (ex.: pChEMBL A2A): RMSE, MAE e R²."""
    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Métricas de classificação binária (ex.: hERG).

    Args:
        y_true: Rótulos 0/1.
        y_proba: Probabilidade da classe positiva (bloqueador hERG).
        threshold: Limiar de decisão.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
        "PR_AUC": float(average_precision_score(y_true, y_proba)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
    }


# -----------------------------------------------------------------------------
# Validação cruzada
# -----------------------------------------------------------------------------
def run_cross_validation(
    estimator,
    X,
    y,
    task: str = "regression",
    n_splits: int = 5,
    n_jobs: int = 2,
) -> dict[str, float]:
    """Validação cruzada K-Fold (estratificada para classificação).

    ``n_jobs=2`` por padrão: em 8 GB RAM, cada worker duplica a matriz X.

    Returns:
        Média e desvio-padrão de cada métrica nos folds.
    """
    if task == "regression":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
    elif task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"ROC_AUC": "roc_auc", "PR_AUC": "average_precision", "MCC": "matthews_corrcoef"}
    else:
        raise ValueError("task deve ser 'regression' ou 'classification'.")

    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    summary: dict[str, float] = {}
    for name in scoring:
        scores = res[f"test_{name}"]
        if name in ("RMSE", "MAE"):  # sklearn retorna valores negativos
            scores = -scores
        summary[f"{name}_mean"] = float(np.mean(scores))
        summary[f"{name}_std"] = float(np.std(scores))
    return summary


# -----------------------------------------------------------------------------
# Persistência e inferência
# -----------------------------------------------------------------------------
def save_model(model: Any, path: PathLike, compress: int = 3) -> Path:
    """Serializa um modelo com joblib (compactado) em ``models/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
    return path


def load_model(path: PathLike) -> Any:
    """Carrega um artefato ``.joblib``."""
    return joblib.load(Path(path))


def predict(model: Any, X, task: str = "regression") -> np.ndarray:
    """Inferência unificada: valor contínuo (regressão) ou P(classe=1)."""
    if task == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
