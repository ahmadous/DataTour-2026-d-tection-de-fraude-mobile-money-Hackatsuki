"""Modèle LightGBM — porté par B1 (rapide, pour itérer sur les features).

Sert aussi à valider rapidement si une nouvelle feature apporte du signal
avant de la pousser dans le feature store.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..validation import SEED, evaluate_ap

PARAMS = dict(
    objective="binary",
    metric="average_precision",
    boosting_type="gbdt",
    num_leaves=63,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def train_oof(
    X: pd.DataFrame, y: np.ndarray, folds, params: dict | None = None
) -> tuple[np.ndarray, list[lgb.LGBMClassifier]]:
    """Entraîne en CV et renvoie (prédictions out-of-fold, liste des modèles)."""
    params = {**PARAMS, **(params or {})}
    oof = np.zeros(len(y))
    models = []
    for train_idx, valid_idx in folds:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        models.append(model)
    print(f"LightGBM OOF AP : {evaluate_ap(y, oof):.4f}")
    return oof, models
