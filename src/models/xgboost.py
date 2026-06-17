"""Modèle XGBoost — porté par C (3e modèle pour la diversité du blend).

Conçu pour être décorrélé de CatBoost (B2) et LightGBM (B1) : features
partiellement différentes + algo différent => le blend final y gagne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..validation import SEED, evaluate_ap

PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",          # aire sous la courbe précision-rappel ~ AP
    max_depth=6,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=100,
)


def train_oof(
    X: pd.DataFrame, y: np.ndarray, folds, params: dict | None = None
) -> tuple[np.ndarray, list[xgb.XGBClassifier]]:
    """Entraîne en CV et renvoie (prédictions out-of-fold, liste des modèles)."""
    params = {**PARAMS, **(params or {})}
    oof = np.zeros(len(y))
    models = []
    for train_idx, valid_idx in folds:
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        models.append(model)
    print(f"XGBoost OOF AP : {evaluate_ap(y, oof):.4f}")
    return oof, models
