"""Modèle de RÉFÉRENCE de l'équipe — CatBoost (porté par B2).

C'est la baseline officielle reproductible que les 4 membres ont sur leur machine.
Entraînement out-of-fold via src/validation.py pour produire les prédictions OOF
nécessaires à la calibration et au blending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..validation import SEED, evaluate_ap

PARAMS = dict(
    loss_function="Logloss",
    eval_metric="PRAUC",      # proche de l'AP, la métrique de la compétition
    depth=6,
    learning_rate=0.05,
    iterations=2000,
    random_seed=SEED,
    od_type="Iter",
    od_wait=100,
    verbose=False,
)


def train_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    folds,
    cat_features: list[str] | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, list[CatBoostClassifier]]:
    """Entraîne en CV et renvoie (prédictions out-of-fold, liste des modèles).

    `folds` : itérable de (train_idx, valid_idx) issu de src/validation.py.
    """
    params = {**PARAMS, **(params or {})}
    oof = np.zeros(len(y))
    models = []
    for train_idx, valid_idx in folds:
        model = CatBoostClassifier(**params)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=(X.iloc[valid_idx], y[valid_idx]),
            cat_features=cat_features,
            use_best_model=True,
        )
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        models.append(model)
    print(f"CatBoost OOF AP : {evaluate_ap(y, oof):.4f}")
    return oof, models
