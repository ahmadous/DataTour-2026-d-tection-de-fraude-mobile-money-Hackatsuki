"""Blending & stacking final — PROPRIÉTÉ DU RÔLE A.

Le blend ne paye que si les modèles sont DIVERSIFIÉS (features ou algos différents).
B1 (LightGBM) + B2 (CatBoost) + C (XGBoost) sont conçus pour être décorrélés.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .validation import evaluate_ap


def rank_average(probas: list[np.ndarray]) -> np.ndarray:
    """Moyenne des rangs — insensible aux échelles de score différentes.

    Souvent plus robuste qu'une moyenne brute pour une métrique de ranking comme l'AP.
    """
    ranks = [np.argsort(np.argsort(p)) / (len(p) - 1) for p in probas]
    return np.mean(ranks, axis=0)


def optimize_weights(
    oof_probas: list[np.ndarray], y_true: np.ndarray
) -> np.ndarray:
    """Cherche les poids de blend qui maximisent l'AP sur l'out-of-fold.

    Renvoie un vecteur de poids positifs sommant à 1.
    """
    n = len(oof_probas)
    stack = np.vstack(oof_probas)

    def neg_ap(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        blend = (w[:, None] * stack).sum(axis=0) / w.sum()
        return -evaluate_ap(y_true, blend)

    res = minimize(neg_ap, x0=np.ones(n) / n, method="Nelder-Mead")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def blend(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Applique des poids à des prédictions de test."""
    stack = np.vstack(probas)
    return (weights[:, None] * stack).sum(axis=0) / weights.sum()


def stack_logistic(
    oof_probas: list[np.ndarray], y_true: np.ndarray
) -> LogisticRegression:
    """Stacking de niveau 2 : régression logistique sur les prédictions OOF."""
    X = np.vstack(oof_probas).T
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X, y_true)
    return meta
