"""Calibration des probabilités — explicitement demandée par la brief.

CatBoost/LightGBM ne sortent pas des scores forcément bien calibrés.
Un calibrateur en post-traitement coûte quasi rien et peut gagner quelques millièmes d'AP.

Toujours fitter le calibrateur sur les prédictions OUT-OF-FOLD, jamais sur le train,
sinon on triche.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_isotonic(oof_proba: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Calibration isotonic (non paramétrique). Robuste si assez de données."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_proba, y_true)
    return iso


def fit_platt(oof_proba: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """Calibration de Platt (sigmoïde). Préférable si peu de données."""
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(oof_proba.reshape(-1, 1), y_true)
    return lr


def apply_isotonic(iso: IsotonicRegression, proba: np.ndarray) -> np.ndarray:
    return iso.predict(proba)


def apply_platt(lr: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    return lr.predict_proba(proba.reshape(-1, 1))[:, 1]
