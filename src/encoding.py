"""Encodages ANTI-FUITE — point 6.3 du rapport, critique pour cette compétition.

Le rapport est explicite :
  - encodages calculés sur le PASSÉ uniquement (fold par fold), jamais sur l'ensemble ;
  - éviter l'encodage direct des IDs de comptes (comptes mixtes -> surapprentissage).

Toute feature de type fréquence / target encoding DOIT passer par ces fonctions,
sinon on fait fuiter le futur dans le train et la CV ment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frequency_encode_oof(
    df: pd.DataFrame, col: str, folds
) -> np.ndarray:
    """Frequency encoding calculé out-of-fold : pour chaque fold de validation,
    la fréquence est apprise UNIQUEMENT sur les lignes d'entraînement du fold.

    `folds` : itérable de (train_idx, valid_idx) issu de src/validation.py.
    Renvoie un vecteur aligné sur df.
    """
    out = np.zeros(len(df))
    values = df[col].values
    for train_idx, valid_idx in folds:
        freq = pd.Series(values[train_idx]).value_counts(normalize=True)
        out[valid_idx] = pd.Series(values[valid_idx]).map(freq).fillna(0).values
    return out


def target_encode_oof(
    df: pd.DataFrame, col: str, y: np.ndarray, folds, smoothing: float = 20.0
) -> np.ndarray:
    """Target encoding lissé, out-of-fold. À utiliser avec PRUDENCE sur les IDs
    de comptes (risque de surapprentissage signalé par le rapport).

    Préférer l'encodage de variables comportementales agrégées, pas l'ID brut.
    """
    out = np.zeros(len(df))
    values = df[col].values
    global_mean = y.mean()
    for train_idx, valid_idx in folds:
        s = pd.DataFrame({"k": values[train_idx], "y": y[train_idx]})
        agg = s.groupby("k")["y"].agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
            agg["count"] + smoothing
        )
        out[valid_idx] = (
            pd.Series(values[valid_idx]).map(smooth).fillna(global_mean).values
        )
    return out


def frequency_encode_full(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Pour le scoring FINAL du test : fréquence apprise sur tout le train,
    appliquée au train (pour le modèle final) et au test."""
    freq = train_col.value_counts(normalize=True)
    return (
        train_col.map(freq).fillna(0).values,
        test_col.map(freq).fillna(0).values,
    )
