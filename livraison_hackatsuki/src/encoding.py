"""Encodages ANTI-FUITE — point 6.3 du rapport, critique pour cette compétition.

Le rapport est explicite :
  - encodages calculés sur le PASSÉ uniquement (fold par fold), jamais sur l'ensemble ;
  - éviter l'encodage direct des IDs de comptes (comptes mixtes -> surapprentissage).

Toute feature de type fréquence / target encoding DOIT passer par ces fonctions,
sinon on fait fuiter le futur dans le train et la CV ment.
"""
from __future__ import annotations

import bisect

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


def target_encode_ref(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    col: str,
    target_col: str,
    smoothing: float = 30.0,
) -> np.ndarray:
    """Target encoding lissé appris sur `ref_df` (= le PASSÉ), appliqué à `df`.

    Fold-safe quand `ref_df` ne contient que des lignes d'entraînement antérieures.
    Renvoie le taux de fraude historique (lissé) de la modalité de `col`.

    ⚠️ Sur l'ID ÉMETTEUR -> piège (comptes mixtes, cf. rapport). À réserver aux
    modalités potentiellement "pures" (destinataire collecteur, couple).
    """
    global_mean = ref_df[target_col].mean()
    agg = ref_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )
    return df[col].map(smooth).fillna(global_mean).to_numpy()


def fit_target_map(
    ref_df: pd.DataFrame, col: str, target_col: str, smoothing: float = 30.0
) -> tuple[dict, float]:
    """Apprend la table modalité -> taux de fraude lissé, sur `ref_df` (= passé)."""
    global_mean = float(ref_df[target_col].mean())
    agg = ref_df.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )
    return smooth.to_dict(), global_mean


def apply_target_map(df: pd.DataFrame, col: str, mapping: dict, global_mean: float) -> np.ndarray:
    return df[col].map(mapping).fillna(global_mean).to_numpy()


def oof_target_encode_train(
    train_df: pd.DataFrame,
    col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Target encoding SANS FUITE pour les lignes d'entraînement (OOF imbriqué).

    Chaque ligne est encodée à partir des AUTRES lignes du train (jamais d'elle-même),
    ce qui supprime la fuite par auto-contribution. À utiliser pour encoder le train ;
    pour valid/test, utiliser fit_target_map(ref) + apply_target_map.
    """
    from sklearn.model_selection import KFold

    out = np.zeros(len(train_df))
    values = train_df[col].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for inner_tr, inner_va in kf.split(train_df):
        mapping, gm = fit_target_map(train_df.iloc[inner_tr], col, target_col, smoothing)
        out[inner_va] = (
            pd.Series(values[inner_va]).map(mapping).fillna(gm).to_numpy()
        )
    return out


def recent_target_rate(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    group_col: str,
    period_col: str,
    target_col: str,
    windows: tuple[int, ...] = (5, 10, 20),
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Taux de fraude RÉCENT d'un groupe (ex. émetteur) sur fenêtres glissantes.

    Pour chaque ligne au temps p, on agrège les lignes de `ref_df` du même groupe
    avec période STRICTEMENT < p (anti-fuite, y compris quand df == ref_df), dans
    [p-w, p). Lissé vers la moyenne globale. Colonnes : recent_rate_{group}_{w}.
    """
    gm = float(ref_df[target_col].mean())
    sub = ref_df[[group_col, period_col, target_col]].sort_values(period_col)
    store: dict = {}
    for key, g in sub.groupby(group_col, sort=False):
        per = g[period_col].to_numpy()
        pref = np.concatenate([[0.0], np.cumsum(g[target_col].to_numpy(dtype=float))])
        store[key] = (per, pref)

    periods = df[period_col].to_numpy()
    groups = df[group_col].to_numpy()
    n = len(df)
    out = {w: np.full(n, gm) for w in windows}
    for i in range(n):
        s = store.get(groups[i])
        if s is None:
            continue
        per, pref = s
        p = periods[i]
        hi = bisect.bisect_left(per, p)  # strictement < p
        for w in windows:
            lo = bisect.bisect_left(per, p - w)
            cnt = hi - lo
            if cnt > 0:
                ssum = pref[hi] - pref[lo]
                out[w][i] = (ssum + gm * smoothing) / (cnt + smoothing)

    f = pd.DataFrame(index=df.index)
    short = group_col.replace("_account", "")
    for w in windows:
        f[f"recent_rate_{short}_{w}"] = out[w]
    return f


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
