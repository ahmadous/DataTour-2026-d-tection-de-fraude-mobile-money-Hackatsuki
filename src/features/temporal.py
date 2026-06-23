"""Features structurelles & temporelles — PROPRIÉTÉ DU RÔLE B2.

Angle : ratios de balance, deltas, features dérivées de `period`, interactions
catégorie d'opération × montant, anomalies vs moyenne du compte. Préfixe : `b2_`.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from ..config import (
    AMOUNT,
    DEST_BAL_AFTER,
    DEST_BAL_BEFORE,
    ORIGIN_ACCT,
    ORIGIN_BAL_AFTER,
    ORIGIN_BAL_BEFORE,
    PERIOD,
)

# Tolérance d'égalité de solde. Soldes en millions -> 1.0 = quasi-égalité exacte.
_BALANCE_EPS = 1.0


def balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features d'incohérence d'arithmétique de solde — row-level, sans fuite.

    ⚠️ MISE À JOUR POST-EDA (notebook 00) : contrairement à ce que suggérait le
    rapport global, l'incohérence de solde est le signal univarié le PLUS fort
    DANS op_03 (Spearman ~0.11 ; les tx incohérentes sont frauduleuses à ~56%
    contre ~30% en moyenne). Le rapport mesurait sur tout le dataset (dilué par
    les opérations sans fraude). On garde donc ces features et on mesure le gain.

    Côté émetteur : attendu après = avant - montant.
    Côté destinataire : attendu après = avant + montant.

    Ajoute aussi des fingerprints connus de la fraude PaySim (nos données en
    dérivent) : émetteur vidé, solde émetteur ≈ 0 après, destinataire non crédité.
    """
    f = pd.DataFrame(index=df.index)

    origin_resid = df[ORIGIN_BAL_AFTER] - (df[ORIGIN_BAL_BEFORE] - df[AMOUNT])
    f["b2_origin_balance_residual"] = origin_resid
    f["b2_origin_resid_log"] = np.sign(origin_resid) * np.log1p(origin_resid.abs())
    f["b2_origin_balance_inconsistent"] = (origin_resid.abs() > _BALANCE_EPS).astype(int)

    dest_resid = df[DEST_BAL_AFTER] - (df[DEST_BAL_BEFORE] + df[AMOUNT])
    f["b2_dest_balance_residual"] = dest_resid
    f["b2_dest_resid_log"] = np.sign(dest_resid) * np.log1p(dest_resid.abs())
    f["b2_dest_balance_inconsistent"] = (dest_resid.abs() > _BALANCE_EPS).astype(int)

    # --- Fingerprints de fraude PaySim (nos données dérivent de PaySim) ---
    # Tolérance relative : montants/soldes vont de centaines à millions.
    rel = lambda a, b: (a - b).abs() <= (b.abs() * 0.01 + _BALANCE_EPS)

    # 1) Émetteur vidé : il envoie ~tout son solde (montant ≈ solde avant).
    f["b2_origin_emptied"] = rel(df[AMOUNT], df[ORIGIN_BAL_BEFORE]).astype(int)
    # 2) Solde émetteur ≈ 0 après la transaction.
    f["b2_origin_after_zero"] = (df[ORIGIN_BAL_AFTER].abs() <= _BALANCE_EPS).astype(int)
    # 3) Destinataire non crédité : solde inchangé malgré réception d'un montant > 0.
    f["b2_dest_not_credited"] = (
        ((df[DEST_BAL_AFTER] - df[DEST_BAL_BEFORE]).abs() <= _BALANCE_EPS)
        & (df[AMOUNT] > _BALANCE_EPS)
    ).astype(int)
    # 4) Ratio montant / solde émetteur avant (continu, ≈1 quand émetteur vidé).
    f["b2_amount_over_origin_before"] = df[AMOUNT] / (df[ORIGIN_BAL_BEFORE].abs() + 1.0)

    return f


def recency_features(
    df: pd.DataFrame, ref_df: pd.DataFrame, windows: tuple[int, ...] = (5, 10)
) -> pd.DataFrame:
    """Dynamique récente par émetteur — apprise sur le PASSÉ STRICT (anti-fuite).

    Pour chaque transaction au temps p, on ne regarde que les tx du même émetteur
    de `ref_df` dont la période est STRICTEMENT < p. Le `< p` garantit l'absence de
    fuite même quand df == ref_df (un fold d'entraînement ne se voit pas lui-même).

    - b2_periods_since_last_origin_tx : écart à la dernière tx de l'émetteur (-1 si aucune)
    - b2_origin_tx_before             : nb total de tx passées de l'émetteur
    - b2_origin_recent_count_{w}      : nb de tx de l'émetteur dans [p-w, p)
    """
    # émetteur -> périodes triées (depuis le passé de référence)
    ref_dict = {
        acc: np.sort(s.values)
        for acc, s in ref_df.groupby(ORIGIN_ACCT)[PERIOD]
    }
    periods = df[PERIOD].to_numpy()
    origins = df[ORIGIN_ACCT].to_numpy()
    n = len(df)

    since_last = np.full(n, -1.0)
    total_before = np.zeros(n)
    recent = {w: np.zeros(n) for w in windows}

    for i in range(n):
        arr = ref_dict.get(origins[i])
        if arr is None:
            continue
        p = periods[i]
        j = bisect.bisect_left(arr, p)  # nb d'éléments strictement < p
        total_before[i] = j
        if j > 0:
            since_last[i] = p - arr[j - 1]
            for w in windows:
                lo = bisect.bisect_left(arr, p - w)
                recent[w][i] = j - lo

    f = pd.DataFrame(index=df.index)
    f["b2_periods_since_last_origin_tx"] = since_last
    f["b2_origin_tx_before"] = total_before
    for w in windows:
        f[f"b2_origin_recent_count_{w}"] = recent[w]
    return f


def rhythm_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Rythme temporel de l'émetteur (hypothèse loi géométrique/Poisson).

    Pour chaque tx au temps p, à partir des tx du même émetteur dans `ref_df` de
    période STRICTEMENT < p (anti-fuite) :
      - b2_origin_age        : p - première période vue de l'émetteur (ancienneté)
      - b2_origin_mean_itv   : intervalle moyen entre ses tx passées
      - b2_interval_vs_mean  : (intervalle depuis la dernière tx) / intervalle moyen
                               -> déviation au rythme habituel (>1 = ralenti, <1 = rafale)
    Valeurs par défaut -1 quand pas d'historique exploitable.
    """
    ref_dict = {acc: np.sort(s.values) for acc, s in ref_df.groupby(ORIGIN_ACCT)[PERIOD]}
    periods = df[PERIOD].to_numpy()
    origins = df[ORIGIN_ACCT].to_numpy()
    n = len(df)
    age = np.full(n, -1.0)
    mean_itv = np.full(n, -1.0)
    itv_vs_mean = np.full(n, -1.0)

    for i in range(n):
        arr = ref_dict.get(origins[i])
        if arr is None:
            continue
        p = periods[i]
        j = bisect.bisect_left(arr, p)  # nb de tx strictement avant p
        if j == 0:
            continue
        first, last = arr[0], arr[j - 1]
        age[i] = p - first
        if j >= 2:
            mi = (last - first) / (j - 1)
            mean_itv[i] = mi
            if mi > 0:
                itv_vs_mean[i] = (p - last) / mi

    f = pd.DataFrame(index=df.index)
    f["b2_origin_age"] = age
    f["b2_origin_mean_itv"] = mean_itv
    f["b2_interval_vs_mean"] = itv_vs_mean
    return f


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features structurelles/temporelles, indexées comme `df`.

    Inclut déjà `balance_features`. Pistes restantes (rapport 6.2) :
      - b2_amount_vs_balance     : montant / origin_balance_before
      - b2_amount_vs_acct_mean   : écart du montant à la moyenne du compte
      - b2_periods_since_last_tx : intervalle depuis la dernière tx de l'émetteur
      - b2_origin_recent_count   : nb de tx de l'émetteur sur les K dernières périodes
      - b2_origin_burst          : rafale d'activité (densité de tx récentes)

    ⚠️ Features "récentes" = calculées sur le passé strict (anti-fuite temporelle).
    """
    feats = balance_features(df)
    # TODO(B2): ajouter les features de dynamique récente (fold-safe)
    return feats
