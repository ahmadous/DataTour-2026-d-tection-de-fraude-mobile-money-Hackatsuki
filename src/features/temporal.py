"""Features structurelles & temporelles — PROPRIÉTÉ DU RÔLE B2.

Angle : ratios de balance, deltas, features dérivées de `period`, interactions
catégorie d'opération × montant, anomalies vs moyenne du compte. Préfixe : `b2_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    AMOUNT,
    DEST_BAL_AFTER,
    DEST_BAL_BEFORE,
    ORIGIN_BAL_AFTER,
    ORIGIN_BAL_BEFORE,
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
