"""Features structurelles & temporelles — PROPRIÉTÉ DU RÔLE B2.

Angle : ratios de balance, deltas, features dérivées de `period`, interactions
catégorie d'opération × montant, anomalies vs moyenne du compte. Préfixe : `b2_`.
"""
from __future__ import annotations

import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features structurelles/temporelles, indexées comme `df`.

    Pistes :
      - b2_balance_ratio_in   : balance après / balance avant (émetteur)
      - b2_balance_delta      : variation de balance sur la transaction
      - b2_amount_vs_balance  : montant / balance avant
      - b2_period_hour        : composante horaire de period
      - b2_amount_vs_acct_mean: écart du montant à la moyenne du compte
    """
    feats = pd.DataFrame(index=df.index)
    # TODO(B2): implémenter après l'EDA commune (jour 1-2)
    return feats
