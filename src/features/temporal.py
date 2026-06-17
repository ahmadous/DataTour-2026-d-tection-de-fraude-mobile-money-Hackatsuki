"""Features structurelles & temporelles — PROPRIÉTÉ DU RÔLE B2.

Angle : ratios de balance, deltas, features dérivées de `period`, interactions
catégorie d'opération × montant, anomalies vs moyenne du compte. Préfixe : `b2_`.
"""
from __future__ import annotations

import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features structurelles/temporelles, indexées comme `df`.

    Pistes (rapport 6.2 — régularité temporelle & dynamique récente) :
      - b2_amount_vs_balance     : montant / origin_balance_before
      - b2_amount_vs_acct_mean   : écart du montant à la moyenne du compte
      - b2_periods_since_last_tx : intervalle depuis la dernière tx de l'émetteur
      - b2_origin_recent_count   : nb de tx de l'émetteur sur les K dernières périodes
      - b2_origin_burst          : rafale d'activité (densité de tx récentes)

    ⚠️ NE PAS s'appuyer sur l'incohérence de solde : le rapport montre que les
       soldes sont bruités (~40% d'arithmétique fausse) et que l'incohérence ne
       corrèle quasiment pas avec la fraude (Spearman +0.04). Piège connu.
    ⚠️ Features "récentes" = calculées sur le passé strict (anti-fuite temporelle).
    """
    feats = pd.DataFrame(index=df.index)
    # TODO(B2): implémenter après l'EDA commune (jour 1-2)
    return feats
