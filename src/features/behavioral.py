"""Features comportementales par compte — PROPRIÉTÉ DU RÔLE B1.

Angle : agrégations par émetteur/destinataire, fréquences, statistiques de montants,
encodages catégoriels. Préfixe de toutes les colonnes : `b1_`.

⚠️ Toute agrégation par compte doit être calculée SANS fuite : si on agrège sur
l'ensemble du dataset, on utilise un encodage out-of-fold (cf. src/validation.py).
"""
from __future__ import annotations

import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features comportementales. Renvoie un DataFrame indexé comme `df`.

    Pistes prioritaires (rapport 6.2 — features comportementales, PAS brutes) :
      - b1_pair_freq             : fréquence du couple (origin, destination)
      - b1_pair_amount_typical   : montant médian du couple (origin, destination)
      - b1_pair_is_new           : relation jamais vue avant cette période
      - b1_origin_tx_count       : nb de transactions de l'émetteur
      - b1_origin_amount_mean    : montant moyen de l'émetteur
      - b1_dest_tx_count         : nb de transactions reçues par le destinataire

    ⚠️ Toute fréquence se calcule fold-by-fold sur le passé (cf. src/encoding.py).
    ⚠️ Ne PAS encoder l'ID de compte brut (comptes mixtes -> surapprentissage).
    """
    feats = pd.DataFrame(index=df.index)
    # TODO(B1): implémenter après l'EDA commune (jour 1-2)
    return feats
