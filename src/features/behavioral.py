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

    À implémenter une fois les colonnes réelles connues. Pistes :
      - b1_sender_tx_count       : nb de transactions par émetteur
      - b1_sender_amount_mean    : montant moyen par émetteur
      - b1_sender_amount_std     : dispersion des montants par émetteur
      - b1_recipient_tx_count    : nb de transactions reçues par destinataire
      - b1_pair_freq             : fréquence du couple (émetteur, destinataire)
    """
    feats = pd.DataFrame(index=df.index)
    # TODO(B1): implémenter après l'EDA commune (jour 1-2)
    return feats
