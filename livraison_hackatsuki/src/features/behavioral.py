"""Features comportementales par compte/couple — PROPRIÉTÉ DU RÔLE B1.

Angle : agrégations par émetteur/destinataire, fréquences, montants typiques,
degrés bipartites. Préfixe de toutes les colonnes : `b1_`.

⚠️ ANTI-FUITE : toutes ces agrégations sont apprises sur un `ref_df` (= le PASSÉ,
les lignes d'entraînement du fold) et appliquées à `df`. On ne calcule JAMAIS sur
l'ensemble train+valid. Utilisation dans la boucle de CV :

    feats_valid = behavioral_features(df=valid_df, ref_df=train_df)
    feats_train = behavioral_features(df=train_df, ref_df=train_df)

Pour le scoring final : ref_df = tout le train op_03.
⚠️ On n'encode JAMAIS l'ID de compte brut (comptes mixtes -> surapprentissage).
"""
from __future__ import annotations

import pandas as pd

from ..config import AMOUNT, DEST_ACCT, ORIGIN_ACCT


def behavioral_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """Agrégations comportementales apprises sur `ref_df`, appliquées à `df`.

    - b1_pair_count        : nb de tx du couple (origin, dest) vu dans le passé
    - b1_pair_is_new       : couple jamais vu dans le passé (relation nouvelle)
    - b1_pair_amount_med   : montant médian historique du couple
    - b1_amount_vs_pair    : montant courant / montant médian du couple
    - b1_origin_count      : activité de l'émetteur (nb tx passées)
    - b1_dest_count        : activité du destinataire (nb tx passées reçues)
    - b1_dest_in_degree    : nb d'émetteurs distincts payant ce destinataire (collecteur)
    - b1_origin_out_degree : nb de destinataires distincts servis par l'émetteur
    """
    f = pd.DataFrame(index=df.index)
    pair = list(zip(df[ORIGIN_ACCT], df[DEST_ACCT]))

    # --- compteurs par couple ---
    ref_pair = ref_df.groupby([ORIGIN_ACCT, DEST_ACCT])
    pair_count = ref_pair.size()
    pair_amount_med = ref_pair[AMOUNT].median()

    pc = pd.Series(pair).map(pair_count.to_dict()).fillna(0).values
    f["b1_pair_count"] = pc
    f["b1_pair_is_new"] = (pc == 0).astype(int)
    pam = pd.Series(pair).map(pair_amount_med.to_dict()).values
    f["b1_pair_amount_med"] = pd.Series(pam, index=df.index).fillna(-1)
    f["b1_amount_vs_pair"] = df[AMOUNT].values / (pd.Series(pam, index=df.index).fillna(df[AMOUNT].median()) + 1.0)

    # --- activité par compte ---
    origin_count = ref_df[ORIGIN_ACCT].value_counts()
    dest_count = ref_df[DEST_ACCT].value_counts()
    f["b1_origin_count"] = df[ORIGIN_ACCT].map(origin_count).fillna(0).values
    f["b1_dest_count"] = df[DEST_ACCT].map(dest_count).fillna(0).values

    # --- degrés bipartites (collecteur / fan-out) ---
    dest_in_degree = ref_df.groupby(DEST_ACCT)[ORIGIN_ACCT].nunique()
    origin_out_degree = ref_df.groupby(ORIGIN_ACCT)[DEST_ACCT].nunique()
    f["b1_dest_in_degree"] = df[DEST_ACCT].map(dest_in_degree).fillna(0).values
    f["b1_origin_out_degree"] = df[ORIGIN_ACCT].map(origin_out_degree).fillna(0).values

    return f


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Variante simple (ref_df = df) pour exploration rapide hors CV.

    ⚠️ NE PAS utiliser pour la CV/scoring : calcule sur l'ensemble -> fuite.
    Passer par `behavioral_features(df, ref_df)` dans la boucle de validation.
    """
    return behavioral_features(df, df)
