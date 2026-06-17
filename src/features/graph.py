"""Features de graphe LÉGÈRES — bipartite uniquement (porté par C).

⚠️ CORRECTION POST-EDA (voir FINDINGS.md) :
Le graphe est BIPARTI — les émetteurs (acc_o_) et destinataires (acc_d_) forment
deux univers totalement séparés, aucun compte n'est les deux. Donc :
  - PAS de chaînes de mules, PAS de cycles à détecter ;
  - PageRank / Louvain / GNN ne sont PAS justifiés (le rapport le dit explicitement).

Seules survivent les features de DEGRÉ bipartite, qui restent du signal potentiel :
  - c_dest_in_degree   : nb d'émetteurs distincts payant ce destinataire (motif "collecteur")
  - c_origin_out_degree: nb de destinataires distincts servis par cet émetteur
  - c_pair_volume_share: part du volume de l'émetteur concentrée sur ce destinataire

Ces features se calculent par simple groupby, networkx est inutile ici.
⚠️ Anti-fuite : calculer ces degrés sur le PASSÉ (cf. logique de src/encoding.py)
si on les dérive des labels ; ici ce sont des comptages structurels, mais rester
vigilant si on agrège quoi que ce soit qui dépende du temps.

GO/NO-GO : si gain < 0.005 AP à la mi-semaine 2, C bascule sur l'analyse d'erreurs.
"""
from __future__ import annotations

import pandas as pd

from ..config import DEST_ACCT, ORIGIN_ACCT


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Features de degré bipartite, indexées comme `df`.

    Implémentation directe par groupby (pas de networkx nécessaire).
    """
    feats = pd.DataFrame(index=df.index)

    # Degré entrant du destinataire : combien d'émetteurs distincts le paient.
    dest_in = df.groupby(DEST_ACCT)[ORIGIN_ACCT].transform("nunique")
    feats["c_dest_in_degree"] = dest_in

    # Degré sortant de l'émetteur : combien de destinataires distincts il sert.
    origin_out = df.groupby(ORIGIN_ACCT)[DEST_ACCT].transform("nunique")
    feats["c_origin_out_degree"] = origin_out

    # TODO(C): c_pair_volume_share (concentration du volume sur un partenaire)
    # et tester le gain AP avant d'aller plus loin (go/no-go mi-S2).
    return feats
