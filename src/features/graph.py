"""Features graphe — PROPRIÉTÉ DU RÔLE C (piste R&D, semaines 1-2).

Le réseau des transactions est un graphe orienté : noeuds = comptes,
arêtes = transactions (émetteur -> destinataire). Préfixe : `c_`.

⚠️ GO/NO-GO : si gain < 0.005 AP à la mi-semaine 2, on abandonne cette piste
et C bascule sur l'analyse d'erreurs. PAS de GNN au début (trop lent à itérer).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame, src_col: str, dst_col: str) -> nx.DiGraph:
    """Construit le graphe orienté des transactions."""
    return nx.from_pandas_edgelist(
        df, source=src_col, target=dst_col, create_using=nx.DiGraph()
    )


def build(df: pd.DataFrame, src_col: str, dst_col: str) -> pd.DataFrame:
    """Calcule des features de graphe par compte, ramenées au niveau transaction.

    Pistes (légères, sans GNN) :
      - c_pagerank          : PageRank du compte
      - c_in_degree         : degré entrant (combien de comptes lui envoient)
      - c_out_degree        : degré sortant
      - c_louvain_comm_size : taille de la communauté Louvain
    """
    feats = pd.DataFrame(index=df.index)
    # TODO(C): implémenter, mesurer le gain AP, puis décider go/no-go mi-S2
    return feats
