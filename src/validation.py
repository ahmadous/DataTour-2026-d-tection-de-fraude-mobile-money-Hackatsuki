"""Stratégie de validation — PROPRIÉTÉ DU RÔLE A.

C'est la source de vérité UNIQUE de l'équipe pour la cross-validation.
Personne ne réimplémente sa propre CV : tout le monde importe `get_folds`.

DÉCISION (figée par l'EDA, voir FINDINGS.md) :
  -> `time_folds` est le schéma de RÉFÉRENCE. Le test est strictement dans le
     futur (périodes 106-143 vs train 0-105) et le taux de fraude varie de 4% à
     18% selon la période. Une CV aléatoire mélange passé/futur et MENT.
  -> `stratified_folds` et `group_folds` sont gardés pour comparaison/diagnostic
     uniquement, pas pour décider.

Rappels critiques de l'EDA :
  - On valide la CV sur les op_03 uniquement (toute la fraude y est).
  - Tout encodage fréquence/target se calcule fold-by-fold sur le PASSÉ
    (cf. src/encoding.py), jamais sur l'ensemble, sous peine de fuite du futur.
  - Si CV et LB public divergent, on croit la CV.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

SEED = 42
N_SPLITS = 5


def stratified_folds(
    y: pd.Series, n_splits: int = N_SPLITS, seed: int = SEED
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Schéma naïf : StratifiedKFold. Référence basse, sujet aux fuites par compte."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), y)


def group_folds(
    y: pd.Series, groups: pd.Series, n_splits: int = N_SPLITS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold sur l'ID émetteur : un compte ne peut pas être dans train ET valid.

    C'est le rempart principal contre le surapprentissage sur les identifiants.
    `groups` = colonne d'ID compte (ex. sender id).
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(np.zeros(len(y)), y, groups=groups)


def time_folds(
    period: pd.Series, n_splits: int = N_SPLITS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Split temporel à fenêtre expansive, sur les frontières de `period`.

    Une même période n'est JAMAIS coupée entre train et valid : on entraîne sur
    les périodes anciennes, on valide sur un bloc de périodes plus récentes.
    Reproduit le décalage réel train (0-105) -> test (106-143).
    """
    period = pd.Series(period).reset_index(drop=True)
    uniq = np.sort(period.unique())
    blocks = np.array_split(uniq, n_splits + 1)
    for i in range(1, n_splits + 1):
        train_periods = np.concatenate(blocks[:i])
        valid_periods = blocks[i]
        train_idx = period.index[period.isin(train_periods)].to_numpy()
        valid_idx = period.index[period.isin(valid_periods)].to_numpy()
        yield train_idx, valid_idx


def evaluate_ap(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Métrique officielle de la compétition : Average Precision."""
    return float(average_precision_score(y_true, y_proba))


def cv_score(
    folds: Iterator[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    oof_proba: np.ndarray,
) -> tuple[float, float]:
    """AP moyenne ± écart-type à partir de prédictions out-of-fold.

    Renvoie (moyenne, std) — à reporter dans experiments/exp_NN.md.
    """
    scores = [evaluate_ap(y[v], oof_proba[v]) for _, v in folds]
    return float(np.mean(scores)), float(np.std(scores))


def summarize_cv(
    y: np.ndarray,
    oof_proba: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    recent_k: int = 2,
) -> dict:
    """Synthèse CV temporelle en privilégiant les folds RÉCENTS.

    Le test est juste après le dernier fold : les folds récents sont le meilleur
    proxy du LB. On ne juge donc PAS sur la moyenne globale (diluée par les vieux
    folds, régime différent), mais sur `recent_mean` / `last`.

    Renvoie {per_fold, global, recent_mean, last}.
    `folds` doit être une LISTE (réutilisée), pas un générateur.
    """
    per_fold = [evaluate_ap(y[v], oof_proba[v]) for _, v in folds]
    return {
        "per_fold": per_fold,
        "global": evaluate_ap(y, oof_proba),
        "recent_mean": float(np.mean(per_fold[-recent_k:])),
        "last": per_fold[-1],
    }
