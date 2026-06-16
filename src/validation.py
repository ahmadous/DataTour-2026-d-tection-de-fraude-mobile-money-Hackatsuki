"""Stratégie de validation — PROPRIÉTÉ DU RÔLE A.

C'est la source de vérité UNIQUE de l'équipe pour la cross-validation.
Personne ne réimplémente sa propre CV : tout le monde importe `get_folds`.

La brief prévient deux fois :
  - éviter le surapprentissage sur les identifiants de comptes ;
  - produire de véritables probabilités bien calibrées.

=> On compare trois schémas et on garde celui dont la corrélation
   CV <-> LB public est la meilleure. Si CV et LB divergent, on croit la CV.
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
    """Split temporel sur `period` : on valide toujours sur le futur.

    Reproduit la réalité (on prédit des transactions postérieures à l'entraînement).
    """
    order = np.argsort(period.values, kind="stable")
    bins = np.array_split(order, n_splits + 1)
    for i in range(1, n_splits + 1):
        train_idx = np.concatenate(bins[:i])
        valid_idx = bins[i]
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
