"""Utilitaires partagés : I/O du feature store, seeds, chemins."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FRAUD_OPERATION, OPERATION

# Racine du projet (= dossier parent de src/)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURE_STORE = ROOT / "features_store"
MODELS = ROOT / "models_saved"
SUBMISSIONS = ROOT / "submissions"

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Fixe toutes les sources d'aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_features(df: pd.DataFrame, name: str) -> Path:
    """Écrit un lot de features dans le feature store partagé (parquet).

    `name` DOIT être préfixé par l'auteur : 'b1_behavioral_v1', 'c_graph_v2'...
    """
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    path = FEATURE_STORE / f"{name}.parquet"
    df.to_parquet(path, index=True)
    return path


def load_features(name: str) -> pd.DataFrame:
    """Recharge un lot de features depuis le feature store."""
    return pd.read_parquet(FEATURE_STORE / f"{name}.parquet")


def op03_mask(df: pd.DataFrame) -> pd.Series:
    """Masque booléen des transactions op_03 (les seules pouvant être frauduleuses).

    Stratégie 6.1 du rapport : on modélise UNIQUEMENT op_03 et on force
    une probabilité nulle pour tout le reste (garanti correct, taux de fraude=0).
    """
    return df[OPERATION] == FRAUD_OPERATION


def assemble_full_proba(df: pd.DataFrame, op03_proba: np.ndarray) -> np.ndarray:
    """Reconstruit le vecteur de probas complet : `op03_proba` sur les op_03,
    0.0 partout ailleurs. `op03_proba` doit être aligné sur df[op03_mask(df)].
    """
    full = np.zeros(len(df))
    full[op03_mask(df).values] = op03_proba
    return full


def make_submission(ids: pd.Series, proba: np.ndarray, name: str) -> Path:
    """Écrit un CSV de soumission au format officiel : colonnes `id`, `target`.

    `proba` doit être une probabilité dans [0, 1].
    """
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS / f"{name}.csv"
    pd.DataFrame({"id": ids, "target": proba}).to_csv(path, index=False)
    return path
