# -*- coding: utf-8 -*-
"""
Solution — Équipe Hackatsuki (DataTour 2026, fraude mobile money)
=================================================================
Régénère `submissions/submission.csv` (le fichier soumis, correspondant au
Private Score, est fourni à la racine : `submission.csv`).

Usage (Windows, macOS, Linux) :
    pip install -r requirements.txt     # Python 3.9 à 3.12
    # placer train.csv, test.csv, sample_submission.csv dans data/
    python solution.py

Recette : champion CatBoost (features + te_origin fold-safe, seed 42)
  -> pseudo-labeling 1 cycle (seuils 0.98/0.02)
  -> blend par moyenne de rangs 25% champion / 75% pseudo
  -> aucune calibration (l'Average Precision est une métrique de rang).
Détails : README.md. Pipeline entièrement déterministe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
assert (ROOT / "src").is_dir(), "dossier src/ introuvable à côté de solution.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C
from src.utils import op03_mask, seed_everything
from src.features.temporal import balance_features, recency_features
from src.features.behavioral import behavioral_features
from src.encoding import (oof_target_encode_train, fit_target_map,
                          apply_target_map, recent_target_rate)

EPS = 1e-6
WINDOWS = (5, 10, 20)
SM = 30


def row_features(df):
    f = pd.DataFrame(index=df.index)
    f["amount_log1p"] = np.log1p(np.maximum(df[C.AMOUNT], 0))
    f["amount_vs_origin_before"] = df[C.AMOUNT] / (np.abs(df[C.ORIGIN_BAL_BEFORE]) + EPS)
    f["amount_vs_dest_before"] = df[C.AMOUNT] / (np.abs(df[C.DEST_BAL_BEFORE]) + EPS)
    f["origin_balance_before"] = df[C.ORIGIN_BAL_BEFORE]
    f["dest_balance_before"] = df[C.DEST_BAL_BEFORE]
    return pd.concat([f, balance_features(df)], axis=1)


def base_build(df, ref):
    X = row_features(df).reset_index(drop=True)
    for col in [C.ORIGIN_ACCT, C.DEST_ACCT]:
        X[f"fq_{col}"] = df[col].map(ref[col].value_counts(normalize=True)).fillna(0).values
    return pd.concat([X,
                      behavioral_features(df, ref).reset_index(drop=True),
                      recency_features(df, ref).reset_index(drop=True),
                      recent_target_rate(df, ref, C.ORIGIN_ACCT, C.PERIOD, C.TARGET,
                                         WINDOWS).reset_index(drop=True)], axis=1)


def ftr(df, ref):
    X = base_build(df, ref)
    X["te"] = oof_target_encode_train(ref, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
    return X


def fap(df, ref):
    X = base_build(df, ref)
    mp, gm = fit_target_map(ref, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
    X["te"] = apply_target_map(df, C.ORIGIN_ACCT, mp, gm)
    return X


def make_cat():
    return CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", depth=6,
                              learning_rate=0.05, iterations=600, random_seed=42,
                              verbose=False)


def rk(x):
    return np.argsort(np.argsort(x)) / (len(x) - 1)


EXPECTED = {"catboost": "1.2.5", "numpy": "1.26.4", "pandas": "2.2.2"}


def check_versions():
    import catboost
    got = {"catboost": catboost.__version__, "numpy": np.__version__,
           "pandas": pd.__version__}
    py = ".".join(map(str, sys.version_info[:2]))
    print(f"Python {py} | " + " | ".join(f"{k} {v}" for k, v in got.items()))
    bad = {k: (v, EXPECTED[k]) for k, v in got.items() if v != EXPECTED[k]}
    if bad or sys.version_info[:2] < (3, 9) or sys.version_info[:2] > (3, 12):
        print("  ⚠️ ATTENTION : versions différentes de requirements.txt :")
        for k, (v, w) in bad.items():
            print(f"     - {k} : installé {v}, attendu {w}")
        if sys.version_info[:2] > (3, 12):
            print(f"     - Python {py} : utiliser 3.9 à 3.12 (cf. README, options B/C)")
        print("  → la reproduction exacte (IDENTIQUE) n'est garantie qu'à versions égales.")


def main():
    seed_everything(42)
    check_versions()
    DATA = ROOT / "data"
    for f in ("train.csv", "test.csv", "sample_submission.csv"):
        assert (DATA / f).exists(), f"data/{f} manquant — y placer les fichiers de la compétition"
    print("Chargement des données...")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    op03 = op03_mask(train).to_numpy()
    y_all = train[C.TARGET].to_numpy()
    te_op = op03_mask(test).to_numpy()
    print(f"train {len(train):,} | test {len(test):,} | "
          f"op03 train {op03.sum():,} / test {te_op.sum():,}")

    # Étape 1 — champion sur tout le train op_03
    print("\n[1/3] Entraînement du champion CatBoost (~5 min)...")
    ref0 = train.iloc[np.where(op03)[0]]
    y0 = y_all[op03]
    test_op = test.iloc[np.where(te_op)[0]].copy()
    m_champion = make_cat().fit(ftr(ref0, ref0), y0)
    pch = m_champion.predict_proba(fap(test_op, ref0))[:, 1]
    # Optionnel — sauvegarder le modèle entraîné (décommenter si besoin) :
    # import joblib; (ROOT / "models").mkdir(exist_ok=True)
    # joblib.dump(m_champion, ROOT / "models" / "champion.pkl")
    # (rechargement : m = joblib.load(...) puis m.predict_proba(features))

    # Étape 2 — pseudo-labeling 1 cycle (0.98 / 0.02) et réentraînement
    print("[2/3] Pseudo-labeling + réentraînement (~6 min)...")
    mf = pch > 0.98
    ml = pch < 0.02
    pse = test_op.iloc[np.where(mf | ml)[0]].copy()
    pse[C.TARGET] = (pch[mf | ml] > 0.98).astype(float)
    aug = pd.concat([ref0, pse], ignore_index=True)
    print(f"      pseudo-labels : {int(mf.sum())} fraudes / {int(ml.sum())} légitimes "
          f"| train augmenté {len(aug):,}")
    m_pseudo = make_cat().fit(ftr(aug, aug), aug[C.TARGET].to_numpy())
    p1 = m_pseudo.predict_proba(fap(test_op, aug))[:, 1]
    # Optionnel — sauvegarder le second modèle (décommenter si besoin) :
    # joblib.dump(m_pseudo, ROOT / "models" / "pseudo.pkl")

    # Étape 3 — blend par moyenne de rangs 25/75 et écriture
    print("[3/3] Blend rank-average 25/75 et écriture de la soumission...")
    WP = 0.75
    blend = (1 - WP) * rk(pch) + WP * rk(p1)
    full = np.zeros(len(test))
    full[te_op] = blend

    out_dir = ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "submission.csv"
    pd.DataFrame({"id": test[C.ID], "target": full}).to_csv(path, index=False)

    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "target"] and len(sub) == len(test)
    assert set(sub["id"]) == set(sample["id"]) and sub["target"].between(0, 1).all()
    print(f"      soumission écrite : {path}")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
