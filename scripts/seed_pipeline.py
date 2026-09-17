"""Pipeline complet champion+pseudo pour une graine donnée (réplique fidèle du notebook 23).

Usage: python scripts/seed_pipeline.py <seed> [<seed2> ...]
Produit, pour chaque graine S :
  - submissions/37_seed{S}_rank_w50.csv  (rank-blend w50 champion×pseudo, recette 28)
  - /tmp/seedpipe/probas_seed{S}.npz     (probas champion et pseudo sur test op03, pour blends multi-graines)
Les features du champion (seed-indépendantes) sont cachées dans /tmp/seedpipe/.
"""
from __future__ import annotations
import sys, gc, pickle, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src import config as C
from src.utils import op03_mask, seed_everything, make_submission
from src.features.temporal import balance_features, recency_features
from src.features.behavioral import behavioral_features
from src.encoding import oof_target_encode_train, fit_target_map, apply_target_map, recent_target_rate

EPS = 1e-6
WINDOWS = (5, 10, 20)
SM = 30
THRESH_FRAUD = 0.98
THRESH_LEGIT = 0.02
CACHE = Path("/tmp/seedpipe")
CACHE.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
        freq = ref[col].value_counts(normalize=True)
        X[f"freq_{col}"] = df[col].map(freq).fillna(0).values
    beh = behavioral_features(df, ref).reset_index(drop=True)
    rec = recency_features(df, ref).reset_index(drop=True)
    rt = recent_target_rate(df, ref, C.ORIGIN_ACCT, C.PERIOD, C.TARGET, WINDOWS).reset_index(drop=True)
    return pd.concat([X, beh, rec, rt], axis=1)

def feats_train(df, ref):
    X = base_build(df, ref)
    X["te_origin"] = oof_target_encode_train(ref, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
    return X

def feats_apply(df, ref):
    X = base_build(df, ref)
    mp, gm = fit_target_map(ref, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
    X["te_origin"] = apply_target_map(df, C.ORIGIN_ACCT, mp, gm)
    return X

def make_cat(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", depth=6,
                              learning_rate=0.05, iterations=600, random_seed=seed,
                              thread_count=4, verbose=False)

def rank01(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return (r + 1.0) / len(x)  # dans (0,1] : aucune ligne op03 ne tombe à 0

def main():
    seeds = [int(s) for s in sys.argv[1:]]
    assert seeds, "donner au moins une graine"
    log(f"Seeds: {seeds}")

    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    op03 = op03_mask(train).to_numpy()
    ref_full = train.iloc[np.where(op03)[0]]
    yf = train[C.TARGET].to_numpy()[op03]
    te_op = op03_mask(test).to_numpy()
    test_op03 = test.iloc[np.where(te_op)[0]].copy()
    test_ids = test[C.ID].copy()
    del train
    gc.collect()
    log(f"train op03={len(ref_full):,}  test op03={len(test_op03):,}")

    # Features champion (seed-indépendantes) — cache
    xf_p, xte_p = CACHE / "Xf.pkl", CACHE / "Xte.pkl"
    if xf_p.exists() and xte_p.exists():
        Xf = pickle.load(open(xf_p, "rb")); Xte = pickle.load(open(xte_p, "rb"))
        log("features champion rechargées du cache")
    else:
        log("build features champion (train)...")
        Xf = feats_train(ref_full, ref_full)
        log("build features champion (test)...")
        Xte = feats_apply(test_op03, ref_full)
        pickle.dump(Xf, open(xf_p, "wb")); pickle.dump(Xte, open(xte_p, "wb"))
        log("features champion cachées")

    for seed in seeds:
        out_npz = CACHE / f"probas_seed{seed}.npz"
        if out_npz.exists():
            log(f"seed {seed}: déjà fait, skip"); continue
        seed_everything(seed)
        log(f"seed {seed}: étape 1 — champion fit...")
        m1 = make_cat(seed).fit(Xf, yf)
        proba_test = m1.predict_proba(Xte)[:, 1]
        del m1; gc.collect()

        pf, pl = proba_test > THRESH_FRAUD, proba_test < THRESH_LEGIT
        log(f"seed {seed}: pseudo-labels fraude={pf.sum():,} légit={pl.sum():,}")
        pm = pf | pl
        pseudo_df = test_op03.iloc[np.where(pm)[0]].copy()
        pseudo_df[C.TARGET] = 0.0
        pseudo_df.loc[pseudo_df.index[pf[pm]], C.TARGET] = 1.0
        train_aug = pd.concat([ref_full, pseudo_df], ignore_index=True)
        y_aug = train_aug[C.TARGET].to_numpy()

        log(f"seed {seed}: étape 2 — features augmentées ({len(train_aug):,})...")
        Xf_aug = feats_train(train_aug, train_aug)
        m2 = make_cat(seed).fit(Xf_aug, y_aug)
        del Xf_aug; gc.collect()
        log(f"seed {seed}: scoring test (features apply)...")
        Xte2 = feats_apply(test_op03, train_aug)
        proba_test2 = m2.predict_proba(Xte2)[:, 1]
        del m2, Xte2, train_aug; gc.collect()

        np.savez(out_npz, champ=proba_test, pseudo=proba_test2)

        blend = 0.5 * rank01(proba_test) + 0.5 * rank01(proba_test2)
        full = np.zeros(len(test_ids))
        full[te_op] = blend
        path = make_submission(test_ids, full, f"37_seed{seed}_rank_w50")
        corr = np.corrcoef(proba_test, proba_test2)[0, 1]
        log(f"seed {seed}: FINI → {path} (corr champ×pseudo={corr:.4f})")

    log("TOUT FINI")

if __name__ == "__main__":
    main()
