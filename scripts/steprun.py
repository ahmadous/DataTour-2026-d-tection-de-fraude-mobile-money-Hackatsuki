"""Runner par étapes pour sandbox à appels limités (45 s).

Usage: python scripts/steprun.py <stage> <seed>
Stages: fit1 | aug_tr | aug_te | fit2 | blend
Tout l'état vit dans /tmp/seedpipe/. Recette = notebook 23 + rank-blend w50 (recette 28).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import config as C
from src.utils import make_submission
from src.features.temporal import balance_features, recency_features
from src.features.behavioral import behavioral_features
from src.encoding import oof_target_encode_train, fit_target_map, apply_target_map, recent_target_rate

CACHE = Path("/tmp/seedpipe")
EPS = 1e-6
SM = 30
TH_F, TH_L = 0.98, 0.02

def cat(seed, snapshot):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC", depth=6,
                              learning_rate=0.05, iterations=600, random_seed=seed,
                              thread_count=4, verbose=False)

def rowbal(df):
    f = pd.DataFrame(index=df.index)
    f["amount_log1p"] = np.log1p(np.maximum(df[C.AMOUNT], 0))
    f["amount_vs_origin_before"] = df[C.AMOUNT] / (np.abs(df[C.ORIGIN_BAL_BEFORE]) + EPS)
    f["amount_vs_dest_before"] = df[C.AMOUNT] / (np.abs(df[C.DEST_BAL_BEFORE]) + EPS)
    f["origin_balance_before"] = df[C.ORIGIN_BAL_BEFORE]
    f["dest_balance_before"] = df[C.DEST_BAL_BEFORE]
    return pd.concat([f, balance_features(df)], axis=1)

def base_build(df, ref):
    X = rowbal(df).reset_index(drop=True)
    for col in [C.ORIGIN_ACCT, C.DEST_ACCT]:
        fr = ref[col].value_counts(normalize=True)
        X[f"freq_{col}"] = df[col].map(fr).fillna(0).values
    X = pd.concat([X,
        behavioral_features(df, ref).reset_index(drop=True),
        recency_features(df, ref).reset_index(drop=True),
        recent_target_rate(df, ref, C.ORIGIN_ACCT, C.PERIOD, C.TARGET, (5, 10, 20)).reset_index(drop=True)], axis=1)
    return X

def rank01(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return (r + 1.0) / len(x)

def get_train_aug(seed):
    ref = pd.read_parquet(CACHE / "ref.parquet")
    tst = pd.read_parquet(CACHE / "tst.parquet")
    proba = np.load(CACHE / f"champ_proba_{seed}.npy")
    pf, pl = proba > TH_F, proba < TH_L
    pm = pf | pl
    pseudo = tst.iloc[np.where(pm)[0]].copy()
    pseudo[C.TARGET] = 0.0
    pseudo.loc[pseudo.index[pf[pm]], C.TARGET] = 1.0
    return pd.concat([ref, pseudo], ignore_index=True), tst

def fit_snapshot(seed, X, y, tag):
    model_p = CACHE / f"{tag}_{seed}.cbm"
    if model_p.exists():
        print(f"{tag} seed {seed}: déjà entraîné")
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(str(model_p))
        return m
    m = cat(seed, None)
    m.fit(X, y, snapshot_file=str(CACHE / f"snap_{tag}_{seed}"), save_snapshot=True, snapshot_interval=5)
    m.save_model(str(model_p))
    print(f"{tag} seed {seed}: fit terminé ({m.tree_count_} arbres)")
    return m

def main():
    stage, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    if stage == "fit1":
        X = pd.read_parquet(CACHE / "champ_X_tr.parquet")
        y = np.load(CACHE / "y_tr.npy")
        m = fit_snapshot(seed, X, y, "fit1")
        Xte = pd.read_parquet(CACHE / "champ_X_te.parquet")
        np.save(CACHE / f"champ_proba_{seed}.npy", m.predict_proba(Xte)[:, 1])
        print("champ_proba sauvé")
    elif stage == "aug_tr":
        train_aug, _ = get_train_aug(seed)
        X = base_build(train_aug, train_aug)
        X["te_origin"] = oof_target_encode_train(train_aug, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
        X.to_parquet(CACHE / f"aug_X_tr_{seed}.parquet")
        np.save(CACHE / f"y_aug_{seed}.npy", train_aug[C.TARGET].to_numpy())
        print(f"aug_tr {X.shape}")
    elif stage == "aug_te":
        train_aug, tst = get_train_aug(seed)
        X = base_build(tst, train_aug)
        mp, gm = fit_target_map(train_aug, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
        X["te_origin"] = apply_target_map(tst, C.ORIGIN_ACCT, mp, gm)
        X.to_parquet(CACHE / f"aug_X_te_{seed}.parquet")
        print(f"aug_te {X.shape}")
    elif stage == "fit2":
        X = pd.read_parquet(CACHE / f"aug_X_tr_{seed}.parquet")
        y = np.load(CACHE / f"y_aug_{seed}.npy")
        m = fit_snapshot(seed, X, y, "fit2")
        Xte = pd.read_parquet(CACHE / f"aug_X_te_{seed}.parquet")
        np.save(CACHE / f"pseudo_proba_{seed}.npy", m.predict_proba(Xte)[:, 1])
        print("pseudo_proba sauvé")
    elif stage == "blend":
        p1 = np.load(CACHE / f"champ_proba_{seed}.npy")
        p2 = np.load(CACHE / f"pseudo_proba_{seed}.npy")
        te_op = np.load(CACHE / "te_op.npy")
        ids = pd.read_parquet(CACHE / "test_ids.parquet")[C.ID]
        blend = 0.5 * rank01(p1) + 0.5 * rank01(p2)
        full = np.zeros(len(ids)); full[te_op] = blend
        path = make_submission(ids, full, f"37_seed{seed}_rank_w50")
        print(f"OK → {path} | corr champ×pseudo = {np.corrcoef(p1, p2)[0,1]:.4f}")
    else:
        raise SystemExit(f"stage inconnu: {stage}")
    print(f"[{stage} seed {seed}] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
