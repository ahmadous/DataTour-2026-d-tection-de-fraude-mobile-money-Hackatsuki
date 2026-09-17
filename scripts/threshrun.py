"""Variantes de seuil du pseudo-labeling sur la recette champion (seed 42).

Source des pseudo-labels : probas du 08 (champion seed42 sur test op03).
Usage: python scripts/threshrun.py <stage> <thf> <thl>
Stages: aug_tr | aug_te | fit2 | blend   (tag = th{thf}_{thl} sans points)
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
from src.encoding import oof_target_encode_train, fit_target_map, apply_target_map
from scripts.steprun import base_build, rank01, fit_snapshot, CACHE, SM

def tag_of(thf, thl):
    return f"th{str(thf).replace('0.','')}_{str(thl).replace('0.','')}"

def get_aug(thf, thl):
    ref = pd.read_parquet(CACHE / "ref.parquet")
    tst = pd.read_parquet(CACHE / "tst.parquet")
    champ = pd.read_csv(ROOT / "submissions/08_te_smooth30.csv")
    m = (champ.target > 0).values
    proba = champ.target.values[m]  # probas 08 sur test op03, ordre test
    pf, pl = proba > thf, proba < thl
    pm = pf | pl
    pseudo = tst.iloc[np.where(pm)[0]].copy()
    pseudo[C.TARGET] = 0.0
    pseudo.loc[pseudo.index[pf[pm]], C.TARGET] = 1.0
    print(f"pseudo: fraude={pf.sum():,} légit={pl.sum():,}")
    return pd.concat([ref, pseudo], ignore_index=True), tst, proba, m, champ

def main():
    stage, thf, thl = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tag = tag_of(thf, thl)
    t0 = time.time()
    if stage == "aug_tr_a":
        from scripts.steprun import rowbal
        from src.features.behavioral import behavioral_features
        aug, _, _, _, _ = get_aug(thf, thl)
        X = rowbal(aug).reset_index(drop=True)
        for col in [C.ORIGIN_ACCT, C.DEST_ACCT]:
            fr = aug[col].value_counts(normalize=True)
            X[f"freq_{col}"] = aug[col].map(fr).fillna(0).values
        X = pd.concat([X, behavioral_features(aug, aug).reset_index(drop=True)], axis=1)
        X.to_parquet(CACHE / f"thaug_A_{tag}.parquet")
        np.save(CACHE / f"thy_aug_{tag}.npy", aug[C.TARGET].to_numpy())
        print("aug_tr_a", X.shape)
    elif stage == "aug_tr_b1":
        from src.features.temporal import recency_features
        aug, _, _, _, _ = get_aug(thf, thl)
        X = recency_features(aug, aug).reset_index(drop=True)
        X.to_parquet(CACHE / f"thaug_B1_{tag}.parquet")
        print("aug_tr_b1", X.shape)
    elif stage == "aug_tr_b2":
        # chunké : arg4 = index de tranche (0..3), tranches de 110k lignes
        from src.encoding import recent_target_rate
        k = int(sys.argv[4])
        slim_p = CACHE / f"thaug_slim_{tag}.parquet"
        if slim_p.exists():
            aug = pd.read_parquet(slim_p)
        else:
            aug, _, _, _, _ = get_aug(thf, thl)
            aug = aug[[C.ORIGIN_ACCT, C.PERIOD, C.TARGET]].copy()
            aug.to_parquet(slim_p)
        lo, hi = k * 110000, min((k + 1) * 110000, len(aug))
        X = recent_target_rate(aug.iloc[lo:hi], aug, C.ORIGIN_ACCT, C.PERIOD, C.TARGET, (5, 10, 20)).reset_index(drop=True)
        X.to_parquet(CACHE / f"thaug_B2_{tag}_part{k}.parquet")
        print(f"aug_tr_b2 part{k} [{lo}:{hi}]", X.shape)
    elif stage == "aug_tr_b2merge":
        parts = sorted(CACHE.glob(f"thaug_B2_{tag}_part*.parquet"))
        X = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        X.to_parquet(CACHE / f"thaug_B2_{tag}.parquet")
        print("aug_tr_b2 merged", X.shape, len(parts), "parts")
    elif stage == "aug_tr_c":
        aug, _, _, _, _ = get_aug(thf, thl)
        A = pd.read_parquet(CACHE / f"thaug_A_{tag}.parquet")
        B1 = pd.read_parquet(CACHE / f"thaug_B1_{tag}.parquet")
        B2 = pd.read_parquet(CACHE / f"thaug_B2_{tag}.parquet")
        X = pd.concat([A, B1, B2], axis=1)
        X["te_origin"] = oof_target_encode_train(aug, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
        X.to_parquet(CACHE / f"thaug_X_tr_{tag}.parquet")
        print("aug_tr_c", X.shape)
    elif stage == "aug_te":
        aug, tst, _, _, _ = get_aug(thf, thl)
        X = base_build(tst, aug)
        mp, gm = fit_target_map(aug, C.ORIGIN_ACCT, C.TARGET, smoothing=SM)
        X["te_origin"] = apply_target_map(tst, C.ORIGIN_ACCT, mp, gm)
        X.to_parquet(CACHE / f"thaug_X_te_{tag}.parquet")
        print("aug_te", X.shape)
    elif stage == "fit2":
        X = pd.read_parquet(CACHE / f"thaug_X_tr_{tag}.parquet")
        y = np.load(CACHE / f"thy_aug_{tag}.npy")
        m2 = fit_snapshot(42, X, y, f"thfit2_{tag}")
        Xte = pd.read_parquet(CACHE / f"thaug_X_te_{tag}.parquet")
        np.save(CACHE / f"thpseudo_proba_{tag}.npy", m2.predict_proba(Xte)[:, 1])
        print("pseudo_proba sauvé")
    elif stage == "blend":
        p2 = np.load(CACHE / f"thpseudo_proba_{tag}.npy")
        champ = pd.read_csv(ROOT / "submissions/08_te_smooth30.csv")
        m = (champ.target > 0).values
        r08 = rank01(champ.target.values[m])
        blend = 0.5 * r08 + 0.5 * rank01(p2)
        full = np.zeros(len(champ)); full[m] = blend
        path = make_submission(champ["id"], full, f"40_08xp{tag}_w50")
        ref28 = pd.read_csv(ROOT / "submissions/28_rank_w50.csv")
        r28 = rank01(ref28.target.values[m])
        print(f"OK → {path} | corr vs 28champion = {np.corrcoef(blend, r28)[0,1]:.4f}")
    print(f"[{stage} {tag}] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
