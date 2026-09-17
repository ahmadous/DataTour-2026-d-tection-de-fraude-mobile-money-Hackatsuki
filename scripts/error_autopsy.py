from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C
from src.encoding import (
    apply_target_map,
    fit_target_map,
    recent_target_rate,
)
from src.features.behavioral import behavioral_features
from src.features.temporal import balance_features, recency_features
from src.utils import DATA, op03_mask, seed_everything
from src.validation import evaluate_ap, time_folds


EPS = 1e-6
WINDOWS = (5, 10, 20)
SMOOTHING = 30


def row_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["amount_log1p"] = np.log1p(np.maximum(df[C.AMOUNT], 0))
    f["amount_vs_origin_before"] = df[C.AMOUNT] / (np.abs(df[C.ORIGIN_BAL_BEFORE]) + EPS)
    f["amount_vs_dest_before"] = df[C.AMOUNT] / (np.abs(df[C.DEST_BAL_BEFORE]) + EPS)
    f["origin_balance_before"] = df[C.ORIGIN_BAL_BEFORE]
    f["dest_balance_before"] = df[C.DEST_BAL_BEFORE]
    return pd.concat([f, balance_features(df)], axis=1)


def add_freq(X: pd.DataFrame, src_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in [C.ORIGIN_ACCT, C.DEST_ACCT]:
        freq = ref_df[col].value_counts(normalize=True)
        X[f"freq_{col}"] = src_df[col].map(freq).fillna(0).values
    return X


def build_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    X = row_features(df).reset_index(drop=True)
    X = add_freq(X, df.reset_index(drop=True), ref_df)
    mp, gm = fit_target_map(ref_df, C.ORIGIN_ACCT, C.TARGET, smoothing=SMOOTHING)
    X["te_origin"] = apply_target_map(df, C.ORIGIN_ACCT, mp, gm)
    return X


def build_features_apply(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    X = row_features(df).reset_index(drop=True)
    X = add_freq(X, df.reset_index(drop=True), ref_df)
    mp, gm = fit_target_map(ref_df, C.ORIGIN_ACCT, C.TARGET, smoothing=SMOOTHING)
    X["te_origin"] = apply_target_map(df, C.ORIGIN_ACCT, mp, gm)
    return X


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_depth=6,
        max_iter=150,
        random_state=42,
    )


def qbin(s: pd.Series, q: int = 5) -> pd.Series:
    try:
        return pd.qcut(s, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(s), index=s.index)


def cbin(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=False)


def add_diagnostic_columns(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["score"] = score
    out["period_bin"] = qbin(out[C.PERIOD], 5)
    out["amount_bin"] = qbin(out[C.AMOUNT], 5)
    out["te_origin_bin"] = qbin(out["te_origin"], 5)
    out["origin_recent_5_bin"] = cbin(
        out["b2_origin_recent_count_5"], [-1, 0, 1, 2, 4, np.inf], ["0", "1", "2", "3-4", "5+"]
    )
    out["dest_in_deg_bin"] = cbin(
        out["b1_dest_in_degree"], [-1, 0, 1, 2, 4, 8, np.inf], ["0", "1", "2", "3-4", "5-8", "9+"]
    )
    out["pair_count_bin"] = cbin(
        out["b1_pair_count"], [-1, 0, 1, 2, 4, 8, np.inf], ["0", "1", "2", "3-4", "5-8", "9+"]
    )
    out["origin_out_deg_bin"] = cbin(
        out["b1_origin_out_degree"], [-1, 0, 1, 2, 4, 8, np.inf], ["0", "1", "2", "3-4", "5-8", "9+"]
    )
    out["origin_inconsistent"] = out["b2_origin_balance_inconsistent"].astype(int)
    out["dest_inconsistent"] = out["b2_dest_balance_inconsistent"].astype(int)
    out["pair_new"] = out["b1_pair_is_new"].astype(int)
    out["origin_emptied"] = out["b2_origin_emptied"].astype(int)
    out["dest_not_credited"] = out["b2_dest_not_credited"].astype(int)
    return out


def slice_summary(df: pd.DataFrame, group_col: str, fraud_low_thr: float, legit_high_thr: float) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        y = g[C.TARGET].to_numpy()
        s = g["score"].to_numpy()
        fraud = y == 1
        legit = y == 0
        rows.append(
            {
                "slice": str(key),
                "n": len(g),
                "fraud_rate": float(y.mean()),
                "mean_score": float(s.mean()),
                "mean_score_fraud": float(s[fraud].mean()) if fraud.any() else np.nan,
                "hard_fraud_rate": float((s[fraud] <= fraud_low_thr).mean()) if fraud.any() else np.nan,
                "hard_legit_rate": float((s[legit] >= legit_high_thr).mean()) if legit.any() else np.nan,
                "lift": float(y.mean() / df[C.TARGET].mean()) if df[C.TARGET].mean() > 0 else np.nan,
            }
        )
    res = pd.DataFrame(rows)
    return res.sort_values(["hard_fraud_rate", "fraud_rate"], ascending=[False, False])


def numeric_slice_report(df: pd.DataFrame, col: str, bins: list[float], labels: list[str], fraud_low_thr: float, legit_high_thr: float) -> pd.DataFrame:
    temp = df.copy()
    temp["bucket"] = cbin(temp[col], bins, labels)
    return slice_summary(temp, "bucket", fraud_low_thr, legit_high_thr)


def frame_block(df: pd.DataFrame, title: str | None = None) -> str:
    text = df.to_string(index=False)
    if title:
        return f"{title}\n{text}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/error_autopsy.md"))
    parser.add_argument("--sample-n", type=int, default=120_000)
    args = parser.parse_args()

    seed_everything(42)
    train = pd.read_csv(DATA / "train.csv")
    op03 = op03_mask(train).to_numpy()
    y_all = train[C.TARGET].to_numpy()
    folds_full = list(time_folds(train[C.PERIOD]))

    if args.sample_n and args.sample_n < op03.sum():
        op03_idx = np.where(op03)[0]
        sample_idx = np.random.RandomState(42).choice(op03_idx, size=args.sample_n, replace=False)
        sample_mask = np.zeros(len(train), dtype=bool)
        sample_mask[sample_idx] = True
        op03 = sample_mask
        y_all = train[C.TARGET].to_numpy()
        folds_full = list(time_folds(train.loc[op03, C.PERIOD]))
        train = train.loc[op03].reset_index(drop=True)
        op03 = np.ones(len(train), dtype=bool)
        y_all = train[C.TARGET].to_numpy()
        folds_full = list(time_folds(train[C.PERIOD]))
        print(f"Using sample of {len(train):,} op_03 rows for fast autopsy")
    else:
        train = train.loc[op03].reset_index(drop=True)
        op03 = np.ones(len(train), dtype=bool)
        y_all = train[C.TARGET].to_numpy()
        folds_full = list(time_folds(train[C.PERIOD]))
        print(f"Using full op_03 set: {len(train):,} rows")

    oof = np.zeros(len(train))
    per_fold = []
    last_model = None
    last_cols = None

    print(f"op_03 rows: {int(op03.sum()):,} / {len(train):,}")
    for k, (tr_idx, va_idx) in enumerate(folds_full):
        tr_op = tr_idx[op03[tr_idx]]
        va_op = va_idx[op03[va_idx]]
        ref = train.iloc[tr_op]
        Xtr = build_features(train.iloc[tr_op], ref)
        Xva = build_features_apply(train.iloc[va_op], ref)
        model = make_model()
        model.fit(Xtr, y_all[tr_op])
        pred = model.predict_proba(Xva)[:, 1]
        oof[va_op] = pred
        score = evaluate_ap(y_all[va_op], pred)
        per_fold.append(score)
        last_model = model
        last_cols = list(Xtr.columns)
        print(f"fold {k}: AP={score:.4f}")

    global_ap = evaluate_ap(y_all[op03], oof[op03])
    recent2 = float(np.mean(per_fold[-2:]))
    last = float(per_fold[-1])
    print(f"global AP: {global_ap:.4f} | recent2: {recent2:.4f} | last: {last:.4f}")

    base = train.loc[op03].reset_index(drop=True).copy()
    base["score"] = oof[op03]
    feats = build_features_apply(base, base)
    df = pd.concat([base, feats], axis=1)

    diag_ref = train.loc[op03].reset_index(drop=True)
    diag = pd.concat(
        [
            behavioral_features(base, diag_ref).reset_index(drop=True),
            recency_features(base, diag_ref).reset_index(drop=True),
            recent_target_rate(
                base, diag_ref, C.ORIGIN_ACCT, C.PERIOD, C.TARGET, WINDOWS
            ).reset_index(drop=True),
        ],
        axis=1,
    )
    df = pd.concat([df, diag], axis=1)
    df = add_diagnostic_columns(df, df["score"].to_numpy())

    fraud = df[C.TARGET].to_numpy() == 1
    legit = ~fraud
    fraud_low_thr = float(np.quantile(df.loc[fraud, "score"], 0.2))
    legit_high_thr = float(np.quantile(df.loc[legit, "score"], 0.8))

    top_error_cols = [
        "period_bin",
        "amount_bin",
        "te_origin_bin",
        "origin_recent_5_bin",
        "dest_in_deg_bin",
        "pair_count_bin",
        "origin_out_deg_bin",
        "origin_inconsistent",
        "dest_inconsistent",
        "pair_new",
        "origin_emptied",
        "dest_not_credited",
    ]

    slices = []
    for col in top_error_cols:
        if df[col].nunique(dropna=False) <= 1:
            continue
        summ = slice_summary(df, col, fraud_low_thr, legit_high_thr)
        summ.insert(0, "feature", col)
        slices.append(summ.head(8))
    all_slices = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()

    report = []
    report.append("# Error Autopsy")
    report.append("")
    report.append("## Model")
    report.append("- CatBoost op_03 only, fold-safe features, OOF target encoding on `te_origin`.")
    report.append(f"- OOF AP global: {global_ap:.4f}")
    report.append(f"- OOF AP recent2: {recent2:.4f}")
    report.append(f"- OOF AP last: {last:.4f}")
    report.append("")
    report.append("## Score thresholds")
    report.append(f"- Fraud low-score threshold (20th pct among fraud): {fraud_low_thr:.4f}")
    report.append(f"- Legit high-score threshold (80th pct among legit): {legit_high_thr:.4f}")
    report.append("")
    report.append("## Hardest slices")
    if not all_slices.empty:
        report.append(
            frame_block(
                all_slices.sort_values(["hard_fraud_rate", "fraud_rate"], ascending=False).head(30),
                "feature slices",
            )
        )
    else:
        report.append("_No slices computed_")
    report.append("")
    report.append("## Top feature signals in the last fold")
    if last_model is not None and last_cols is not None:
        if hasattr(last_model, "feature_importances_"):
            imp = pd.Series(last_model.feature_importances_, index=last_cols).sort_values(ascending=False)
            report.append(frame_block(imp.head(15).to_frame("importance"), "importance"))
    report.append("")
    report.append("## Slice tables")

    for name, col, bins, labels in [
        ("amount", C.AMOUNT, [0, 10_000, 20_000, 50_000, 100_000, np.inf], ["0-10k", "10-20k", "20-50k", "50-100k", "100k+"]),
        ("period", C.PERIOD, [0, 20, 40, 60, 80, np.inf], ["0-20", "20-40", "40-60", "60-80", "80+"]),
        ("te_origin", "te_origin", [0, 0.05, 0.1, 0.2, 0.4, 1.0], ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", "0.4-1"]),
    ]:
        report.append(f"### {name}")
        report.append(frame_block(numeric_slice_report(df, col, bins, labels, fraud_low_thr, legit_high_thr).head(6)))
        report.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
