import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from src import config as C
from src.validation import time_folds, evaluate_ap
from src.utils import op03_mask, seed_everything
from src.features.temporal import balance_features, recency_features
from src.features.behavioral import behavioral_features
from src.encoding import oof_target_encode_train, fit_target_map, apply_target_map, recent_target_rate
from catboost import CatBoostClassifier
seed_everything(42)
train = pd.read_csv('data/train.csv')
op03 = op03_mask(train).to_numpy(); y_all = train[C.TARGET].to_numpy()
folds_full = list(time_folds(train[C.PERIOD]))
EPS=1e-6; WINDOWS=(5,10,20); SM=30

def bb(df, r):
    f=pd.DataFrame(index=df.index)
    f['amount_log1p']=np.log1p(np.maximum(df[C.AMOUNT],0))
    f['amount_vs_origin_before']=df[C.AMOUNT]/(np.abs(df[C.ORIGIN_BAL_BEFORE])+EPS)
    f['amount_vs_dest_before']=df[C.AMOUNT]/(np.abs(df[C.DEST_BAL_BEFORE])+EPS)
    f['origin_balance_before']=df[C.ORIGIN_BAL_BEFORE]; f['dest_balance_before']=df[C.DEST_BAL_BEFORE]
    X=pd.concat([f,balance_features(df)],axis=1).reset_index(drop=True)
    for col in [C.ORIGIN_ACCT,C.DEST_ACCT]:
        freq=r[col].value_counts(normalize=True)
        X[f'freq_{col}']=df[col].map(freq).fillna(0).values
    return pd.concat([X,behavioral_features(df,r).reset_index(drop=True),recency_features(df,r).reset_index(drop=True),
                      recent_target_rate(df,r,C.ORIGIN_ACCT,C.PERIOD,C.TARGET,WINDOWS).reset_index(drop=True)],axis=1)

cat=lambda:CatBoostClassifier(loss_function='Logloss',eval_metric='PRAUC',depth=6,learning_rate=0.05,iterations=600,random_seed=42,verbose=False)

def run(add_dest, label):
    pf=[]
    for tr_idx, va_idx in folds_full:
        trop=tr_idx[op03[tr_idx]]; vaop=va_idx[op03[va_idx]]; ref=train.iloc[trop]
        Xt=bb(train.iloc[trop],ref); Xv=bb(train.iloc[vaop],ref)
        Xt['te_origin']=oof_target_encode_train(ref,C.ORIGIN_ACCT,C.TARGET,smoothing=SM)
        mp,gm=fit_target_map(ref,C.ORIGIN_ACCT,C.TARGET,smoothing=SM)
        Xv['te_origin']=train.iloc[vaop][C.ORIGIN_ACCT].map(mp).fillna(gm).values
        if add_dest:
            Xt['te_dest']=oof_target_encode_train(ref,C.DEST_ACCT,C.TARGET,smoothing=SM)
            mpd,gmd=fit_target_map(ref,C.DEST_ACCT,C.TARGET,smoothing=SM)
            Xv['te_dest']=train.iloc[vaop][C.DEST_ACCT].map(mpd).fillna(gmd).values
        m=cat().fit(Xt,y_all[trop])
        pf.append(evaluate_ap(y_all[vaop],m.predict_proba(Xv)[:,1]))
        print(f'{label} fold done: {pf[-1]:.4f}', flush=True)
    print(f'{label}: recent2={np.mean(pf[-2:]):.4f} last={pf[-1]:.4f} all={[round(x,4) for x in pf]}')
    return pf, m, Xt.columns.tolist()

pfA, mA, colsA = run(False, 'A-champion')
pfB, mB, colsB = run(True,  'B-te_dest')
print(f'DELTA recent2 = {np.mean(pfB[-2:])-np.mean(pfA[-2:]):+.4f}')
impB=pd.Series(mB.get_feature_importance(),index=colsB).sort_values(ascending=False)
print('Top10 importance B:')
print(impB.head(10).to_string())
