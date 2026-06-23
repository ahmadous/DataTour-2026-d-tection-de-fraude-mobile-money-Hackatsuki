# Correction des notebooks de Cissokho (D2026 + fast-testing)

Salut Cissokho 👋 Tes intuitions sont bonnes (historique de fraude des comptes, séquence,
timing « géométrique », angle graphe). Le problème n'est pas l'idée mais l'**implémentation** :
il y a de la **fuite de données** qui gonfle artificiellement ton score. Voici quoi corriger
pour que ton modèle soit **blendable** avec notre CatBoost (qui est à **LB 0.3569, 3e place**).

> Objectif : ton modèle graphe est d'une **famille différente** du nôtre → s'il est propre,
> son blend avec le nôtre peut nous faire passer **1er** (le top n'est qu'à 0.3579).

---

## 🔴 1. Fuite de données à supprimer (le bug principal — D2026)

Ces features utilisent `fraud_flag` sur **tout le dataset** et le remettent sur chaque ligne du
compte. Le modèle « voit » donc le label (présent ET futur). **À supprimer ou recoder fold-safe :**

- `origin_time_to_first_fraud`, `origin_num_txns_to_first_fraud` (+ destination)
- `p_geometric_origin_period`, `p_geometric_*` (dérivées des précédentes → fuite héritée)
- `origin_account_previously_fraud`, `destination_account_previously_fraud`
  (`isin(comptes frauduleux de tout df)` = encode le label, y compris la ligne courante)

**Symptôme** : avec ces features + un `train_test_split` aléatoire, ton AP paraît énorme
(0.6–0.9) mais s'effondre sur le vrai test (LB réel du peloton ≈ 0.353–0.358).

---

## 🟠 2. Les 3 règles non négociables (validées par nos 12 expériences)

1. **Validation TEMPORELLE, pas aléatoire.** Le test est dans le futur (périodes 106–143 vs
   train 0–105). `train_test_split(stratify=y)` ment. Utilise un split par période :
   entraîne sur périodes anciennes, valide sur récentes.
2. **Métrique = Average Precision (PR-AUC)**, PAS ROC AUC. (`average_precision_score`)
3. **Restreins à `op_03`** : 100 % de la fraude y est. Prédis 0 ailleurs.
4. **Pas de calibration** sur la soumission ! (l'isotonic nous a coûté **−0.013** d'AP : c'est
   une métrique de rang, la calibration crée des ex-æquo qui la cassent.)

---

## ✅ 3. La bonne version (fold-safe) de ton idée « previously_fraud »

Ton intention « ce compte a-t-il un historique de fraude » est EXACTE — c'est notre feature la
plus forte. Mais il faut l'apprendre **uniquement sur le passé**. Version correcte (standalone) :

```python
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

TARGET = "fraud_flag"

# ---- Target encoding du compte émetteur, SANS FUITE (OOF imbriqué pour le train) ----
def fit_target_map(ref, col, smoothing=30):
    gm = ref[TARGET].mean()
    agg = ref.groupby(col)[TARGET].agg(["mean", "count"])
    smooth = (agg["mean"]*agg["count"] + gm*smoothing) / (agg["count"] + smoothing)
    return smooth.to_dict(), gm

def oof_target_encode(train_df, col, smoothing=30, n_splits=5, seed=42):
    from sklearn.model_selection import KFold
    out = np.zeros(len(train_df)); vals = train_df[col].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(train_df):
        mp, gm = fit_target_map(train_df.iloc[tr], col, smoothing)
        out[va] = pd.Series(vals[va]).map(mp).fillna(gm).to_numpy()
    return out
# train : te = oof_target_encode(train_op03, "origin_account")
# test  : mp, gm = fit_target_map(train_op03, "origin_account"); te = test["origin_account"].map(mp).fillna(gm)
```

C'est `te_origin` : il vaut **+0.0067** sur le LB chez nous (le seul vrai levier). Tes features
de séquence/timing peuvent rester, mais **calculées sur le passé strict** (`groupby(...).cumcount()`
en triant par `period`, et `period.diff()` — pas de référence aux labels).

---

## 🐛 4. Bug fast-testing (GraphSAGE)

`F.binary_cross_entropy(pred, y, pos_weight=...)` → `binary_cross_entropy` n'accepte PAS
`pos_weight`. Utilise `F.binary_cross_entropy_with_logits(logits, y, pos_weight=...)`
(et enlève le `Sigmoid` final du modèle, il est inclus dans `_with_logits`).

⚠️ Note : le graphe est **biparti** (émetteurs et destinataires disjoints, aucun cycle). Donc
Node2Vec (les marches meurent) et GraphSAGE « identité de compte » sont fragiles. Le plus solide
chez toi = **Test 1 (degrés in/out + XGBoost)**. Garde ça, valide-le temporellement.

---

## 📦 5. Ce qu'on a besoin que tu nous livres (pour blender)

Avec ton modèle graphe corrigé (XGBoost sur degrés + embeddings, op_03, CV temporelle, AP) :

1. **`oof_cissokho.csv`** : colonnes `id, oof` = tes prédictions out-of-fold sur le **train op_03**
   (probabilité, non calibrée). → nous permet d'**optimiser le poids du blend** de façon fiable.
2. **`test_cissokho.csv`** : colonnes `id, target` = tes prédictions sur le **test** (op_03 ; 0 ailleurs).

Avec ça, on fait `rank_average(notre_catboost, ton_graphe)` et on mesure le gain en CV avant de
soumettre. Si ton modèle est assez décorrélé (même un peu plus faible), **le blend nous met en tête**.

---

### Récap rapide
- ❌ supprime les features dérivées de `fraud_flag` (fuite)
- ✅ validation temporelle + AP + op_03 + pas de calibration
- ✅ recode l'historique compte en **target encoding fold-safe**
- 🐛 corrige `binary_cross_entropy_with_logits`
- 📦 livre **OOF (train) + prédictions test** en CSV `id`-alignés
