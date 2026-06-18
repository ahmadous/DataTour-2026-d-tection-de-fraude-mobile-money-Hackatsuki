# FINDINGS — Découvertes EDA (à lire AVANT de coder)

> Source : analyse exploratoire (rapport Opus, 1.29 M train / 430 K test, taux de fraude 10 %).
> Ces findings **priment sur la stratégie générique**. Plusieurs idées « évidentes » sont des pièges.

---

## 🎯 Les 5 faits qui décident de tout

### 1. 100 % de la fraude est dans `op_03`
- `op_01, op_02, op_04, op_05` → taux de fraude **exactement 0**.
- `op_03` → 32 % des transactions, **31,2 % de fraude interne**, **100 % du risque**.
- **Action** : prédire `target = 0` hors `op_03`, modéliser uniquement les ~415 K `op_03`.
  Helper : `utils.op03_mask()` + `utils.assemble_full_proba()`.
- **Conséquence métrique** : « op_03 = risqué » donne déjà **AP ≈ 0,31 gratuitement**.
  Le top leaderboard est ~0,82. **Tout se joue dans le classement INTERNE à op_03.**

### 2. Le graphe est BIPARTI (pas de cycles, pas de mules)
- Émetteurs (`acc_o_`) et destinataires (`acc_d_`) sont deux univers disjoints.
- ❌ PageRank, Louvain, GNN, détection de cycles → **non justifiés**, ne pas y passer de temps.
- ✅ Seules survivent les **features de degré bipartite** (collecteur, fan-out) → `src/features/graph.py`.

### 3. Validation TEMPORELLE obligatoire
- Train = périodes 0–105, Test = **106–143, strictement dans le futur**.
- Taux de fraude variable selon la période (**4 % → 18 %**).
- ❌ CV aléatoire → score optimiste et **trompeur**.
- ✅ `validation.time_folds` (fenêtre expansive sur `period`) = schéma de référence.

### 4. Les comptes sont MIXTES → l'ID est un piège
- **96 %** des transactions viennent de comptes faisant **à la fois** fraude ET légitime.
- ❌ Target/Frequency encoding sur l'**ID de compte brut** → CV flatteuse, **ne généralise pas**.
  C'est exactement le surapprentissage des identifiants contre lequel la brief met en garde.
- ✅ Encodages **fold-by-fold sur le passé** uniquement → `src/encoding.py`.

### 5. Soldes bruités globalement, MAIS l'incohérence est le top signal DANS op_03
- Soldes négatifs (~2,5 %) = artefact d'anonymisation, pas un bug.
- ⚠️ **Correction (notebook 00, vérifié sur les données)** : sur *tout* le dataset, l'arithmétique
  est fausse ~40 % du temps et corrèle +0,04 (dilué par les opérations sans fraude). **Mais dans
  op_03**, seules ~4 % des tx sont incohérentes et elles sont **frauduleuses à ~56 %** (vs ~30 %
  en moyenne) → Spearman ≈ **0,11**, le **signal univarié le plus fort**.
- ✅ On garde `balance_features` (résidu signé + flag, émetteur ET destinataire). Voir
  `src/features/temporal.py`. Leçon : toujours mesurer dans le bon périmètre (op_03), pas en global.

---

## 📉 Pourquoi c'est dur

Dans `op_03`, fraude et légitime se ressemblent énormément. **Aucune variable seule** ne sépare
(toutes les corrélations |Spearman| < 0,1). Le montant des fraudes est normal vs l'historique du
compte (médiane montant/moyenne compte : 0,70 fraude vs 0,68 légitime).

→ Le signal est **multivarié** : agrégations comportementales fines captées par gradient boosting.

---

## ✅ Plan d'attaque dérivé (ce qui paye)

1. **Restreindre à op_03** (gratuit, correct).
2. **Features comportementales / temporelles** (pas brutes) :
   régularité temporelle, dynamique récente, comportement par couple émetteur-destinataire,
   degrés bipartites légers.
3. **Validation temporelle sans fuite** + encodages calculés sur le passé.
4. **Gradient boosting** (CatBoost réf., LightGBM, XGBoost) + **calibration isotonic**.
5. **Ensemble diversifié**, poids optimisés sur l'out-of-fold.

> L'avantage compétitif vient des **features + validation + calibration**, bien plus que du choix du modèle.

---

## 🗂️ Colonnes (réelles)

`id, period, operation, amount, origin_account, origin_balance_before, origin_balance_after,
destination_account, destination_balance_before, destination_balance_after, fraud_flag`

Soumission : `id, target` (probabilité ∈ [0, 1]).
Constantes centralisées dans `src/config.py`.
