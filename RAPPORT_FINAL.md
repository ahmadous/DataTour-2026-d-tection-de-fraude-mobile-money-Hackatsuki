# Rapport final — DataTour 2026, détection de fraude mobile money

**Équipe Hackatsuki · 3e place, LB public 0.35686** (1er à 0.35791, à 0.001).
Métrique : Average Precision (PR-AUC). Classement final sur le **privé (100% du test)**.

---

## 1. La recette gagnante (modèle 08, LB 0.3569)

Quatre décisions, dans l'ordre d'impact :

| # | Décision | Gain | Pourquoi |
|---|---|---|---|
| 1 | **Pas de calibration** sur la soumission | **+0.013** | L'AP est une métrique de RANG ; l'isotonic crée des ex-æquo qui la cassent |
| 2 | **`te_origin`** = taux de fraude historique de l'émetteur, **fold-safe** (OOF imbriqué) | **+0.0067** | Le seul vrai signal ; 33% de l'importance |
| 3 | **Restriction à `op_03`** + proba 0 ailleurs | gratuit | 100% de la fraude y est |
| 4 | **Validation temporelle** (`time_folds`) | fiabilité | Test dans le futur ; la CV aléatoire ment |

Modèle : **CatBoost** (depth 6, lr 0.05, 600 itér, graine 42), features = montant/ratios + incohérences
de solde (fingerprints PaySim) + fréquences + degrés bipartites + dynamique récente + `te_origin`.

---

## 2. La boussole CV↔LB (notre outil décisif)

Pour un modèle **non calibré** : **LB ≈ CV(last fold) − 0.004**. Vérifié :

| Soumission | CV last | LB réel |
|---|---|---|
| 01 (calibré) | 0.3483 | 0.3332 |
| 05 (calibré) | 0.3571 | 0.3399 |
| **08 (non calibré)** | 0.3607 | **0.3569** (écart 0.0038) |

Conséquence : on a pu **itérer en local et ne soumettre que du gagnant**. Règle d'or : **croire la CV
(recent2), pas le public** (30%, bruité).

---

## 3. Journal des 17 expériences

**Ce qui a marché :** `te_origin` fold-safe (+0.0067), suppression calibration (+0.013).

**Ce qui n'a RIEN donné (mesuré, pas supposé) :**

| Piste | Verdict |
|---|---|
| Incohérence de solde / fingerprints PaySim | +0.0017 (marginal) |
| Features comportementales (paires, degrés) | ~0 |
| Dynamique récente / taux récent | +0.0026 |
| TE destinataire / TE couple | inutile (destinataires aussi mixtes à 96.7%) |
| Features de rythme (hypothèse délai/loi géométrique) | négatif |
| Balayage du lissage (3/10/30) | 30 optimal, le reste pire |
| Seed-bagging, +itérations | nul/négatif (08 = bonne graine) |
| LightGBM, XGBoost, ensemble décorrélé | corrélés (~0.93), n'aident pas |
| MLP + blend (cat+NN) | CV +0.001 (bruit) ; LB **pire** (0.3551) |
| Modèle graphe-XGBoost (degrés + voisinage) | corr 0.929, blend = cat |

---

## 4. Pourquoi 0.357 est un plafond structurel

`te_origin` (la propension de fraude du compte) est **quasiment tout le signal exploitable**. D'où :
- tout modèle **fort** s'appuie dessus → corrélation ~0.93 → **le blend n'aide pas** ;
- tout modèle qui **l'ignore** est trop faible → tire le blend vers le bas.

Il n'existe donc pas de modèle « fort ET décorrélé » : le signal est **unidimensionnel**. C'est
pourquoi **tout le leaderboard est collé entre 0.353 et 0.358**, et l'écart en tête (0.001) est
essentiellement du **bruit** (variance de seed / des 30% publics). L'EDA l'avait annoncé :
dans op_03, fraude et légitime se ressemblent (toutes corrélations |ρ| < 0.11).

---

## 5. Le seul levier restant pour le 1er

Une **vraie diversité de modèle** d'un coéquipier — typiquement les **embeddings de graphe**
(Node2Vec/GraphSAGE) de Cissokho, en version **corrigée** (sans fuite, temporelle, AP, op_03 :
voir `CORRECTION_CISSOKHO.md` + `notebooks/13`). À blender avec le CatBoost via leurs OOF.
⚠️ Bridé par le graphe **biparti sans cycle** : faible probabilité, mais seule piste non épuisée.

---

## 6. Soumission finale recommandée

**`submissions/08_te_smooth30.csv`** (CatBoost, te_origin, non calibré) — meilleur sur le public
ET sur recent2 (proxy privé). À **sélectionner comme soumission finale** pour le scoring privé.

> Reproductible via `notebooks/08_te_smoothing.ipynb`. Tout le code réutilisable est dans `src/`.
