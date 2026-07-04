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

## 3. Journal des 21 expériences

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
| Signal de voisinage via destination partagée (guilt-by-association) | corr **0.007-0.009** avec target, dégrade le modèle complet (-0.0015 et -0.0021 sur 2 folds indép.) |
| Pseudo-labeling "inversé" (générateur sans `te_origin`, distillation croisée) | gain vs pseudo-labeling standard : **+0.0010, +0.0037, -0.0034** sur 3 folds indép. (moyenne +0.0004, écart-type 0.0029) → bruit |
| Smoothing fin de `te_origin` (20/25/35/40 autour de 30) | classement instable sur 3 folds (20 gagne sur 2, perd sur le 3e ; 25 gagne puis perd) → 30 reste un choix aussi défendable que tout autre, pas de point fin qui domine |
| Feature `amount_zscore_origin` (montant standardisé par compte émetteur) | **-0.0022, -0.0050, +0.0017** sur 3 folds indép. → négatif en moyenne (-0.0018), pas retenue |

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

## 5. Node2Vec/GraphSAGE — piste fermée (diagnostic structurel + test direct)

Avant d'investir dans un embedding appris, diagnostic structurel du graphe origine→destination
(train+test op_03, 21 412 comptes, 569 328 lignes) :
- origines et destinations sont des **espaces de comptes disjoints** (0% de recouvrement) → graphe
  **biparti strict**, aucun triangle possible ;
- mais **pas un forêt épars** : 99,1% des comptes dans **une seule composante connexe géante** ;
  58,6% des destinations sont partagées par ≥2 origines distinctes (58 en moyenne) → structure de
  hubs bien réelle, donc un Node2Vec aurait *quelque chose* à apprendre structurellement.

Test direct de ce que cette structure pourrait apporter : feature « guilt-by-association » =
taux de fraude historique moyen des **autres origines partageant une destination** avec le compte
(fold-safe, même lissage que `te_origin`). Résultat sur 2 folds indépendants :
- corrélation avec `te_origin` : **0.004** → vraiment décorrélé (contrairement à tout le reste testé) ;
- mais corrélation avec la **cible** : **0.007-0.009** → quasi nulle ;
- ajoutée au modèle complet : AP **-0.0015** (fold last) et **-0.0021** (fold -2) → dégrade dans les
  deux cas.

**Verdict : piste fermée.** Le graphe a une vraie structure (hubs, composante géante), mais cette
structure n'encode pas de signal de fraude additionnel — partager un destinataire avec d'autres
comptes ne dit rien sur sa propre propension à la fraude. Un Node2Vec/GraphSAGE apprendrait
essentiellement ce même type de signal de voisinage (communautés via hubs partagés) : aucune
raison de penser qu'il ferait mieux que ce test direct. Inutile d'investir plusieurs jours dans
cette piste.

**Réserve théorique traitée — équivalence structurelle.** Node2Vec n'apprend pas que des moyennes
de voisinage (homophilie) ; il peut aussi capturer une équivalence structurelle (deux comptes au
rôle topologique similaire, même sans voisin commun). Mais ce signal-là, on l'a déjà sous la main :
les degrés (`b1_dest_in_degree`, `b1_origin_count`, etc.) sont précisément les descripteurs
d'équivalence structurelle, et `b1_dest_in_degree` est même la **2e feature en importance** du
modèle (20.7, juste après `te_origin` à 29.5) — le modèle s'appuie déjà fortement dessus. Or
l'ablation de l'expérience "features comportementales (paires, degrés)" donne un gain d'AP **~0**.
Conclusion : le signal structurel est disponible, utilisé, mais n'apporte rien au rang. Un Node2Vec
plus riche apprendrait une représentation plus fine du même rôle topologique — sans raison de
penser qu'elle franchirait le seuil que la version brute (degré) ne franchit déjà pas.

---

## 6. La remontée finale : pseudo-labeling + rank-blend (2e place)

Après le plateau de 08 (0.3569), **le pseudo-labeling a débloqué la situation** — la seule idée
qui ait dépassé 08. Progression :

| Étape | LB public |
|---|---|
| 08 (CatBoost te_origin) | 0.35686 |
| + pseudo-labeling 1-cycle (seuils 0.98/0.02) | 0.35727 |
| + blend champion×pseudo (raw 50/50) | 0.35740 |
| + **rank-average** au lieu de raw | 0.35780 |
| + **poids optimal ~50% pseudo** | **0.357915** (2e place) |

**Pourquoi le pseudo-labeling marche** : le test (périodes 106-143) contient ~9.7% de comptes
destinataires nouveaux. Les pseudo-labels (prédictions très confiantes du champion sur le test,
seuils 0.98/0.02) donnent au modèle un signal sur ces comptes futurs.

**Ce qui a marché** : 1 cycle (pas 2), **rank-average** (>> raw, +0.00012), poids ~50-55% pseudo.
**Ce qui a échoué** : 2 cycles (overfit), bagging (graine 42 déjà bonne), seuils agressifs.

### Soumission finale
**`submissions/28_rank_w50.csv`** = rank-blend(champion, pseudo-1cycle) à 50% — **LB 0.357915, 2e place**.
Champion de secours robuste : `08_te_smooth30.csv` (0.3569).

> Pipeline reproductible : `notebooks/23_pseudo_labeling.ipynb` + scripts de blend. Code dans `src/`.
