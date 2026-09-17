# DataTour 2026 — Détection de fraude Mobile Money

> **Équipe Hackatsuki (Sénégal) · 2ᵉ place nationale**
> Coupe d'Afrique des Nations en Science des Données (CANSD), organisée par Data Afrique Hub.
> Juin – juillet 2026 · plus de 18 nationalités en lice.

Ce dépôt contient **l'intégralité de notre travail** : les 46 notebooks d'exploration, le journal
des 64 soumissions, les pistes qui ont marché **et celles qui ont échoué**, et le code exact de la
solution finale.

---

## 🏁 Résultat

Classement définitif du Sénégal, après l'audit de reproductibilité du comité (le code de chaque
équipe a été réexécuté et le score régénéré) :

| # | Équipe | Score plateforme | Score régénéré |
|---|--------|------------------|----------------|
| 1 | DataXel | 0.358509 | 0.358509 |
| **2** | **Hackatsuki** | **0.356839** | **0.356825** |
| 3 | Data King | 0.356327 | 0.356327 |

Écart de régénération : **1,4 × 10⁻⁵**. Reproductibilité validée par le comité.

**Métrique : Average Precision** (aire sous la courbe précision-rappel). Une métrique de *rang* :
elle ne récompense que si les vraies fraudes remontent en haut du classement de risque.

---

## 🎯 Le problème

Estimer, pour chaque transaction mobile money d'un jeu anonymisé, la probabilité qu'elle soit
frauduleuse. Les classes sont fortement déséquilibrées, et la difficulté n'est pas de repérer les
catégories d'opération à risque — c'est de **distinguer les fraudes des transactions normales à
l'intérieur même du comportement à risque**.

---

## 🔑 Les cinq découvertes qui ont tout décidé

Détail complet dans **[FINDINGS.md](FINDINGS.md)**, écrit avant la première ligne de modélisation.

1. **100 % de la fraude est dans `op_03`.** Prédire 0 ailleurs est gratuit et correct. Tout se joue
   dans le classement *interne* à cette opération.
2. **Le graphe des comptes est biparti**, sans cycles ni mules. PageRank, Louvain et les GNN sont
   hors sujet ; seuls des degrés bipartites simples ont du sens.
3. **La validation doit être temporelle.** Le test est strictement postérieur au train. Une
   validation croisée aléatoire mélange passé et futur et gonfle les scores.
4. **Les comptes sont mixtes** (ils font du frauduleux *et* du légitime). Encoder l'identifiant brut
   donne une CV flatteuse qui ne généralise pas.
5. **L'incohérence d'arithmétique de solde**, inutile sur l'ensemble du jeu, devient le meilleur
   signal univarié *à l'intérieur* d'`op_03`. Toujours revérifier dans le bon périmètre.

---

## ⚙️ La solution finale

Fichier soumis : `28_rank_w75.csv`. Code exécutable : **[`livraison_hackatsuki/solution.ipynb`](livraison_hackatsuki/solution.ipynb)**.

**1. Le modèle champion** — CatBoost (depth 6, lr 0.05, 600 itérations, graine 42), entraîné
uniquement sur `op_03`. Features : montants et ratios, incohérences de solde, fréquences de comptes,
comportement par compte et par paire, dynamique récente, et surtout `te_origin` — le taux de fraude
historique de l'émetteur, encodé **fold-safe** en out-of-fold imbriqué. À lui seul, 33 % de
l'importance du modèle.

**2. Pseudo-labeling, un cycle** — les prédictions très confiantes du champion sur le test
(probabilité > 0,98 → fraude, < 0,02 → légitime) sont réinjectées dans le train, puis le modèle est
réentraîné. Motivation : environ 9,7 % des comptes destinataires du test sont nouveaux, invisibles à
l'entraînement.

**3. Blend par moyenne de rangs** — 25 % champion, 75 % modèle pseudo. L'AP étant une métrique de
rang, la moyenne de rangs bat la moyenne brute de probabilités.

**Et une décision négative qui a rapporté le plus : aucune calibration.** L'isotonic regression crée
des ex-æquo qui cassent le classement. La retirer a valu **+0,013 d'AP** — plus que n'importe quelle
feature ajoutée.

---

## 💻 Contrainte de calcul : zéro GPU

Nous n'avions pas de machine dédiée. Tout a été entraîné en **CPU, sur les quotas gratuits de Kaggle
et Google Colab**. Cette contrainte a façonné la méthode plus que n'importe quel choix théorique :

- **Des arbres, pas du deep learning.** CatBoost et LightGBM tiennent sur CPU dans le temps d'une
  session gratuite.
- **Un cache de prédictions partagé.** Les sessions expirent et le travail est perdu. Chaque sortie
  de modèle (out-of-fold et test) est donc écrite sur disque, et un notebook par soumission la
  recharge — jamais deux fois le même entraînement.
- **Le blend au niveau des fichiers.** Comme réentraîner coûtait cher, les modèles sont combinés par
  moyenne de rangs directement sur les CSV sauvegardés : coût de calcul quasi nul. C'est précisément
  cette méthode née de la contrainte qui a rapporté le plus de points en fin de course.

---

## 📉 Ce qui n'a pas marché

La partie la plus utile de ce dépôt est peut-être celle-là. Journal complet dans
**[RAPPORT_FINAL.md](RAPPORT_FINAL.md)** — chaque piste a été mesurée, pas supposée.

| Piste | Verdict |
|-------|---------|
| Target encoding du destinataire et du couple | Inutile : les destinataires sont mixtes à 96,7 % |
| Features de rythme temporel (loi géométrique) | Négatif |
| Seed-bagging, itérations supplémentaires | Nul à négatif — la graine 42 était déjà un bon tirage |
| LightGBM, XGBoost, ensemble décorrélé | Corrélés à ~0,93 au champion : le blend n'apporte rien |
| MLP et blend réseau + arbres | +0,001 en CV (bruit), pire sur le leaderboard |
| Modèle graphe (degrés + voisinage à 1 saut) | Corrélation 0,929 : redondant |
| Guilt-by-association via destinataire partagé | Corrélation 0,007 avec la cible : dégrade le modèle |
| Embeddings SVD du graphe biparti | **+0,002 en CV, −0,0015 sur le leaderboard.** Transductifs : ils aident sur une validation « train-like », pas sur le vrai futur |
| Seuils de pseudo-labeling plus stricts ou plus larges | On perd dans les deux sens ; 0,98 / 0,02 est l'optimum |

**La leçon la plus chère** : la validation croisée peut mentir dans une direction précise. Les
embeddings calculés sur train + test brillaient en CV et se sont effondrés sur le test réel.

---

## 🧭 La boussole CV ↔ leaderboard

Pour un modèle non calibré : **LB ≈ CV(dernier fold) − 0,004**. Vérifié sur l'ensemble de nos
soumissions. Cette relation nous a permis d'itérer en local et de ne soumettre que ce qui avait des
chances de gagner, plutôt que de brûler des soumissions à l'aveugle.

Règle d'or associée : **quand la CV et le leaderboard public divergent, on croit la CV.** Le public
ne couvre que 30 % du test et il est bruité.

---

## 🔁 Reproduire la soumission

```bash
git clone https://github.com/ahmadous/DataTour-2026-d-tection-de-fraude-mobile-money-Hackatsuki.git
cd DataTour-2026-d-tection-de-fraude-mobile-money-Hackatsuki

python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Placez `train.csv` et `test.csv` dans `data/` (dossier gitignored : les données de la compétition ne
sont pas redistribuées ici), puis exécutez
[`livraison_hackatsuki/solution.ipynb`](livraison_hackatsuki/solution.ipynb) de bout en bout.

Le pipeline est **déterministe** — toutes les graines sont fixées à 42, aucune donnée externe, aucun
accès réseau pendant l'exécution. Sortie : `submissions/submission.csv`.

---

## 📁 Structure du dépôt

```
FINDINGS.md              ← les découvertes EDA, écrites avant de modéliser
RAPPORT_FINAL.md         ← journal complet des expériences et des échecs
features.md              ← dictionnaire des features

src/                     ← code réutilisable
  validation.py          ← CV temporelle — source de vérité unique
  encoding.py            ← target encoding fold-safe (OOF imbriqué)
  blending.py            ← blend et moyenne de rangs
  calibration.py         ← isotonic / Platt (finalement écartée)
  features/              ← behavioral · temporal · graph
  models/                ← catboost_ref · lightgbm · xgboost

notebooks/               ← 46 notebooks : 00 (EDA) → 41, un par expérience
scripts/                 ← pipelines par graine, balayages, autopsie d'erreurs
experiments/             ← une fiche par run
submissions/             ← les 91 CSV soumis
livraison_hackatsuki/    ← dossier de livraison officiel (solution reproductible)
reports/                 ← analyse des erreurs du modèle final
```

Les notebooks sont numérotés dans l'ordre chronologique : chacun porte une hypothèse, sa mesure et
sa conclusion, y compris quand la conclusion est « piste refermée ».

---

## 👥 L'équipe

**Hackatsuki** — Sénégal.

| Membre | Rôle |
|--------|------|
| **DIOP Pape Malick** | Leader d'équipe |
| **Sow Papa Ahmadou Seydou** | Modélisation, validation temporelle, encodages anti-fuite, blending |
| **Cissokho Mamadou** | Exploration de l'angle graphe et réseau entre comptes |

**Comment nous avons travaillé à trois.** Des pistes explorées en parallèle — modélisation
comportementale d'un côté, approche par graphe de l'autre — puis confrontées et fusionnées quand
elles apportaient réellement de la diversité. Une règle de validation unique, partagée
(`src/validation.py`), pour que les scores de chacun restent comparables. Et une relecture croisée
systématique du code, qui a servi à autre chose qu'à la forme : c'est elle qui a permis
d'intercepter une fuite de données dans un pipeline — des features dérivées de la cible — avant
qu'elle ne produise des scores flatteurs et intenables (voir
[CORRECTION_CISSOKHO.md](CORRECTION_CISSOKHO.md) et le notebook
[`13_graph_corrige_cissokho.ipynb`](notebooks/13_graph_corrige_cissokho.ipynb)).

---

## 📊 En chiffres

- **64 soumissions** sur la plateforme, **91 CSV** générés
- **46 notebooks**, un par hypothèse testée
- **21 expériences** documentées avec leur verdict
- **+0,013 d'AP** pour la décision la plus rentable — retirer la calibration
- **1,4 × 10⁻⁵** d'écart à la régénération du score par le comité
