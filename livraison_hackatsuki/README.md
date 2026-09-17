# Équipe Hackatsuki — DataTour 2026 (détection de fraude mobile money)

**Fichier soumis (Private Score) : `submission.csv`** — métrique : Average Precision.

---

## 1. Instructions de reproduction

> **Python 3.9 à 3.12 requis** (avec Python 3.13+, `catboost==1.2.5` n'a pas de
> roue précompilée). Si besoin, installer Python 3.12 depuis
> [python.org/downloads](https://www.python.org/downloads/).

Les données de la compétition (`train.csv`, `test.csv`, `sample_submission.csv`)
sont **déjà incluses dans `data/`** — aucune intervention manuelle nécessaire.

```bash
pip install -r requirements.txt

jupyter notebook solution.ipynb    # puis « Run All »
# (équivalent sans jupyter : python solution.py)
```

Durée : ~12-15 min sur CPU (2 entraînements CatBoost). La soumission est
écrite dans **`submissions/submission.csv`**.

Le pipeline est **entièrement déterministe** : toutes les graines aléatoires
sont fixées (seed 42 — `numpy`, `random` et `random_seed` CatBoost), aucune
donnée externe n'est utilisée et aucune connexion Internet n'est nécessaire
pendant l'exécution. La première cellule affiche les versions installées et
signale tout écart avec `requirements.txt`.

---

## 2. Méthodologie

### Étape 0 — Restriction au périmètre de la fraude
100% des fraudes du train se trouvent dans la catégorie d'opération `op_03`
(taux interne : 31.2%). Le modèle n'est entraîné et appliqué que sur `op_03` ;
les autres transactions reçoivent une probabilité 0.

### Étape 1 — Modèle champion
CatBoost (depth 6, lr 0.05, 600 itérations, seed 42) sur des features construites
**sans aucune fuite de données** :
- montants et ratios montant/soldes, incohérences de solde (résidus comptables) ;
- fréquences des comptes émetteur/destinataire ;
- comportement par compte et par paire (compteurs, nouveauté de la paire, degrés) ;
- dynamique récente de l'émetteur (fenêtres glissantes strictement passées) ;
- **`te_origin`** : taux de fraude historique du compte émetteur, la feature
  dominante (~33% d'importance), encodée **fold-safe** — out-of-fold imbriqué
  sur le train pour éviter tout auto-encodage, mapping appris sur le train
  seulement pour le test (lissage bayésien m=30).

**Aucune calibration** : l'Average Precision est une métrique de rang, la
calibration isotonique dégradait le score d'environ 0.013 (mesuré).

### Étape 2 — Pseudo-labeling, 1 cycle
Les prédictions très confiantes du champion sur le test (proba > 0.98 → fraude,
< 0.02 → légitime ; ~5 600 transactions) sont ajoutées au train comme
pseudo-labels, puis le modèle est réentraîné.
Motivation : ~9.7% des comptes destinataires du test n'existent pas dans le
train ; le pseudo-labeling fournit au modèle un signal sur ces comptes futurs.
Un seul cycle : le 2e cycle et les seuils plus permissifs dégradaient (mesuré).

### Étape 3 — Blend par moyenne de rangs (25% champion / 75% pseudo)
Les deux vecteurs de prédictions sont convertis en rangs normalisés puis
moyennés (25/75). La moyenne de rangs surpasse la moyenne de probabilités
brutes pour une métrique de rang ; la pondération a été choisie par balayage
validé sur le leaderboard public.

### Validation
Toute la sélection de modèle a été faite en **validation temporelle**
(expanding window sur les périodes, 5 folds — `src/validation.py`), le test
étant strictement dans le futur du train (périodes 106-143 vs 0-105).
Aucun split aléatoire n'a été utilisé.

---

## 3. Contenu de l'archive

```
├── solution.ipynb       ← LE notebook (Run All) : régénère la soumission de bout en bout
├── solution.py          ← le même pipeline en script (alternative sans jupyter)
├── README.md            ← ce fichier (instructions + méthodologie)
├── requirements.txt     ← versions exactes des dépendances
├── submission.csv       ← le fichier soumis, correspondant au Private Score
├── src/                 ← modules du pipeline (features, encodage fold-safe, utilitaires)
├── data/                ← train.csv / test.csv / sample_submission.csv (inclus)
└── submissions/         ← la soumission régénérée est écrite ici à l'exécution
```

---

*Équipe Hackatsuki — DataTour 2026*
