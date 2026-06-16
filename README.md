# DataTour 2026 — Détection de fraude Mobile Money (Hackatsuki)

Compétition data science · détection de fraude sur transactions mobile money.
**Métrique : Average Precision (AP).** Objectif : probabilités **bien calibrées**, sans
surapprentissage sur les identifiants de comptes.

---

## ⚡ Démarrage rapide (à faire par chaque membre, jour 1)

```bash
git clone https://github.com/ahmadous/DataTour-2026-d-tection-de-fraude-mobile-money-Hackatsuki.git
cd DataTour-2026-d-tection-de-fraude-mobile-money-Hackatsuki

python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# OBLIGATOIRE : nettoie les sorties des notebooks avant chaque commit
nbstripout --install
```

Placez ensuite `train.csv` / `test.csv` dans `data/` (dossier **gitignored**, jamais commité).

---

## 👥 Équipe & rôles (stratégie à 3 + doublage du Feature Engineering = 4)

| Rôle | Membre | Responsabilité | Fichiers propriétaires |
|------|--------|----------------|------------------------|
| **A — Lead / Validation** | _à remplir_ | Stratégie de CV, calibration, blending, soumissions, leaderboard interne | `src/validation.py`, `src/calibration.py`, `src/blending.py` |
| **B1 — FE comportemental** | _à remplir_ | Agrégations par compte émetteur/destinataire, fréquences, stats montants. Modèle **LightGBM**. | `src/features/behavioral.py`, `src/models/lightgbm.py` |
| **B2 — FE structurel/temporel** | _à remplir_ | Ratios, deltas de balance, features `period`, interactions. Modèle **CatBoost** (référence). | `src/features/temporal.py`, `src/models/catboost_ref.py` |
| **C — R&D graphe + analyse d'erreurs** | _à remplir_ | S1-2 : features graphe (PageRank, degrés, Louvain). S2-3 : analyse FP/FN, stacking. | `src/features/graph.py`, `src/models/xgboost.py` |

> ⏱️ **Go/No-Go graphe** : si gain < 0.005 AP à la mi-semaine 2, C bascule 100 % sur l'analyse d'erreurs.

---

## 📁 Structure du repo

```
src/                  ← code RÉUTILISABLE (.py) — testable, diffable
  validation.py       ← stratégie de CV (propriété A) — source de vérité unique
  calibration.py      ← isotonic / Platt
  blending.py         ← blend & stacking final
  utils.py            ← I/O parquet, seeds, helpers
  features/           ← un module par auteur, préfixe par auteur
  models/             ← un module par modèle
notebooks/            ← EXPLORATION (.ipynb) — jamais du code de prod
experiments/          ← un exp_NN.md par run (voir template)
submissions/          ← CSV de soumission (seuls CSV commités)
data/                 ← gitignored
features_store/       ← gitignored, parquet partagés
models_saved/         ← gitignored
```

---

## 📐 Règles non négociables

1. **`.py` pour tout code réutilisé, `.ipynb` pour explorer.** Dès qu'une fonction sert deux fois → elle va dans `src/`.
2. **Une seule stratégie de validation**, dans `src/validation.py`. Personne ne réimplémente sa propre CV.
3. **Feature store partagé en Parquet, préfixe par auteur** : `b1_freq_sender`, `b2_ratio_balance`, `c_pagerank`… + une ligne par feature dans `features.md`.
4. **Si CV et LB public divergent, on croit la CV.**
5. **Budget de soumissions** : on ne soumet que si la CV bat la meilleure de plus du seuil convenu (à figer jour 1).
6. **Un `experiments/exp_NN.md` par run.** Pas de run sans trace.

---

## 🌿 Workflow Git

- `main` : code stable et intégré. **Personne ne pousse directement.**
- Branches courtes : `feat/<auteur>-<sujet>` (ex. `feat/b1-behavioral-v2`).
- Merge vers `main` via **Pull Request** + relecture croisée par un autre ingénieur (≥ 2×/semaine).
- La relecture croisée sert aussi à attraper les fuites de données et bugs de CV.

---

## 🗓️ Phases

| Phase | Durée | Contenu |
|-------|-------|---------|
| **Commune** | Jour 1-2 | EDA partagée · figer la **CV** · baseline CatBoost reproductible · format du feature store |
| **Parallèle** | Jour 3 → S2 | FE + modèles individuels en parallèle · expériences tracées |
| **Convergence** | S2 → S3 | Go/No-Go graphe · analyse d'erreurs · calibration · blending · soumissions finales |

Détail des rôles et de la stratégie : voir `experiments/exp_000_template.md` pour le format de run.
