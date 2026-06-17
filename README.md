# DataTour 2026 — Détection de fraude Mobile Money (Hackatsuki)

Compétition data science · détection de fraude sur transactions mobile money.
**Métrique : Average Precision (AP).** Objectif : probabilités **bien calibrées**, sans
surapprentissage sur les identifiants de comptes.

> 📌 **Lire [FINDINGS.md](FINDINGS.md) AVANT de coder.** L'EDA a révélé que 100 % de la
> fraude est dans `op_03`, que le graphe est biparti (PageRank/Louvain inutiles), que la
> validation doit être temporelle et que l'ID de compte est un piège. Ces faits priment.

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
| **A — Lead / Validation** | _à remplir_ | CV temporelle (`time_folds`), encodages anti-fuite, calibration, blending, soumissions, leaderboard interne | `src/validation.py`, `src/encoding.py`, `src/calibration.py`, `src/blending.py` |
| **B1 — FE comportemental** | _à remplir_ | Comportement par couple émetteur-destinataire, fréquences, stats montants (fold-safe). Modèle **LightGBM**. | `src/features/behavioral.py`, `src/models/lightgbm.py` |
| **B2 — FE temporel / dynamique** | _à remplir_ | Régularité temporelle, dynamique récente, montant vs habitude du compte. Modèle **CatBoost** (référence). | `src/features/temporal.py`, `src/models/catboost_ref.py` |
| **C — Degrés bipartites + analyse d'erreurs** | _à remplir_ | Features de degré bipartite (collecteur / fan-out) **time-boxées**, puis analyse FP/FN + stacking. Modèle **XGBoost**. | `src/features/graph.py`, `src/models/xgboost.py` |

> ⚠️ **La piste graphe lourde est ABANDONNÉE** (graphe biparti, sans cycles — cf. FINDINGS.md).
> Rôle C : tester vite les degrés bipartites, puis basculer dès la S2 sur l'**analyse d'erreurs**,
> qui est le vrai levier ici. Go/No-Go degrés : si gain < 0.005 AP, on coupe immédiatement.

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
