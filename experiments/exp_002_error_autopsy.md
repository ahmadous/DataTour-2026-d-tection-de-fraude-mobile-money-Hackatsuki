# Exp 002 — Autopsie des erreurs du modèle op_03

Auteur : équipe
Date : 2026-06-24
Branche : analysis/error-autopsy
Notebook/script : `scripts/error_autopsy.py`

## Périmètre
- [x] Analyse centrée sur `op_03`
- [x] Validation temporelle sur folds récents
- [x] Autopsie des faux négatifs / faux positifs par régime

## Modèle de diagnostic
- Baseline légère de diagnostic: row + balances + fréquences + `te_origin`
- Modèle utilisé pour l’autopsie: HistGradientBoostingClassifier
- Échantillon rapide: 120k lignes `op_03`

## Résultats
- OOF AP global: 0.3533
- OOF AP recent2: 0.3551
- OOF AP last: 0.3496

## Constats clés
- Le pire régime est le début de période:
  - `period_bin (0-23]` : fraudes très mal classées
  - `hard_fraud_rate` très élevé, le modèle manque surtout les fraudes précoces
- Les transactions avec `origin_emptied`, `origin_inconsistent`, `dest_inconsistent`, `dest_not_credited` sont surreprésentées parmi les fraudes, mais restent imparfaitement classées
- Les comptes avec faible profondeur historique (`origin_out_deg_bin` 2, 3-4, `origin_recent_5_bin` faible) sont des zones de faiblesse
- `te_origin` reste le signal central, mais il ne suffit pas sur les régimes de faible historique

## Hypothèse suivante
- Tester des features qui mesurent explicitement la **quantité de preuve disponible**:
  - âge du compte
  - densité d’historique à plusieurs fenêtres
  - interaction `te_origin × ancienneté`
  - fallback de prior par régime temporel / période
- Tester une segmentation "early history vs enough history" plutôt qu’un modèle unique

## Décision
- Stopper les variantes qui ne changent que la famille `te_origin` sans traiter le régime “début de période / faible historique”.
- Construire une nouvelle famille de features orientée **history depth** et **regime switch**.
