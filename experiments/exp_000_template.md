# Exp 000 — TEMPLATE (copier en exp_NN.md, ne pas modifier celui-ci)

Auteur : A / B1 / B2 / C
Date : 2026-MM-JJ
Branche : feat/xxx

## Périmètre
- [ ] Modélisation restreinte à op_03 (proba 0 ailleurs)
- Schéma de validation : time_folds (référence) / group_folds / stratified (diagnostic)

## Features utilisées
- ...
- (préfixées par auteur : b1_, b2_, c_)

## Modèle
CatBoost / LightGBM / XGBoost — hyperparams : depth=, lr=, iterations=, seed=42

## Résultats
CV temporelle (5 folds) : 0.____ ± 0.____
LB public (30 %)        : 0.____

## Anti-fuite (checklist obligatoire)
- [ ] Aucun encodage d'ID de compte brut
- [ ] Encodages fréquence/target calculés fold-by-fold sur le passé
- [ ] Aucune feature dérivée du futur

## Commentaires
+ ...
- ...
```
```
