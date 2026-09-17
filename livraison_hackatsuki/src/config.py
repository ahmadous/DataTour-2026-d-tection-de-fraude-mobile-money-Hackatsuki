"""Constantes du projet — alignées sur les VRAIES colonnes et les findings de l'EDA.

Source : data/datatour_column_descriptions.csv + rapport_datatour_2026 (EDA Opus).
Voir FINDINGS.md pour le détail des découvertes.
"""
from __future__ import annotations

# --- Colonnes du dataset ---
ID = "id"
PERIOD = "period"
OPERATION = "operation"
AMOUNT = "amount"
ORIGIN_ACCT = "origin_account"
ORIGIN_BAL_BEFORE = "origin_balance_before"
ORIGIN_BAL_AFTER = "origin_balance_after"
DEST_ACCT = "destination_account"
DEST_BAL_BEFORE = "destination_balance_before"
DEST_BAL_AFTER = "destination_balance_after"
TARGET = "fraud_flag"          # cible dans train.csv
SUBMISSION_TARGET = "target"   # colonne attendue dans la soumission

# --- Findings structurels (EDA) ---
FRAUD_OPERATION = "op_03"      # 100% des fraudes sont ici (taux interne 31.2%)
TRAIN_PERIOD_MAX = 105         # train : périodes 0..105
TEST_PERIOD_MIN = 106          # test  : périodes 106..143 (strictement dans le futur)

# AP "gratuite" si on prédit seulement op_03=risqué (baseline triviale)
BASELINE_AP_OP03_ONLY = 0.31
