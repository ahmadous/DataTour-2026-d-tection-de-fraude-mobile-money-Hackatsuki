# Feature Store — registre partagé

Une ligne par feature poussée dans `features_store/`. **Préfixe par auteur obligatoire**
(`b1_`, `b2_`, `c_`). Évite que deux personnes recréent la même variable sous deux noms.

| Feature | Auteur | Calcul | Anti-fuite OK ? | Gain AP (vs sans) |
|---------|--------|--------|-----------------|-------------------|
| _ex : b1_pair_freq_ | B1 | fréquence du couple (origin, destination), fold-by-fold | ✅ | +0.00x |
|  |  |  |  |  |
