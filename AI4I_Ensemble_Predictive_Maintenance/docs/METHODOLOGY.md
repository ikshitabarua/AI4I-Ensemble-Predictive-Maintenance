# Methodology

1. Read AI4I 2020 data and normalize known column-name variants (with/without units).
2. Exclude UID, Product ID and failure-mode columns TWF/HDF/PWF/OSF/RNF to avoid leakage.
3. Engineer temperature difference, mechanical power, and interaction features.
4. Split once into stratified train and untouched test sets. Fit imputation values from training only.
5. Split a validation partition from the training set. Apply SMOTE only to the inner fitting partition.
6. Tune decision thresholds on validation data according to a declared criterion: among thresholds with F1 at least 95% of maximum validation F1, choose the smallest precision–recall gap.
7. Refit models using SMOTE on the full training set and evaluate once on the untouched test set.

The threshold procedure aims to explore a precision–recall balance but cannot guarantee close test metrics. ROC-AUC and PR-AUC are based on probabilities and do not depend on the chosen classification threshold.
