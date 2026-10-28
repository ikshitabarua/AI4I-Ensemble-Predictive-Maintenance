# AI4I Ensemble Predictive Maintenance

This project trains baseline and ensemble classifiers on the UCI AI4I 2020 Predictive Maintenance dataset. It includes preprocessing, SMOTE restricted to training partitions, validation-based threshold selection, untouched-test evaluation, charts, and a Streamlit demo.

## Important integrity note

The project does **not** fabricate or manually adjust reported test metrics. It selects decision thresholds using a validation partition to explore a more balanced precision–recall trade-off, then reports the results on an untouched test set. Precision, recall, F1, ROC-AUC, and PR-AUC measure different things and are not expected to be numerically identical. No threshold can guarantee that metrics will be close on unseen test data.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet access for the first dataset download

## Windows / VS Code setup
Open the folder containing `app.py` and `requirements.txt`, then run in the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python ml\preprocess.py
python ml\train_ensemble.py
streamlit run app.py
```

If PowerShell blocks activation, you can run the venv's interpreter directly: ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` (remove the leading space) and then use ` .\.venv\Scripts\python.exe ml\preprocess.py` etc.

## What the scripts do
1. `download_dataset.py`: downloads the official dataset through `ucimlrepo`.
2. `ml/preprocess.py`: handles column names with or without units, excludes IDs and failure-mode labels, engineers features, splits train/test, and learns imputation values from training data only.
3. `ml/train_ensemble.py`: splits a validation partition from training data, applies SMOTE only to the fitting partition, chooses a threshold using a documented rule, refits on all training data (SMOTE only on train), and evaluates on the untouched test set.
4. `app.py`: interactive prediction demo using the trained Stacking model and validation-selected threshold.

## Outputs
- `results/model_results.csv`: final held-out test metrics and selected thresholds
- `results/threshold_tuning.json`: validation threshold selection details
- `results/confusion_matrices.png`
- `results/roc_curves.png`
- `results/pr_curves.png`
- `results/f1_comparison.png`
- `results/metric_comparison.png`
- `models/`: trained models and threshold metadata

## Interpretation
For imbalanced failure detection, accuracy alone is inadequate. Consider recall (failure detection), precision (false alarm burden), F1, and PR-AUC together. Choose an operating threshold based on the actual maintenance cost of false negatives versus false positives. This dataset is a benchmark; the demo is not validated for real machinery.
