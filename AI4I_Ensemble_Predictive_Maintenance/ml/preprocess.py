"""Normalize AI4I columns, engineer features, split data, and save artifacts.
SMOTE is intentionally deferred to training so validation/test data stay untouched.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
if not RAW.exists():
    raise FileNotFoundError(f"Dataset not found at {RAW}. Run: python download_dataset.py")

df = pd.read_csv(RAW)
df.columns = [str(c).strip() for c in df.columns]
# UCI versions vary: some include units in column names and some do not.
aliases = {
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
}
df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns and new not in df.columns})
required = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Machine failure"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}. Found: {list(df.columns)}")

# Explicitly exclude identifiers and failure-mode indicators to avoid target leakage.
y = pd.to_numeric(df["Machine failure"], errors="raise").astype(int)
exclude = ["UID", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
X = df.drop(columns=[c for c in exclude if c in df.columns]).copy()
# Feature engineering based on available sensor inputs.
X["Temperature Difference"] = X["Process temperature"] - X["Air temperature"]
X["Mechanical Power [W]"] = X["Torque"] * X["Rotational speed"] * 2 * np.pi / 60
X["Torque_RPM"] = X["Torque"] * X["Rotational speed"]
X["ToolWear_Torque"] = X["Tool wear"] * X["Torque"]
X["ToolWear_RPM"] = X["Tool wear"] * X["Rotational speed"]
X = pd.get_dummies(X, columns=["Type"], drop_first=False, dtype=int)
X = X.replace([np.inf, -np.inf], np.nan)

# Split before learning imputation values; the test set remains untouched.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
medians = X_train.median(numeric_only=True).fillna(0)
X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

X_train.to_csv(OUT / "X_train.csv", index=False)
X_test.to_csv(OUT / "X_test.csv", index=False)
pd.DataFrame({"Machine failure": y_train}).to_csv(OUT / "y_train.csv", index=False)
pd.DataFrame({"Machine failure": y_test}).to_csv(OUT / "y_test.csv", index=False)
summary = {
    "rows": int(len(df)), "source_columns": list(df.columns), "model_features": list(X.columns),
    "original_distribution": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
    "train_distribution": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
    "test_distribution": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
    "note": "Test set is untouched. SMOTE is applied only within model-training folds/partitions."
}
(OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("=" * 62)
print("PREPROCESSING COMPLETE")
print(f"Dataset rows: {len(df)} | Features: {X.shape[1]}")
print("Train distribution (before SMOTE):\n", y_train.value_counts().sort_index().to_string())
print("Test distribution (untouched):\n", y_test.value_counts().sort_index().to_string())
print(f"Saved files to: {OUT}")
