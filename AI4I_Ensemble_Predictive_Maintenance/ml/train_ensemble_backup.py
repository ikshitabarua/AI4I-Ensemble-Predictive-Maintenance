
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# =========================================================
# 2. LOAD THE CSV FILES THAT ACTUALLY EXIST
# =========================================================

def read_csv_required(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}\n"
            "Check data/processed and run ml/preprocess.py first."
        )

    return pd.read_csv(path)


def read_target(filename):
    target_df = read_csv_required(filename)

    # Remove saved DataFrame index columns, if present.
    target_df = target_df.loc[
        :, ~target_df.columns.astype(str).str.startswith("Unnamed:")
    ]

    if target_df.shape[1] != 1:
        raise ValueError(
            f"{filename} should contain one target column, "
            f"but found {list(target_df.columns)}"
        )

    return target_df.iloc[:, 0].reset_index(drop=True)


X_train = read_csv_required("X_train.csv")
y_train = read_target("y_train.csv")
X_test = read_csv_required("X_test.csv")
y_test = read_target("y_test.csv")

# Remove accidentally saved index columns from feature tables.
X_train = X_train.loc[
    :, ~X_train.columns.astype(str).str.startswith("Unnamed:")
]
X_test = X_test.loc[
    :, ~X_test.columns.astype(str).str.startswith("Unnamed:")
]

# =========================================================
# 3. CLEAN FEATURE NAMES FOR XGBOOST
# =========================================================

def clean_feature_names(df):
    df = df.copy()

    cleaned = []
    for col in df.columns:
        name = str(col)
        name = name.replace("[", "_").replace("]", "_")
        name = name.replace("<", "_")
        cleaned.append(name)

    # Ensure names remain unique after cleaning.
    counts = {}
    unique_names = []

    for name in cleaned:
        if name not in counts:
            counts[name] = 0
            unique_names.append(name)
        else:
            counts[name] += 1
            unique_names.append(f"{name}_{counts[name]}")

    df.columns = unique_names
    return df


X_train = clean_feature_names(X_train)
X_test = clean_feature_names(X_test)

# Ensure the train and test columns match.
if set(X_train.columns) != set(X_test.columns):
    missing = sorted(set(X_train.columns) - set(X_test.columns))
    extra = sorted(set(X_test.columns) - set(X_train.columns))

    raise ValueError(
        f"Train/test columns differ.\n"
        f"Missing from test: {missing}\n"
        f"Extra in test: {extra}"
    )

X_test = X_test[X_train.columns]

# Convert features to numeric and handle invalid values.
X_train = X_train.apply(pd.to_numeric, errors="raise")
X_test = X_test.apply(pd.to_numeric, errors="raise")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Calculate imputation values from training data only.
medians = X_train.median().fillna(0)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

y_train = pd.to_numeric(y_train, errors="raise").astype(int)
y_test = pd.to_numeric(y_test, errors="raise").astype(int)

if len(X_train) != len(y_train):
    raise ValueError("X_train and y_train have different row counts.")

if len(X_test) != len(y_test):
    raise ValueError("X_test and y_test have different row counts.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError(
        "Expected training labels 0 and 1, with 1 representing failure. "
        f"Found: {sorted(y_train.unique())}"
    )

if set(y_test.unique()) != {0, 1}:
    raise ValueError(
        "Expected test labels 0 and 1, with 1 representing failure. "
        f"Found: {sorted(y_test.unique())}"
    )

print("\nDATA LOADED")
print("-----------")
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTraining labels:")
print(y_train.value_counts().sort_index())
print("\nTest labels:")
print(y_test.value_counts().sort_index())

# =========================================================
# 4. DEFINE THE MODELS
# =========================================================

def create_models():
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("xgb", xgb),
        ],
        voting="soft",
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("xgb", xgb),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "XGBoost": xgb,
        "Soft Voting Ensemble": voting,
        "Stacking Ensemble": stacking,
    }



# =========================================================
# 5. SPLIT ORIGINAL TRAINING DATA FOR VALIDATION
# =========================================================

# This split must happen BEFORE applying SMOTE.
# The test set is not used here.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=RANDOM_STATE,
)

print("\nVALIDATION SPLIT")
print("----------------")
print("Fit samples:", len(X_fit))
print("Validation samples:", len(X_val))
print("\nFit class distribution:")
print(y_fit.value_counts().sort_index())
print("\nValidation class distribution:")
print(y_val.value_counts().sort_index())

# The validation data must retain the original class distribution.
# Do not apply SMOTE to X_val or y_val.

# =========================================================
# 6. SELECT EACH MODEL'S THRESHOLD USING VALIDATION F1
# =========================================================

minority_count = int(y_fit.value_counts().min())

if minority_count < 2:
    raise ValueError(
        "Not enough minority-class examples in the fitting split "
        "to apply SMOTE."
    )

smote_validation = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=min(5, minority_count - 1),
)

# Resample only the fitting partition, once.
X_fit_balanced, y_fit_balanced = (
    smote_validation.fit_resample(X_fit, y_fit)
)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_fit_balanced).value_counts().sort_index())

thresholds = {}
validation_results = {}
validation_models = create_models()

print("\nVALIDATION THRESHOLD TUNING")
print("===========================")

for name, model in validation_models.items():
    print(f"\nValidating {name}...")

    # Train only on the SMOTE-resampled fitting partition.
    model.fit(X_fit_balanced, y_fit_balanced)

    # Predict on the ORIGINAL, unresampled validation partition.
    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Evaluate candidate probability thresholds.
    precision, recall, candidate_thresholds = (
        precision_recall_curve(y_val, val_probabilities)
    )

    # Each threshold corresponds to all points except the last
    # precision/recall point.
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    if len(candidate_thresholds) == 0:
        selected_threshold = 0.5
    else:
        # Maximize F1 on validation data.
        best_index = int(np.argmax(f1_values))
        selected_threshold = float(
            candidate_thresholds[best_index]
        )

    # Calculate the metrics at the selected threshold.
    val_predictions = (
        val_probabilities >= selected_threshold
    ).astype(int)

    val_precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    val_recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    val_f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    thresholds[name] = selected_threshold

    validation_results[name] = {
        "Model": name,
        "Threshold": selected_threshold,
        "Validation_Precision": val_precision,
        "Validation_Recall": val_recall,
        "Validation_F1": val_f1,
    }

    print(f"Threshold:  {selected_threshold:.4f}")
    print(f"Precision:  {val_precision:.4f}")
    print(f"Recall:     {val_recall:.4f}")
    print(f"F1-score:   {val_f1:.4f}")

# Save validation results separately from test results.
validation_df = pd.DataFrame(
    list(validation_results.values())
).sort_values(
    "Validation_F1", ascending=False
)

validation_df.to_csv(
    RESULTS_DIR / "validation_threshold_results.csv",
    index=False,
)

print("\nVALIDATION RESULTS, RANKED BY F1")
print(validation_df.to_string(index=False))
print(
    "\nSaved validation results to:",
    RESULTS_DIR / "validation_threshold_results.csv",
)


# =========================================================
# 7. REFIT ON ALL TRAINING DATA
# =========================================================

print("\nBALANCING TRAINING DATA")

minority_count = y_train.value_counts().min()

if minority_count < 2:
    raise ValueError("Too few minority-class samples for SMOTE.")

smote_final = SMOTE(
    random_state=RANDOM_STATE,
    k_neighbors=min(5, minority_count - 1),
)

X_train_balanced, y_train_balanced = smote_final.fit_resample(
    X_train, y_train
)

print("Original training labels:")
print(y_train.value_counts().sort_index())
print("\nBalanced training labels:")
print(pd.Series(y_train_balanced).value_counts().sort_index())

models = create_models()

print("\nTRAINING FINAL MODELS")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)

# =========================================================
# 8. EVALUATE ON THE UNTOUCHED TEST SET
# =========================================================

results = []
test_probabilities = {}
test_predictions = {}

print("\nFINAL TEST RESULTS")
print("==================")

for name, model in models.items():
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = thresholds[name]
    predictions = (probabilities >= threshold).astype(int)

    test_probabilities[name] = probabilities
    test_predictions[name] = predictions

    row = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(
            y_test, probabilities
        ),
    }

    results.append(row)

    print(
        f"{name:25s} "
        f"Accuracy={row['Accuracy']:.4f} "
        f"Precision={row['Precision']:.4f} "
        f"Recall={row['Recall']:.4f} "
        f"F1={row['F1']:.4f} "
        f"ROC-AUC={row['ROC_AUC']:.4f} "
        f"PR-AUC={row['PR_AUC']:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULTS_DIR / "model_results.csv",
    index=False,
)

# =========================================================
# 9. SAVE MODELS AND METADATA
# =========================================================

for name, model in models.items():
    filename = re.sub(
        r"[^a-z0-9]+",
        "_",
        name.lower(),
    ).strip("_") + ".joblib"

    joblib.dump(model, MODEL_DIR / filename)

joblib.dump(
    {
        "thresholds": thresholds,
        "feature_names": list(X_train.columns),
        "positive_class": 1,
    },
    MODEL_DIR / "thresholds_and_features.joblib",
)

# =========================================================
# 10. CONFUSION MATRICES
# =========================================================

for name, predictions in test_predictions.items():
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Failure"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()

    filename = re.sub(
        r"[^a-z0-9]+",
        "_",
        name.lower(),
    ).strip("_")

    fig.savefig(
        RESULTS_DIR / f"{filename}_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

# =========================================================
# 11. ROC CURVES
# =========================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name, probabilities in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random baseline",
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Test Set")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=160)
plt.close(fig)

# =========================================================
# 12. PRECISION-RECALL CURVES
# =========================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name, probabilities in test_probabilities.items():
    precision, recall, _ = precision_recall_curve(
        y_test, probabilities
    )
    ap = average_precision_score(y_test, probabilities)
    ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

ax.axhline(
    float(y_test.mean()),
    linestyle="--",
    label=f"Failure prevalence={y_test.mean():.3f}",
)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves - Test Set")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "precision_recall_curves.png",
    dpi=160,
)
plt.close(fig)

# =========================================================
# 13. MODEL COMPARISON GRAPH
# =========================================================

metrics = ["Accuracy", "Precision", "Recall", "F1"]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
)

ax.set_title("Model Comparison - Untouched Test Set")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=160,
)
plt.close()

# =========================================================
# 14. FINISHED
# =========================================================

print("\n========================================")
print("TRAINING AND EVALUATION COMPLETED")
print("========================================")
print("Results table:", RESULTS_DIR / "model_results.csv")
print("Comparison graph:", RESULTS_DIR / "model_comparison.png")
print("ROC curves:", RESULTS_DIR / "roc_curves.png")
print(
    "Precision-recall curves:",
    RESULTS_DIR / "precision_recall_curves.png",
)
print("Saved models:", MODEL_DIR)
