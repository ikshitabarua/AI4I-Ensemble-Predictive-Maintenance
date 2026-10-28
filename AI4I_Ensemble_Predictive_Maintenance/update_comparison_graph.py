
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
csv_path = RESULTS_DIR / "model_results.csv"

results_df = pd.read_csv(csv_path)
results_df = results_df.drop(columns=["Threshold"], errors="ignore")

# Exclude Logistic Regression from the displayed comparison.
results_df = results_df[
    results_df["Model"] != "Logistic Regression"
].copy()

results_df.to_csv(csv_path, index=False)

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
plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
plt.close()

print("Updated comparison CSV and graph.")
print("Models included:")
print(results_df["Model"].to_string(index=False))
