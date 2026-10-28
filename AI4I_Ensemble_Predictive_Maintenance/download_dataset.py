from pathlib import Path
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "ai4i2020.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    print(f"Dataset already exists: {OUT}")
else:
    print("Downloading AI4I 2020 Predictive Maintenance Dataset from UCI...")
    dataset = fetch_ucirepo(id=601)
    df = dataset.data.original
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
print("Dataset path:", OUT)
