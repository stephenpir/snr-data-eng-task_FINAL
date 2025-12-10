# src/part1_prepare_data.py
from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

if __name__ == "__main__":
    tx = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["txn_timestamp"])
    labels = pd.read_csv(DATA_DIR / "labels.csv")

    tx["clean_desc"] = tx["description"].fillna("").apply(clean_text)

    agg = (
        tx.groupby("customer_id")
        .agg(
            txn_count=("transaction_id", "count"),
            total_debit=("amount", lambda x: x[x < 0].sum()),
            total_credit=("amount", lambda x: x[x > 0].sum()),
            avg_amount=("amount", "mean"),
            all_desc=("clean_desc", lambda x: " ".join(x)),
        )
        .reset_index()
    )

    # super simple text feature: indicator words
    keywords = ["rent", "netflix", "tesco", "payroll", "bonus"]
    for kw in keywords:
        agg[f"kw_{kw}"] = agg["all_desc"].str.contains(rf"\b{kw}\b").astype(int)

    df = agg.merge(labels, on="customer_id", how="left")
    df.to_csv(ARTIFACTS_DIR / "training_set.csv", index=False)
    print("wrote artifacts/training_set.csv")