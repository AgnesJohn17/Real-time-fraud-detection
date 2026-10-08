from pathlib import Path
import json
import time
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

# Recreate the held-out test set to avoid replaying training data
df = pd.read_csv(ROOT / "data/raw/creditcard.csv").drop_duplicates()

_, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Class"]
)

# Mix legitimate and fraud examples so both appear in this demo.
# This deliberately increased fraud rate is NOT the real dataset rate.
demo = pd.concat([
    test_data[test_data["Class"] == 0].head(15),
    test_data[test_data["Class"] == 1].head(5)
]).sample(frac=1, random_state=42)

for number, (_, row) in enumerate(demo.iterrows(), start=1):
    payload = {
        "Time": float(row["Time"]),
        "Amount": float(row["Amount"]),
        "features": [
            float(row[f"V{i}"]) for i in range(1, 29)
        ]
    }

    request = Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=30) as response:
        result = json.load(response)

    actual = "FRAUD" if row["Class"] == 1 else "LEGITIMATE"

    print(
        f"{number:02d} | Amount: {row['Amount']:.2f}"
        f" | Score: {result['fraud_score']:.4f}"
        f" | Decision: {result['decision']}"
        f" | Actual: {actual}"
    )

    time.sleep(1)