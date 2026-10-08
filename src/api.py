from fastapi.responses import FileResponse

from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "models/fraud_model.joblib")

app = FastAPI(title="Fraud Detection API")

DB_PATH = ROOT / "data" / "predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as connection:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            fraud_score REAL NOT NULL,
            decision TEXT NOT NULL
        )
    """)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: FiniteFloat = Field(ge=0)
    Amount: FiniteFloat = Field(ge=0)
    features: list[FiniteFloat] = Field(
        min_length=28,
        max_length=28,
        description="Values for V1 through V28, in that order"
    )


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/predict")
def predict(transaction: Transaction):
    row = {
        "Time": transaction.Time,
        "Amount": transaction.Amount
    }

    for index, value in enumerate(transaction.features, start=1):
        row[f"V{index}"] = value

    inputs = pd.DataFrame([row], columns=bundle["features"])

    score = float(bundle["model"].predict_proba(inputs)[0, 1])
    threshold = float(bundle["threshold"])
    flagged = score >= threshold
    decision = "REVIEW" if flagged else "PASS"

    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO predictions
                (timestamp, amount, fraud_score, decision)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                transaction.Amount,
                score,
                decision
            )
        )

    return {
        "fraud_score": round(score, 4),
        "threshold": threshold,
        "flagged": flagged,
        "decision": decision
    }


@app.get("/transactions")
def get_transactions():
    with sqlite3.connect(DB_PATH, timeout=30) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT id, timestamp, amount, fraud_score, decision
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return [dict(row) for row in rows]

@app.get("/")
def dashboard():
    return FileResponse(ROOT / "src" / "dashboard.html")