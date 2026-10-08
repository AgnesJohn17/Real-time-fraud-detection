import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load transaction data
df = pd.read_csv("data/raw/creditcard.csv")

print("Dataset loaded:", df.shape)

# Remove duplicate transactions
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Clean dataset:", df.shape)
print("Features:", X.shape)
print("Target:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest trained successfully")

from pathlib import Path
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)

# Evaluate on transactions the model did not train on
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

print("\nAverage precision:")
print(round(average_precision_score(y_test, y_scores), 4))

# Save the model and the feature order needed for future predictions
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "threshold": 0.5
    },
    "models/fraud_model.joblib"
)

print("\nModel saved to models/fraud_model.joblib")