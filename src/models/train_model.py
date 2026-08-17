"""
Step 3 — Train, evaluate, and persist the model.

Loads the processed train/test split, trains a RandomForestClassifier with
hyperparameters from config.yaml, evaluates it, and saves both the model
artifact and a metrics.json so results are tracked over time.

Run:
    python -m src.models.train_model
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import load_config, resolve


def train_model():
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    processed_dir = resolve(data_cfg["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    target_col = data_cfg["target_column"]
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = RandomForestClassifier(**model_cfg["params"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Persist the model
    model_path = resolve(model_cfg["artifact_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"Model saved to {model_path}")

    # Persist metrics so accuracy over time is tracked in git history
    metrics_path = resolve(model_cfg["metrics_path"])
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return model, acc


if __name__ == "__main__":
    train_model()
