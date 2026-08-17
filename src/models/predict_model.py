"""
Step 4 — Load the trained model and run a prediction.

This is the same loading logic the API uses under the hood; kept as a
standalone script so you can sanity-check predictions from the CLI without
starting a server.

Run:
    python -m src.models.predict_model
"""
import joblib
import pandas as pd

from src.config import load_config, resolve


def load_model():
    config = load_config()
    model_path = resolve(config["model"]["artifact_path"])
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


def predict(features: dict) -> str:
    model, feature_cols = load_model()
    row = pd.DataFrame([features])[feature_cols]
    prediction = model.predict(row)[0]
    return prediction


if __name__ == "__main__":
    example = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }
    result = predict(example)
    print(f"Input: {example}")
    print(f"Predicted species: {result}")
