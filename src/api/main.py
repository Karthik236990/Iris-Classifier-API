"""
Step 5 — Serve the trained model behind a REST API.

Run directly:
    uvicorn src.api.main:app --reload

Then open http://localhost:8000/docs for interactive Swagger UI, or:

    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config, resolve

_model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on every request.
    config = load_config()
    model_path = resolve(config["model"]["artifact_path"])
    if model_path.exists():
        bundle = joblib.load(model_path)
        _model_bundle["model"] = bundle["model"]
        _model_bundle["feature_cols"] = bundle["feature_cols"]
    else:
        print(f"WARNING: no model found at {model_path}. Run `make pipeline` first.")
    yield
    _model_bundle.clear()


app = FastAPI(
    title="Iris Classifier API",
    description="Example production-style ML serving layer.",
    version="1.0.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class PredictionResponse(BaseModel):
    species: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _model_bundle}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if "model" not in _model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")

    model = _model_bundle["model"]
    feature_cols = _model_bundle["feature_cols"]

    row = pd.DataFrame([{
        "sepal length (cm)": features.sepal_length,
        "sepal width (cm)": features.sepal_width,
        "petal length (cm)": features.petal_length,
        "petal width (cm)": features.petal_width,
    }])[feature_cols]

    prediction = model.predict(row)[0]
    confidence = float(max(model.predict_proba(row)[0]))

    return PredictionResponse(species=prediction, confidence=confidence)
