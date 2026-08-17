"""
Tests for the pipeline. Run with:

    pytest
    # or
    make test

These run the pipeline end-to-end against small in-memory checks, so they
also double as a smoke test that the whole thing still works after changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import predict


def test_make_dataset_has_expected_shape():
    df = make_dataset()
    assert len(df) == 150
    assert "species" in df.columns
    assert df["species"].nunique() == 3


def test_build_features_creates_split():
    build_features()  # requires make_dataset to have run
    from src.config import load_config, resolve
    config = load_config()
    processed_dir = resolve(config["data"]["processed_dir"])
    assert (processed_dir / "train.csv").exists()
    assert (processed_dir / "test.csv").exists()


def test_train_model_meets_accuracy_bar():
    # Iris is an easy dataset; a healthy model should comfortably clear 90%.
    _, accuracy = train_model()
    assert accuracy >= 0.9


def test_predict_returns_valid_species():
    example = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }
    result = predict(example)
    assert result in ("setosa", "versicolor", "virginica")
