"""
Step 1 — Data ingestion.

In a real project this would query a database, call an API, or pull a file
from S3. Here we load the classic Iris dataset from scikit-learn so the
project runs with zero external dependencies or credentials.

Run:
    python -m src.data.make_dataset
"""
from sklearn.datasets import load_iris
import pandas as pd

from src.config import load_config, resolve


def make_dataset() -> pd.DataFrame:
    config = load_config()

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=["target"])

    raw_path = resolve(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} rows to {raw_path}")
    return df


if __name__ == "__main__":
    make_dataset()
