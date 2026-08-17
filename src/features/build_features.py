"""
Step 2 — Feature engineering + train/test split.

Reads the raw CSV, does any cleaning/feature work, and writes separate
train/test CSVs so training is always reproducible from the same split.

Run:
    python -m src.features.build_features
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    data_cfg = config["data"]

    raw_path = resolve(data_cfg["raw_path"])
    df = pd.read_csv(raw_path)

    # --- Feature engineering goes here ---
    # This dataset needs none, but this is where you'd add things like
    # scaling, encoding categoricals, creating ratio features, etc.

    train_df, test_df = train_test_split(
        df,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=df[data_cfg["target_column"]],
    )

    processed_dir = resolve(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Saved to {processed_dir}")
    return train_df, test_df


if __name__ == "__main__":
    build_features()
