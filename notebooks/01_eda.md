# Exploratory Data Analysis — Iris Dataset

This would normally be a Jupyter notebook (`01_eda.ipynb`). It's kept as
markdown notes here to keep the template lightweight — in a real project,
replace this with an actual notebook that:

1. Loads `data/raw/iris.csv`
2. Checks for missing values, class balance, and outliers
3. Plots feature distributions and pairwise relationships (e.g. `seaborn.pairplot`)
4. Checks correlation between features
5. Documents any decisions that feed into `src/features/build_features.py`
   (e.g. "petal length and width are highly correlated — consider dropping one
   if using a linear model")

## Findings (example)

- 150 rows, 4 numeric features, 3 balanced classes (50 each) — no class
  imbalance handling needed.
- No missing values.
- `petal_length` and `petal_width` are the most predictive features by eye —
  confirmed later by feature importances from the trained model.
