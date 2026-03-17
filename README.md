# Titanic Assignment

This project explores the Titanic training dataset, performs data cleaning and feature engineering, and runs feature selection with a Random Forest model.

## Project Structure

```text
.
|-- data/
|   |-- train.csv
|   `-- test.csv
|-- notebooks/
|   `-- Titanic_Feature_Engineering.ipynb
|-- scripts/
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   `-- feature_selection.py
|-- requirements.txt
`-- README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Notebook Workflow

The notebook `notebooks/Titanic_Feature_Engineering.ipynb` follows this flow:

1. Load and inspect the training dataset.
2. Check shape, data types, missing values, and duplicates.
3. Handle missing values (for example, `Age` and `Embarked`).
4. Create engineered features such as `FamilySize`, `IsAlone`, and title-based features.
5. Encode categorical variables.
6. Train a Random Forest model to inspect feature importances.
7. Save cleaned output to `data/train_cleaned.csv`.

## Script Modules

### `scripts/data_cleaning.py`

Provides `clean_data(df)` to:

- fill missing values,
- drop `Cabin` when present,
- remove duplicates,
- normalize `Sex`,
- clip `Fare` outliers.

### `scripts/feature_engineering.py`

Provides `create_family_features(df)` to add:

- `FamilySize`
- `IsAlone`

### `scripts/feature_selection.py`

Provides:

- `prepare_features(df)`
- `get_feature_importance(df)`
- `select_top_features(df, top_n=10)`

`prepare_features(df)` now handles key preprocessing steps internally (missing values, basic engineered features, dropping non-useful columns, and one-hot encoding), so feature selection can run directly on raw Titanic data as long as `Survived` exists.

## Notes

- The target column for modeling is `Survived`.
- Feature-selection functions raise a clear error if `Survived` is missing.