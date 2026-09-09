"""
preprocessing.py
-----------------
Step 2 of the pipeline: clean the raw data, engineer date features,
encode categoricals, and produce the train/test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    CATEGORICAL_COLUMNS,
    DATE_COLUMN,
    DATE_FORMAT,
    ID_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and impute missing values.

    Numeric columns -> median. Categorical columns -> mode.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocessing] Removed {before - len(df)} duplicate rows")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"[preprocessing] Filled '{col}' missing values with median={median}")

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode)
            print(f"[preprocessing] Filled '{col}' missing values with mode='{mode}'")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose the claim submission date and drop ID columns that
    carry no predictive signal.
    """
    df = df.copy()

    dates = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    df["Submission_Year"] = dates.dt.year
    df["Submission_Month"] = dates.dt.month
    df["Submission_Day"] = dates.dt.day
    df = df.drop(columns=[DATE_COLUMN])
    print(f"[preprocessing] Expanded '{DATE_COLUMN}' into Submission_Year/Month/Day")

    existing_id_cols = [c for c in ID_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_id_cols)
    print(f"[preprocessing] Dropped ID columns: {existing_id_cols}")

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns the encoded frame and
    a dict of fitted LabelEncoders (useful for decoding predictions
    later or encoding new incoming data consistently).
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"[preprocessing] Label-encoded columns: {list(encoders.keys())}")
    return df, encoders


def encode_new_data(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode categorical columns on NEW data using encoders already
    fitted during training (transform only, never fit_transform).

    This is what Predict.py must use instead of encode_features(): reusing
    fit_transform on new data would assign fresh, inconsistent codes (e.g.
    'Female' -> 0 on one run, -> 1 on another) instead of the codes the
    model was actually trained on.

    Categories the model has never seen (a new Patient_State, a typo,
    etc.) can't be transformed and would raise ValueError, so they are
    mapped to a stable fallback (the encoder's alphabetically-first known
    class) with a printed warning rather than crashing.
    """
    df = df.copy()

    for col, le in encoders.items():
        if col not in df.columns:
            continue

        values = df[col].astype(str)
        known = set(le.classes_)
        unseen_mask = ~values.isin(known)

        if unseen_mask.any():
            fallback = le.classes_[0]
            unseen_values = sorted(values[unseen_mask].unique())
            print(
                f"[preprocessing] WARNING: '{col}' has {unseen_mask.sum()} rows with "
                f"values unseen during training {unseen_values} -> mapped to "
                f"fallback '{fallback}'"
            )
            values = values.where(~unseen_mask, fallback)

        df[col] = le.transform(values)

    return df


def split_data(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """Stratified 80/20 train/test split."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[preprocessing] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def preprocess(df: pd.DataFrame):
    """Run the full preprocessing pipeline and return a train/test split
    plus the fitted encoders.
    """
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    from load_data import load_data

    raw = load_data()
    X_train, X_test, y_train, y_test, encoders = preprocess(raw)
    print(X_train.head())
