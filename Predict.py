"""
Predict.py

Usage:
    python Predict.py new_claims.csv

Scores new claims for fraud with the trained Random Forest model.

Unlike the original version, this runs incoming data through the SAME
cleaning -> feature-engineering -> encoding pipeline used at training
time (preprocessing.py) before calling the model. Feeding the model a
raw CSV directly (with ID columns, a raw date string, and un-encoded
text categories) is what caused the earlier
"feature names should match those that were passed during fit" error.
"""

import sys
import pickle

import pandas as pd

from config import ID_COLUMNS, MODEL_PATH, OUTPUT_DIR, TARGET_COLUMN
from preprocessing import clean_data, encode_new_data, engineer_features

ENCODERS_PATH = f"{OUTPUT_DIR}/encoders.pkl"


def main():
    if len(sys.argv) != 2:
        print("Usage: python Predict.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        print(f"[Predict] ERROR: {ENCODERS_PATH} not found.")
        print("[Predict] Run 'python fit_and_save_encoders.py' once first.")
        sys.exit(1)

    raw = pd.read_csv(csv_file)

    # The target column won't be present for real new claims, but if
    # someone runs this on a labeled/held-out file for a sanity check,
    # drop it so it isn't accidentally treated as a feature.
    df = raw.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Same steps used during training, in the same order.
    df = clean_data(df)
    df = engineer_features(df)              # drops ID cols, decomposes the date
    df = encode_new_data(df, encoders)       # transform-only, using TRAINING encoders

    # Guarantee the exact column set/order the model was fit on.
    missing = [c for c in model.feature_names_in_ if c not in df.columns]
    if missing:
        print(f"[Predict] ERROR: input CSV is missing required columns: {missing}")
        sys.exit(1)
    X = df[list(model.feature_names_in_)]

    predictions = model.predict(X)
    probabilities = (
        model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    )

    results = raw.copy()
    if probabilities is not None:
        results["Fraud_Probability"] = probabilities.round(4)
    results["Prediction"] = predictions
    results["Prediction_Label"] = results["Prediction"].map({0: "Not Fraud", 1: "Fraud"})

    print("\nPredictions\n")
    id_cols_present = [c for c in ID_COLUMNS if c in results.columns]
    preview_cols = id_cols_present + ["Prediction_Label"] + (
        ["Fraud_Probability"] if probabilities is not None else []
    )
    print(results[preview_cols].to_string(index=False))

    output_file = "predictions_output.csv"
    results.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
