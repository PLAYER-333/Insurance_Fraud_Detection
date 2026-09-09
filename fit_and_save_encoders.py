"""
fit_and_save_encoders.py
--------------------------
One-off fix: the original pipeline (main.py / train_model.py) trained
random_forest_model.pkl but never saved the LabelEncoders it used, so
Predict.py had no consistent way to encode new categorical data.

This script reconstructs those encoders WITHOUT retraining the model.

Why that's safe: LabelEncoder.fit() sorts a column's unique values
alphabetically and assigns 0..n-1 in that order. That mapping depends
only on the *set* of category values in insurance_fraud_training.csv,
not on row order or anything random. So re-running the same
clean -> engineer -> encode steps against the same training CSV
reproduces byte-for-byte identical encoders to the ones used when the
model was originally trained.

Run once (or any time insurance_fraud_training.csv changes and the
model is retrained):

    python fit_and_save_encoders.py
"""

import pickle

from config import DATA_PATH, OUTPUT_DIR
from load_data import load_data
from preprocessing import clean_data, encode_features, engineer_features

ENCODERS_PATH = f"{OUTPUT_DIR}/encoders.pkl"

df = load_data(DATA_PATH)
df = clean_data(df)
df = engineer_features(df)
_, encoders = encode_features(df)

with open(ENCODERS_PATH, "wb") as f:
    pickle.dump(encoders, f)

print(f"\n[fit_and_save_encoders] Saved {len(encoders)} encoders to {ENCODERS_PATH}")
for col, le in encoders.items():
    print(f"  {col}: {list(le.classes_)}")
