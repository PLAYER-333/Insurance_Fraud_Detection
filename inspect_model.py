"""
inspect_model.py

Prints the exact feature names (and count) the trained Random Forest
expects, straight from the pickled model itself.

Run:
    python inspect_model.py outputs/random_forest_model.pkl
"""

import pickle
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/random_forest_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    print(f"Model expects {len(model.feature_names_in_)} features, in this exact order:\n")
    for i, name in enumerate(model.feature_names_in_, 1):
        print(f"{i:2d}. {name}")
else:
    print("This model has no feature_names_in_ attribute — it was likely fit on a")
    print("plain NumPy array rather than a DataFrame, so column names weren't recorded.")
    print(f"n_features_in_: {getattr(model, 'n_features_in_', 'unknown')}")