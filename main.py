"""
main.py
-------
Entry point. Runs the full pipeline exactly as laid out in the project
notes:

    Load CSV -> Clean Data -> Feature Engineering -> EDA ->
    Train/Test Split -> Random Forest Training -> Prediction ->
    Evaluation -> Save Graphs -> Save Model

Run with:
    python main.py
"""

import pickle

from config import MODEL_PATH, OUTPUT_DIR
from load_data import load_data
from preprocessing import clean_data, encode_features, engineer_features, split_data
from eda import run_eda
from train_model import save_model, train_models
from evaluate import compare_models, evaluate_model, plot_primary_model_diagnostics


def main():
    print("=" * 60)
    print("HEALTHCARE INSURANCE FRAUD DETECTION DS/ML MODEL")
    print("=" * 60)

    # 1. Load CSV
    df = load_data()

    # 2. Clean Data
    df = clean_data(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. EDA (on cleaned, human-readable categorical values)
    run_eda(df)

    # 5. Encode + Train/Test Split
    df_encoded, encoders = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Persist the fitted encoders so Predict.py can encode new data the
    # same way later (transform, not fit_transform) instead of guessing.
    encoders_path = f"{OUTPUT_DIR}/encoders.pkl"
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"[main] Saved encoders to {encoders_path}")

    # 6. Random Forest Training (+ Logistic Regression / Decision Tree baselines)
    models = train_models(X_train, y_train)

    # 7. Prediction + Evaluation for every model
    all_metrics = [
        evaluate_model(model, X_test, y_test, name)
        for name, model in models.items()
    ]
    comparison = compare_models(all_metrics)

    # 8. Save Graphs (confusion matrix + feature importance for the primary model)
    primary_model = models["Random Forest"]
    plot_primary_model_diagnostics(primary_model, X_test, y_test)

    # 9. Save Model
    save_model(primary_model, MODEL_PATH)

    print("=" * 60)
    print(f"Pipeline complete. Best model by ROC-AUC:")
    print(comparison["roc_auc"].idxmax(), "->", round(comparison["roc_auc"].max(), 3))
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
