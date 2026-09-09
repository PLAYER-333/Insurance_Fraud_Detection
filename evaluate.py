"""
evaluate.py
-----------
Step 5 of the pipeline: evaluate trained models on the held-out test
set (accuracy, precision, recall, F1, ROC-AUC), and plot the confusion
matrix + feature importance for the primary Random Forest model.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import OUTPUT_DIR
from visualize import plot_confusion_matrix_graph, plot_feature_importance_graph


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute the standard classification metrics for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"[evaluate] {model_name}: "
          f"accuracy={metrics['accuracy']:.3f} "
          f"precision={metrics['precision']:.3f} "
          f"recall={metrics['recall']:.3f} "
          f"f1={metrics['f1_score']:.3f} "
          f"roc_auc={metrics['roc_auc']:.3f}")

    return metrics


def compare_models(all_metrics: list) -> pd.DataFrame:
    """Build and save a comparison table across all trained models."""
    comparison = pd.DataFrame(all_metrics).set_index("model")
    path = f"{OUTPUT_DIR}/model_comparison.csv"
    comparison.to_csv(path)
    print(f"[evaluate] Saved model comparison to {path}")
    print(comparison)
    return comparison


def plot_primary_model_diagnostics(model, X_test, y_test, model_name="Random Forest"):
    """Confusion matrix + feature importance for the primary model."""
    y_pred = model.predict(X_test)

    plot_confusion_matrix_graph(
        y_test, y_pred, "confusion_matrix.png",
        title=f"Confusion Matrix - {model_name}",
    )

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        plot_feature_importance_graph(
            importances, "feature_importance.png",
            title=f"Feature Importance - {model_name}",
        )
        imp_path = f"{OUTPUT_DIR}/feature_importance.csv"
        importances.sort_values(ascending=False).to_csv(imp_path, header=["importance"])
        print(f"[evaluate] Saved feature importances to {imp_path}")


if __name__ == "__main__":
    from load_data import load_data
    from preprocessing import preprocess
    from train_model import train_models

    df = load_data()
    X_train, X_test, y_train, y_test, encoders = preprocess(df)
    models = train_models(X_train, y_train)

    all_metrics = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]
    compare_models(all_metrics)
    plot_primary_model_diagnostics(models["Random Forest"], X_test, y_test)
