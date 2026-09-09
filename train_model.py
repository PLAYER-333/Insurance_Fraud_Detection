"""
train_model.py
---------------
Step 4 of the pipeline: train the primary RandomForestClassifier plus
Logistic Regression and Decision Tree baselines for comparison.
"""

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_PATH, RANDOM_STATE


def train_models(X_train, y_train) -> dict:
    """Train RandomForest (primary) plus two baseline models.

    Returns a dict {model_name: fitted_model}.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        # Scaled, since unscaled claim-amount-scale features make lbfgs
        # struggle to converge for logistic regression.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                max_iter=1000,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[train_model] Trained {name}")

    return models


def save_model(model, path: str = MODEL_PATH):
    """Pickle the trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train_model] Saved model to {path}")


if __name__ == "__main__":
    from load_data import load_data
    from preprocessing import preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, encoders = preprocess(df)
    models = train_models(X_train, y_train)
    save_model(models["Random Forest"])
