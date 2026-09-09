# Healthcare Insurance Fraud Detection

A complete, working fraud-detection pipeline built from your project notes.
Primary model is a `RandomForestClassifier`, benchmarked against Logistic
Regression and a Decision Tree, trained on `insurance_fraud.csv`
(10,000 real claim records, ~8.3% fraud rate).

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

That's it — this runs the entire pipeline (load → clean → feature
engineer → EDA → split → train → evaluate → save graphs → save model)
and populates `outputs/`.

## Project structure

```
Insurance_Fraud_Detection/
├── main.py              # Orchestrates the full pipeline
├── config.py             # Shared paths, random seed, column roles
├── load_data.py          # Step 1: load CSV, report shape/missing/dupes
├── preprocessing.py      # Step 2: clean, engineer dates, encode, split
├── eda.py                # Step 3: exploratory graphs + summary stats
├── train_model.py        # Step 4: train RF / Logistic Regression / Decision Tree
├── evaluate.py            # Step 5: metrics, confusion matrix, feature importance
├── visualize.py           # Shared plotting helpers used by eda.py & evaluate.py
├── insurance_fraud.csv    # Your dataset
├── requirements.txt
└── outputs/               # Everything the pipeline produces (see below)
```

Each module also runs standalone for debugging, e.g. `python eda.py`
or `python train_model.py`.

## What each pipeline step does

1. **Load** — reads the CSV, prints shape, missing-value counts, duplicate count.
2. **Clean** — drops duplicate rows, fills numeric NaNs with the median,
   fills categorical NaNs with the mode.
3. **Feature engineering** — splits `Claim_Submission_Date` into
   `Submission_Year` / `Submission_Month` / `Submission_Day`, and drops
   `Provider_ID` / `Claim_ID` (identifiers, no predictive signal).
4. **EDA** — saves distribution and correlation graphs, plus
   `outputs/summary_statistics.csv`.
5. **Encode + split** — label-encodes categorical columns, stratified
   80/20 train/test split (`random_state=42`, since fraud is rare and
   we want the same ratio in both sets).
6. **Train** — fits Random Forest (`n_estimators=200`,
   `class_weight="balanced"`), Logistic Regression (scaled), and a
   Decision Tree, all class-weight-balanced to compensate for the
   ~8% fraud rate.
7. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC for every
   model, written to `outputs/model_comparison.csv`.
8. **Save graphs** — confusion matrix and feature importance for the
   Random Forest.
9. **Save model** — the trained Random Forest is pickled to
   `outputs/random_forest_model.pkl`.

## Outputs (`outputs/`)

| File | What it shows |
|---|---|
| `fraud_distribution.png` | Class balance (fraud vs. not) |
| `age_distribution.png` | Patient age histogram |
| `claim_distribution.png` | Claim amount histogram |
| `approved_distribution.png` | Approved amount histogram |
| `claim_boxplot.png` | Claim amount outliers |
| `correlation_heatmap.png` | Numeric feature correlations |
| `confusion_matrix.png` | Random Forest predictions vs. actual on the test set |
| `feature_importance.png` | Which features drive the Random Forest's predictions |
| `summary_statistics.csv` | `df.describe()` for every column |
| `model_comparison.csv` | Metrics table for all 3 models |
| `feature_importance.csv` | Feature importances, sorted |
| `random_forest_model.pkl` | The trained, ready-to-load model |

## Current results (on your data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.972 | 0.887 | 0.759 | 0.818 | 0.993 |
| Logistic Regression | 0.977 | 0.790 | 0.976 | 0.873 | 0.999 |
| Decision Tree | 0.975 | 0.862 | 0.825 | 0.843 | 0.907 |

All three separate fraud very cleanly on this dataset — `Claim_Status`
and the Claim/Approved-amount gap turn out to be the dominant signal
(see `feature_importance.png`). Logistic Regression edges out Random
Forest on ROC-AUC here, but Random Forest was kept as the saved/primary
model per the original spec — swap `primary_model` in `main.py` if you'd
rather ship Logistic Regression instead.

## Loading the saved model later

```python
import pickle
with open("outputs/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

model.predict(X_new)          # X_new must have the same encoded columns as X_train
model.predict_proba(X_new)    # fraud probability
```

Note: new incoming data needs the same cleaning/encoding steps
(`preprocessing.py`) applied before calling `.predict()` — the label
encoders are not currently saved to disk. If you'll be scoring new
claims regularly, let me know and I can add encoder persistence
(`pickle` the `encoders` dict from `preprocess()`) so new data can be
transformed consistently without retraining.

## Using your own updated data later

Replace `insurance_fraud.csv` with a new export using the same column
names and re-run `python main.py` — nothing else needs to change.
