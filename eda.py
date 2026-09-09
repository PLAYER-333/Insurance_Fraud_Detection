"""
eda.py
------
Step 3 of the pipeline: exploratory data analysis. Produces the
distribution/summary graphs and prints summary statistics. Actual
plotting is delegated to visualize.py.
"""

import pandas as pd
from config import OUTPUT_DIR, TARGET_COLUMN
from visualize import (
    plot_boxplot,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_histogram,
)


def run_eda(df: pd.DataFrame):
    """Generate all EDA graphs and a summary_statistics.csv into outputs/."""

    plot_class_distribution(
        df[TARGET_COLUMN], "fraud_distribution.png",
        title="Fraud vs Non-Fraud Claims", xlabel="Is_Fraud",
        labels={0: "Not Fraud", 1: "Fraud"},
    )

    plot_histogram(
        df["Patient_Age"], "age_distribution.png",
        title="Patient Age Distribution", xlabel="Age",
    )

    plot_histogram(
        df["Claim_Amount"], "claim_distribution.png",
        title="Claim Amount Distribution", xlabel="Claim Amount ($)",
    )

    plot_histogram(
        df["Approved_Amount"], "approved_distribution.png",
        title="Approved Amount Distribution", xlabel="Approved Amount ($)",
    )

    plot_boxplot(
        df["Claim_Amount"], "claim_boxplot.png",
        title="Claim Amount Box Plot", xlabel="Claim Amount ($)",
    )

    plot_correlation_heatmap(df, "correlation_heatmap.png")

    summary = df.describe(include="all").transpose()
    summary_path = f"{OUTPUT_DIR}/summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"[eda] Saved summary statistics to {summary_path}")

    print("[eda] EDA complete.")


if __name__ == "__main__":
    from load_data import load_data

    df = load_data()
    run_eda(df)
