"""
visualize.py
------------
Reusable plotting helpers. Every graph in the project is produced by a
function in this module and saved into outputs/, so eda.py and
evaluate.py stay focused on *what* to plot rather than *how*.
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import OUTPUT_DIR

sns.set_style("whitegrid")


def _save(fig, filename):
    path = f"{OUTPUT_DIR}/{filename}"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[visualize] Saved {path}")


def plot_class_distribution(series: pd.Series, filename: str, title: str,
    xlabel: str, labels=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = series.value_counts().sort_index()
    if labels:
        counts.index = [labels.get(i, i) for i in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _save(fig, filename)


def plot_histogram(series: pd.Series, filename: str, title: str, xlabel: str, bins=30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.dropna(), bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    _save(fig, filename)


def plot_boxplot(series: pd.Series, filename: str, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(series.dropna(), vert=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    _save(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename: str, title: str = "Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title(title)
    _save(fig, filename)


def plot_confusion_matrix_graph(y_true, y_pred, filename: str, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, filename)


def plot_feature_importance_graph(importances: pd.Series, filename: str,
    title: str = "Feature Importance", top_n: int = 15):
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    _save(fig, filename)
