"""
load_data.py
------------
Step 1 of the pipeline: load the raw claims CSV and report basic
shape/health info so problems are visible immediately.
"""

import pandas as pd
from config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw insurance claims CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[Load Data] Loaded '{path}'")
    print(f"[Load Data] Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    missing = df.isnull().sum() #Count Missing Values
    missing = missing[missing > 0]
    if len(missing):
        print("[Load Data] Missing values by column:")
        for col, n in missing.items():
            print(f"\n{col}: {n}")
    else:
        print("[Load Data] No missing values detected.")

    dupes = df.duplicated().sum()
    print(f"[Load Data] Duplicate rows: {dupes}")

    return df


df = load_data()

print("Heads of Data: ")
print(df.head(),end=" ") #First 5 
print("Tails of Data: ")
print(df.tail(),end=" ") #Last 5 