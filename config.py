"""
config.py
---------
Shared constants used across the pipeline so every module agrees on
paths, the random seed, and column roles.
"""

import os

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_PATH = os.path.join(os.path.dirname(__file__), "insurance_fraud_training.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")

TARGET_COLUMN = "Is_Fraud"

# Columns that identify a record but carry no predictive signal
ID_COLUMNS = ["Provider_ID", "Claim_ID"]

# Raw date column that gets decomposed into Submission_Year/Month/Day
DATE_COLUMN = "Claim_Submission_Date"
DATE_FORMAT = "%d-%m-%Y"  # source data is day-first, e.g. 01-09-2024

# Categorical columns that need label encoding before modeling
CATEGORICAL_COLUMNS = [
    "Patient_Gender",
    "Diagnosis_Code",
    "Insurance_Type",
    "Provider_Specialty",
    "Patient_State",
    "Claim_Status",
    "Visit_Type",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
