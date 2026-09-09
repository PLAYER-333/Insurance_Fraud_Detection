"""
generate_csv.py
-----------------
Generates a synthetic batch of NEW insurance claims for Predict.py to
score. Matches the real project schema (see config.py) exactly:
same column names, same categories, same numeric ranges as
insurance_fraud_training.csv.

Run:
    python generate_csv.py

Output:
    new_claims.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_STATE = 7          # different seed from training data on purpose
N_ROWS = 200

rng = np.random.default_rng(RANDOM_STATE)

GENDERS = ["Male", "Female"]
DIAGNOSIS_CODES = ["E11.9", "E78.5", "F41.9", "I10", "I25.10", "J06.9", "J18.9", "K21.9", "M54.5", "N39.0"]
PROCEDURE_CODES = [36415, 99213, 93000, 85025, 80053, 97110, 71046, 87086, 99214]
INSURANCE_TYPES = ["Medicaid", "Medicare", "Private", "Self-Pay"]
PROVIDER_SPECIALTIES = ["Cardiology", "General Practice", "Internal Medicine", "Neurology", "Orthopedics", "Pulmonology"]
PATIENT_STATES = ["CA", "FL", "GA", "IL", "NY", "OH", "PA", "TX"]
CLAIM_STATUSES = ["Approved", "Pending", "Rejected"]
VISIT_TYPES = ["Emergency", "Inpatient", "Outpatient"]

n = N_ROWS

provider_ids = [f"P{rng.integers(0, 400):04d}" for _ in range(n)]
claim_ids = [f"C1{i:06d}" for i in range(n)]  # distinct prefix from training's C0...

patient_age = rng.integers(1, 96, size=n)
patient_gender = rng.choice(GENDERS, size=n)
diagnosis_code = rng.choice(DIAGNOSIS_CODES, size=n)
procedure_code = rng.choice(PROCEDURE_CODES, size=n)
insurance_type = rng.choice(INSURANCE_TYPES, size=n)
provider_specialty = rng.choice(PROVIDER_SPECIALTIES, size=n)
patient_state = rng.choice(PATIENT_STATES, size=n)
visit_type = rng.choice(VISIT_TYPES, size=n)
chronic_condition_flag = rng.choice([0, 1], size=n, p=[0.7, 0.3])
prior_visits_12m = rng.integers(0, 13, size=n)
length_of_stay = rng.integers(0, 10, size=n)
days_between_service_and_claim = rng.integers(0, 30, size=n)
number_of_claims_per_provider_monthly = rng.integers(1, 150, size=n)

claim_amount = np.round(rng.lognormal(mean=6.3, sigma=0.8, size=n), 2)
claim_amount = np.clip(claim_amount, 20, 20000)
approved_amount = np.round(claim_amount * rng.uniform(0.5, 1.0, size=n), 2)

claim_status = rng.choice(CLAIM_STATUSES, size=n, p=[0.6, 0.2, 0.2])

start_date = datetime(2025, 1, 1)
random_days = rng.integers(0, 250, size=n)
submission_dates = [(start_date + timedelta(days=int(d))).strftime("%d-%m-%Y") for d in random_days]

df = pd.DataFrame({
    "Provider_ID": provider_ids,
    "Claim_ID": claim_ids,
    "Patient_Age": patient_age,
    "Patient_Gender": patient_gender,
    "Diagnosis_Code": diagnosis_code,
    "Procedure_Code": procedure_code,
    "Claim_Amount": claim_amount,
    "Approved_Amount": approved_amount,
    "Insurance_Type": insurance_type,
    "Claim_Submission_Date": submission_dates,
    "Days_Between_Service_and_Claim": days_between_service_and_claim,
    "Number_of_Claims_Per_Provider_Monthly": number_of_claims_per_provider_monthly,
    "Provider_Specialty": provider_specialty,
    "Patient_State": patient_state,
    "Claim_Status": claim_status,
    "Length_of_Stay": length_of_stay,
    "Visit_Type": visit_type,
    "Chronic_Condition_Flag": chronic_condition_flag,
    "Prior_Visits_12m": prior_visits_12m,
})

OUT_PATH = "new_claims.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} new (unlabeled) claims to {OUT_PATH}")
print(df.head())
