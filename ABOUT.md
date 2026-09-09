# About: Healthcare Insurance Fraud Detection

This repository contains a complete, end-to-end Machine Learning pipeline designed to detect fraudulent healthcare insurance claims. 

## Project Overview

Healthcare insurance fraud is a significant financial challenge that impacts the entire medical ecosystem. This project leverages historical claim data (containing 10,000 real claim records with an ~8.3% fraud rate) to train machine learning models capable of identifying suspicious patterns and flagging potentially fraudulent claims. 

## How It Works

The automated pipeline handles everything from raw data ingestion to final model serialization. The workflow is broken down into modular steps:

1. **Data Loading & Preprocessing**: Reads the raw data, handles missing values (imputation using median/mode), removes duplicates, engineers new features from dates (e.g., extracting year/month/day), and encodes categorical text variables into a machine-readable format.
2. **Exploratory Data Analysis (EDA)**: Automatically generates visualizations to help understand data distributions, outliers, and feature correlations (saved in the `outputs/` folder).
3. **Model Training**: Trains multiple baseline algorithms to handle the severe class imbalance (fraud is rare). It specifically evaluates a **Random Forest Classifier** (the primary model), Logistic Regression, and a Decision Tree. 
4. **Evaluation**: Tests the models on a hold-out test set using metrics critical for imbalanced datasets: Precision, Recall, F1-Score, and ROC-AUC. 
5. **Serialization & Diagnostics**: Saves the best-performing model (`random_forest_model.pkl`) to disk so it can be deployed on new data without retraining. It also exports diagnostic charts like the Confusion Matrix and a Feature Importance graph to explain *why* the model makes its decisions.

## Technology Stack

*   **Language**: Python 3
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn
*   **Data Visualization**: Matplotlib, Seaborn

## More Details

This is a complete, well-structured machine learning pipeline designed to detect healthcare insurance fraud. Based on the code in the project, here is a comprehensive breakdown of everything from the algorithms and libraries used to a detailed file-by-file explanation of how the logic works.

1. Algorithms Used
The project uses three different machine learning classification algorithms to predict whether a claim is fraudulent. It compares them and selects the best one:

Random Forest Classifier (Primary Model): An ensemble learning method that constructs multiple decision trees and merges them together to get a more accurate and stable prediction. It is configured with n_estimators=200 (200 trees) and class_weight="balanced" to handle the minority class (fraud cases which are only ~8.3% of the data).
Logistic Regression: A statistical model that models the probability of a binary class. In this project, it is used as a baseline for comparison. Because Logistic Regression is sensitive to the scale of input features, it is wrapped in a Pipeline that first scales the data using StandardScaler.
Decision Tree Classifier: A simpler algorithm that splits the data into branches based on feature values. It is used as another baseline model to benchmark the Random Forest.

2. Libraries Used and How They Are Applied
Scikit-Learn (sklearn): The core machine learning library used for modeling and data preparation.
Preprocessing: Uses LabelEncoder to convert categorical text data into numbers, and StandardScaler to normalize numerical data before passing it to Logistic Regression.
Splitting: Uses train_test_split with stratify=y to split the data (80% training, 20% testing) while maintaining the 8.3% fraud ratio in both sets.
Modeling: Imports the algorithms RandomForestClassifier, LogisticRegression, and DecisionTreeClassifier.
Metrics: Uses it to calculate performance scores (Accuracy, Precision, Recall, F1, ROC-AUC) and the Confusion Matrix.
Pandas (pandas): Used extensively for data manipulation. It handles loading the CSV dataset into a DataFrame, filling missing values (imputation), dropping duplicates, engineering new date features, creating summary statistics, and writing results to CSV files.
Matplotlib (matplotlib.pyplot) & Seaborn (seaborn): Used for Data Visualization (EDA and Evaluation graphs). Seaborn provides an aesthetically pleasing layer over Matplotlib to draw heatmaps (for correlations and confusion matrices) and distributions.
Pickle (pickle): A built-in Python library used to serialize ("save") the trained Random Forest model to disk as a .pkl file so it can be loaded later without retraining.

3. Evaluation Metrics and Matrices
Because fraud detection is highly imbalanced (most claims are not fraud), Accuracy alone isn't enough. The project uses the following metrics:

Confusion Matrix: A grid showing True Positives (correctly predicted fraud), True Negatives (correctly predicted non-fraud), False Positives (falsely accused of fraud), and False Negatives (missed fraud).
Precision: Out of all claims the model flagged as fraud, what percentage were actually fraud?
Recall: Out of all the actual fraud cases in the data, what percentage did the model successfully find? (Highly critical in fraud detection).
F1-Score: The harmonic mean of Precision and Recall, giving a single balance score.
ROC-AUC: Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between the Fraud and Non-Fraud classes. A score near 1.0 means perfect separation.

4. File-by-File Breakdown & Logic Flow
Here is exactly how the pipeline logic executes, broken down by file:

config.py (The Settings File)
Acts as the central configuration hub. It stores constants like file paths, the RANDOM_STATE (for reproducibility), and lists categorizing which columns are IDs, Dates, Categoricals, and the Target (Is_Fraud). This ensures all other files agree on the rules.

main.py (The Orchestrator)
This is the entry point. When you run python main.py, it executes the pipeline sequentially:

Calls load_data()
Calls clean_data() and engineer_features()
Calls run_eda()
Calls encode_features() and split_data()
Calls train_models()
Calls evaluate_model() and compares them
Calls plot_primary_model_diagnostics() and save_model()
load_data.py (Step 1: Ingestion)
Loads insurance_fraud.csv using Pandas. It prints out a preliminary "health check" of the data: the shape (rows and columns), the number of missing values per column, and the number of duplicate rows.

preprocessing.py (Step 2: Cleaning & Prep)
Contains the heavy lifting for data transformations:

clean_data(): Removes duplicate rows. It fills missing numeric values with the median of that column, and missing categorical values with the mode (most frequent item).
engineer_features(): Takes the Claim_Submission_Date (e.g., "01-09-2024") and splits it into three new columns: Submission_Year, Submission_Month, and Submission_Day. It then drops the original date column and drops Provider_ID and Claim_ID since random identifiers do not help predict fraud.
encode_features(): Uses Scikit-learn's LabelEncoder to convert text-based categories (like Patient_Gender or Claim_Status) into integers so the math algorithms can process them.
split_data(): Splits the cleaned, encoded dataset into training (80%) and testing (20%) sets.
eda.py (Step 3: Exploratory Data Analysis)
Analyzes the pre-encoded data. It calls plotting functions to visualize the balance of Fraud vs. Non-Fraud, histograms of patient ages and claim amounts, a boxplot to spot outliers in claim amounts, and a correlation heatmap. It also generates a statistical summary (df.describe()) and saves it as a CSV.

visualize.py (The Plotting Helper)
A utility file containing all the Matplotlib/Seaborn code. By moving all graphing logic here, files like eda.py and evaluate.py stay uncluttered. It uses an "Agg" backend to safely generate and save plots directly to the outputs/ folder as .png images without attempting to open pop-up windows.

train_model.py (Step 4: Machine Learning)
Initializes the three models (Random Forest, Logistic Regression, Decision Tree). It applies StandardScaler strictly to the Logistic Regression pipeline. It fits (trains) all three models using the 80% X_train and y_train data. Finally, it provides a save_model() function that uses pickle to save the trained Random Forest to disk.

evaluate.py (Step 5: Scoring)
Takes the trained models and tests them using the unseen 20% X_test data.

It generates predictions and calculates the Accuracy, Precision, Recall, F1, and ROC-AUC.
It builds a comparison CSV table of all three models.
For the Random Forest model specifically, it generates the Confusion Matrix graph and a Feature Importance graph (which tells you mathematically which data columns—like Claim Status or the gap between claim/approved amounts—were most suspicious to the AI).