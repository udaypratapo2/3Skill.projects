import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# 1. Load Data
data = pd.read_csv("dataset_12000_records.csv")

# 2. Initial Exploration & Data Leakage Check (Project Step A & B)
print("Dataset Shape:", data.shape)
print("\nTarget Distribution (Class Balance):\n", data["Readmitted_30_Days"].value_counts(normalize=True))

# ⚠️ CRITICAL PROJECT REQUIREMENT: Check for Data Leakage!
# Look at data.columns. If there are columns like 'readmission_diagnosis', 'days_until_readmission', 
# or 'post_discharge_follow_up_date', you MUST drop them before splitting.
# Example: data = data.drop(columns=['leaky_column_name'], errors='ignore')

for i in data.columns :
    if i in ['readmission_diagnosis', 'days_until_readmission','post_discharge_follow_up_date'] :
        data=data.drop(columns=f"{i}")

# 3. Separate Features and Target
X = data.drop(columns="Readmitted_30_Days")
y = data["Readmitted_30_Days"]

# 4. Preprocessing: Handle Missing Values (Step B)
# You must check for missing values first: print(X.isnull().sum())
# Example fix: X = X.fillna(X.median(numeric_only=True)) 

# 5. Preprocessing: One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# 6. Train/Test Split (Project Step C)
# 🛠️ FIX 1: Corrected the variable unpacking order (was X_train, Y_train, X_test, Y_test)
# 🛠️ FIX 2: Added stratify=y to ensure the train/test split maintains the same class imbalance ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# 7. Feature Scaling (Crucial for KNN, beneficial for Logistic Regression)
scaler = StandardScaler()
# 🛠️ FIX 3: Fit ONLY on training data to prevent data leakage from the test set into the scaler.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Logistic Regression (Project Step D)
# Added class_weight='balanced' to handle class imbalance, and max_iter to ensure convergence.
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 9. Predictions (Project Step E)
# We need predictions for BOTH train and test to diagnose overfitting/underfitting (Project Step F)
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

# We need probabilities for ROC-AUC
y_test_prob = lr_model.predict_proba(X_test_scaled)[:, 1] 

# 10. Evaluation Function (To be reused for KNN and Decision Tree)
def evaluate_model(model_name, y_true_train, y_pred_train, y_true_test, y_pred_test, y_prob_test):
    print(f"========== {model_name} ==========")
    print(f"Train Accuracy: {accuracy_score(y_true_train, y_pred_train):.4f}")
    print(f"Test Accuracy:  {accuracy_score(y_true_test, y_pred_test):.4f}")
    print("-" * 30)
    print(f"Test Precision: {precision_score(y_true_test, y_pred_test):.4f}")
    print(f"Test Recall:    {recall_score(y_true_test, y_pred_test):.4f}")
    print(f"Test F1-Score:  {f1_score(y_true_test, y_pred_test):.4f}")
    print(f"Test ROC-AUC:   {roc_auc_score(y_true_test, y_prob_test):.4f}")
    
    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(y_true_test, y_pred_test)
    print(cm)
    print("Interpretation: [[TN, FP],\n               [FN, TP]]")
    print("TN: Correctly predicted NOT readmitted")
    print("FP: Incorrectly predicted readmitted (False Alarm)")
    print("FN: Incorrectly predicted NOT readmitted (Missed High-Risk Patient) <- WORST in healthcare")
    print("TP: Correctly predicted readmitted\n")

# Run evaluation for Logistic Regression
evaluate_model("Logistic Regression", y_train, y_train_pred, y_test, y_test_pred, y_test_prob)

