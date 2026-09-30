from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)


from sklearn.metrics import accuracy_score


# ==========================================
# 1. Train Decision Tree with Constraints (Answers Q5)
# ==========================================
# We set constraints to prevent the tree from growing too deep and overfitting.
# class_weight='balanced' helps the tree handle the imbalanced readmission target.
dt_model = DecisionTreeClassifier(
    max_depth=5,                # Limits the maximum number of splits (prevents overly complex trees)
    min_samples_split=20,       # A node must have at least 20 samples to be considered for a split
    min_samples_leaf=10,        # A leaf node must have at least 10 samples (smooths the model)
    class_weight='balanced',    # Penalizes mistakes on the minority class (readmitted) more heavily
    random_state=42
)
data = pd.read_csv("dataset_12000_records.csv")
X = data.drop(columns="Readmitted_30_Days")
y = data["Readmitted_30_Days"]
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()

# CRITICAL: fit_transform on TRAIN, and only transform on TEST
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_model.fit(X_train_scaled, y_train) # Note: Scaling isn't strictly needed for Trees, but harmless and keeps pipeline consistent

# ==========================================
# 2. Predictions and Evaluation
# ==========================================
y_train_pred_dt = dt_model.predict(X_train_scaled)
y_test_pred_dt = dt_model.predict(X_test_scaled)
y_test_prob_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate using the function from the previous step

# ==========================================
# 3. (Optional but highly recommended) Feature Importance
# ==========================================
# This helps with model interpretability, which is crucial in healthcare.
feature_importances = pd.Series(dt_model.feature_importances_, index=X_encoded.columns)
top_features = feature_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Most Important Features for Decision Tree:")
print(top_features)
def evaluate_model(model_name, y_true_train, y_pred_train, y_true_test, y_pred_test, y_prob_test):
    print(f"========== {model_name} ==========")
    print(f"Train Accuracy: {accuracy_score(y_true_train, y_pred_train)}")
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

# Plot the top features
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title('Top 10 Feature Importances (Decision Tree)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis() # Highest importance at the top
plt.show()
evaluate_model("Decision Tree", y_train, y_train_pred_dt, y_test, y_test_pred_dt, y_test_prob_dt)
