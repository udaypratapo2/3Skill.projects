import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#--- ASSUMING YOU ALREADY DID THIS PART CORRECTLY ---
data = pd.read_csv("dataset_12000_records.csv")
X = data.drop(columns="Readmitted_30_Days")
y = data["Readmitted_30_Days"]
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
#----------------------------------------------------

# ==========================================
# 1. CORRECT SCALING STEP (Fixes the ValueError)
# ==========================================
scaler = StandardScaler()

# CRITICAL: fit_transform on TRAIN, and only transform on TEST
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quick check to ensure it's a 2D array (e.g., shape like (8400, 45))
print("X_train_scaled shape:", X_train_scaled.shape)
print("Type of X_train_scaled:", type(X_train_scaled))

# ==========================================
# 2. Finding the Optimal 'k' for KNN
# ==========================================
k_range = range(1, 11)
train_scores = []
test_scores = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    
    knn_temp.fit(X_train_scaled, y_train)
    
    train_scores.append(accuracy_score(y_train, knn_temp.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn_temp.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(k_range, train_scores, label='Train Accuracy', marker='o')
plt.plot(k_range, test_scores, label='Test Accuracy', marker='s')
plt.title('KNN: Train vs Test Accuracy for different values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.legend()
plt.grid(True)
plt.show()

optimal_k = 9  
print(f"\nSelected optimal k: {optimal_k}")

# ==========================================
# 3. Train Final KNN Model
# ==========================================
knn_model = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

print("KNN Model trained successfully!")