import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer
target_column_name = 'diagnosis' 

# --- 1. Load and Prepare the Dataset ---
print("--- 1. Loading Dataset ---")

try:
    df = pd.read_csv('breast-cancer.csv')
    print(f"Successfully loaded external CSV: 'breast-cancer.csv'")
except FileNotFoundError:
    print("CSV file not found. Falling back to Scikit-learn's built-in Breast Cancer Dataset.")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    df = df.rename(columns={'target': target_column_name})
    print("Successfully loaded Scikit-learn's built-in dataset.")

print("\n--- Data Head ---")
print(df.head())

# --- 2. Data Cleaning and Preparation ---
print("\n--- 2. Data Preparation ---")

columns_to_drop = ['id', 'Unnamed: 32'] 
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

# X (Features): Drop the target column and any unnecessary columns
X = df.drop(columns=[target_column_name] + existing_cols_to_drop, axis=1)

# y (Target): The column we want to predict (e.g., 'diagnosis')
y = df[target_column_name]

# Convert target variable to numeric if it's categorical (e.g., 'M' and 'B')
if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=target_column_name)
    print("Encoded target variable to numeric (0 and 1).")

print(f"Target Variable distribution ({target_column_name}):")
print(y.value_counts())


# --- 3. Split Data into Training and Testing Sets ---
print("\n--- 3. Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y 
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("\n--- 4. Feature Scaling ---")

scaler = StandardScaler()

# Fit the scaler only on the training data and transform both sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data splitting and feature scaling complete.")


# --- 5. Train and Evaluate Linear SVM ---
print("\n--- 5. Training and Evaluating Linear SVM ---")

# Initialize and train the SVM classifier with a linear kernel
linear_svc = SVC(kernel='linear', C=1.0, random_state=42)
linear_svc.fit(X_train, y_train)

# Prediction and Evaluation (Linear SVM)
y_pred_linear = linear_svc.predict(X_test)

print("\n[Linear SVM] Evaluation on Test Set:")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))

# Classification Report (Metrics)
print("\nClassification Report (Accuracy, Precision, Recall, F1-score):")
print(classification_report(y_test, y_pred_linear, digits=4))


print("\n--- Training RBF Kernel SVM ---")

# Initialize the SVM classifier with an RBF kernel
# C=1.0 and gamma='scale' are good default starting points
rbf_svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model
rbf_svc.fit(X_train, y_train)

print("RBF Kernel SVM training complete.")


# --- 7. Prediction and Evaluation (RBF Kernel SVM) ---

# Predict the labels on the test set
y_pred_rbf = rbf_svc.predict(X_test)

# Evaluate the model performance
print("\n[RBF Kernel SVM] Evaluation on Test Set:")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))

# Classification Report (Metrics)
print("\nClassification Report (Accuracy, Precision, Recall, F1-score):")
print(classification_report(y_test, y_pred_rbf, digits=4))