# svm_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer # Importing this as a backup/reference

try:
    df = pd.read_csv('breast_cancer.csv')
except FileNotFoundError:
    print("CSV file not found. Falling back to Scikit-learn's built-in Breast Cancer Dataset.")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    df = df.rename(columns={'target': 'diagnosis'}) 
    print("Successfully loaded Scikit-learn's built-in dataset.")

print("\n--- Data Head ---")
print(df.head())

print("\n--- Data Info ---")
df.info()
target_column_name = 'diagnosis' # Check your CSV and change this if needed

if target_column_name in df.columns:
    print(f"\nTarget Variable distribution ({target_column_name}):")
    print(df[target_column_name].value_counts())
else:
    print(f"\nWARNING: Target column '{target_column_name}' not found. Please inspect your CSV columns and update 'target_column_name'.")
