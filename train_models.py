"""
Telco Customer Churn - ML Classification Training Script
BITS Pilani M.Tech AIML - Machine Learning Assignment 2
Dataset: Telco Customer Churn (19 features, 7043 instances, binary classification)
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# 1. Load Dataset
print("=" * 65)
print("BITS Pilani – ML Assignment 2: Telco Customer Churn Classification")
print("=" * 65)
print("\nStep 1: Loading Telco Customer Churn dataset from local CSV...")

DATA_PATH = "raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Drop customerID (not a predictive feature)
df = df.drop(columns=['customerID'])

# Fix TotalCharges: some rows have whitespace instead of a number
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

print(f"  After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Features: {list(df.columns[:-1])}")
print(f"  Target distribution:\n{df['Churn'].value_counts().to_string()}")

# 2. Preprocessing
print("\nStep 2: Preprocessing...")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')

label_encoders = {}
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_le = LabelEncoder()
df_enc['Churn'] = target_le.fit_transform(df['Churn'])   # No=0, Yes=1
label_encoders['Churn'] = target_le

X = df_enc.drop('Churn', axis=1)
y = df_enc['Churn']
feature_names = list(X.columns)

print(f"  Feature count : {len(feature_names)}")
print(f"  Class mapping : {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}  (No=0, Yes=1)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 3. Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, C=1.0),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7, metric='minkowski'),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42, C=1.0, cache_size=800),
}

model_files = {
    'Logistic Regression': 'models/logistic_regression.joblib',
    'Decision Tree':       'models/decision_tree.joblib',
    'KNN':                 'models/knn.joblib',
    'Naive Bayes':         'models/naive_bayes.joblib',
    'SVM':                 'models/svm.joblib',
}

os.makedirs('models', exist_ok=True)
results = {}

# 4. Train, Evaluate, Save
print("\nStep 3: Training models…\n")
print(f"{'Model':<22} {'Acc':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        'Accuracy':  round(accuracy_score(y_test, y_pred),                            4),
        'AUC':       round(roc_auc_score(y_test, y_proba),                             4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0),           4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0),              4),
        'F1':        round(f1_score(y_test, y_pred, zero_division=0),                  4),
        'MCC':       round(matthews_corrcoef(y_test, y_pred),                          4),
    }
    results[name] = metrics

    print(f"{name:<22} {metrics['Accuracy']:>6} {metrics['AUC']:>6} "
          f"{metrics['Precision']:>6} {metrics['Recall']:>6} "
          f"{metrics['F1']:>6} {metrics['MCC']:>6}")

    joblib.dump(model, model_files[name])

print("-" * 65)

# 5. Save Preprocessing Artifacts
joblib.dump(scaler,         'models/scaler.joblib')
joblib.dump(label_encoders, 'models/label_encoders.joblib')
joblib.dump(feature_names,  'models/feature_names.joblib')
print("\nSaved: scaler, label_encoders, feature_names")

# 6. Save Test Data & Results
test_df = X_test.copy()
test_df['Churn'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print(f"Saved: test_data.csv  ({len(test_df)} rows)")

results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("Saved: model_results.csv")

print("\n  All done! Paste the metrics below into README.md\n")
print(results_df.to_string())
