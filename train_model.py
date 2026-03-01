"""
Personal Loan Predictor - Model Training
Generates synthetic data and trains a Random Forest classifier.
Run this script first to create the model file: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import pickle
import json

np.random.seed(42)
N = 5000

# Simulate realistic loan applicant data
age = np.random.randint(21, 70, N)
income = np.random.normal(60000, 25000, N).clip(15000, 250000)
employment_years = np.random.randint(0, 30, N)
credit_score = np.random.normal(680, 80, N).clip(300, 850).astype(int)
existing_loans = np.random.randint(0, 5, N)
loan_amount = np.random.normal(15000, 8000, N).clip(1000, 50000)
loan_term = np.random.choice([12, 24, 36, 48, 60], N)
debt_to_income = (existing_loans * 500 + loan_amount / loan_term) / (income / 12)
debt_to_income = debt_to_income.clip(0, 1)
education = np.random.choice([0, 1, 2, 3], N, p=[0.15, 0.30, 0.40, 0.15])  # 0=HS,1=Some,2=BS,3=Grad
home_ownership = np.random.choice([0, 1, 2], N, p=[0.35, 0.45, 0.20])  # 0=rent,1=own,2=mortgage

# Build approval probability with realistic rules
approval_score = (
    (credit_score - 300) / 550 * 0.35
    + (income / 250000) * 0.20
    + (employment_years / 30) * 0.10
    + (1 - debt_to_income) * 0.20
    + (education / 3) * 0.05
    + (home_ownership / 2) * 0.05
    + ((age - 21) / 49) * 0.05
)
noise = np.random.normal(0, 0.08, N)
approval_prob = (approval_score + noise).clip(0, 1)
approved = (approval_prob > 0.48).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "employment_years": employment_years,
    "credit_score": credit_score,
    "existing_loans": existing_loans,
    "loan_amount": loan_amount.astype(int),
    "loan_term": loan_term,
    "debt_to_income": debt_to_income.round(4),
    "education": education,
    "home_ownership": home_ownership,
    "approved": approved,
})

X = df.drop("approved", axis=1)
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_prob)

print("=== Model Performance ===")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {auc:.4f}")

# Feature importances
feature_names = X.columns.tolist()
importances = model.named_steps["clf"].feature_importances_
feat_imp = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

# Save model + metadata
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

metadata = {
    "accuracy": round(report["accuracy"], 4),
    "auc_roc": round(auc, 4),
    "precision": round(report["1"]["precision"], 4),
    "recall": round(report["1"]["recall"], 4),
    "f1": round(report["1"]["f1-score"], 4),
    "feature_importances": feat_imp,
    "approval_rate": round(float(y.mean()), 4),
    "training_samples": len(X_train),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved to model.pkl")
print("Metadata saved to model_metadata.json")
print(f"Feature importances: {feat_imp}")
