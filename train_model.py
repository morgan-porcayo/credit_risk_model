#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:43:58 2025

@author: morganporcayo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load dataset
path = "/Users/morganporcayo/Downloads/loan_data.csv"
df = pd.read_csv(path)

# 2. Split features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 3. Identify numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# 4. Preprocessing
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# 5. Build pipeline with preprocessing + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 9. Predict for a new applicant
new_applicant = pd.DataFrame([{
    "person_age": 26,
    "person_gender": "male",
    "person_education": "Master",
    "person_income": 23000,
    "person_emp_exp": 0,
    "person_home_ownership": "RENT",
    "loan_amnt": 35000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 6.7,
    "loan_percent_income": (35000)/(23000*12),
    "cb_person_cred_hist_length": 5,
    "credit_score": 600,
    "previous_loan_defaults_on_file": "No"
}])

prediction = model.predict(new_applicant)
print("Approved!" if prediction[0] == 1 else "Rejected")
                  
"""
import joblib

# Assuming `pipeline` is your trained scikit-learn Pipeline
joblib.dump(model, "credit_model.joblib")
"""