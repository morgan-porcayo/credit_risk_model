#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:06:44 2025

@author: morganporcayo
"""

import joblib
import pandas as pd

# Load the trained pipeline
model_path = "/Users/morganporcayo/Desktop/credit_model_risk/credit_model.joblib"
model = joblib.load(model_path)

# New applicant
new_applicant = pd.DataFrame([{
    "person_age": 40,
    "person_gender": "male",
    "person_education": "Master",
    "person_income": 10000,
    "person_emp_exp": 5,
    "person_home_ownership": "OWN",
    "loan_amnt": 35000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 6.7,
    "loan_percent_income": (35000)/(10000*12),
    "cb_person_cred_hist_length": 5,
    "credit_score": 600,
    "previous_loan_defaults_on_file": "No"
}])

# Make prediction
prediction = model.predict(new_applicant)
print("Approved!" if prediction[0] == 1 else "Rejected")